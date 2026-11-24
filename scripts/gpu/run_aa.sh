#!/usr/bin/env bash
# A/A noise-floor measurement for one pinned system under test on a single-GPU Linux host.
#
# Usage: scripts/gpu/run_aa.sh <sut-config.json> <workload.json[.gz]> [runs]
#
# Environment (defaults in parentheses):
#   SLOGATE_MAX_MINUTES  wall-clock cap; on RunPod the pod is stopped when it expires (60)
#   SLOGATE_STOP_POD     1 stops the RunPod pod when this script exits (1)
#   SLOGATE_OUT          output directory (/workspace/slogate-runs/aa-<UTC time>)
#   SLOGATE_WORK         virtual environments and model cache (/workspace/slogate-work)
#   SLOGATE_PORT         local engine port (8000)
#   SLOGATE_TRACE        1 captures worker step/CUDA diagnostics; leave off for E0 (0)
#   ENGINE_PYTHON        interpreter for the engine virtual environment (python3.12)
set -euo pipefail

usage="usage: $0 <sut-config.json> <workload.json[.gz]> [runs]"
SUT_CONFIG=$(realpath "${1:?$usage}")
WORKLOAD=$(realpath "${2:?$usage}")
RUNS=${3:-6}
MAX_MINUTES=${SLOGATE_MAX_MINUTES:-60}
STOP_POD=${SLOGATE_STOP_POD:-1}
WORK=${SLOGATE_WORK:-/workspace/slogate-work}
OUT=${SLOGATE_OUT:-/workspace/slogate-runs/aa-$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${SLOGATE_PORT:-8000}
ENGINE_PYTHON=${ENGINE_PYTHON:-python3.12}
TRACE=${SLOGATE_TRACE:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

# Defines SUT_NAME, MODEL_ID, MODEL_REVISION and the SERVE_ARGS, ENGINE_REQUIREMENTS and
# ENGINE_PIP_ARGS arrays.
eval "$(python3 "$REPO/scripts/gpu/sut_env.py" "$SUT_CONFIG")"

mkdir -p "$OUT/host" "$WORK"
export HF_HOME="$WORK/hf"
SERVER_PID=""
WATCHDOG_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  local status=$?
  stop_server
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  echo "slogate: finished with status $status; results in $OUT" >&2
  if [[ "$STOP_POD" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
    runpodctl pod stop "$RUNPOD_POD_ID" || true
  fi
}
trap cleanup EXIT

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  (
    sleep "$((MAX_MINUTES * 60))"
    echo "slogate: ${MAX_MINUTES}-minute cap reached; stopping the pod" >&2
    runpodctl pod stop "$RUNPOD_POD_ID"
  ) &
  WATCHDOG_PID=$!
else
  echo "slogate: RUNPOD_POD_ID is unset, so no pod-level cost cap is active" >&2
fi

echo "slogate: preparing engine environment for $SUT_NAME" >&2
ENGINE_VENV="$WORK/venv-$SUT_NAME"
case "$TRACE" in
  0) unset SLOGATE_TRACE_DIR ;;
  1)
    if [[ -v VLLM_PLUGINS && ",${VLLM_PLUGINS:-}," != *,slogate,* ]]; then
      echo "SLOGATE_TRACE=1 requires slogate in VLLM_PLUGINS (or leave the list unset)" >&2
      exit 1
    fi
    ENGINE_VENV="${ENGINE_VENV}-trace"
    mkdir "$OUT/trace"
    SLOGATE_TRACE_DIR=$(cd "$OUT/trace" && pwd)
    export SLOGATE_TRACE_DIR
    ;;
  *) echo "SLOGATE_TRACE must be 0 (baseline) or 1 (diagnostic)" >&2; exit 1 ;;
esac
if [[ ! -x "$ENGINE_VENV/bin/python" ]]; then
  "$ENGINE_PYTHON" -m venv "$ENGINE_VENV"
fi
"$ENGINE_VENV/bin/python" -m pip install --quiet --upgrade pip
"$ENGINE_VENV/bin/python" -m pip install --quiet "${ENGINE_PIP_ARGS[@]}" "${ENGINE_REQUIREMENTS[@]}"
if [[ "$TRACE" == "1" ]]; then
  "$ENGINE_VENV/bin/python" -m pip install --quiet --no-deps "$REPO"
fi
"$ENGINE_VENV/bin/python" -m pip check
"$ENGINE_VENV/bin/python" -m pip freeze > "$OUT/host/engine-pip-freeze.txt"
"$ENGINE_VENV/bin/python" "$REPO/scripts/gpu/probe_engine.py" "$SUT_CONFIG" > "$OUT/host/engine.json"

echo "slogate: preparing client environment" >&2
CLIENT_VENV="$WORK/venv-slogate-client"
if [[ ! -x "$CLIENT_VENV/bin/python" ]]; then
  "$ENGINE_PYTHON" -m venv "$CLIENT_VENV"
fi
"$CLIENT_VENV/bin/python" -m pip install --quiet -c "$REPO/configs/client/e0-py312-linux.txt" "$REPO"
"$CLIENT_VENV/bin/python" -m pip check
"$CLIENT_VENV/bin/python" -m pip list --format=freeze > "$OUT/host/client-packages.txt"

python3 "$REPO/scripts/gpu/collect_host.py" "$SUT_CONFIG" "$OUT/host/engine.json" \
  > "$OUT/sut-info.json"
cp "$SUT_CONFIG" "$OUT/host/sut-config.json"

# A per-run key keeps the local endpoint closed to anything but this client.
VLLM_API_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
export VLLM_API_KEY

echo "slogate: starting $MODEL_ID@$MODEL_REVISION" >&2
"$ENGINE_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" \
  --host 127.0.0.1 --port "$PORT" "${SERVE_ARGS[@]}" \
  > "$OUT/host/server.log" 2>&1 &
SERVER_PID=$!

"$CLIENT_VENV/bin/slogate" aa \
  --base-url "http://127.0.0.1:$PORT" \
  --workload "$WORKLOAD" \
  --out "$OUT/aa" \
  --runs "$RUNS" \
  --cooldown-s 10 \
  --wait-ready-s 1200 \
  --api-key-env VLLM_API_KEY \
  --sut-info "$OUT/sut-info.json" \
  --label "$SUT_NAME"

if [[ "$TRACE" == "1" ]]; then
  stop_server
  "$CLIENT_VENV/bin/python" -m slogate.check_trace "$SLOGATE_TRACE_DIR"
fi
