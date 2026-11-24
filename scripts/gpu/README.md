# Running on a GPU host

`run_aa.sh` measures the A/A noise floor of one pinned system under test (SUT). It builds an
engine environment from the SUT config, verifies versions and hardware, starts the engine on
localhost with a per-run API key, runs `slogate aa`, and stores everything under one folder.

The [weeks 1-2 GPU result](../../docs/weeks-01-02.md) used a capture-specific launcher
patch that timed out only the experiment processes. The general launcher in this directory
does **not** have that same stop behavior: see [cost and safety controls](#cost-and-safety-controls)
before running it on a Pod with ephemeral storage.

## RunPod

1. Create a pod with one RTX 4090, an NVIDIA driver of 580 or newer (the pinned Torch build
   uses CUDA 13.0; the predecessor lab ran 580.159 and 580.178), Python 3.12 available, and a
   `/workspace` large enough for the model and both virtual environments. Use a persistent
   volume, or arrange a verified off-Pod export before stopping (see below).
2. Copy this repository and the workload file onto the pod, for example to
   `/workspace/slogate` and `/workspace/workloads/`. Check the workload digest with
   `slogate workload info` against [`workloads/README.md`](../../workloads/README.md).
   The E0 release bundle already contains the workload inside `slogate/workloads/`; after
   extracting it, use `workloads/azure-2024-05-12-smoke-30s.json.gz` from the repo root
   instead of the separate `/workspace/workloads/` path in the example.
3. Run:

   ```bash
   cd /workspace/slogate
   SLOGATE_TRACE=0 SLOGATE_MAX_MINUTES=60 bash scripts/gpu/run_aa.sh \
     configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json \
     /workspace/workloads/azure-2024-05-12-smoke-30s.json.gz 6
   ```

4. Download `/workspace/slogate-runs/aa-<time>/`. Start with `aa/noise.md`; `sut-info.json`
   and `host/` record exactly what ran. Keep the whole directory: if E0 runs without
   per-request SLO bounds, goodput must be computed later from the saved request records.

The E0 client uses [the qualified Linux dependency versions](../../configs/client/e0-py312-linux.txt).
These constraints do not apply to vLLM's engine environment. Confirm that each
`aa/run-*/run.json` records `meta.replay_config.transport` as `aiohttp` version `3.14.5`;
`host/client-packages.txt` lists the installed client packages. Read `aa/noise.json` for
included/excluded runs, not just the process exit code.

## Separate diagnostic rerun (not E0)

Use the same command with `SLOGATE_TRACE=1` and a **fresh output directory**. The launcher
installs slogate without changing dependencies in a separate `-trace` engine venv. If you
already set `VLLM_PLUGINS`, its allowlist must include `slogate` or the launcher stops before
installing the engine. The launcher records
`worker_step_trace_requested` in `sut-info.json` and checks the artifacts on completion.
Do not pool this run with E0: event recording, hashing and file writes can affect latency.

Each `trace/steps-<pid>.jsonl` covers that worker's lifetime, including client warmups and
all repeats. The `header` (schema 1) pairs host monotonic and wall clocks. Within a file:

| Record | How to read it |
|---|---|
| `step` | `step_id`, host start/end (monotonic ns), `ok`, scheduled requests/tokens, attention backend class names and the dispatched graph mode/padded tokens when available. `schedule_key` hashes sorted `(request ID, pre-step position, scheduled tokens)` for a future scheduler join. No raw IDs, token IDs or prompts are written. |
| `cuda` | Join by `step_id` **in the same file**. Only `status: ready` supplies `cuda_elapsed_s`. This is elapsed time on the original current stream around `execute_model`, **not** kernel busy time, sampling time, or end-to-end latency. |

Unknown graph/position layouts produce null fields, not guessed values. An unknown runner
signature is skipped with a warning. A full event queue or CUDA error leaves host data;
missing/unresolved timings, including the tail of a killed worker, are **unknown**, never
zero. The smoke check prints timed/successful step counts and fails if any observed worker
has no usable timing or if no worker file exists. It does not certify complete coverage.
To repeat it after downloading: `python -m slogate.check_trace <run-dir>/trace`.

Adapter references (checked 2026-11-17/18):
[lab recorder, `8e92371`](https://github.com/dlinlabs/vllm-serving-lab/blob/8e92371/prefill_lab/plugin.py),
[vLLM v0.30.0 plugin loading](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/plugins/__init__.py),
[V1 runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py)
and [V2 runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/model_runner.py).
These are source/CPU contract checks; the pinned GPU path and observer overhead still need measurement.

## Cost and safety controls

- The watchdog attempts to stop the Pod after `SLOGATE_MAX_MINUTES` while the launcher is
  running; this requires `RUNPOD_POD_ID` and a working `runpodctl pod stop`. Normal exit
  also attempts to stop it unless `SLOGATE_STOP_POD=0`. Container disk is ephemeral: if
  `/workspace` has no persistent mount, set `SLOGATE_STOP_POD=0`, export the complete result
  directory and verify its checksum off-Pod, then manually stop the Pod. The watchdog still
  applies during execution and does not stay active after the launcher exits.
- The engine binds to 127.0.0.1 and requires a random key generated for each invocation; the
  key is passed through the environment and never written to disk.
- Results are refused rather than produced on the wrong hardware: `collect_host.py` fails if
  the GPU model or count differs from the config, and `probe_engine.py` fails if vLLM, Torch or
  the CUDA runtime differ from the pins.
