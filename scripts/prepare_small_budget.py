"""Prepare the E0 vs small-budget trial locally. No command starts or stops an engine.

    python scripts/prepare_small_budget.py --out trial --workload "$WORKLOAD"
    python scripts/prepare_small_budget.py slot trial --slot pair-001-A --receipt-sha256 HEX

Save the two printed pins separately BEFORE any capture. Transfer the new trial directory
byte-for-byte, along with the reviewed slogate release. `slot` prints a Bash recipe for
just the named next slot; it does not reserve it, contact a service or require Linux.
`begin` and `record` are for the operator of an already authorized, exclusive Linux host.
Use the same prepared engine/client venvs, local model cache, physical GPU, private
environment and port throughout. Do not run the exact-token service at the same time.
Keep the entire trial, including unfinished operations; never replace/recycle a slot.
The eager entry point shares the same slot protocol but requires its own candidate and pins.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from slogate import workload
from slogate.paired_plan import digest, file_digest, freeze, load_plan, pin, require, slots
from slogate.report import RECORDS_FILE, RUN_FILE, write_json

if __package__:
    from .week3_candidates import BASE, candidates
else:  # Direct execution: sys.path[0] is scripts/, not the repo root.
    if sys.flags.safe_path:  # -I omits even the script's own directory.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from week3_candidates import BASE, candidates

ROOT = Path(__file__).resolve().parents[1]
PROC = Path("/proc")
BASE_DIGEST = "3cdd93540fe56725eae5c2bcfa8b1b69c38bf4fc4709a6c993b6625c8841d0f8"
WORK_DIGEST = "821ec458b144aa45b4126af56d360f055615aafeb34ac3467c604144664aa13c"
WORK = "workload.json.gz"
CONFIG = {"A": "baseline.json", "B": "candidate.json"}
FILES = (*CONFIG.values(), WORK, "ab/plan.json")
FLAG = "--max-num-batched-tokens"
TRIALS = {
    "small-budget": (FLAG, "scripts/prepare_small_budget.py"),
    "eager": ("--enforce-eager", "scripts/prepare_eager.py"),
}
URL = "http://127.0.0.1:8000"
KEY = "VLLM_API_KEY"
ENV = {"HF_HUB_OFFLINE": "1", "VLLM_BATCH_INVARIANT": "0", "PYTHONNOUSERSITE": "1"}
TRANSPORT = {"name": "aiohttp", "version": "3.14.5"}
DESIGN = {"pairs": 6, "block_pairs": 1, "margin_ms": 2000, "warmup_requests": 2}
FIDELITY = {
    "max_launch_lag_s": 0.05,
    "max_event_loop_lag_s": 0.05,
    "max_gc_pause_s": 0.05,
    "require_all_launched": True,
}
# Estimates from the six saved E0 run.json wall_s (152.154..153.123 s) and ONE
# E0 server.log (non-default args -> Starting vLLM server: 212 s, not full readiness).
COST = {
    "design_target_gpu_min": 60,
    "basis": "E0 A only: ~152.9 s replay wall; one 212 s startup log span, not time to readiness",
    "replay_gpu_min_if_b_like_e0": round(12 * 152.9 / 60, 1),
    "startup_gpu_min_if_all_12_like_one_cold_e0": round(12 * 212 / 60, 1),
    "excludes": ["probes", "warmups", "switching/teardown", "separate correctness", "export"],
    "fit_guaranteed": False,
}


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def checked_work(path: Path) -> workload.Workload:
    w = workload.load(path)
    require(
        len(w.requests) == 787 and w.duration_s == 30 and w.digest == WORK_DIGEST,
        "expected the exact 787-request, 30 s E0 workload (canonical SHA-256)",
    )
    return w


def checked_path(root: Path, rel: str) -> Path:
    # All rels are program constants or names from paired_plan.slots, never receipt paths.
    path = root
    for part in Path(rel).parts:
        path = path / part
        require(not path.is_symlink(), f"unexpected symlink in {rel}")
    require(path.resolve().is_relative_to(root.resolve()), f"path outside trial: {rel}")
    return path


def receipt(
    root: Path, trusted_files: dict | None = None, *, candidate: str = "small-budget"
) -> dict:
    require(candidate in TRIALS, "unsupported trial")
    for name in FILES:
        require(checked_path(root, name).is_file(), f"missing trial input: {name}")
    hashes = {name: file_digest(root / name) for name in FILES}
    if trusted_files is not None:
        require(hashes == trusted_files, "trial file SHA-256 mismatch")
    a, b = (read(root / name) for name in CONFIG.values())
    require(digest(a) == BASE_DIGEST, "not the pinned E0 baseline")
    require(b == candidates(a)[candidate], f"not the prescribed {candidate} candidate")
    w = checked_work(root / WORK)
    plan, sha = load_plan(root / "ab")
    expected = {
        **DESIGN,
        "kind": "vllm",
        "candidate_ceiling_ms": None,
        "change": [TRIALS[candidate][0]],
        "arms": {"A": pin("vllm", a), "B": pin("vllm", b)},
        "workload": w.summary(),
        "layout": [[r.id, r.arrival_s, r.max_tokens] for r in w.requests],
        "fidelity_policy": FIDELITY,
        "replay_config": {
            "model": w.model,
            "ignore_eos": True,
            "warmup_requests": 2,
            "request_timeout_s": 300.0,
            "drain_timeout_s": 300.0,
            "max_in_flight": 4096,
            "transport": TRANSPORT,
        },
    }
    for k, v in expected.items():
        require(plan[k] == v, f"not the preregistered {k}")
    return {
        "schema": f"{candidate}-performance-v1",
        "ab_plan_sha256": sha,
        "file_sha256": hashes,
        "workload_sha256": w.digest,
        "configs": CONFIG,
        "slots": [name for name, _ in slots(plan)],
        "decision": "TTFT p99 B minus A; +2000 ms reference margin, not a product SLO",
        "base_url": URL,
        "api_key_env": KEY,
        "env": ENV,
        "cost_estimate_only": COST,
    }


def prepare(out: Path, work: Path, *, candidate: str = "small-budget") -> dict:
    require(version("aiohttp") == TRANSPORT["version"], "prepare with E0 aiohttp==3.14.5")
    require(candidate in TRIALS, "unsupported trial")
    w = checked_work(work)
    a = read(BASE)
    require(digest(a) == BASE_DIGEST, "not the pinned E0 baseline")
    b = candidates(a)[candidate]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / CONFIG["A"], a)
    write_json(out / CONFIG["B"], b)
    if work.suffix == ".gz":
        shutil.copyfile(work, out / WORK)
    else:
        workload.save(w, out / WORK)
    sha = freeze(out / "ab", w, kind="vllm", baseline=a, candidate=b, ceiling_ms=None, **DESIGN)
    doc = receipt(out, candidate=candidate)
    require(doc["ab_plan_sha256"] == sha, "plan changed during preparation")
    write_json(out / "receipt.json", doc)  # Last: a partial preparation cannot authorize a slot.
    return {"receipt_sha256": file_digest(out / "receipt.json"), "ab_plan_sha256": sha}


def checked_slot(
    root: Path, trusted: str, slot: str, *, recording: bool = False, candidate: str = "small-budget"
) -> tuple[str, Path]:
    path = checked_path(root, "receipt.json")
    require(file_digest(path) == trusted, "receipt differs from the separately saved SHA-256")
    doc = read(path)
    require(
        doc == receipt(root, doc["file_sha256"], candidate=candidate),
        "receipt does not bind these trial files",
    )
    order = doc["slots"]
    require(slot in order, "unknown preassigned slot")
    prior = set(order[: order.index(slot)])
    ab = root / "ab"
    ops = checked_path(root, "ab/operations")
    require(
        {p.name for p in ab.glob("pair-*")} == prior,
        "not the next slot: missing, occupied or unplanned capture",
    )
    require(
        ({p.name for p in ops.iterdir()} if ops.exists() else set())
        == (prior | {slot} if recording else prior),
        "missing, unfinished or unplanned operation; retain this experiment",
    )
    for name in prior:
        path = checked_path(root, f"ab/{name}/seal.json")
        seal = read(path)
        done = read(checked_path(root, f"ab/operations/{name}/result.json"))
        require(
            done["receipt_sha256"] == trusted
            and (done["exit"] == 0 or (name.endswith("-B") and done["exit"] == 2))
            and done["begin_sha256"] == file_digest(root / f"ab/operations/{name}/begin.json")
            and done["seal_sha256"] == file_digest(path)
            and seal["plan_sha256"] == doc["ab_plan_sha256"]
            and seal["slot"] == name
            and seal["run_sha256"] == file_digest(checked_path(root, f"ab/{name}/{RUN_FILE}"))
            and seal["records_sha256"]
            == file_digest(checked_path(root, f"ab/{name}/{RECORDS_FILE}")),
            f"invalid prior slot {name}; do not replace it",
        )
    return slot[-1], checked_path(root, f"ab/operations/{slot}")


def engine_args(config: dict) -> list[str]:
    model = config["model"]
    return [
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["id"],
        "--revision",
        model["revision"],
        "--tokenizer-revision",
        model["revision"],
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        *config["serve_args"],
    ]


def recipe(
    slot: str, arm: str, trusted: str, config: dict, *, candidate: str = "small-budget"
) -> str:
    guard = f'"$CLIENT_PYTHON" {TRIALS[candidate][1]}'
    args = f'"$TRIAL" --slot {slot} --receipt-sha256 {trusted}'
    return f"""# ONE slot ({slot}); manual steps, not an unattended launch script.
# Only on an authorized exclusive Linux host with host PID/GPU visibility, same operator,
# the model revision cached locally, and no other performance or correctness service.
# If finishing in 60 GPU-min is mandatory, STOP. E0-like replay ~30.6 + all-cold
# startup ~42.4 = ~73.0 GPU-min BEFORE probes, warmups, switching and correctness.
# Actual B, cached startup and correctness costs are unknown; no quota is assumed.
# In the release repo root start a clean interactive Bash ONCE for the whole trial:
# env -i HOME="$HOME" PATH="$PATH" LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}" \\
#   /bin/bash --noprofile --norc
# Export TRIAL, ENGINE_PYTHON, CLIENT_PYTHON, HF_HOME as absolute paths in that shell.
# If the reference needs a separate CUDA 13 toolchain, export the same CUDA_HOME,
# CUDA_PATH, CPATH and LIBRARY_PATH for both arms before reserving the first slot.
# Use the pinned engine venv and the separate E0-constrained CPython 3.12.3 client.
# Privately provision/export VLLM_API_KEY once (e.g. read -rs VLLM_API_KEY; export VLLM_API_KEY).
# Never enable shell tracing, a profiler, special GC, or log/print the environment.
export HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=0 PYTHONNOUSERSITE=1
umask 077
set -o noclobber
# 1. begin checks the idle GPU/port and reserves this slot once.
D="$TRIAL/ab/operations/{slot}"
CFG="$TRIAL/{CONFIG[arm]}"
if {guard} begin {args} &&
   "$ENGINE_PYTHON" scripts/gpu/probe_engine.py "$CFG" > "$D/engine.json" 2> "$D/engine.stderr" &&
   "$CLIENT_PYTHON" scripts/gpu/collect_host.py "$CFG" "$D/engine.json" \\
     > "$D/sut-info.json" 2> "$D/host.stderr"
then
  "$ENGINE_PYTHON" {shlex.join(engine_args(config))} > "$D/server.log" 2>&1 &
  SERVER_PID=$!
else
  unset SERVER_PID
  printf '%s\\n' 'STOP: reservation or probe failed; do not launch.' >&2
fi
# 2. Wait for startup complete in THIS job's log. ONLY THEN run the following line
# (remove its leading comment). record delegates to the existing slogate ab record:
# {guard} record {args} --server-pid "${{SERVER_PID:?no owned server}}" > "$D/record.log" 2>&1
# 3. Inspect the exit and retain ALL artifacts (including failures). In this interactive
# shell, use jobs -l, foreground ONLY your verified server job with fg %JOB, then Ctrl-C.
# Wait for that process and workers to exit and the port/GPU to be idle before continuing.
# Never kill by name, reassign a slot, bypass a failed check, or truncate/drop a pair.
# Re-run 'slot' for the next assigned name. On completion OR budget stop, on the client:
# "$CLIENT_PYTHON" -m slogate ab report "$TRIAL/ab"
# This only evaluates performance. Export private evidence separately from public summaries."""


def environment(env: dict[str, str]) -> str:
    shell = {"_", "SHLVL", "PWD", "OLDPWD"}
    allowed = {
        "HOME",
        "PATH",
        "LD_LIBRARY_PATH",
        "TERM",
        "LANG",
        "LC_CTYPE",
        "LC_ALL",
        "HF_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
        "CPATH",
        "LIBRARY_PATH",
    }
    paths = {"TRIAL", "CLIENT_PYTHON", "ENGINE_PYTHON", "HF_HOME"}
    require(not (env.keys() - shell - allowed - paths - ENV.keys() - {KEY}), "need a clean shell")
    require(all(env.get(k) == v for k, v in ENV.items()), "wrong offline/performance environment")
    require(all(Path(env.get(k, "")).is_absolute() for k in paths), "absolute host paths required")
    key = env.get(KEY, "")
    require(bool(key) and all("!" <= c <= "~" for c in key), "nonempty private API key required")
    # Only an opaque comparison of non-secret settings/venv paths leaves the process.
    return digest({k: v for k, v in env.items() if k not in shell | {KEY}})


def client_stamp(root: Path) -> str:
    require(sys.platform == "linux" and sys.version_info[:3] == (3, 12, 3), "need E0 Linux client")
    env = dict(os.environ)
    stamp = environment(env)
    require(
        env["CLIENT_PYTHON"] == sys.executable
        and env["ENGINE_PYTHON"] != sys.executable
        and Path(env["TRIAL"]).samefile(root),
        "wrong trial, engine or client interpreter",
    )
    require(
        gc.isenabled()
        and gc.get_threshold() == (700, 10, 10)
        and sys.getprofile() is None
        and sys.gettrace() is None,
        "unprofiled client with normal CPython 3.12 GC required",
    )
    lock = ROOT / "configs/client/e0-py312-linux.txt"
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            pkg, ver = line.split("==")
            if pkg not in {"click", "starlette", "uvicorn"}:  # Optional fake-engine extra.
                require(version(pkg) == ver, f"E0 client constraint mismatch: {pkg}")
    return stamp


def query(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=10).stdout


def gpu_pids() -> list[int]:
    rows = query("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    return [int(line) for line in rows.strip().splitlines()]


def begin(root: Path, trusted: str, slot: str, op: Path) -> None:
    stamp = client_stamp(root)
    if slot != "pair-001-A":
        first = read(root / "ab/operations/pair-001-A/begin.json")
        require(stamp == first["env_sha256"], "environment differs from first performance slot")
    major_minor = "import sys; print('%d.%d' % sys.version_info[:2])"
    require(
        query(os.environ["ENGINE_PYTHON"], "-c", major_minor).strip() == "3.12",
        "need the pinned Python 3.12 engine interpreter",
    )
    require(not gpu_pids(), "GPU already occupied; do not launch or kill another job")
    require(not query("ss", "-H", "-ltn", "sport = :8000").strip(), "performance port occupied")
    op.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_json(op / "begin.json", {"slot": slot, "receipt_sha256": trusted, "env_sha256": stamp})


def check_engine(pid: int, config: dict, stamp: str) -> None:
    require(pid > 0, "need the owned API server PID")
    proc = PROC / str(pid)
    exe = os.environ["ENGINE_PYTHON"]
    argv = [os.fsdecode(s) for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
    require(
        argv == [exe, *engine_args(config)] and (proc / "exe").samefile(exe),
        "wrong live engine binary, revision, port or arm",
    )
    env = dict(
        os.fsdecode(s).split("=", 1) for s in (proc / "environ").read_bytes().split(b"\0") if s
    )
    require(environment(env) == stamp and env[KEY] == os.environ[KEY], "live env/auth mismatch")
    apps = gpu_pids()
    require(bool(apps), "cannot attribute any GPU compute process to the server")
    for child in apps:
        seen = set()
        while child != pid:
            require(child > 1 and child not in seen, "unrelated or unobservable GPU process")
            seen.add(child)
            child = int((PROC / str(child) / "stat").read_bytes().rsplit(b")", 1)[1].split()[1])


def record(root: Path, trusted: str, slot: str, arm: str, op: Path, pid: int) -> int:
    with (op / "record.started").open("x"):
        pass  # Exclusive, retained even if attestation or warmup fails before a seal exists.
    stamp = client_stamp(root)
    require(
        read(op / "begin.json") == {"slot": slot, "receipt_sha256": trusted, "env_sha256": stamp},
        "wrong operation or environment changed since begin",
    )
    config = read(root / CONFIG[arm])
    info = read(op / "sut-info.json")
    require(
        info["config"] == config
        and info["worker_step_trace_requested"] is False
        and info["engine"] == read(op / "engine.json")
        and all(info["engine"][k] == v for k, v in config["engine"]["expect"].items()),
        "wrong per-slot engine probe, trace mode or manifest",
    )
    check_engine(pid, config, stamp)
    # A fresh uninstrumented client, not a replay inside the preparer's Python heap.
    code = subprocess.run(
        [
            sys.executable,
            "-m",
            "slogate",
            "ab",
            "record",
            str(root / "ab"),
            "--arm",
            arm,
            "--base-url",
            URL,
            "--workload",
            str(root / WORK),
            "--sut-info",
            str(op / "sut-info.json"),
            "--server-pid",
            str(pid),
            "--api-key-env",
            KEY,
        ],
        check=False,
    ).returncode
    seal = root / "ab" / slot / "seal.json"
    write_json(
        op / "result.json",
        {
            "receipt_sha256": trusted,
            "exit": code,
            "begin_sha256": file_digest(op / "begin.json"),
            "seal_sha256": file_digest(seal) if seal.is_file() else None,
        },
    )
    return code


def main(argv: list[str] | None = None, *, candidate: str = "small-budget") -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
        if candidate == "small-budget"
        else "E0 vs --enforce-eager: prepare, preview or record the frozen trial.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ops = parser.add_subparsers(dest="command", required=True)
    prep = ops.add_parser("prepare", help="freeze a never-run trial in a NEW directory")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--workload", type=Path, required=True)
    for cmd, help_ in (
        ("slot", "print a recipe; no launch or reservation"),
        ("begin", "reserve once on the authorized host"),
        ("record", "delegate one capture to slogate ab"),
    ):
        sub = ops.add_parser(cmd, help=help_)
        sub.add_argument("trial", type=Path)
        sub.add_argument("--slot", required=True)
        sub.add_argument("--receipt-sha256", required=True)
        if cmd == "record":
            sub.add_argument("--server-pid", type=int, required=True)
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--out", "--workload"):
        args.insert(0, "prepare")
    opts = parser.parse_args(args)
    try:
        if opts.command == "prepare":
            print(json.dumps(prepare(opts.out, opts.workload, candidate=candidate), sort_keys=True))
        else:
            root, trusted, slot = opts.trial, opts.receipt_sha256, opts.slot
            arm, op = checked_slot(
                root, trusted, slot, recording=opts.command == "record", candidate=candidate
            )
            if opts.command == "slot":
                print(recipe(slot, arm, trusted, read(root / CONFIG[arm]), candidate=candidate))
            elif opts.command == "begin":
                begin(root, trusted, slot, op)
            else:
                return record(root, trusted, slot, arm, op, opts.server_pid)
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"{candidate}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
