"""Private worker for sustained_replay: one port on stdin = one ordinary CLI run."""

import gc
import hashlib
import os
import platform
import sys
import tracemalloc
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

FLAGS = (
    "--model",
    "fake-model",
    "--local-fixture-clock",
    "--wait-ready-s",
    "15",
    "--warmup-requests",
    "2",
)


def main(out: Path, workload: Path, pkg: Path, name: str) -> int:
    # No fixture, source records, report analysis or RSS sampler in the measured process.
    sys.path.insert(0, str(pkg.parent))
    import slogate
    from slogate.cli import main as cli
    from slogate.report import write_json

    thresholds = gc.get_threshold()
    if Path(slogate.__file__).resolve().parent != pkg:
        raise RuntimeError(f"expected {pkg}, imported {slogate.__file__}")
    if (
        not gc.isenabled()
        or not thresholds[0]
        or tracemalloc.is_tracing()
        or sys.getprofile()
        or sys.gettrace()
    ):
        raise RuntimeError("expected automatic GC and no active profiler")
    packages = ("httpx", "httpcore", "aiohttp", "numpy", "uvicorn", "starlette")
    write_json(
        out / "client.json",
        {
            "pid": os.getpid(),
            "package": str(pkg),
            "slogate": slogate.__version__,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {p: version(p) for p in packages if find_spec(p)},
            "gc_thresholds": thresholds,
            "gc_enabled": gc.isenabled(),
            "hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
            "replay_sha256": hashlib.sha256((pkg / "replay.py").read_bytes()).hexdigest(),
        },
    )
    return run_commands(sys.stdin, out, workload, name, cli, write_json)


def run_commands(ports, out, workload, name, cli, write_json) -> int:
    thresholds = gc.get_threshold()
    worst = 0
    for i, port in enumerate(ports, 1):
        wave = out / f"wave-{i:03}"
        rc = cli(
            [
                "run",
                "--base-url",
                f"http://127.0.0.1:{int(port)}",
                "--workload",
                str(workload),
                "--out",
                str(wave / name),
                *FLAGS,
            ]
        )
        write_json(wave / "done.json", {"pid": os.getpid(), "cli_exit": rc})
        if not gc.isenabled() or gc.get_threshold() != thresholds:
            raise RuntimeError("client GC policy changed")
        if rc not in (0, 3):
            return rc
        worst = max(rc, worst)
    return worst


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
