"""Reproduce E0 client pressure on Linux loopback; never contacts a user-supplied server."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import uvicorn

from slogate.fake_engine import EngineProfile, FakeEngine
from slogate.recorded_stream import RecordedStreamEngine, RecordedStreams
from slogate.report import write_json
from slogate.workload import load

DIGEST = "821ec458b144aa45b4126af56d360f055615aafeb34ac3467c604144664aa13c"
ORDER = "ABBAAB"
TTFT_S = {"A": 0.05, "B": 23.5}


def _serve(sock: socket.socket, profile: EngineProfile, fixture: RecordedStreams | None) -> None:
    engine = RecordedStreamEngine(profile, fixture) if fixture is not None else FakeEngine(profile)
    config = uvicorn.Config(engine.app(), log_level="warning", lifespan="off")
    uvicorn.Server(config).run(sockets=[sock])


@contextmanager
def local_engine(
    profile: EngineProfile, fixture: RecordedStreams | None = None
) -> Generator[str, None, None]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1024)
        port = sock.getsockname()[1]
        server = multiprocessing.get_context("spawn").Process(
            target=_serve, args=(sock, profile, fixture)
        )
        server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        if server.is_alive():
            server.terminate()
        server.join(5)
        if server.is_alive():
            server.kill()
            server.join()
        server.close()


def rss_kib(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None  # The client may already be a zombie.


def run_arm(
    root: Path,
    name: str,
    workload: Path,
    profile: EngineProfile,
    fixture: RecordedStreams | None = None,
) -> None:
    samples = []
    timeout_s = 180 if fixture is None else 240
    with local_engine(profile, fixture) as url, (root / f"{name}.log").open("x") as log:
        command = [
            sys.executable,
            "-m",
            "slogate",
            "run",
            "--base-url",
            url,
            "--workload",
            str(workload),
            "--model",
            profile.model,
            "--out",
            str(root / name),
            "--client-timing",
            "--wait-ready-s",
            "15",
            "--warmup-requests",
            "2",
        ]
        if fixture is not None and fixture.shared_clock:
            command.append("--local-fixture-clock")
        client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_s
        try:
            while client.poll() is None:
                rss = rss_kib(client.pid)
                if rss is not None:
                    samples.append((time.time(), rss))
                    if rss > 1024 * 1024:
                        raise RuntimeError(f"{name}: client exceeded 1 GiB RSS")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name}: client exceeded {timeout_s} seconds")
                time.sleep(0.2)
        finally:
            if client.poll() is None:
                client.kill()
            client.wait()
            write_json(root / f"{name}-rss.json", {"unix_s_rss_kib": samples})
        if client.returncode not in (0, 3) or not (root / name / "run.json").is_file():
            raise RuntimeError(f"{name}: client exit {client.returncode}; see {log.name}")
    print(f"{name}: client exit {client.returncode}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--recorded-aa",
        type=Path,
        help="instead of ABBAAB, replay the six saved E0 client-observed schedules, one per run",
    )
    parser.add_argument(
        "--shared-clock",
        action="store_true",
        help="anchor recorded chunks to the client replay epoch instead of HTTP arrival",
    )
    parser.add_argument(
        "--workload", type=Path, default=Path("workloads/azure-2024-05-12-smoke-30s.json.gz")
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run under Linux (e.g. WSL); RSS is sampled via procfs")
    if args.shared_clock and not args.recorded_aa:
        parser.error("--shared-clock requires --recorded-aa")
    workload = load(args.workload)
    if workload.digest != DIGEST:
        parser.error("expected the pinned 787-request E0 workload")
    args.out.mkdir(parents=True, exist_ok=False)
    profile = EngineProfile(itl_s=0.010, jitter=0, seed=1)
    protocol = {
        "threshold_ms": 50,
        "workload_digest": DIGEST,
        "gc_thresholds": gc.get_threshold(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("httpx", "httpcore", "numpy", "uvicorn", "starlette")},
        "isolation": "fresh client and separate fake-server processes per arm; automatic GC",
    }
    if args.recorded_aa:
        names = [f"run-{i:03d}" for i in range(1, 7)]
        fixtures = [
            replace(
                RecordedStreams.load(args.recorded_aa / n, workload), shared_clock=args.shared_clock
            )
            for n in names
        ]
        write_json(
            args.out / "protocol.json",
            {
                **protocol,
                "mode": "recorded-client-content-v1",
                "anchor": "shared" if args.shared_clock else "request",
                "names": names,
                "source_records_sha256": {f.source: f.records_sha256 for f in fixtures},
                "match_limits": {
                    "ttft_p99_abs_ms": 50,
                    "chunk_offset_p99_abs_ms": 50,
                    "global_chunk_p99_abs_ms": 50,
                    "chunk_gap_p99_abs_ms": 10,
                    "wall_abs_s": 2,
                    "peak_rel": 0.05,
                    "request_seconds_rel": 0.02,
                },
            },
        )
        for i, (name, fixture) in enumerate(zip(names, fixtures, strict=True)):
            print(f"{name}: replaying source {fixture.records_sha256[:12]}", flush=True)
            run_arm(args.out, name, args.workload, profile, fixture)
            if i + 1 < len(names):
                time.sleep(2)
        return
    write_json(
        args.out / "protocol.json",
        {
            **protocol,
            "order": ORDER,
            "synthetic_ttft_s": TTFT_S,
            "other_fake_settings": profile.to_dict(),
            "peak_target": 659,
            "peak_target_band": [625, 695],
        },
    )
    for i, arm in enumerate(ORDER, 1):
        name = f"run-{i:03d}-{arm}"
        print(f"{name}: starting", flush=True)
        run_arm(args.out, name, args.workload, replace(profile, ttft_s=TTFT_S[arm]))
        if i < len(ORDER):
            time.sleep(2)


if __name__ == "__main__":
    main()
