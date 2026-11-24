"""Diagnostic-only: inventory the live client heap near a recorded E0 stall on Linux loopback."""

from __future__ import annotations

import argparse
import asyncio
import gc
import subprocess
import sys
import time
import tracemalloc
import types
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cpu_pressure import local_engine

from slogate.cli import main as cli
from slogate.diagnostics import ClientDiagnostics
from slogate.fake_engine import EngineProfile
from slogate.recorded_stream import RecordedStreams
from slogate.report import write_json
from slogate.workload import load


def census() -> Counter[str]:
    counts: Counter[str] = Counter()
    for obj in gc.get_objects():
        cls = type(obj)
        if cls is types.CoroutineType or cls is types.AsyncGeneratorType:
            code = obj.cr_code if cls is types.CoroutineType else obj.ag_code
            counts[f"frame: {Path(code.co_filename).name}:{code.co_qualname}"] += 1
        else:
            counts[f"{cls.__module__}.{cls.__qualname__}"] += 1
    return counts


def allocation_sites() -> list[dict]:
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    sites = snapshot.filter_traces((tracemalloc.Filter(False, __file__),))
    return [
        {"site": str(s.traceback[0]), "bytes": s.size, "blocks": s.count}
        for s in sites.statistics("lineno")[:50]
    ]


def profile_client(out: Path, url: str, workload: Path, at_s: float) -> None:
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("fixture must be on loopback")
    start = ClientDiagnostics.start

    def instrument(monitor: ClientDiagnostics, origin: float) -> None:
        before = census()
        tracemalloc.start(1)

        def capture() -> None:
            offset = time.perf_counter() - origin
            # Drop the snapshot itself before counting GC-tracked objects.
            sites = allocation_sites()
            after = census()
            diff = after.copy()
            diff.subtract(before)
            write_json(
                out / "heap.json",
                {
                    "diagnostic_only": True,
                    "target_s": at_s,
                    "offset_s": offset,
                    "python": sys.version,
                    "gc_enabled": gc.isenabled(),
                    "gc_thresholds": gc.get_threshold(),
                    "tracked_before": sum(before.values()),
                    "tracked_at_capture": sum(after.values()),
                    "tracked_growth_by_type": diff.most_common(60),
                    "suspended_stack_growth": [
                        [label, n]
                        for label, n in diff.most_common()
                        if label.startswith("frame:") and n > 5
                    ][:50],
                    "live_allocations_since_start": sites,
                },
            )

        start(monitor, origin)
        asyncio.get_running_loop().call_later(max(0, origin + at_s - time.perf_counter()), capture)

    with patch.object(ClientDiagnostics, "start", instrument):
        rc = cli(
            [
                "run",
                "--base-url",
                url,
                "--workload",
                str(workload),
                "--model",
                "fake-model",
                "--out",
                str(out / "profiled-run"),
                "--local-fixture-clock",
                "--wait-ready-s",
                "15",
                "--warmup-requests",
                "2",
            ]
        )
    if not (out / "heap.json").exists():
        raise RuntimeError("heap capture missing")
    print(f"Profiled run exited {rc}; its timings are not qualification evidence.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--at-s", type=float, default=23)
    parser.add_argument(
        "--workload", type=Path, default=Path("workloads/azure-2024-05-12-smoke-30s.json.gz")
    )
    parser.add_argument("--client-url", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.client_url:
        profile_client(args.out, args.client_url, args.workload, args.at_s)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    f = replace(RecordedStreams.load(args.source, load(args.workload)), shared_clock=True)
    with local_engine(EngineProfile(), f) as url:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--client-url", url], check=True, timeout=300
        )


if __name__ == "__main__":
    main()
