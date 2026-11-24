from __future__ import annotations

import asyncio
import gc
import time

import pytest

from slogate import workload
from slogate.cli import main
from slogate.diagnostics import ClientDiagnostics
from slogate.pressure_report import overdue_windows, summarize_run
from slogate.report import read_run, write_json


def test_common_clock_trace_captures_real_collection_and_wakes() -> None:
    async def capture():
        monitor = ClientDiagnostics.create(timing_trace=True)
        origin = time.perf_counter()
        monitor.start(origin)
        try:
            await asyncio.sleep(0.03)
            lo = time.perf_counter() - origin
            gc.collect(2)
            hi = time.perf_counter() - origin
            await asyncio.sleep(0.02)
        finally:
            await monitor.stop()
        return monitor.summary(), lo, hi, time.perf_counter() - origin

    summary, lo, hi, end = asyncio.run(capture())
    trace = summary["timing_trace"]
    assert any(lo <= s <= s + d <= hi and gen == 2 for s, gen, d in trace["gc_start_gen_duration"])
    assert 0 < min(trace["loop_due_s"]) <= max(trace["loop_due_s"]) <= end
    assert len(trace["loop_due_s"]) == len(trace["loop_lag_s"]) == summary["event_loop_lag_s"]["n"]
    assert max(trace["loop_lag_s"]) == summary["event_loop_lag_s"]["max"]
    assert len(trace["gc_start_gen_duration"]) == summary["gc"]["collections"]
    assert "timing_trace" not in ClientDiagnostics.create().summary()


def test_overlap_uses_only_excess_after_deadline() -> None:
    windows = overdue_windows(
        [1.0, 2.0, 3.0],
        [0.08, 0.05, 0.06],
        [(0.8, 1, 0.05), (0.99, 2, 0.05), (1.05, 0, 0.04), (3.1, 2, 0.01)],
        0.05,
    )
    assert [w["due_s"] for w in windows] == [1, 3]
    assert windows[0]["gc_overlap_s_by_generation"] == pytest.approx({2: 0.04, 0: 0.03})
    assert windows[1]["gc_overlap_s_by_generation"] == {}
    with pytest.raises(ValueError):
        overdue_windows([1.0], [], [], 0.05)


def test_cli_trace_can_be_reanalyzed_offline(tmp_path, fake_engine) -> None:
    w = workload.synthetic(
        name="timing", duration_s=0.4, rate_rps=12, input_tokens=[8], max_tokens=2, seed=3
    )
    path = workload.save(w, tmp_path / "input.json")
    out = tmp_path / "trial"
    code = main(
        [
            "run",
            "--base-url",
            fake_engine(ttft_s=0.01, jitter=0),
            "--workload",
            str(path),
            "--out",
            str(out),
            "--client-timing",
            "--warmup-requests",
            "0",
        ]
    )
    doc = read_run(out)
    assert code == (0 if doc["fidelity"]["valid"] else 3)
    t0 = doc["replay"]["started_unix_s"]
    write_json(tmp_path / "trial-rss.json", {"unix_s_rss_kib": [[t0 - 5, 9999], [t0 + 0.1, 2048]]})
    row = summarize_run(tmp_path, "trial")
    assert row["requests"]["succeeded"] == len(w.requests)
    assert row["workload_digest"] == w.digest
    assert row["peak_open"] >= 1
    assert row["max_sampled_client_rss_mib"] == 2  # Excludes the pre-replay sample.
