from __future__ import annotations

import math

import pytest

from slogate.metrics import SLO, distribution, summarize
from slogate.replay import ReplayResult, RequestRecord


def record(identifier: str, **fields: object) -> RequestRecord:
    base: dict[str, object] = {"id": identifier, "scheduled_s": 0.0, "launched_s": 0.0}
    base.update(fields)
    return RequestRecord(**base)  # type: ignore[arg-type]


def result(records: list[RequestRecord], duration_s: float = 10.0) -> ReplayResult:
    return ReplayResult(
        workload_name="unit",
        workload_digest="0" * 64,
        model="m",
        duration_s=duration_s,
        started_unix_s=0.0,
        wall_s=duration_s,
        records=records,
        in_flight_limit_hit=False,
        diagnostics=None,
    )


def test_multi_token_chunks_split_gaps_per_token() -> None:
    r = record(
        "a",
        launched_s=0.5,
        first_token_s=1.0,
        finished_s=1.4,
        chunk_times_s=[1.0, 1.1, 1.3],
        chunk_tokens=[1, 1, 2],
        output_tokens=4,
        success=True,
    )
    assert r.token_gaps_s() == pytest.approx([0.1, 0.1, 0.1])
    assert r.tpot_s == pytest.approx(0.1)
    assert r.ttft_s == pytest.approx(0.5)
    assert r.e2e_s == pytest.approx(0.9)


def test_distribution_handles_empty_and_values() -> None:
    assert distribution([])["p99"] is None
    summary = distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["max"] == 4.0


def test_goodput_counts_failures_as_misses() -> None:
    fast = record(
        "fast",
        first_token_s=0.1,
        finished_s=0.5,
        chunk_times_s=[0.1, 0.2, 0.3],
        chunk_tokens=[1, 1, 1],
        output_tokens=3,
        success=True,
    )
    slow = record(
        "slow",
        first_token_s=2.0,
        finished_s=2.5,
        chunk_times_s=[2.0, 2.1],
        chunk_tokens=[1, 1],
        output_tokens=2,
        success=True,
    )
    failed = record("failed", error="HTTP500", error_phase="status")
    metrics = summarize(result([fast, slow, failed]), SLO(ttft_s=0.5, tpot_s=0.2))
    assert metrics["requests"]["succeeded"] == 2
    assert metrics["requests"]["errors"] == {"HTTP500": 1}
    assert metrics["goodput"]["met"] == 1
    assert metrics["goodput"]["attainment"] == pytest.approx(1 / 3)
    assert metrics["goodput"]["rps"] == pytest.approx(0.1)


def test_usage_mismatch_is_reported() -> None:
    mismatched = record(
        "m",
        first_token_s=0.1,
        finished_s=0.3,
        chunk_times_s=[0.1, 0.2],
        chunk_tokens=[1, 1],
        output_tokens=5,
        success=True,
    )
    metrics = summarize(result([mismatched]))
    assert metrics["requests"]["usage_token_mismatch"] == 1
    assert "goodput" not in metrics


def test_throughput_uses_completed_span() -> None:
    first = record(
        "a",
        launched_s=1.0,
        first_token_s=1.1,
        finished_s=2.0,
        chunk_times_s=[1.1, 1.5],
        chunk_tokens=[1, 1],
        output_tokens=2,
        success=True,
    )
    second = record(
        "b",
        launched_s=2.0,
        first_token_s=2.1,
        finished_s=3.0,
        chunk_times_s=[2.1, 2.5],
        chunk_tokens=[1, 1],
        output_tokens=2,
        success=True,
    )
    throughput = summarize(result([first, second]))["throughput"]
    assert throughput["output_tokens"] == 4
    assert math.isclose(throughput["output_tokens_per_s"], 2.0)
