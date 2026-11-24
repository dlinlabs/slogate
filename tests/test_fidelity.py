from __future__ import annotations

import pytest

from slogate.fidelity import FidelityPolicy, assess
from slogate.replay import ReplayResult, RequestRecord


def result(
    lags: list[float | None], diagnostics: dict | None = None, limit_hit: bool = False
) -> ReplayResult:
    records = []
    for index, lag in enumerate(lags):
        scheduled = float(index)
        records.append(
            RequestRecord(
                id=str(index),
                scheduled_s=scheduled,
                launched_s=None if lag is None else scheduled + lag,
            )
        )
    return ReplayResult(
        workload_name="unit",
        workload_digest="0" * 64,
        model="m",
        duration_s=10.0,
        started_unix_s=0.0,
        wall_s=10.0,
        records=records,
        in_flight_limit_hit=limit_hit,
        diagnostics=diagnostics,
    )


def test_clean_run_is_valid() -> None:
    verdict = assess(result([0.001, 0.002, 0.004]))
    assert verdict.valid
    assert verdict.stats["launch_lag_s"]["max"] == pytest.approx(0.004)


def test_late_launch_invalidates_measurement() -> None:
    verdict = assess(result([0.001, 0.080]), FidelityPolicy(max_launch_lag_s=0.05))
    assert not verdict.valid
    assert "launch lag max 80.0 ms" in verdict.reasons[0]


def test_unlaunched_requests_and_limit_are_reported() -> None:
    verdict = assess(result([0.001, None], limit_hit=True))
    assert not verdict.valid
    assert any("not launched" in reason for reason in verdict.reasons)
    assert any("in-flight limit" in reason for reason in verdict.reasons)


def test_client_pauses_invalidate_measurement() -> None:
    diagnostics = {
        "event_loop_lag_s": {"max": 0.120},
        "gc": {"pause_max": 0.070, "dropped": 0},
    }
    verdict = assess(result([0.001], diagnostics))
    assert not verdict.valid
    assert any("event loop" in reason for reason in verdict.reasons)
    assert any("GC pause" in reason for reason in verdict.reasons)
