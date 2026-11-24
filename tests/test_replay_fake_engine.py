"""Integration tests against the fake engine.

Timing bounds are deliberately loose: Windows timers tick at roughly 15 ms, and these tests
must pass on developer laptops as well as on Linux CI.
"""

from __future__ import annotations

import asyncio
import statistics

from slogate import workload
from slogate.fake_engine import free_port
from slogate.fidelity import assess
from slogate.metrics import summarize
from slogate.replay import ReplayConfig, replay


def small_workload(duration_s: float = 2.0, rate_rps: float = 8.0, max_tokens: int = 8):
    return workload.synthetic(
        name="it",
        duration_s=duration_s,
        rate_rps=rate_rps,
        input_tokens=[16],
        max_tokens=max_tokens,
        seed=3,
    )


def run(url: str, load: workload.Workload, **config: object):
    return asyncio.run(replay(load, ReplayConfig(base_url=url, **config)))  # type: ignore[arg-type]


def test_replay_records_complete_timelines(fake_engine) -> None:
    url = fake_engine(ttft_s=0.03, itl_s=0.02, jitter=0.0, seed=1)
    load = small_workload()
    result = run(url, load)
    assert len(result.records) == len(load.requests)
    for record in result.records:
        assert record.success, record.error
        assert record.output_tokens == 8
        assert record.streamed_tokens == 8
        assert record.prompt_tokens == 16
        assert record.finish_reason == "length"
    metrics = summarize(result)
    assert 0.02 <= metrics["ttft_s"]["p50"] <= 0.25
    assert 0.015 <= metrics["itl_s"]["p50"] <= 0.08
    assert metrics["requests"]["usage_token_mismatch"] == 0
    assert assess(result).valid


def test_replay_is_open_loop_under_slow_responses(fake_engine) -> None:
    url = fake_engine(ttft_s=1.0, itl_s=0.01, jitter=0.0)
    load = small_workload(duration_s=1.0, rate_rps=12.0, max_tokens=2)
    result = run(url, load)
    lags = [r.launch_lag_s for r in result.records]
    assert all(lag is not None and lag < 0.05 for lag in lags)
    first_token_after_last_launch = [
        r for r in result.records if r.first_token_s is not None and r.first_token_s > 1.0
    ]
    assert first_token_after_last_launch, "responses should arrive after later launches"


def test_injected_itl_regression_is_visible(fake_engine) -> None:
    load = small_workload(duration_s=1.5, max_tokens=10)
    baseline = run(fake_engine(ttft_s=0.02, itl_s=0.02, jitter=0.0), load)
    regressed = run(fake_engine(ttft_s=0.02, itl_s=0.02, jitter=0.0, extra_itl_s=0.04), load)
    base_itl = statistics.median(g for r in baseline.records for g in r.token_gaps_s())
    slow_itl = statistics.median(g for r in regressed.records for g in r.token_gaps_s())
    assert slow_itl - base_itl > 0.025


def test_eos_stops_outputs_early_when_not_ignored(fake_engine) -> None:
    url = fake_engine(ttft_s=0.01, itl_s=0.005, jitter=0.0, seed=5)
    result = run(url, small_workload(max_tokens=20), ignore_eos=False)
    lengths = {r.output_tokens for r in result.records}
    assert all(r.success for r in result.records)
    assert len(lengths) > 1
    assert max(lengths) <= 20


def test_connection_failures_are_recorded_not_raised() -> None:
    url = f"http://127.0.0.1:{free_port()}"
    load = small_workload(duration_s=0.5, rate_rps=6.0)
    result = run(url, load, request_timeout_s=5.0)
    assert all(not r.success for r in result.records)
    assert {r.error_phase for r in result.records} == {"headers"}
    assert summarize(result)["requests"]["failed"] == len(load.requests)
