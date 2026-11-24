from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from slogate.noise import MIN_RUNS, noise_floor, to_markdown


def runs(values: list[float]) -> list[dict]:
    return [{"ttft_s": {"p99": v}, "itl_s": {"p99": 2 * v}} for v in values]


def row(report: dict, metric: str) -> dict:
    return next(r for r in report["metrics"] if r["metric"] == metric)


def test_mde_follows_the_documented_formula() -> None:
    values = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]
    report = noise_floor(runs(values))
    ttft = row(report, "ttft_s.p99")
    cv = float(np.std(values, ddof=1) / np.mean(values))
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    assert ttft["status"] == "ok"
    assert ttft["cv"] == pytest.approx(cv)
    assert ttft["mde_rel"]["1"] == pytest.approx(z * math.sqrt(2) * cv)
    assert ttft["mde_rel"]["5"] == pytest.approx(z * math.sqrt(2) * cv / math.sqrt(5))


def test_scaled_metric_has_same_relative_noise() -> None:
    report = noise_floor(runs([1.0, 1.2, 0.8, 1.1]))
    assert row(report, "itl_s.p99")["cv"] == pytest.approx(row(report, "ttft_s.p99")["cv"])
    assert row(report, "ttft_s.p99")["status"].startswith("low confidence")


def test_too_few_runs_are_flagged() -> None:
    report = noise_floor(runs([1.0] * (MIN_RUNS - 1)))
    assert row(report, "ttft_s.p99")["status"].startswith("insufficient")
    assert row(report, "goodput.rps")["runs"] == 0


def test_drift_detects_monotonic_change() -> None:
    report = noise_floor(runs([1.0, 1.0, 1.0, 1.3, 1.3, 1.3]))
    assert row(report, "ttft_s.p99")["drift_rel"] > 0.2


def test_markdown_lists_every_metric() -> None:
    text = to_markdown(noise_floor(runs([1.0, 1.1, 0.9])))
    assert "ttft_s.p99" in text
    assert "goodput.rps" in text
