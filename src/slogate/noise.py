"""A/A noise floor: how much do run-level metrics move when nothing changed?

Repeated runs of the same build on the same host estimate run-to-run variation. That
variation sets the smallest regression a paired comparison can detect (the MDE), which the
gate later uses to choose run counts and thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

METRICS: tuple[tuple[str, str], ...] = (
    ("ttft_s", "p50"),
    ("ttft_s", "p99"),
    ("itl_s", "p50"),
    ("itl_s", "p99"),
    ("tpot_s", "p50"),
    ("tpot_s", "p99"),
    ("e2e_s", "p99"),
    ("throughput", "output_tokens_per_s"),
    ("goodput", "rps"),
)
MIN_RUNS = 3
RECOMMENDED_RUNS = 6
PAIR_COUNTS = (1, 3, 5)


def _value(metrics: dict[str, Any], path: tuple[str, str]) -> float | None:
    section = metrics.get(path[0])
    if not isinstance(section, dict):
        return None
    value = section.get(path[1])
    return float(value) if isinstance(value, (int, float)) else None


def noise_floor(
    run_metrics: Sequence[dict[str, Any]], *, alpha: float = 0.05, power: float = 0.80
) -> dict[str, Any]:
    """Estimate per-metric noise and minimum detectable effect from A/A runs.

    The MDE assumes run-level metrics of two independent runs differ with variance
    ``2 * sigma^2`` and that the gate averages ``n`` paired differences:
    ``MDE(n) = (z_{1-alpha/2} + z_{power}) * sqrt(2) * sigma / sqrt(n)``, reported
    relative to the mean.
    """
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    rows = []
    for path in METRICS:
        values = [v for v in (_value(m, path) for m in run_metrics) if v is not None]
        row: dict[str, Any] = {"metric": f"{path[0]}.{path[1]}", "runs": len(values)}
        if len(values) < MIN_RUNS:
            row["status"] = f"insufficient runs (need >= {MIN_RUNS})"
            rows.append(row)
            continue
        data = np.asarray(values, dtype=float)
        mean = float(data.mean())
        std = float(data.std(ddof=1))
        row.update(mean=mean, std=std, min=float(data.min()), max=float(data.max()))
        if mean == 0:
            row["status"] = "mean is zero; relative noise undefined"
            rows.append(row)
            continue
        cv = std / abs(mean)
        half = len(data) // 2
        row["cv"] = cv
        row["range_rel"] = float((data.max() - data.min()) / abs(mean))
        row["drift_rel"] = float((data[half:].mean() - data[:half].mean()) / abs(mean))
        row["mde_rel"] = {str(n): z * math.sqrt(2.0) * cv / math.sqrt(n) for n in PAIR_COUNTS}
        row["status"] = (
            "ok"
            if len(values) >= RECOMMENDED_RUNS
            else f"low confidence (< {RECOMMENDED_RUNS} runs)"
        )
        rows.append(row)
    return {
        "alpha": alpha,
        "power": power,
        "runs": len(run_metrics),
        "pair_counts": list(PAIR_COUNTS),
        "metrics": rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    """Render a compact table for humans; JSON stays the source of truth."""
    header = [
        f"A/A noise floor over {report['runs']} runs "
        f"(alpha={report['alpha']}, power={report['power']}).",
        "",
        "| metric | runs | mean | CV | range | drift | MDE n=1 | MDE n=3 | MDE n=5 | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines = []
    for row in report["metrics"]:
        if "cv" not in row:
            lines.append(f"| {row['metric']} | {row['runs']} | | | | | | | | {row['status']} |")
            continue
        mde = row["mde_rel"]
        lines.append(
            f"| {row['metric']} | {row['runs']} | {row['mean']:.6g} | {row['cv']:.1%} | "
            f"{row['range_rel']:.1%} | {row['drift_rel']:+.1%} | {mde['1']:.1%} | "
            f"{mde['3']:.1%} | {mde['5']:.1%} | {row['status']} |"
        )
    return "\n".join(header + lines) + "\n"
