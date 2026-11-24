"""A sign interval for the median of independent, run-level block means."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from statistics import mean
from typing import Any


def _finite_ms(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        ms = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the finite float range") from exc
    if not math.isfinite(ms):
        raise ValueError(f"{name} must be a finite real number")
    return ms


def _rank_and_coverage(n: int) -> tuple[int, float | None]:
    """Innermost valid one-based order-statistic rank, or zero for no finite interval."""
    outcomes = 1 << n
    rank, tail, term = 0, 0, 1
    # For rank r, one tail has sum(C(n, j) for j < r) of the 2**n sign patterns.
    # Spend at most 1/40 per tail; decide ranks in integers, not rounded probabilities.
    while 40 * (tail + term) <= outcomes:
        tail += term
        rank += 1
        term = term * (n - rank + 1) // rank  # C(n, rank)
    return rank, (outcomes - 2 * tail) / outcomes if rank else None


def paired_interval(
    deltas_ms: Sequence[float], *, margin_ms: float, block_pairs: int = 1
) -> dict[str, Any]:
    """Compare a median B-minus-A effect to a positive margin (larger is worse).

    Supply one delta per run pair in collection order. Fix the count and block size
    before collection; never drop pairs or choose a grouping based on the verdict.
    The default assumes independent pairs. With blocking, it is the *block means*
    that must be independent with a common median; grouping can't guarantee that.

    Coverage is the attained two-sided sign confidence level (conservative with ties),
    or None if no finite 95% interval exists. Invalid input raises ValueError.
    """
    margin = _finite_ms(margin_ms, "margin_ms")
    if margin <= 0:
        raise ValueError("margin_ms must be positive")
    if type(block_pairs) is not int or block_pairs < 1:
        raise ValueError("block_pairs must be a positive integer")
    if not isinstance(deltas_ms, Sequence) or isinstance(deltas_ms, (str, bytes, bytearray)):
        raise ValueError("deltas_ms must be an ordered sequence of finite real numbers")
    if len(deltas_ms) % block_pairs:
        raise ValueError("block_pairs must divide the number of run pairs; no partial blocks")

    values = [_finite_ms(d, f"deltas_ms[{i}]") for i, d in enumerate(deltas_ms)]
    block_means = values
    if block_pairs != 1:
        block_means = [
            mean(values[i : i + block_pairs]) for i in range(0, len(values), block_pairs)
        ]
    ordered = sorted(block_means)
    n = len(ordered)
    # Mean of the central one or two values; statistics.mean avoids overflow on finite inputs.
    estimate = mean(ordered[(n - 1) // 2 : n // 2 + 1]) if n else None
    rank, coverage = _rank_and_coverage(n)
    if not rank:
        lower = upper = None
        verdict = "inconclusive"
        reason = (
            f"Only {n} blocks; finite two-sided 95% bounds require at least 6 independent blocks"
        )
    else:
        lower, upper = ordered[rank - 1], ordered[-rank]
        if lower > margin:
            verdict, reason = "regression", "lower bound is strictly above the margin"
        elif upper < margin:
            verdict, reason = "pass", "upper bound is strictly below the margin"
        else:
            verdict, reason = "inconclusive", "interval touches or crosses the margin"
    return {
        "verdict": verdict,
        "estimate_ms": estimate,
        "lower_ms": lower,
        "upper_ms": upper,
        "coverage": coverage,
        "block_means_ms": block_means,
        "reason": reason,
    }
