from __future__ import annotations

import json
import math
import sys
from collections import Counter
from itertools import product

import pytest

from slogate.paired_stats import paired_interval


def test_first_finite_interval_and_json_contract() -> None:
    deltas = (12, 3, 7, 9, 4, 10)
    result = paired_interval(deltas, margin_ms=2)
    assert set(result) == {
        "verdict",
        "estimate_ms",
        "lower_ms",
        "upper_ms",
        "coverage",
        "block_means_ms",
        "reason",
    }
    assert result["verdict"] == "regression"
    assert result["estimate_ms"] == 8
    assert (result["lower_ms"], result["upper_ms"], result["coverage"]) == (3, 12, 62 / 64)
    assert result["block_means_ms"] == list(deltas)
    assert isinstance(result["reason"], str) and result["reason"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("blocks", range(6))
def test_too_few_independent_blocks_never_decide(blocks: int) -> None:
    # Up to ten raw pairs, all far below the margin: still not six independent blocks.
    result = paired_interval([0, 2] * blocks, margin_ms=100, block_pairs=2)
    assert result["verdict"] == "inconclusive"
    assert result["block_means_ms"] == [1] * blocks
    assert result["estimate_ms"] == (1 if blocks else None)
    assert result["lower_ms"] is result["upper_ms"] is result["coverage"] is None
    assert "6" in result["reason"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_group_in_collection_order_then_take_median_not_grand_mean() -> None:
    # Sorting before grouping, taking the raw median, and taking the grand mean all differ.
    deltas = [0, 200, 1, 2, -40, 44, 0, 4, -100, 110, 0, 12]
    original = deltas.copy()
    result = paired_interval(deltas, margin_ms=5, block_pairs=2)
    assert deltas == original
    assert result["block_means_ms"] == [100, 1.5, 2, 2, 5, 6]
    assert result["estimate_ms"] == 3.5
    assert (result["lower_ms"], result["upper_ms"]) == (1.5, 100)
    assert result["verdict"] == "inconclusive"


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ([-20] * 6, "pass"),
        ([math.nextafter(5, -math.inf)] * 6, "pass"),
        ([math.nextafter(5, math.inf)] * 6, "regression"),
        ([5] * 6, "inconclusive"),
        ([4] * 5 + [5], "inconclusive"),
        ([5] + [6] * 5, "inconclusive"),
        ([4] + [6] * 5, "inconclusive"),
    ],
)
def test_verdict_uses_strict_bounds_not_the_estimate(deltas: list[float], expected: str) -> None:
    assert paired_interval(deltas, margin_ms=5)["verdict"] == expected


def test_nine_blocks_exact_balanced_sign_experiment() -> None:
    # At the true median, each of the 512 sign patterns is equally likely. The two
    # excluded tails each contain the all-one-sign pattern and nine single exceptions.
    counts = Counter(
        paired_interval(sample, margin_ms=5)["verdict"] for sample in product((4, 6), repeat=9)
    )
    assert counts == {"pass": 10, "regression": 10, "inconclusive": 492}
    r = paired_interval(range(9, 0, -1), margin_ms=5)
    assert (r["lower_ms"], r["estimate_ms"], r["upper_ms"], r["coverage"]) == (2, 5, 8, 492 / 512)


@pytest.mark.parametrize("n", [7, 8, 12, 20, 35, 128, 1100])
def test_order_statistics_spend_no_more_than_five_percent(n: int) -> None:
    result = paired_interval(range(n), margin_ms=1)
    trimmed = int(result["lower_ms"])
    assert result["upper_ms"] == n - 1 - trimmed
    patterns = 2**n
    tail = sum(math.comb(n, j) for j in range(trimmed + 1))
    assert 40 * tail <= patterns < 40 * (tail + math.comb(n, trimmed + 1))
    assert result["coverage"] == (patterns - 2 * tail) / patterns


def test_finite_extremes_produce_finite_means_and_median() -> None:
    largest = sys.float_info.max
    result = paired_interval([largest] * 12, margin_ms=1, block_pairs=2)
    assert result["lower_ms"] == result["estimate_ms"] == result["upper_ms"] == largest
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("margin", [False, True, None, "3", 0, -1, math.nan, math.inf, -math.inf])
def test_invalid_margin_rejected(margin) -> None:
    with pytest.raises(ValueError, match="margin_ms"):
        paired_interval([], margin_ms=margin)


@pytest.mark.parametrize(
    "value", [True, False, "1", None, 1j, [1], math.nan, math.inf, -math.inf, 10**400]
)
def test_invalid_delta_never_filtered_even_in_small_sample(value) -> None:
    with pytest.raises(ValueError, match=r"deltas_ms\[1\]"):
        paired_interval([0, value], margin_ms=3)


@pytest.mark.parametrize("invalid", [None, 4, {1, 2}, {1: 2}, "123456", b"123456", bytearray()])
def test_reject_nonsequences_and_text(invalid) -> None:
    with pytest.raises(ValueError, match="deltas_ms"):
        paired_interval(invalid, margin_ms=3)


@pytest.mark.parametrize(
    "size", [True, False, 0, -1, 1.5, 2.0, None, "2", math.nan, math.inf, 4, 7]
)
def test_invalid_or_nondivisible_block_size_rejected(size) -> None:
    with pytest.raises(ValueError, match="block_pairs"):
        paired_interval([0] * 6, margin_ms=3, block_pairs=size)
