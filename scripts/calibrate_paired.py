"""Offline operating-characteristic check of the paired median gate. Prints aggregates only."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import median

from slogate.paired_stats import paired_interval

PAIRS = 12
MARGIN_MS = 100.0
SIGMA_MS = 100.0
RHO = 0.9
E0_CSV = Path(__file__).resolve().parents[1] / "docs" / "results" / "e0-2026-11-23.csv"


def _correlated(rng: random.Random) -> list[float]:
    """Independent two-pair clusters: same marginal SD as IID, covariance rho * SD**2."""
    noise = []
    for _ in range(PAIRS // 2):
        shared = rng.gauss(0, SIGMA_MS * math.sqrt(RHO))
        noise.extend(shared + rng.gauss(0, SIGMA_MS * math.sqrt(1 - RHO)) for _ in range(2))
    return noise


def _e0_run_ms() -> list[float]:
    with E0_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (
        len(rows) != 6
        or len({r["run"] for r in rows}) != 6
        or any(r["fidelity_valid"] != "true" for r in rows)
    ):
        raise ValueError("expected all six distinct, fidelity-valid E0 summary rows")
    values = [1000 * float(r["ttft_p99_s"]) for r in rows]
    if not all(math.isfinite(ms) and ms > 0 for ms in values):
        raise ValueError("invalid E0 run TTFT p99")
    return values


def _rate(count: int, n: int) -> str:
    """Count/denominator and 95% Wilson score limits (also defined at 0/n and n/n)."""
    p, z = count / n, 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return f"{count}/{n} [{max(0, center - half):.2%}, {min(1, center + half):.2%}]"


def _report(model: str, draws: list[list[float]], block_pairs: int) -> None:
    n = len(draws)
    for effect in (0.0, MARGIN_MS, 3 * MARGIN_MS):
        passed = regressed = covered = 0
        widths = []
        for noise in draws:
            r = paired_interval(
                [effect + d for d in noise], margin_ms=MARGIN_MS, block_pairs=block_pairs
            )
            passed += r["verdict"] == "pass"
            regressed += r["verdict"] == "regression"
            lo, hi = r["lower_ms"], r["upper_ms"]
            covered += lo <= effect <= hi
            widths.append(hi - lo)
        print(
            f"{model} | {block_pairs} | {effect:g} | {passed}/{n} | {n - passed - regressed}/{n} | "
            f"{_rate(regressed, n)} | {_rate(covered, n)} | {median(widths):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--trials", type=int, default=2000, help="experiments per row")
    parser.add_argument("--e0", action="store_true", help="also resample the six published E0 p99s")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    rng = random.Random(args.seed)
    print(f"seed={args.seed}, trials={args.trials}, pairs={PAIRS}, toy margin={MARGIN_MS:g} ms")
    print(f"Gaussian B-A noise SD={SIGMA_MS:g} ms; cluster size=2, within-cluster rho={RHO}.")
    print("Effects: zero (A/A), boundary (regression is false), and a clear injected shift.")
    print("Wilson 95% limits are per row, for Monte Carlo error only. Widths are descriptive.")
    print("The same noise draws are reused across shifts and across analyses of the same model.")
    print(
        "model | block_pairs | effect_ms | pass/n | inconclusive/n | regression/n [95%] | "
        "covers true effect/n [95%] | median CI width_ms"
    )
    iid = [[rng.gauss(0, SIGMA_MS) for _ in range(PAIRS)] for _ in range(args.trials)]
    _report("IID", iid, 1)
    clusters = [_correlated(rng) for _ in range(args.trials)]
    _report("clusters: predeclared blocks", clusters, 2)
    _report("same clusters: wrongly assumed IID", clusters, 1)
    print("Grouping is fixed, not selected by verdict. Ignoring clustering is a diagnostic only.")
    if args.e0:
        values = _e0_run_ms()
        print(
            "Exploratory E0: independently resampling A and B with replacement from six run p99s."
        )
        print("This discrete model permits ties and discards time order. It is not held-out")
        print("calibration and establishes no historical margin or power target.")
        draws = [
            [rng.choice(values) - rng.choice(values) for _ in range(PAIRS)]
            for _ in range(args.trials)
        ]
        _report("E0 resample", draws, 1)


if __name__ == "__main__":
    main()
