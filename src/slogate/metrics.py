"""Run-level serving metrics computed from client-observed request timelines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .replay import ReplayResult, RequestRecord

PERCENTILES = (50, 90, 99)


@dataclass(frozen=True)
class SLO:
    """Per-request service-level objective; unset bounds are not enforced."""

    ttft_s: float | None = None
    tpot_s: float | None = None

    @property
    def defined(self) -> bool:
        return self.ttft_s is not None or self.tpot_s is not None

    def met_by(self, record: RequestRecord) -> bool:
        if not record.success:
            return False
        if self.ttft_s is not None and (record.ttft_s is None or record.ttft_s > self.ttft_s):
            return False
        if self.tpot_s is not None:
            tpot = record.tpot_s
            if tpot is not None and tpot > self.tpot_s:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Summary statistics; percentiles use NumPy's default linear interpolation."""
    if not values:
        return {"n": 0, "mean": None, **{f"p{p}": None for p in PERCENTILES}, "max": None}
    data = np.asarray(values, dtype=float)
    summary: dict[str, float | int | None] = {"n": int(data.size), "mean": float(data.mean())}
    for p in PERCENTILES:
        summary[f"p{p}"] = float(np.percentile(data, p))
    summary["max"] = float(data.max())
    return summary


def summarize(result: ReplayResult, slo: SLO | None = None) -> dict[str, Any]:
    records = result.records
    ok = [r for r in records if r.success]
    errors = Counter(r.error for r in records if not r.success and r.error)
    gaps: list[float] = []
    for record in ok:
        gaps.extend(record.token_gaps_s())
    output_tokens = sum(
        r.output_tokens if r.output_tokens is not None else r.streamed_tokens for r in ok
    )
    usage_mismatch = sum(
        1 for r in ok if r.output_tokens is not None and r.output_tokens != r.streamed_tokens
    )
    span = 0.0
    if ok:
        span = max(r.finished_s or 0.0 for r in ok) - min(r.launched_s or 0.0 for r in ok)

    metrics: dict[str, Any] = {
        "requests": {
            "offered": len(records),
            "launched": sum(1 for r in records if r.launched_s is not None),
            "succeeded": len(ok),
            "failed": len(records) - len(ok),
            "errors": dict(sorted(errors.items())),
            "usage_token_mismatch": usage_mismatch,
        },
        "ttft_s": distribution([r.ttft_s for r in ok if r.ttft_s is not None]),
        "itl_s": distribution(gaps),
        "tpot_s": distribution([r.tpot_s for r in ok if r.tpot_s is not None]),
        "e2e_s": distribution([r.e2e_s for r in ok if r.e2e_s is not None]),
        "throughput": {
            "output_tokens": output_tokens,
            "output_tokens_per_s": output_tokens / span if span > 0 else None,
            "completed_rps": len(ok) / span if span > 0 else None,
        },
    }
    if slo is not None and slo.defined:
        met = sum(1 for r in records if slo.met_by(r))
        metrics["goodput"] = {
            "slo": slo.to_dict(),
            "met": met,
            "attainment": met / len(records) if records else None,
            "rps": met / result.duration_s,
        }
    return metrics
