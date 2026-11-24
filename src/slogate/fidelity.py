"""Load-fidelity verdicts: was the offered load actually delivered on schedule?

A run that fails fidelity is reported as invalid measurement (an infrastructure flake),
never as a server regression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .replay import ReplayResult


@dataclass(frozen=True)
class FidelityPolicy:
    max_launch_lag_s: float = 0.050
    max_event_loop_lag_s: float = 0.050
    max_gc_pause_s: float = 0.050
    require_all_launched: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FidelityVerdict:
    valid: bool
    reasons: tuple[str, ...]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reasons": list(self.reasons), "stats": self.stats}


def _ms(value: float) -> str:
    return f"{value * 1000:.1f} ms"


def assess(result: ReplayResult, policy: FidelityPolicy | None = None) -> FidelityVerdict:
    policy = policy or FidelityPolicy()
    reasons: list[str] = []
    lags = [r.launch_lag_s for r in result.records if r.launch_lag_s is not None]
    not_launched = sum(1 for r in result.records if r.launched_s is None)
    stats: dict[str, Any] = {"offered": len(result.records), "not_launched": not_launched}

    if lags:
        data = np.asarray(lags, dtype=float)
        stats["launch_lag_s"] = {
            "p50": float(np.percentile(data, 50)),
            "p99": float(np.percentile(data, 99)),
            "max": float(data.max()),
        }
        if data.max() > policy.max_launch_lag_s:
            reasons.append(
                f"launch lag max {_ms(float(data.max()))} exceeds {_ms(policy.max_launch_lag_s)}"
            )
    else:
        reasons.append("no request was launched")

    if not_launched and policy.require_all_launched:
        reasons.append(f"{not_launched} of {len(result.records)} requests were not launched")
    if result.in_flight_limit_hit:
        reasons.append("client in-flight limit was reached")

    diagnostics = result.diagnostics or {}
    loop = diagnostics.get("event_loop_lag_s") or {}
    if loop.get("max") is not None:
        stats["event_loop_lag_max_s"] = loop["max"]
        if loop["max"] > policy.max_event_loop_lag_s:
            reasons.append(
                f"client event loop lag max {_ms(loop['max'])} exceeds "
                f"{_ms(policy.max_event_loop_lag_s)}"
            )
    gc_stats = diagnostics.get("gc") or {}
    if gc_stats.get("pause_max") is not None:
        stats["gc_pause_max_s"] = gc_stats["pause_max"]
        if gc_stats["pause_max"] > policy.max_gc_pause_s:
            reasons.append(
                f"client GC pause max {_ms(gc_stats['pause_max'])} exceeds "
                f"{_ms(policy.max_gc_pause_s)}"
            )
    if gc_stats.get("dropped"):
        reasons.append(f"{gc_stats['dropped']} GC pauses were not recorded")

    return FidelityVerdict(valid=not reasons, reasons=tuple(reasons), stats=stats)
