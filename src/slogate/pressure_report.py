"""Offline analysis of the paired CPU pressure diagnostic (no server or GPU required)."""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .report import read_records, read_run, write_json


def overdue_windows(
    due_s: Sequence[float],
    lag_s: Sequence[float],
    pauses: Sequence[tuple[float, int, float]],
    threshold_s: float,
) -> list[dict[str, Any]]:
    """Intersect GC spans with [deadline, wake), *not* with the intentional sleep."""
    windows = []
    for due, lag in zip(due_s, lag_s, strict=True):
        if lag <= threshold_s:
            continue
        by_generation: dict[int, float] = {}
        for start, generation, duration in pauses:
            overlap = min(due + lag, start + duration) - max(due, start)
            if overlap > 0:
                by_generation[generation] = by_generation.get(generation, 0.0) + overlap
        windows.append({"due_s": due, "lag_s": lag, "gc_overlap_s_by_generation": by_generation})
    return windows


def occupancy(records: list[dict[str, Any]], at_s: Sequence[float]) -> tuple[int, list[int], float]:
    starts = sorted(r["launched_s"] for r in records if r["launched_s"] is not None)
    ends = sorted(r["finished_s"] for r in records if r["finished_s"] is not None)

    def active(t: float) -> int:
        return bisect_right(starts, t) - bisect_right(ends, t)

    area = sum(
        r["finished_s"] - r["launched_s"]
        for r in records
        if r["finished_s"] is not None and r["launched_s"] is not None
    )
    return max((active(t) for t in starts), default=0), [active(t) for t in at_s], area


def summarize_run(root: Path, name: str) -> dict[str, Any]:
    run = read_run(root / name)
    records = read_records(root / name)
    replay = run["replay"]
    diag = replay["diagnostics"]
    trace = diag["timing_trace"]
    policy = run["meta"]["fidelity_policy"]
    pauses = trace["gc_start_gen_duration"]
    if (
        len(pauses) != diag["gc"]["collections"]
        or len(trace["loop_lag_s"]) != diag["event_loop_lag_s"]["n"]
    ):
        raise ValueError(f"{name}: trace and diagnostic sample counts disagree")
    windows = overdue_windows(
        trace["loop_due_s"], trace["loop_lag_s"], pauses, policy["max_event_loop_lag_s"]
    )
    peak, samples, area = occupancy(records, [replay["duration_s"], *(w["due_s"] for w in windows)])
    for window, n in zip(windows, samples[1:], strict=True):
        window["open_at_due"] = n
    overdue = sum(w["lag_s"] for w in windows)
    covered = sum(sum(w["gc_overlap_s_by_generation"].values()) for w in windows)
    rss = json.loads((root / f"{name}-rss.json").read_text(encoding="utf-8"))["unix_s_rss_kib"]
    t0 = replay["started_unix_s"]
    rss_during_replay = [kib / 1024 for ts, kib in rss if t0 <= ts <= t0 + replay["wall_s"]]
    return {
        "name": name,
        "valid": run["fidelity"]["valid"],
        "reasons": run["fidelity"]["reasons"],
        "requests": run["metrics"]["requests"],
        "workload_digest": run["meta"]["workload"]["digest"],
        "peak_open": peak,
        "open_at_end_of_arrivals": samples[0],
        "request_seconds": area,
        "wall_s": replay["wall_s"],
        "max_sampled_client_rss_mib": max(rss_during_replay, default=None),
        "launch_max_ms": 1000 * run["fidelity"]["stats"]["launch_lag_s"]["max"],
        "loop_max_ms": 1000 * diag["event_loop_lag_s"]["max"],
        "gc_max_ms": 1000 * diag["gc"]["pause_max"] if pauses else None,
        "gc_dropped": diag["gc"]["dropped"],
        "gc_collections_by_generation": diag["gc"]["by_generation"],
        "long_gc_generations": [
            generation for _, generation, duration in pauses if duration > policy["max_gc_pause_s"]
        ],
        "late_windows": windows,
        "windows_with_gc": sum(bool(w["gc_overlap_s_by_generation"]) for w in windows),
        "overdue_s": overdue,
        "gc_overlap_s": covered,
        "gc_share_of_overdue": covered / overdue if overdue else None,
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def main(root: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("mode") == "recorded-client-content-v1":
        raise ValueError(
            "use python -m slogate.timeline_report OUTPUT SOURCE_AA for recorded timing"
        )
    rows = [summarize_run(root, f"run-{i:03d}-{arm}") for i, arm in enumerate(protocol["order"], 1)]
    if any(row["workload_digest"] != protocol["workload_digest"] for row in rows):
        raise ValueError("Workload digest differs from the declared pressure protocol")
    write_json(root / "pressure.json", {"protocol": protocol, "runs": rows})
    low, high = protocol["peak_target_band"]
    stressed = [r for r in rows if r["name"].endswith("-B")]
    reached = sum(low <= r["peak_open"] <= high for r in stressed)
    lines = [
        "CPU-only loopback pressure diagnostic. All attempts shown; this is not a GPU baseline.",
        "",
        f"B target: {low}-{high} outstanding (GPU E0: {protocol['peak_target']}); "
        f"{reached}/{len(stressed)} runs reached the band.",
        "",
        "| Run | OK/offered | Valid | Peak open | Open at 30 s | Launch max ms | Loop max ms "
        "| GC max ms | Late wakes with GC / all late wakes | GC share of overdue | RSS MiB* |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        req = r["requests"]
        share = r["gc_share_of_overdue"]
        ratio = "-" if share is None else f"{share:.1%}"
        lines.append(
            f"| {r['name']} | {req['succeeded']}/{req['offered']} | {r['valid']} "
            f"| {r['peak_open']} | {r['open_at_end_of_arrivals']} | {r['launch_max_ms']:.1f} "
            f"| {r['loop_max_ms']:.1f} | {_fmt(r['gc_max_ms'])} | {r['windows_with_gc']} / "
            f"{len(r['late_windows'])} | {ratio} | {_fmt(r['max_sampled_client_rss_mib'])} |"
        )
    lines.extend(["", "* Peak sampled client RSS within replay; 200 ms external polling."])
    report = "\n".join(lines) + "\n"
    (root / "pressure.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
