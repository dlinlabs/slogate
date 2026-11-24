"""Verify recorded-schedule reproduction separately from the client's load-fidelity verdict."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .pressure_report import occupancy, summarize_run
from .report import RECORDS_FILE, read_records, read_run, write_json


def timing_errors(source: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    a = {r["id"]: r for r in source}
    b = {r["id"]: r for r in actual}
    if len(a) != len(source) or len(b) != len(actual):
        raise ValueError("duplicate IDs cannot be paired")
    keys = ("scheduled_s", "chunk_tokens", "prompt_tokens", "output_tokens", "finish_reason")
    errors: dict[str, list[float]] = {
        k: [] for k in ("ttft", "chunk_offset", "global_chunk", "chunk_gap", "end")
    }
    wrong = set(a).symmetric_difference(b)
    for rid in a.keys() & b.keys():
        s, r = a[rid], b[rid]
        if not (s["success"] and r["success"]) or any(s[k] != r[k] for k in keys):
            wrong.add(rid)
            continue
        t, u = np.asarray(s["chunk_times_s"]), np.asarray(r["chunk_times_s"])
        if not len(t) or len(t) != len(u) or len(t) != len(s["chunk_tokens"]):
            wrong.add(rid)
            continue
        # Individual request latency errors vs. cross-request alignment to the arrival schedule.
        delta = u - t
        offset_delta = delta - (r["launched_s"] - s["launched_s"])
        errors["ttft"].append(float(abs(offset_delta[0])))
        errors["chunk_offset"].extend(np.abs(offset_delta).tolist())
        errors["global_chunk"].extend(np.abs(delta).tolist())
        errors["chunk_gap"].extend(np.abs(np.diff(delta)).tolist())
        errors["end"].append(
            abs((r["finished_s"] - r["launched_s"]) - (s["finished_s"] - s["launched_s"]))
        )
    return {
        "source_requests": len(a),
        "actual_requests": len(b),
        "unmatched_ids": sorted(wrong),
        "matched_content_chunks": len(errors["global_chunk"]),
        **{
            f"{k}_p99_abs_ms": float(np.percentile(v, 99)) * 1000 if v else None
            for k, v in errors.items()
        },
    }


def compare_run(root: Path, source: Path, name: str, protocol: dict[str, Any]) -> dict[str, Any]:
    with (source / name / RECORDS_FILE).open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    measured = read_run(root / name)
    expected = read_run(source / name)
    server = measured["meta"]["server"]["version"]
    anchor = protocol.get("anchor", "request")
    if (
        checksum != protocol["source_records_sha256"][name]
        or server["source_records_sha256"] != checksum
        or server["source"] != name
        or measured["meta"]["workload"]["digest"] != protocol["workload_digest"]
        or expected["meta"]["workload"]["digest"] != protocol["workload_digest"]
        or server.get("anchor", "request") != anchor
        or bool(measured["meta"]["replay_config"].get("local_fixture_clock"))
        != (anchor == "shared")
    ):
        raise ValueError(f"{name}: source/run identity mismatch")
    original = read_records(source / name)
    fit = timing_errors(original, read_records(root / name))
    peak, _, area = occupancy(original, [])
    wall = expected["replay"]["wall_s"]
    diag = summarize_run(root, name)
    fit.update(
        peak_rel=abs(diag["peak_open"] - peak) / peak,
        request_seconds_rel=abs(diag["request_seconds"] - area) / area,
        wall_abs_s=abs(diag["wall_s"] - wall),
        source_peak=peak,
        source_request_seconds=area,
        source_wall_s=wall,
    )
    missed = [
        k for k, limit in protocol["match_limits"].items() if fit[k] is None or fit[k] > limit
    ]
    if fit["unmatched_ids"]:
        missed.append("request/chunk identity")
    return {
        "name": name,
        "source_sha256": checksum,
        "timing_match": not missed,
        "missed_match_checks": missed,
        "timing": fit,
        "client": diag,
    }


def main(root: Path, source: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("mode") != "recorded-client-content-v1":
        raise ValueError("recorded-client-content-v1 protocol required")
    rows = [compare_run(root, source, name, protocol) for name in protocol["names"]]
    write_json(root / "timeline.json", {"protocol": protocol, "runs": rows})
    matched = [row["client"] for row in rows if row["timing_match"]]
    joint = sum(bool(c["long_gc_generations"]) and bool(c["late_windows"]) for c in matched)
    lines = [
        f"Loopback replay of E0 client-observed content times (no GPU; "
        f"{protocol.get('anchor', 'request')} clock).",
        "Timing match is independent of the client's unchanged 50 ms fidelity gate.",
        f"Timing matches: {len(matched)}/{len(rows)}. Within matched runs: "
        f"{sum(c['valid'] for c in matched)} valid, {joint} with both loop and GC over threshold "
        "(not necessarily at the same instant).",
        "",
        "| Run | Timing match | Content chunks | TTFT / chunk-offset / global / gap "
        "p99 absolute error (ms) |"
        " Peak open | Wall s | Valid | Launch / loop / GC max (ms) | Late wakes with GC / total |",
        "|---|---|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        c, t = row["client"], row["timing"]
        err = " / ".join(
            "-" if t[k] is None else f"{t[k]:.1f}"
            for k in (
                "ttft_p99_abs_ms",
                "chunk_offset_p99_abs_ms",
                "global_chunk_p99_abs_ms",
                "chunk_gap_p99_abs_ms",
            )
        )
        maxima = " / ".join(
            "-" if c[k] is None else f"{c[k]:.1f}"
            for k in ("launch_max_ms", "loop_max_ms", "gc_max_ms")
        )
        match = "yes" if row["timing_match"] else "NO: " + ", ".join(row["missed_match_checks"])
        lines.append(
            f"| {row['name']} | {match} | {t['matched_content_chunks']} | {err} | {c['peak_open']} "
            f"| {c['wall_s']:.2f} | {c['valid']} | {maxima} | "
            f"{c['windows_with_gc']} / {len(c['late_windows'])} |"
        )
    lines.extend(
        [
            "",
            "Filler text and empty/control events are synthetic. Widths and final usage"
            " are taken from the capture; count mismatches are intentionally preserved.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (root / "timeline.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
