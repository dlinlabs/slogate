"""Local known-answer controls for the performance gate. No model download or GPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cpu_pressure import local_engine

from slogate import __version__, workload
from slogate.fake_engine import EngineProfile
from slogate.paired_gate import EXIT, evaluate, markdown, record
from slogate.paired_plan import freeze
from slogate.report import write_json


def identity(profile: EngineProfile) -> dict:
    return {"version": f"slogate-fake-engine {__version__}", "profile": profile.to_dict()}


def collect(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    w = workload.synthetic(
        name="ttft-control",
        duration_s=1.5,
        rate_rps=18,
        input_tokens=[32, 64],
        max_tokens=4,
        seed=20261123,
    )
    work = workload.save(w, root / "workload.json")
    a = EngineProfile(ttft_s=0.025, itl_s=0.002, jitter=0.0, seed=7)
    b = replace(a, ttft_s=0.085)
    cases = [
        ("unchanged", a, 6, "pass"),
        ("delayed", b, 6, "fail"),
        ("underpowered", b, 2, "inconclusive"),
        ("wrong-arm", b, 1, "invalid_measurement"),
    ]
    # Freeze every expectation and margin before contacting either server.
    write_json(
        root / "controls.json",
        {
            "scope": "CPU fake engine, not vLLM calibration",
            "practical_margin_ms": 10,
            "cases": {name: expected for name, _, _, expected in cases},
        },
    )
    for name, profile, pairs, _ in cases:
        freeze(
            root / name,
            w,
            kind="fake",
            baseline=identity(a),
            candidate=identity(profile),
            pairs=pairs,
            block_pairs=1,
            margin_ms=10,
            ceiling_ms=None,
            warmup_requests=1,
        )
    with local_engine(a) as url_a, local_engine(b) as url_b:
        for name, profile, pairs, _ in cases:
            directory = root / name
            for _ in range(pairs):
                record(directory, "A", url_a, work, None, None)
                # The last control intentionally supplies the wrong pinned profile.
                endpoint = url_b if profile != a and name != "wrong-arm" else url_a
                record(directory, "B", endpoint, work, None, None)


def score(root: Path) -> int:
    controls = json.loads((root / "controls.json").read_bytes())
    scores = []
    for name, expected in controls["cases"].items():
        directory = root / name
        result = evaluate(directory)
        write_json(directory / "gate.json", result)
        (directory / "gate.md").write_text(markdown(result), encoding="utf-8")
        relative = result["relative"]
        scores.append(
            {
                "case": name,
                "expected": expected,
                "observed": result["verdict"],
                "exit_code": EXIT[result["verdict"]],
                "recorded_runs": len(result["runs"]),
                "interval_ms": [relative.get("lower_ms"), relative.get("upper_ms")],
                "reasons": result["invalid_reasons"],
            }
        )
        print(f"{name}: expected {expected}, observed {result['verdict']}", flush=True)
    w = workload.load(root / "workload.json")
    write_json(root / "scorecard.json", {"requests_per_run": len(w.requests), "cases": scores})
    return 0 if all(c["observed"] == c["expected"] for c in scores) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="new directory (existing for --report-only)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        collect(args.out)
    return 0 if args.collect_only else score(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
