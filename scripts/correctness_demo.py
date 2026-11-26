"""Create and score invented token evidence without running a language model."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from slogate.correctness import check
from slogate.report import write_json


def synthetic_documents(mode: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if mode not in ("exact", "drift"):
        raise ValueError("expected exact or drift")
    identity = {
        f"{key}_sha256": hashlib.sha256(f"synthetic {key}".encode()).hexdigest()
        for key in ("model", "tokenizer", "suite")
    }
    protocol: dict[str, Any] = {
        "decoding": "greedy",
        "temperature": 0,
        "ignore_eos": False,
        "max_new_tokens": 4,
    }
    suite = [{"id": "alpha", "prompt_ids": [11, 12]}, {"id": "beta", "prompt_ids": [13]}]
    rows = [
        {**deepcopy(case), "output_ids": output, "complete": True}
        for case, output in zip(suite, ([21, 22], [24]), strict=True)
    ]
    plan = {
        "schema": "correctness-plan-v1",
        "mode": mode,
        "backend": "synthetic",
        "identity": identity,
        "protocol": protocol,
        "cases": suite,
    }
    if mode == "drift":
        protocol["scoring"] = "teacher-forced-raw-ln"
        plan["limits"] = {
            "max_abs_logprob_delta": 0.1,
            "min_candidate_correct": 1,
            "max_lost_correct": 0,
        }
        for case, obs, gold, scores in zip(
            suite, rows, ([21, 22], [23]), ([-1, -2], [-3]), strict=True
        ):
            case["expected_ids"] = gold
            obs["teacher_forced"] = {"input_ids": [*obs["prompt_ids"], *gold], "logprobs": scores}
    ref = {
        "schema": "correctness-evidence-v1",
        "role": "reference",
        "identity": deepcopy(identity),
        "protocol": deepcopy(protocol),
        "provenance": {"backend": "synthetic"},
        "cases": rows,
    }
    return plan, ref, {**deepcopy(ref), "role": "candidate"}


def demo(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=False)
    scores = []
    for mode in ("exact", "drift"):
        d = root / mode
        d.mkdir()
        plan, ref, same = synthetic_documents(mode)
        p, r = d / "plan.json", d / "reference.json"
        write_json(r, ref)
        write_json(p, {**plan, "reference_sha256": hashlib.sha256(r.read_bytes()).hexdigest()})
        changed = deepcopy(same)
        trials = [("unchanged", same, "pass")]
        if mode == "exact":
            changed["cases"][0]["output_ids"][1] = 99
            trials.append(("changed-token", changed, "fail"))
        else:
            changed["cases"][0]["teacher_forced"]["logprobs"][1] = -20
            missing, wrong_answer = deepcopy(same), deepcopy(same)
            missing["cases"][0]["teacher_forced"]["logprobs"].pop()
            wrong_answer["cases"][0]["output_ids"] = [21, 99]
            trials.extend(
                [
                    ("changed-score", changed, "fail"),
                    ("missing-score", missing, "unverified"),
                    ("lost-answer", wrong_answer, "fail"),
                ]
            )
        for name, candidate, expected in trials:
            c = d / f"{name}.json"
            write_json(c, candidate)
            report = check(p, r, c)
            write_json(d / f"{name}-report.json", report)
            scores.append(
                {"mode": mode, "case": name, "expected": expected, "observed": report["status"]}
            )
            print(f"{mode}/{name}: {report['status']}")
    write_json(root / "scorecard.json", {"backend": "synthetic", "cases": scores})
    return 0 if all(c["observed"] == c["expected"] for c in scores) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new local directory")
    raise SystemExit(demo(parser.parse_args().out))
