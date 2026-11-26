"""Synthetic token IDs and invented scores only. No inference, network, replay or timing."""

from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from decimal import localcontext
from pathlib import Path
from typing import Any

import pytest

from slogate.cli import main as slogate_main
from slogate.correctness import check

type Doc = dict[str, Any]
type Files = tuple[Path, Path, Path]

_example = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "correctness_demo.py")
)["synthetic_documents"]


def _toy(mode: str = "exact", backend: str = "synthetic") -> tuple[Doc, Doc, Doc]:
    plan, ref, cand = _example(mode)
    if backend == "vllm":  # Mocked declarations to exercise validation, not a real SUT.
        plan["backend"] = backend
        provenance: Doc = {
            "backend": backend,
            "version": "0.30.0",
            "hardware": "toy",
            "attention_backend": "toy",
            "dtype": "bf16",
        }
        if mode == "exact":
            provenance.update(VLLM_BATCH_INVARIANT="1", batch_invariance_supported=True)
            plan.update(
                schema="correctness-plan-v2",
                collection={
                    "spec_sha256": "e" * 64,
                    "reference_arm_pin_sha256": "a" * 64,
                    "candidate_arm_pin_sha256": "b" * 64,
                },
            )
            for doc, pin in ((ref, "a" * 64), (cand, "b" * 64)):
                doc.update(
                    schema="correctness-evidence-v2",
                    collection={"spec_sha256": "e" * 64, "arm_pin_sha256": pin},
                )
        ref["provenance"], cand["provenance"] = provenance, deepcopy(provenance)
    return plan, ref, cand


def _dump(path: Path, doc: Doc) -> None:
    path.write_bytes((json.dumps(doc, indent=2) + "\n").encode())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(root: Path, plan: Doc, ref: Doc, cand: Doc) -> Files:
    p, r, c = root / "plan.json", root / "reference.json", root / "candidate.json"
    _dump(r, ref)
    _dump(p, {"reference_sha256": _sha(r), **plan})
    if cand["schema"] == "correctness-evidence-v2":
        cand = {**cand, "collection": {**cand["collection"], "plan_sha256": _sha(p)}}
    _dump(c, cand)
    return p, r, c


def _cli(*argv: str) -> int:
    return slogate_main(["correctness", *argv])


def _run(files: Files, *args: str) -> int:
    p, r, c = files
    return _cli("--plan", str(p), "--reference", str(r), "--candidate", str(c), *args)


def test_synthetic_controls(tmp_path_factory, capsys) -> None:
    """Exercise each mode through the registered public command."""
    root = tmp_path_factory.mktemp("synthetic", numbered=False)
    for mode in ("exact", "drift"):
        folder = root / mode
        folder.mkdir()
        plan, ref, cand = _toy(mode)
        p, r, c = _freeze(folder, plan, ref, cand)
        assert _run((p, r, c), "--plan-sha256", _sha(p)) == 0
        report = json.loads(capsys.readouterr().out)
        assert (report["status"], report["backend"], report["planned_cases"]) == (
            "pass",
            "synthetic",
            2,
        )
        assert report["sha256"] == {"plan": _sha(p), "reference": _sha(r), "candidate": _sha(c)}

        changed = deepcopy(cand)
        if mode == "exact":
            changed["cases"][0]["output_ids"][1] = 99
        else:
            changed["cases"][0]["teacher_forced"]["logprobs"][1] = -20
        _dump(folder / "changed.json", changed)
        assert _run((p, r, folder / "changed.json")) == 2
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "fail"
        row = report["cases"][0]
        if mode == "exact":
            assert row == {
                "id": "alpha",
                "reference_length": 2,
                "candidate_length": 2,
                "first_difference": {
                    "index": 1,
                    "reference_token_id": 22,
                    "candidate_token_id": 99,
                },
            }
        else:
            assert report["eval"]["pass"] and not report["logprob_pass"]
            assert row["first_excess"] == {
                "index": 1,
                "target_id": 22,
                "reference_logprob": "-2",
                "candidate_logprob": "-20",
                "abs_delta": "18",
            }

        missing = deepcopy(cand)
        if mode == "exact":
            del missing["cases"][0]["output_ids"]
        else:
            missing["cases"][0]["teacher_forced"]["logprobs"].pop()
        _dump(folder / "missing.json", missing)
        assert _run((p, r, folder / "missing.json")) == 4
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "unverified" and "candidate: alpha" in report["reason"]


@pytest.mark.parametrize(("output", "left", "right"), [([21], 22, None), ([21, 22, 3], None, 3)])
def test_length_is_part_of_exact_match(tmp_path, output, left, right) -> None:
    plan, ref, cand = _toy()
    cand["cases"][0]["output_ids"] = output
    result = check(*_freeze(tmp_path, plan, ref, cand))
    assert result["status"] == "fail"
    row = result["cases"][0]
    assert row["first_difference"] == {
        "index": min(2, len(output)),
        "reference_token_id": left,
        "candidate_token_id": right,
    }
    assert row["candidate_length"] == len(output)


def test_case_order_irrelevant_and_report_repeatable(tmp_path, capsys) -> None:
    plan, ref, cand = _toy()
    plan["cases"].reverse()
    cand["cases"].reverse()
    files = _freeze(tmp_path, plan, ref, cand)
    before = [_sha(f) for f in files]
    assert _run(files) == 0
    text = capsys.readouterr().out
    assert [c["id"] for c in json.loads(text)["cases"]] == ["alpha", "beta"]
    assert _run(files) == 0 and capsys.readouterr().out == text
    assert before == [_sha(f) for f in files]


@pytest.mark.parametrize("mode", ["exact", "drift"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "different_prompt"])
def test_symmetric_bad_coverage_never_passes(tmp_path, mode, change) -> None:
    plan, ref, cand = _toy(mode)
    for arm in (ref, cand):
        cases = arm["cases"]
        if change == "missing":
            cases.pop()
        elif change == "duplicate":
            cases.append(cases[0])
        elif change == "extra":
            cases.append({**cases[0], "id": "extra"})
        else:
            cases[0]["prompt_ids"] = [99]
    result = check(*_freeze(tmp_path, plan, ref, cand))
    assert result["status"] == "unverified"
    assert result["reason"].startswith("reference:")


@pytest.mark.parametrize("key", ["model_sha256", "tokenizer_sha256", "suite_sha256"])
def test_identity_must_match_external_plan(tmp_path, key) -> None:
    plan, ref, cand = _toy()
    ref["identity"][key] = cand["identity"][key] = "f" * 64
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 0.1),
        ("temperature", False),
        ("ignore_eos", 0),
        ("ignore_eos", True),
        ("decoding", "sampling"),
        ("max_new_tokens", 7),
        ("max_new_tokens", True),
        ("stop", ["x"]),
    ],
)
def test_mismatched_or_unsupported_sampling(tmp_path, key, value) -> None:
    plan, ref, cand = _toy()
    cand["protocol"][key] = value
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize("bad", [None, [], "21,22", [True], [-1], [2.0], [1, 1, 1, 1, 1]])
def test_invalid_output_ids_not_an_exact_mismatch(tmp_path, bad) -> None:
    plan, ref, cand = _toy()
    cand["cases"][0]["output_ids"] = bad
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


def test_text_or_usage_is_not_a_substitute_for_ids(tmp_path) -> None:
    plan, ref, cand = _toy()
    for arm in (ref, cand):
        del arm["cases"][0]["output_ids"]
        arm["cases"][0].update(text="same", usage={"completion_tokens": 2})
    result = check(*_freeze(tmp_path, plan, ref, cand))
    assert result["status"] == "unverified" and "missing output_ids" in result["reason"]


def test_matching_partial_generations_are_not_complete_sequences(tmp_path) -> None:
    plan, ref, cand = _toy()
    for arm in (ref, cand):
        arm["cases"][0]["complete"] = False
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


def test_sampled_token_or_processed_logprobs_not_teacher_forced_raw_scores(tmp_path) -> None:
    plan, ref, cand = _toy("drift")
    cand["protocol"]["scoring"] = "sampled-token-logprobs"
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("backend", "another-engine"),
        ("backend", "synthetic"),
        ("VLLM_BATCH_INVARIANT", None),
        ("VLLM_BATCH_INVARIANT", "0"),
        ("VLLM_BATCH_INVARIANT", 1),
        ("batch_invariance_supported", False),
        ("batch_invariance_supported", 1),
        ("hardware", "other-host"),
        ("version", "other-version"),
        ("dtype", "other-precision"),
        ("attention_backend", ""),
    ],
)
def test_scheduler_requires_comparable_supported_provenance(tmp_path, key, value) -> None:
    plan, ref, cand = _toy(backend="vllm")
    if value is None:
        del cand["provenance"][key]
    else:
        cand["provenance"][key] = value
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


def test_modes_distinguish_version_kernel_precision_changes(tmp_path) -> None:
    assert check(*_freeze(tmp_path, *_toy(backend="vllm")))["status"] == "pass"
    plan, ref, cand = _toy("drift", "vllm")
    cand["provenance"].update(version="0.31.0", attention_backend="toy2", dtype="fp16")
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "pass"
    cand["provenance"]["hardware"] = "other-host"
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize(
    ("role", "key"),
    [
        ("reference", "arm_pin_sha256"),
        ("reference", "spec_sha256"),
        ("candidate", "arm_pin_sha256"),
        ("candidate", "spec_sha256"),
        ("candidate", "plan_sha256"),
    ],
)
def test_native_exact_checks_every_role_binding(tmp_path, role, key) -> None:
    plan, ref, cand = _toy(backend="vllm")
    if role == "reference":
        ref["collection"][key] = cand["collection"][key] if key == "arm_pin_sha256" else "f" * 64
    p, r, c = _freeze(tmp_path, plan, ref, cand)
    if role == "candidate":
        doc = json.loads(c.read_bytes())
        doc["collection"][key] = ref["collection"][key] if key == "arm_pin_sha256" else "f" * 64
        _dump(c, doc)
    result = check(p, r, c, plan_sha256=_sha(p))
    assert result["status"] == "unverified" and result["reason"].startswith(f"{role}: collection")


def test_native_v2_does_not_accept_content_only_candidate(tmp_path) -> None:
    p, r, c = _freeze(tmp_path, *_toy(backend="vllm"))
    doc = json.loads(c.read_bytes())
    doc.pop("collection")
    doc["schema"] = "correctness-evidence-v1"
    _dump(c, doc)
    result = check(p, r, c, plan_sha256=_sha(p))
    assert (
        result["status"] == "unverified"
        and "candidate: evidence: missing collection" in result["reason"]
    )


def test_unresolved_backend_on_both_arms_does_not_establish_support(tmp_path) -> None:
    plan, ref, cand = _toy(backend="vllm")
    ref["provenance"]["attention_backend"] = cand["provenance"]["attention_backend"] = "auto"
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize(
    "path", [[11, 99, 21, 22], [11, 12, 22, 21], [12, 21, 22], [True, 12, 21, 22]]
)
def test_symmetric_misaligned_teacher_forcing_is_not_numeric_evidence(tmp_path, path) -> None:
    plan, ref, cand = _toy("drift")
    for arm in (ref, cand):
        arm["cases"][0]["teacher_forced"]["input_ids"] = path
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


def test_gold_path_not_generated_path_and_inclusive_eval_limits(tmp_path) -> None:
    plan, ref, cand = _toy("drift")
    # B gets beta right but loses alpha. Matching accuracy alone would hide the regression.
    cand["cases"][0]["output_ids"] = [51, 52]
    cand["cases"][1]["output_ids"] = [23]
    result = check(*_freeze(tmp_path, plan, ref, cand))
    assert result["status"] == "fail" and result["logprob_pass"]
    assert result["eval"] == {
        "pass": False,
        "cases": 2,
        "reference_correct": 1,
        "candidate_correct": 1,
        "lost_correct": 1,
        "min_candidate_correct": 1,
        "max_lost_correct": 0,
    }
    plan["limits"]["max_lost_correct"] = 1
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "pass"
    plan["limits"]["min_candidate_correct"] = 2
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "fail"


def test_logprob_comparison_uses_declared_decimal_boundary(tmp_path) -> None:
    plan, ref, cand = _toy("drift")
    cand["cases"][0]["teacher_forced"]["logprobs"][0] = -1.1
    p, r, c = _freeze(tmp_path, plan, ref, cand)
    assert check(p, r, c)["status"] == "pass"  # |-1.1 - (-1)| == 0.1
    c.write_bytes(c.read_bytes().replace(b"-1.1", b"-1.10000000000000000000000000000001"))
    result = check(p, r, c)
    assert result["status"] == "fail"
    assert result["cases"][0]["first_excess"]["abs_delta"] == "0.10000000000000000000000000000001"


def test_external_decimal_context_does_not_change_verdict(tmp_path) -> None:
    plan, ref, cand = _toy("drift")
    cand["cases"][0]["teacher_forced"]["logprobs"][0] = -10000
    files = _freeze(tmp_path, plan, ref, cand)
    with localcontext() as ctx:
        ctx.Emax = 2
        assert check(*files)["cases"][0]["first_excess"]["abs_delta"] == "9999"


@pytest.mark.parametrize(
    "bad", [[], [-1], [-1, -2, -3], [0, True], [-1, 1], [-1, None], [-1, "-2"]]
)
def test_drift_requires_a_valid_score_at_every_target(tmp_path, bad) -> None:
    plan, ref, cand = _toy("drift")
    for arm in (ref, cand):
        arm["cases"][0]["teacher_forced"]["logprobs"] = bad
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize("key", ["teacher_forced", "output_ids"])
def test_absent_scoring_or_eval_on_both_arms_is_unverified(tmp_path, key) -> None:
    plan, ref, cand = _toy("drift")
    for arm in (ref, cand):
        del arm["cases"][1][key]
    cand["cases"][0]["output_ids"][0] = 100
    assert check(*_freeze(tmp_path, plan, ref, cand))["status"] == "unverified"


@pytest.mark.parametrize("bad", [b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e-999"])
def test_nonfinite_and_out_of_range_numbers(tmp_path, bad) -> None:
    p, r, c = _freeze(tmp_path, *_toy("drift"))
    c.write_bytes(c.read_bytes().replace(b"-3", bad))
    assert check(p, r, c)["status"] == "unverified"
    p.write_bytes(p.read_bytes().replace(b"0.1", bad))
    assert check(p, r, c)["status"] == "error"


@pytest.mark.parametrize(
    "fault", ["limits", "negative", "bool", "gold", "empty", "duplicate", "sha", "unknown"]
)
def test_bad_plan_is_an_error_not_a_measured_failure(tmp_path, capsys, fault) -> None:
    plan, ref, cand = _toy("drift")
    if fault == "limits":
        del plan["limits"]
    elif fault == "negative":
        plan["limits"]["max_abs_logprob_delta"] = -1
    elif fault == "bool":
        plan["limits"]["max_lost_correct"] = False
    elif fault == "gold":
        del plan["cases"][0]["expected_ids"]
    elif fault == "empty":
        plan["cases"] = []
    elif fault == "duplicate":
        plan["cases"].append(plan["cases"][0])
    elif fault == "sha":
        plan["reference_sha256"] = "mutable-label"
    else:
        plan["schema"] = "future-plan"
    assert _run(_freeze(tmp_path, plan, ref, cand)) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_immutable_files_distinct_arms_and_external_plan_pin(tmp_path) -> None:
    p, r, c = _freeze(tmp_path, *_toy())
    old_pin = _sha(p)
    assert check(p, r, r)["status"] == "unverified"
    p.write_bytes(p.read_bytes() + b"\n")
    assert check(p, r, c, plan_sha256=old_pin)["status"] == "error"
    r.write_bytes(r.read_bytes() + b"\n")
    result = check(p, r, c)
    assert result["status"] == "unverified" and "reference: file SHA-256" in result["reason"]


def test_missing_file_bad_json_and_duplicate_json_keys(tmp_path) -> None:
    p, r, c = _freeze(tmp_path, *_toy())
    assert check(tmp_path, r, c)["status"] == "error"
    assert check(p, tmp_path / "absent", c)["status"] == "unverified"
    assert check(p, r, tmp_path / "absent")["status"] == "unverified"
    c.write_bytes(
        c.read_bytes().replace(b'"temperature": 0', b'"temperature": 9, "temperature": 0')
    )
    result = check(p, r, c)
    assert result["status"] == "unverified" and "duplicate JSON object key" in result["reason"]
    for data in (b'"\xff"', b'{"truncated"', b"null"):
        c.write_bytes(data)
        assert check(p, r, c)["status"] == "unverified"


def test_argparse_syntax_error_distinct_from_a_failed_comparison(tmp_path, capsys) -> None:
    files = _freeze(tmp_path, *_toy())
    with pytest.raises(SystemExit) as usage:
        _run(files, "--unsupported")
    assert usage.value.code == 1 and "usage:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as missing:
        _cli()
    assert missing.value.code == 1
    with pytest.raises(SystemExit) as help_requested:
        _cli("--help")
    assert help_requested.value.code == 0 and "--plan" in capsys.readouterr().out
