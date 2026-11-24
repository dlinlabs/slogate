"""Read-only comparison of separately collected correctness evidence. See docs/correctness.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Never

from ._cli_args import UsageParser

EXIT = {"pass": 0, "error": 1, "fail": 2, "unverified": 4}
_SHA256 = re.compile("[0-9a-f]{64}")
_CASE_ID = re.compile("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_IDENTITY = "model_sha256 tokenizer_sha256 suite_sha256"
_UNRESOLVED = {"", "auto", "default", "latest", "n/a", "unknown", "unspecified", "unsupported"}


class _Invalid(ValueError):
    """A diagnostic about the document, not a measured difference."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Invalid("duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(_: str) -> Never:
    raise _Invalid("non-finite JSON number")


def _read(
    path: Path, role: str, digests: dict[str, str], expected_sha256: str | None = None
) -> Any:
    try:
        if not path.is_file():
            raise _Invalid("missing or not a regular file")
        data = path.read_bytes()
    except OSError as exc:
        raise _Invalid(f"cannot read file ({type(exc).__name__})") from exc
    digest = digests[role] = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise _Invalid("file SHA-256 differs from frozen digest")
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except _Invalid:
        raise
    except (ValueError, InvalidOperation, RecursionError) as exc:
        raise _Invalid(f"invalid UTF-8 JSON ({type(exc).__name__})") from exc


def _fields(value: Any, names: str, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _Invalid(f"{where}: expected an object")
    required = set(names.split())
    if missing := sorted(required - value.keys()):
        raise _Invalid(f"{where}: missing {', '.join(missing)}")
    if value.keys() - required:
        raise _Invalid(f"{where}: unexpected field(s)")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _Invalid(f"{where}: expected a lowercase SHA-256 hex digest")
    return value


def _tokens(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(t) is not int or t < 0 for t in value):
        raise _Invalid(f"{where}: expected a nonempty array of nonnegative integer token IDs")
    return value


def _number(value: Any, where: str) -> Decimal:
    if type(value) is not int and not isinstance(value, Decimal):
        raise _Invalid(f"{where}: expected a JSON number")
    n = Decimal(value)
    binary64 = float(n)
    if not math.isfinite(binary64) or (n and not binary64):
        raise _Invalid(f"{where}: number outside finite binary64 range")
    return n if n else Decimal(0)


def _protocol(value: Any, mode: str) -> int:
    names = "decoding temperature ignore_eos max_new_tokens"
    p = _fields(value, names + (" scoring" if mode == "drift" else ""), "protocol")
    if (
        p["decoding"] != "greedy"
        or _number(p["temperature"], "temperature") != 0
        or p["ignore_eos"] is not False
    ):
        raise _Invalid("protocol: requires greedy, temperature 0 and natural EOS")
    if mode == "drift" and p["scoring"] != "teacher-forced-raw-ln":
        raise _Invalid("protocol.scoring: requires teacher-forced raw natural-log probabilities")
    limit = p["max_new_tokens"]
    if type(limit) is not int or limit < 1:
        raise _Invalid("protocol.max_new_tokens: expected a positive integer")
    return limit


def _by_id(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise _Invalid("cases: expected a nonempty array")
    indexed = {}
    for row in value:
        case_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
            raise _Invalid("cases: invalid or missing case ID")
        if case_id in indexed:
            raise _Invalid(f"cases: duplicate ID {case_id}")
        indexed[case_id] = row
    return indexed


@dataclass(frozen=True)
class _Case:
    prompt: list[int]
    gold: list[int]


@dataclass(frozen=True)
class _Limits:
    max_abs_logprob_delta: Decimal
    min_candidate_correct: int
    max_lost_correct: int


@dataclass(frozen=True)
class _Plan:
    mode: str
    backend: str
    reference_sha256: str
    identity: dict[str, str]
    max_new_tokens: int
    cases: dict[str, _Case]
    limits: _Limits | None


def _plan(value: Any) -> _Plan:
    if not isinstance(value, dict) or value.get("schema") != "correctness-plan-v1":
        raise _Invalid("expected schema correctness-plan-v1")
    mode = value.get("mode")
    if mode not in ("exact", "drift"):
        raise _Invalid("mode must be exact or drift")
    fields = "schema mode backend reference_sha256 identity protocol cases"
    p = _fields(value, fields + (" limits" if mode == "drift" else ""), "plan")
    if p["backend"] not in ("synthetic", "vllm"):
        raise _Invalid("backend must be synthetic or vllm")
    ref_sha = _digest(p["reference_sha256"], "reference_sha256")
    identity = _fields(p["identity"], _IDENTITY, "identity")
    for name, digest in identity.items():
        _digest(digest, name)
    cap = _protocol(p["protocol"], mode)
    cases = {}
    for key, row in _by_id(p["cases"]).items():
        _fields(row, "id prompt_ids" + (" expected_ids" if mode == "drift" else ""), key)
        prompt = _tokens(row["prompt_ids"], f"{key}.prompt_ids")
        gold = _tokens(row["expected_ids"], f"{key}.expected_ids") if mode == "drift" else []
        if len(gold) > cap:
            raise _Invalid(f"{key}: expected output exceeds max_new_tokens")
        cases[key] = _Case(prompt, gold)
    limits = None
    if mode == "drift":
        tol = _fields(
            p["limits"], "max_abs_logprob_delta min_candidate_correct max_lost_correct", "limits"
        )
        delta = _number(tol["max_abs_logprob_delta"], "max_abs_logprob_delta")
        if delta < 0:
            raise _Invalid("max_abs_logprob_delta must be nonnegative")
        for name in ("min_candidate_correct", "max_lost_correct"):
            if type(tol[name]) is not int or not 0 <= tol[name] <= len(cases):
                raise _Invalid(f"{name} must be an integer in [0, case count]")
        limits = _Limits(delta, tol["min_candidate_correct"], tol["max_lost_correct"])
    return _Plan(mode, p["backend"], ref_sha, identity, cap, cases, limits)


def _provenance(value: Any, plan: _Plan) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("backend") != plan.backend:
        raise _Invalid("provenance.backend differs from plan or is unsupported")
    fields = "backend"
    if plan.backend == "vllm":
        fields += " version hardware attention_backend dtype"
        if plan.mode == "exact":
            fields += " VLLM_BATCH_INVARIANT batch_invariance_supported"
    p = _fields(value, fields, "provenance")
    if plan.backend == "vllm":
        for name in ("version", "hardware", "attention_backend", "dtype"):
            label = p[name]
            if not isinstance(label, str) or label.strip().lower() in _UNRESOLVED:
                raise _Invalid(f"provenance.{name}: expected a resolved identifier")
        if plan.mode == "exact" and (
            p["VLLM_BATCH_INVARIANT"] != "1" or p["batch_invariance_supported"] is not True
        ):
            raise _Invalid(
                "vLLM exact mode requires supported batch invariance and VLLM_BATCH_INVARIANT=1"
            )
    return p


@dataclass(frozen=True)
class _Observation:
    output: list[int]
    logprobs: list[Decimal]


@dataclass(frozen=True)
class _Evidence:
    provenance: dict[str, Any]
    cases: dict[str, _Observation]


def _evidence(value: Any, role: str, plan: _Plan) -> _Evidence:
    e = _fields(value, "schema role provenance identity protocol cases", "evidence")
    if e["schema"] != "correctness-evidence-v1" or e["role"] != role:
        raise _Invalid(f"expected correctness-evidence-v1 with role {role}")
    provenance = _provenance(e["provenance"], plan)
    identity = _fields(e["identity"], _IDENTITY, "identity")
    for name, pinned in plan.identity.items():
        if identity[name] != pinned:
            raise _Invalid(f"identity.{name} differs from plan")
    if _protocol(e["protocol"], plan.mode) != plan.max_new_tokens:
        raise _Invalid("protocol.max_new_tokens differs from plan")
    rows = _by_id(e["cases"])
    if rows.keys() != plan.cases.keys():
        missing = sorted(plan.cases.keys() - rows.keys())
        extra = sorted(rows.keys() - plan.cases.keys())
        raise _Invalid(f"case IDs differ from plan: missing={missing}, extra={extra}")
    cases = {}
    for key in sorted(rows):
        expected = plan.cases[key]
        names = "id prompt_ids output_ids complete"
        row = _fields(rows[key], names + (" teacher_forced" if plan.limits else ""), key)
        if row["complete"] is not True:
            raise _Invalid(f"{key}: greedy generation not confirmed complete")
        if _tokens(row["prompt_ids"], f"{key}.prompt_ids") != expected.prompt:
            raise _Invalid(f"{key}: generation prompt IDs differ from plan")
        output = _tokens(row["output_ids"], f"{key}.output_ids")
        if len(output) > plan.max_new_tokens:
            raise _Invalid(f"{key}: output exceeds declared max_new_tokens")
        scores = []
        if plan.limits:
            tf = _fields(row["teacher_forced"], "input_ids logprobs", f"{key}.teacher_forced")
            if (
                _tokens(tf["input_ids"], f"{key}.teacher_forced.input_ids")
                != expected.prompt + expected.gold
            ):
                raise _Invalid(f"{key}: teacher-forced prefix or target token IDs differ from plan")
            raw = tf["logprobs"]
            if not isinstance(raw, list) or len(raw) != len(expected.gold):
                raise _Invalid(f"{key}: expected exactly one logprob per frozen target token")
            for i, score in enumerate(raw):
                n = _number(score, f"{key}.logprobs[{i}]")
                if n > 0:
                    raise _Invalid(f"{key}.logprobs[{i}]: a log probability cannot be positive")
                scores.append(n)
        cases[key] = _Observation(output, scores)
    return _Evidence(provenance, cases)


def _first_difference(a: list[int], b: list[int]) -> dict[str, Any] | None:
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else None
        right = b[i] if i < len(b) else None
        if left != right:
            return {"index": i, "reference_token_id": left, "candidate_token_id": right}
    return None


def _delta(a: Decimal, b: Decimal) -> Decimal:
    # Align the two JSON decimals without rounding a token just across a frozen threshold.
    digits = max(len(x.as_tuple().digits) for x in (a, b)) + abs(a.adjusted() - b.adjusted()) + 1
    return Context(prec=digits).subtract(a, b).copy_abs()


def _score(plan: _Plan, ref: _Evidence, cand: _Evidence) -> dict[str, Any]:
    if plan.mode == "exact" and ref.provenance != cand.provenance:
        changed = sorted(k for k, v in ref.provenance.items() if cand.provenance[k] != v)
        raise _Invalid(f"scheduler-only provenance differs between arms: {', '.join(changed)}")
    if ref.provenance.get("hardware") != cand.provenance.get("hardware"):
        raise _Invalid("hardware differs between arms")

    rows = []
    if plan.limits is None:
        for key in sorted(plan.cases):
            a, b = ref.cases[key].output, cand.cases[key].output
            rows.append(
                {
                    "id": key,
                    "reference_length": len(a),
                    "candidate_length": len(b),
                    "first_difference": _first_difference(a, b),
                }
            )
        failed = any(row["first_difference"] is not None for row in rows)
        return {"status": "fail" if failed else "pass", "cases": rows}

    limits = plan.limits
    reference_correct = candidate_correct = lost = 0
    drift_failed = False
    for key in sorted(plan.cases):
        gold = plan.cases[key].gold
        a, b = ref.cases[key], cand.cases[key]
        a_correct, b_correct = a.output == gold, b.output == gold
        reference_correct += a_correct
        candidate_correct += b_correct
        lost += a_correct and not b_correct
        worst = Decimal(0)
        first = None
        for i, (left, right) in enumerate(zip(a.logprobs, b.logprobs, strict=True)):
            delta = _delta(left, right)
            worst = max(worst, delta)
            if first is None and delta > limits.max_abs_logprob_delta:
                drift_failed = True
                first = {
                    "index": i,
                    "target_id": gold[i],
                    "reference_logprob": str(left),
                    "candidate_logprob": str(right),
                    "abs_delta": str(delta),
                }
        rows.append(
            {
                "id": key,
                "reference_correct": a_correct,
                "candidate_correct": b_correct,
                "scored_tokens": len(gold),
                "max_abs_logprob_delta": str(worst),
                "first_excess": first,
            }
        )
    eval_failed = candidate_correct < limits.min_candidate_correct or lost > limits.max_lost_correct
    return {
        "status": "fail" if drift_failed or eval_failed else "pass",
        "logprob_pass": not drift_failed,
        "max_abs_logprob_delta_allowed": str(limits.max_abs_logprob_delta),
        "eval": {
            "pass": not eval_failed,
            "cases": len(rows),
            "reference_correct": reference_correct,
            "candidate_correct": candidate_correct,
            "lost_correct": lost,
            "min_candidate_correct": limits.min_candidate_correct,
            "max_lost_correct": limits.max_lost_correct,
        },
        "cases": rows,
    }


def check(
    plan_path: Path, reference_path: Path, candidate_path: Path, *, plan_sha256: str | None = None
) -> dict[str, Any]:
    """Check snapshots without modifying them. A bad plan is not a failed model comparison."""
    digests: dict[str, str] = {}
    report: dict[str, Any] = {"schema": "correctness-report-v1", "sha256": digests}
    try:
        if plan_sha256 is not None:
            _digest(plan_sha256, "plan_sha256")
        plan = _plan(_read(plan_path, "plan", digests, plan_sha256))
    except _Invalid as exc:
        return {**report, "status": "error", "reason": f"plan: {exc}"}
    report.update(mode=plan.mode, backend=plan.backend, planned_cases=len(plan.cases))
    phase = "reference"
    try:
        ref = _evidence(_read(reference_path, phase, digests, plan.reference_sha256), phase, plan)
        phase = "candidate"
        cand = _evidence(_read(candidate_path, phase, digests), phase, plan)
        phase = "comparison"
        return {**report, **_score(plan, ref, cand)}
    except _Invalid as exc:
        return {**report, "status": "unverified", "reason": f"{phase}: {exc}"}


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True, help="frozen correctness-plan-v1 JSON")
    parser.add_argument(
        "--plan-sha256", help="optional independently saved digest of the plan bytes"
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="digest-pinned reference JSON"
    )
    parser.add_argument("--candidate", type=Path, required=True, help="separate candidate JSON")
    parser.set_defaults(handler=_command)


def _command(args: argparse.Namespace) -> int:
    report = check(args.plan, args.reference, args.candidate, plan_sha256=args.plan_sha256)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return EXIT[report["status"]]


def main(argv: list[str] | None = None) -> int:
    parser = UsageParser(prog="slogate correctness", description=__doc__, allow_abbrev=False)
    _configure(parser)
    return _command(parser.parse_args(argv))


def add_parser(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "correctness", help="check offline token-ID or logprob/eval evidence", allow_abbrev=False
    )
    _configure(parser)


if __name__ == "__main__":
    raise SystemExit(main())
