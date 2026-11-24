# Offline correctness gate

`slogate correctness --plan PLAN --reference A --candidate B` compares three local
UTF-8 JSON files and writes a JSON report to stdout. `python -m slogate.correctness`
accepts the same flags independently of the top-level CLI. Neither command runs a
model, contacts a server, reads a performance replay, or modifies the inputs.

## Try the synthetic controls

From an installed checkout, use a **new** output directory. The demo leaves small,
readable inputs, each verdict, and a `scorecard.json`. All identities, tokens and scores
are invented.

```powershell
python scripts\correctness_demo.py --out runs\correctness-demo
$e = "runs\correctness-demo\exact"
$d = "runs\correctness-demo\drift"
slogate correctness --plan "$e\plan.json" --reference "$e\reference.json" --candidate "$e\unchanged.json"
slogate correctness --plan "$e\plan.json" --reference "$e\reference.json" --candidate "$e\changed-token.json"
slogate correctness --plan "$d\plan.json" --reference "$d\reference.json" --candidate "$d\changed-score.json"
slogate correctness --plan "$d\plan.json" --reference "$d\reference.json" --candidate "$d\missing-score.json"
```

Expected exits in order: **0** (identical), **2** (`alpha` output index 1: `22` vs `99`),
**2** (same target `22`, logprob `-2` vs `-20`; eval unchanged), **4** (missing score).
These test the **checker**, not vLLM, a GPU, or inference quality.

## What to freeze

1. Choose the model, tokenizer, prompt suite, case IDs, tokenized prompts, protocol
   and mode independently of the observed outputs. For drift, fix gold answer IDs and
   all three limits below. Use non-sensitive cases; token IDs can reveal prompt content.
2. Collect **separate correctness evidence** for the reference. Freeze the plan with
   `reference_sha256` set to the SHA-256 of the exact reference file bytes. Archive both
   before inspecting B. Do not discover coverage by intersecting A and B.
3. Collect the candidate independently. Supply the frozen plan from trusted storage.
   Optionally pass `--plan-sha256 HEX` from a separately saved receipt (PowerShell:
   `(Get-FileHash -Algorithm SHA256 $planFile).Hash.ToLowerInvariant()`). Even a line-ending
   change alters a file pin. The report includes digests of the files actually read.
   A digest detects replacement relative to a trusted pin; it is not a signature,
   timestamp, or proof that two independent collections took place.

## V1 JSON contract

| Document | Required top-level fields |
|---|---|
| Plan | `schema: "correctness-plan-v1"`, `mode: "exact"` or `"drift"`, `backend: "synthetic"` or `"vllm"`, `reference_sha256`, `identity`, `protocol`, `cases`; **also** `limits` in drift mode |
| Either evidence file | `schema: "correctness-evidence-v1"`, `role: "reference"` or `"candidate"` as applicable, `identity`, `protocol`, `provenance`, `cases` |

- **Identity:** `{"model_sha256": "...", "tokenizer_sha256": "...", "suite_sha256": "..."}`.
  Use digests of immutable artifacts or manifests identifying the base weights/config,
  tokenizer (including special tokens/template), and prompt suite. All SHA-256s are 64
  lowercase hex digits. Both arms must report the plan's identities. The checker does
  not download or authenticate these artifacts.
- **Protocol:** `{"decoding":"greedy","temperature":0,"ignore_eos":false,"max_new_tokens":4}`
  (replace 4 with a positive integer). V1 means one greedy completion, natural EOS or
  the declared cap; no custom stop rules, minimum length, penalties, constrained decoding
  or other logit modifiers. Report complete generated IDs, including a generated EOS;
  do not infer IDs by retokenizing text. In drift mode add
  `"scoring":"teacher-forced-raw-ln"` to this object in **all three** files.
- **Provenance:** synthetic evidence has exactly `{"backend":"synthetic"}`. vLLM evidence
  requires `backend: "vllm"` plus resolved identifiers `version`, `hardware`,
  `attention_backend`, `dtype`. Use the actual build, an opaque hardware fingerprint
  (not a credential), selected backend and compute dtype. Blank labels and `auto`,
  `default`, `latest`, `n/a`, `unknown`, `unspecified`, `unsupported` (ignoring case
  and surrounding whitespace) are not resolved identifiers. Both arms must declare the same
  hardware. **Exact** additionally requires `"VLLM_BATCH_INVARIANT":"1"` and
  `"batch_invariance_supported":true` on each arm, with all provenance fields matching.
  This is a collector attestation for the actual configuration, not a lookup based on
  the model name or a check of the scorer's own environment. Do not attest unknown support.
  Drift permits the declared version, attention backend and dtype to differ.
- **Coverage:** `cases` is a nonempty array, indexed by `id`, not array order. IDs match
  `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` and must occur exactly once per document; missing
  **or extra** cases invalidate the evidence. All token arrays below are nonempty,
  ordered, nonnegative JSON integers (not booleans). Output and gold lengths cannot
  exceed the cap. Duplicate JSON keys, unknown fields, wrong types and non-finite
  numbers are not accepted.

### Exact: scheduler-only changes

Plan case: `{"id":"alpha","prompt_ids":[11,12]}`. Evidence case:
`{"id":"alpha","prompt_ids":[11,12],"output_ids":[21,22],"complete":true}`.
`prompt_ids` must be the actual generation input, including any special tokens.
`complete` attests normal termination, not a partial stream, timeout or usage count.
Compare the entire output sequence and length. Each case reports both lengths and the
first differing zero-based index and IDs (`null` on the side that ended); an identical
case has `first_difference: null`.

### Drift: kernel, version or precision changes

Add `"expected_ids":[21,22]` to that plan case. These are both the gold answer to a
small fixed-answer task and the continuation to score. Corresponding evidence case:

```json
{
  "id":"alpha", "prompt_ids":[11,12], "output_ids":[21,22], "complete":true,
  "teacher_forced":{"input_ids":[11,12,21,22],"logprobs":[-1,-2]}
}
```

`input_ids` is the full **actual input to the scoring pass**: it must equal the frozen
prompt concatenated with the frozen target IDs. At target index `i` supply exactly one
raw conditional natural log probability, `ln P(target[i] | prompt, target[:i])`.
Do not copy selected-token scores from a different generated path, substitute a top-k
neighbor, or skip positions. `output_ids` separately records the completed, **free-running**
greedy answer for evaluation; it need not follow the teacher-forced path.

Example plan limits: `{"max_abs_logprob_delta":0.1,"min_candidate_correct":1,"max_lost_correct":0}`.
Every target's absolute A/B logprob difference must be **at most** the first limit.
The report gives each case's worst delta and first excess. Scores must be nonpositive;
the tolerance must be nonnegative. Numbers that overflow binary64 or underflow to zero
are rejected, but comparisons use the supplied JSON decimals, not rounded binary floats.
Report scores and deltas are decimal strings.

An eval answer is correct iff its complete output IDs equal `expected_ids`. B must
have **at least** `min_candidate_correct` correct answers and lose **at most**
`max_lost_correct` cases that A got right. Gains cannot hide losses. Both limits are
integers from zero through the frozen case count; there are no implicit tolerances.
The report gives both arms' counts and the loss count. This measures only the declared
fixed-answer suite, not semantic equivalence, general accuracy, or a statistical bound.

## Exit contract

| Exit | Status | Meaning |
|---:|---|---|
| 0 | `pass` | Full, comparable evidence meets every declared check. |
| 2 | `fail` | Full, comparable evidence shows an ID/length difference or a breached numeric/eval limit. |
| 4 | `unverified` | Missing, mutated reference, malformed, unsupported, incomplete or noncomparable evidence. Includes missing scores or eval outputs, even if missing on both arms. |
| 1 | `error` | Missing/malformed plan, unsupported plan, or wrong plan pin. CLI syntax errors also exit 1, with usage on stderr instead of JSON. |

`--help` exits 0 without a verdict. Evidence validation precedes scoring: no partial
verdict from a subset of cases. Reports contain no timings and cannot establish which
change caused a difference. Collection and independent verification of runtime provenance
are outside the checker.

Sources for the vLLM collection requirements: [v0.30.0 sampling parameters](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/sampling_params.py)
(temperature zero is greedy), [batch invariance](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/batch_invariance.md)
(opt-in flag and support limitations), and [reproducibility](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/usage/reproducibility.md)
(same hardware/version caveat). None is evidence that a particular SUT ran with those settings.
