# Paired TTFT gate: one decision, one primary metric

This first increment compares **configuration changes** on one host. Its primary metric is
the run-level p99 time to first *visible* token (TTFT). A supplied margin is the increase
worth blocking, in milliseconds; it is not the noise floor or a user-facing SLO.
Performance and output correctness are separate checks. A performance pass alone is not
permission to deploy.

## Try a known answer without a GPU

From an installed dev checkout, use a **new** output directory:

```bash
PYTHONPATH=src python scripts/paired_demo.py --out runs/week3-controls
```

The script freezes all plans before collecting data. A has a fake 25 ms first-token delay.
B either has the same profile or a fake 85 ms delay. The same 25 arrivals, output lengths,
warmup and 50 ms client-fidelity checks are used throughout. The **10 ms margin is for this
teaching fixture only**. Each arm uses a localhost endpoint; no language model is invoked.

| Control | What should happen | Reason |
|---|---|---|
| Six A/A pairs | Relative pass if the entire interval is below +10 ms | No change |
| Six delayed B pairs | Fail if the entire interval is above +10 ms | Known +60 ms server setting |
| Only two predeclared pairs | Inconclusive | Too few independent observations for the interval |
| Deliberately use A's endpoint for a B pinned to a different profile | Invalid measurement | A label is not proof of which build ran |

These are **expectations**, not hardcoded verdicts. Inspect `scorecard.json` and each
experiment's `gate.json` / `gate.md`. If the client falls behind or the observations are
noisy, keep the actual outcome. Do not rerun until a control passes.

The [local control measurements](results/week3-cpu-pairs-2026-11-23.csv) contain every
preassigned pair, including the deliberately wrong arm. Values are rounded only for display.
Replay a saved experiment with `slogate ab report DIRECTORY` without contacting any server.

## Collect a comparison

```bash
slogate ab init --kind fake --workload "$WORKLOAD" \
  --baseline "$A_VERSION_JSON" --candidate "$B_VERSION_JSON" \
  --pairs 6 --block-pairs 1 --warmup-requests 1 --margin-ms 10 --out "$EXPERIMENT"

# Record one slot at a time, in the frozen order: A1, B1, A2, B2, ...
slogate ab record "$EXPERIMENT" --arm A --base-url "$A_URL" --workload "$WORKLOAD"
slogate ab record "$EXPERIMENT" --arm B --base-url "$B_URL" --workload "$WORKLOAD"
# After recording all the planned slots:
slogate ab report "$EXPERIMENT"
```

For fake engines, the pin files are the complete expected `/version` JSON documents,
including the full profile. The plan, each run and its compressed request records have
content digests. Nonoverlapping run times, ordered request IDs/arrival schedules, model,
client/transport, output lengths and host identity must agree. An unchanged profile is
allowed for A/A. Otherwise one setting may differ. Old `slogate aa` results are **not**
silently paired after the fact. Seals catch accidental mix-ups, not a malicious operator
rewriting the entire experiment.

`ab report` recomputes the primary metric from the request records. The independent unit
is a full pair, never a request, a token or a within-run time slice. For each pair the
effect is **B p99 minus A p99**. `--block-pairs K` averages each consecutive group of K
preassigned effects first; groups must divide the plan exactly. The estimate is the
**median** of these block means, with a conservative, two-sided, order-statistic interval
of at least 95% nominal coverage under independent blocks. Six blocks are the smallest
number with finite bounds here. The estimand differs from the *mean*-based E0 MDE estimate.
Choose block size and sample count before collection. Blocking is not proof of
independence, and fixed AB order does not remove systematic within-pair time trends.

| Exit (`ab report`) | Interpretation |
|---|---|
| 0 | Performance pass: upper bound strictly below the margin; any configured absolute ceiling also passed |
| 2 | Relative regression, a candidate request failure, or a breached absolute ceiling; see the separate fields for which |
| 3 | Invalid evidence: e.g. load-fidelity failure, reference request failure, wrong arm/host, modified or overlapping runs |
| 4 | Inconclusive: unfinished plan, insufficient independent blocks, or interval touching/crossing the margin |
| 1 | Usage, plan or I/O error |

Each attempted slot remains on disk, including invalid measurements. A reserved slot
without a completed seal is invalid, not a slot that may be overwritten. The MVP requires
all planned pairs: no post hoc exclusions, early statistical decisions or replacement
samples. Set `--candidate-ceiling-ms` **at init** to also check an absolute per-run
TTFT p99 requirement on B. Without a supplied ceiling the absolute SLO is `not_configured`.
The [correctness command](correctness.md) checks different evidence; volume of streamed
tokens is not a substitute for actual token IDs or scores on matched prefixes.

## Statistical sanity check (not GPU calibration)

```bash
PYTHONPATH=src python scripts/calibrate_paired.py --seed 20261123 --trials 2000 --e0
```

This separate simulation uses 12 pairs, Gaussian noise with a 100 ms standard deviation
and a **100 ms toy margin**, not the 10 ms teaching margin above. The boundary case sets the
true effect exactly to the margin: reporting a regression is then a false alarm.

| Model | A/A false alarms | False alarms at the margin | Detected +300 ms effect |
|---|---:|---:|---:|
| Independent pairs | 0/2,000 | 43/2,000 | 1,996/2,000 |
| Pairs correlated in predeclared two-pair blocks | 0/2,000 | 36/2,000 | 1,760/2,000 |
| Same correlated draws, incorrectly treated as independent | 2/2,000 | 136/2,000 | 1,990/2,000 |

Ignoring the known correlation made the interval narrower but pushed false alarms at the
boundary to 6.8%. The script prints per-row Wilson limits for sampling error. `--e0`
resamples the six published run p99s with replacement; it does **not** preserve temporal
dependence and is not a held-out trial. Neither result certifies a production false-alarm
rate or the proposed 80% detection-power target at the E0 MDE.

## Pinned vLLM trials

```bash
PYTHONPATH=src python scripts/week3_candidates.py --out runs/week3-candidates
```

Use the [E0 SUT config](../configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json)
as A; use **one** generated file as B in each new `ab init --kind vllm` plan. Freeze a
GPU-relevant margin separately. The generator does not start engines or spend GPU time.

| Candidate | Only change from E0 | Status |
|---|---|---|
| `eager` | add `--enforce-eager` at the original 2048 budget | [Frozen protocol; not measured](#second-trial-eager-execution) |
| `small-graphs` | set `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'` | Not measured |
| `small-budget` | set `--max-num-batched-tokens` from 2048 to 512 | [Measured: 6/6 pairs, relative TTFT pass](#first-real-trial-small-budget) |

These flags may worsen a chosen metric, improve it, or have no material effect. They are
not three detected GPU regressions. Option names were checked against [vLLM v0.30.0
arguments](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py)
and its [compilation config](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/compilation.py)
on 2026-11-23.

GPU lifecycle is **not** automated by this CLI. Run exactly one untraced reference service
at a time on the same physical device/loopback port, using the pinned `python -m
vllm.entrypoints.openai.api_server` command. For each slot, record with `--server-pid PID
--sut-info FILE --api-key-env ENV_NAME` on the SUT host. Supply a fresh manifest generated
by `scripts/gpu/collect_host.py`, not one copied from E0. On Linux the collector checks
the shared loopback namespace, listener ownership, process start identity, exact launch
flags, GPU fingerprint and driver
before and after the run; unrecognizable process titles, proxies, inaccessible procfs, or
missing evidence are not substitutes for these checks. Remote/tunneled endpoints,
multi-GPU, engine upgrades, power calibration and real token/logprob collection are not
qualified by the CPU demo.

### First real trial: small budget

**What changed.** In [vLLM v0.30.0's scheduler config](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/scheduler.py),
`max_num_batched_tokens` limits the tokens processed **across sequences in a single
iteration**. This [pinned configuration](../configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json)
uses chunked prefill with no speculative decoding. A used `2048`; B changed that one
performance flag to `512`, a quarter of A's budget. The separate `max_num_seqs=32`,
`max_model_len=8192` and performance workload's `max_tokens=128` did not change.
These are vLLM settings and workload limits; slogate did not modify the scheduler or kernels.

For example, suppose 16 ongoing sequences each need one decode token this step, with
room for a new sequence and sufficient KV cache. After scheduling those 16 tokens, A
has at most `2048 - 16 = 2032` tokens left for prompt work; B has `512 - 16 = 496`.
An uncached 1,200-token prompt **could** fit in one A step, but must span at least three
B steps. This is a budget example, not a trace of an observed step. It illustrates the
[chunked-prefill trade-off](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/configuration/optimization.md#chunked-prefill):
larger chunks can finish incoming prompts sooner; smaller chunks can leave ongoing
decodes waiting less between outputs.

**Decision fixed before collection.** Would B increase run-level TTFT p99 by at least
2,000 ms on this heavily queued workload? TTFT was the only performance gate. The
plan called for six fresh pairs in fixed A1/B1 through A6/B6 order, one pair per
statistical block and two warmup requests per run. No absolute latency SLO was set.

**What was measured (2026-11-25 UTC).** On the same RTX 4090 and pinned Qwen3/vLLM
build, the client scheduled 787 requests over 30 seconds each run, then waited for all
responses. All 12 runs passed the original 50 ms client-fidelity checks; 9,444/9,444
requests completed. No worker trace was enabled, no old E0 run was substituted, and no
pair was excluded.

- **Primary verdict:** B had a longer p99 wait for the first token in every pair. The
  median of the six paired TTFT p99 differences was **+1,039.38 ms**, with a two-sided sign interval of
  **[+960.42, +1,207.35] ms** (96.875% nominal coverage assuming independent pairs).
  Its upper bound was below the **+2,000 ms** practical margin frozen before B: **pass**.
  This is neither equality nor an absolute serving SLO. Fixed A-before-B order cannot
  rule out a time-order effect.
- **Secondary trade-off:** comparing the medians of each arm's six run p99s, TPOT went
  from **53.34 to 50.63 ms/token** and visible ITL from **152.45 to 58.82 ms**. These were
  not gate criteria. On 178 A streams and 159 B streams, visible token counts differed
  from final usage; ITL is client-observed, not an exact GPU timestamp for every token.
  No step or kernel was timed to establish the cause.
- **Separate [correctness check](correctness.md#native-vllm-collection-for-the-small-budget-trial):**
  A and B produced identical native token IDs for all four frozen cases (24 output tokens
  each), including a 2,081-token prompt. Running each case alone or in a reverse-order
  batch also gave identical IDs within each arm. This covers the suite, not all prompts.

[Per-pair data (CSV, milliseconds)](results/small-budget-2026-11-25.csv). Paired differences
are calculated before rounding; the median paired difference need not equal the difference
of the two arm medians. Trial source: [`e9a8c0f`](https://github.com/dlinlabs/slogate/commit/e9a8c0fbdea56002c1299ffdf4533c8dba6dc7a1).

#### Reproducing the trial

From an installed client checkout, these steps prepare the trial and preview the first
performance slot. They do **not** contact a server or reserve a GPU:

```bash
python scripts/prepare_small_budget.py prepare --out trial \
  --workload workloads/azure-2024-05-12-smoke-30s.json.gz
# Save the printed plan and receipt pins separately. For a delivered trial, verify
# its trusted pins and skip preparation; never regenerate a new plan on the Pod.
python scripts/prepare_small_budget.py slot trial --slot pair-001-A \
  --receipt-sha256 "$RECEIPT_SHA"
```

The preview checks the files, the next slot, and previous slot receipts. The printed
recipe is **manual**, for an already-authorized, supervised Linux host. `begin` checks an
idle GPU/port and reserves that slot; `record` checks the owned listener, environment,
new SUT manifest and whether the GPU processes belong to that server before calling
`slogate ab record`. It does not manage Pod billing or guarantee a wall-clock cutoff.
Use the real GPU and process namespace; if the engine binary or GPU processes cannot be
identified, do not bypass the check. No cloud credentials belong in the performance
process environment. Keep model downloads and engine bootstrap outside measured slots.

Before collection, the estimate was 30.6 GPU-minutes for 12 E0-like replays or about
73 minutes with 12 E0-like cold starts, before other overhead. The observed single-GPU
window from host preflight through off-host archive verification was **59.89 minutes**,
not a billing reading or a guarantee that a repeat fits in 60 minutes. If an approved
window is insufficient, retain partial slots and report incomplete evidence rather
than changing the number of pairs. Preparation itself needs no Pod.

### Second trial: eager execution

**Decision, set before seeing B:** with the token budget back at **2048 on both arms**,
does forcing eager execution increase run-level TTFT p99 by **2,000 ms** or more?
A is the pinned E0 launch; B is exactly A plus `--enforce-eager`. In
[vLLM 0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/vllm.py)
the flag disables **both torch.compile and CUDA Graphs**. A uses the default execution
policy; we do not assume every step replays a graph. A difference would measure the
effect of the flag, not isolate either mechanism. No worker trace is part of this gate.

| Fixed before measurement | Rule |
|---|---|
| System and load | Same single RTX 4090, software/model pins, 32 sequences, 8,192-token context. Replay all 787 arrivals over 30 s with `max_tokens=128`. |
| Sample | Six **new** adjacent A/B pairs in fixed order, two warmup requests per run, one pair per block. Never borrow an A from E0 or the 512 trial. |
| Performance | Unchanged 50 ms client checks and normal GC. Compare paired TTFT p99 (B minus A) against **+2,000 ms**, no absolute SLO. TPOT/ITL are secondary, not alternate ways to pass. |
| Correctness | Capture fresh native output IDs for A and B on separate correctness-only services. Same frozen four prompts, greedy, at most 24 output tokens. This checks the suite in batch-invariant mode, not all production outputs. |

Local preparation and **read-only** preview (fresh output directory, no server):

```bash
"$CLIENT_PYTHON" scripts/prepare_eager.py prepare --out "$TRIAL" \
  --workload workloads/azure-2024-05-12-smoke-30s.json.gz
"$CLIENT_PYTHON" -m slogate vllm-exact prepare --ab "$TRIAL/ab" \
  --suite configs/correctness/qwen3-small-budget.json --out "$TRIAL/exact"
# Save the printed receipt, AB plan and exact spec digests independently.
"$CLIENT_PYTHON" scripts/prepare_eager.py slot "$TRIAL" --slot pair-001-A \
  --receipt-sha256 "$RECEIPT_SHA"
```

For a delivered, frozen trial, **verify its supplied pins and skip both `prepare`
commands**. The suite filename is historical; it contributes only the prompts and
decoding protocol, not the arm flags or previous outputs. Use only `prepare_eager.py`
for performance slots; it rejects the 512 receipt and vice versa.

Only after separate host and time-budget authorization, follow the
[native-ID capture steps](correctness.md#native-vllm-collection-for-the-small-budget-trial)
on the **new** exact spec. Save A's new reference plan digest off-host before B.
If correctness fails or is unverified, preserve the attempt and stop. Otherwise
use each performance `slot` preview's `begin`/probe/start/`record` steps in order.
For every slot record a fresh manifest and actual listener PID, then release only its
own server. Finish with `"$CLIENT_PYTHON" -m slogate ab report "$TRIAL/ab"`.
Read the [exit codes](#collect-a-comparison): an interval touching +2,000 ms is
inconclusive, bad measurements are invalid, and an empty plan is **not a pass**.
The previous trial's elapsed time does not guarantee this one fits in 60 GPU-minutes.
At the approved deadline keep partial evidence; never rerun a slot, extend the budget
or change the decision rule to finish. Export and verify before any Pod lifecycle action.
