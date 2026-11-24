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

## Pinned vLLM trials (prepared, not measured)

```bash
PYTHONPATH=src python scripts/week3_candidates.py --out runs/week3-candidates
```

Use the [E0 SUT config](../configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json)
as A; use **one** generated file as B in each new `ab init --kind vllm` plan. Freeze a
GPU-relevant margin separately. The generator does not start engines or spend GPU time.

| Candidate | Only change from E0 | Status |
|---|---|---|
| `eager` | add `--enforce-eager` | Not measured |
| `small-graphs` | set `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'` | Not measured |
| `small-budget` | set `--max-num-batched-tokens` from 2048 to 512 | Not measured |

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
