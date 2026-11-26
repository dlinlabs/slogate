# slogate

**An SLO regression gate for LLM inference engines.**

slogate replays a fixed workload before a serving change ships. It checks whether the
client delivered the intended load, then compares baseline and candidate latency against
a margin chosen before the trial. Locating a regression in a specific runtime layer is
still a separate, unvalidated step.

**Two vLLM changes, two different user experiences.** On a single RTX 4090 under heavy
queueing, a smaller scheduling budget delayed the first token but shortened the pauses
that followed. Forced eager execution also delayed the first token, without that
streaming benefit. Both passed the same preselected **+2 s regression limit**.
Neither result says that the service was fast enough for a customer.

[Read the experiments in order](#experiment-guide) | [See the two GPU results side by side](#gpu-results-side-by-side)
| [Try the gate locally without a GPU](#quickstart-cpu-only)

## How slogate talks to vLLM

The operator starts **either A or B**, one vLLM server at a time on the same RTX 4090.
slogate sends requests, times the responses and scores the comparison. It does **not**
set the scheduler budget through an HTTP request or patch vLLM's source code.

[![Interaction diagram: slogate sends the same streaming chat requests to vLLM A and B. Only the server's per-step token budget changes. slogate times the returned chunks and judges TTFT; a separate non-streaming call checks output token IDs.](docs/figures/slogate-vllm.svg)](docs/figures/slogate-vllm.svg)

The diagram uses the **2048 vs 512** trial as its example. The eager trial uses the
same request and reporting paths but changes a different startup flag.

| Where the value is set | What we set |
|---|---|
| **vLLM at startup: changed** | Budget trial: `--max-num-batched-tokens` **2048 vs 512**. Eager trial: both **2048**, only B adds `--enforce-eager`. One change per trial. |
| **vLLM at startup: unchanged** | `--max-num-seqs 32` (sequences per step), `--max-model-len 8192` (context), `--tensor-parallel-size 1` and `--no-enable-prefix-caching`. |
| **slogate sends in each performance request** | `max_tokens=128` caps **one answer**; `stream=true` asks for chunks as the answer is generated. Neither sets the shared scheduler budget. |
| **slogate only: never sent to vLLM** | Six A/B pairs; 50 ms checks on the **client** (launch lag, event-loop lag, GC pauses); **+2,000 ms** allowed TTFT p99 increase for the **relative** verdict. TPOT and ITL are reported, not used to pass the gate. |

See the [actual A startup flags](configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json)
and the [one-change B configurations](scripts/week3_candidates.py).

## Experiment guide

**A/A** means testing the same configuration repeatedly to see how much the measurement
moves on its own. **A/B** means comparing the reference (A) with one proposed change (B).
**p99** describes the slow end: roughly 99 out of 100 observations are at or below it.

### 1. Can we trust the clock? CPU A/A and the GPU baseline

- **Goal and expectation:** Before testing a change, see whether the load generator can
  send the same requests on time. An unchanged server should give similar, not identical,
  numbers. A successful HTTP response does not prove that the test was valid.
- **Method and command:** `slogate aa` sends the same workload repeatedly to `--base-url`;
  `--runs` is the number of repeats, not requests. The [CPU quickstart](#quickstart-cpu-only)
  shows the full command. `slogate noise "$AA_DIR"` rebuilds the noise report from an
  existing A/A directory without sending more requests. Use a working copy for re-reporting.
- **Observed:** Three unchanged fake-engine runs gave TTFT p99s of **69.1, 67.7 and
  74.0 ms**. In the first GPU baseline (E0), all **4,722** requests succeeded, yet **0/6**
  runs were valid: the client's event loop and garbage collector both stalled past 50 ms.
  With the revised client, **6/6** fresh GPU runs passed at the original limits. The mean
  of their six TTFT p99s was **120.139 s**, with **0.318%** run-to-run variation (CV).
- **Conclusion:** We now have a [noise baseline](docs/results/e0-2026-11-23.csv) for this
  heavily queued workload. That two-minute wait includes queueing; it is not GPU compute
  time or a product SLO. The failed and qualified baselines ran on different hosts, so
  comparing their latencies is **not** an engine speedup measurement.

### 2. Was the load generator itself stalling?

- **Goal and expectation:** Reproduce the failed GPU test on CPU. If the stream reader
  is part of the problem, changing the client should reduce stalls without disabling GC,
  reducing the load or relaxing the 50 ms limit.
- **Method and command:** `scripts/cpu_pressure.py --recorded-aa ... --shared-clock`
  replays recorded chunk times on one clock for all requests; simply matching the
  number of open requests was not enough. `scripts/sustained_replay.py --waves 3`
  holds one client process open for three replays. [Full CPU commands below](#cpu-diagnostic-client-pressure).
- **Observed:** After changing the reader from HTTPX to aiohttp, qualifying waves went
  from **0/3 to 3/3**. Worst event-loop delay fell from **107.2 to 36.7 ms**. A separate
  test using six different schedules still had one **130.9 ms** stall; it remains invalid.
- **Conclusion:** The measuring tool improved, not the model. The [before/after checks](docs/weeks-01-02.md)
  justify using the revised client; they do not explain every pause on the original GPU host.

### 3. Does the gate recognize a known answer?

**Method:** From an installed dev checkout, `python scripts/paired_demo.py --out runs/cpu-controls`
starts local fake engines, freezes the plan and scores four controls. `--out` must be
a new directory. Here the teaching margin is **+10 ms**, not the real GPU margin.

| Control and purpose | Expected | Recorded result |
|---|---|---|
| Same build, six pairs: check that ordinary noise does not block this control | Pass | Interval -2.65 to +3.94 ms; **pass** |
| Add 60 ms to B's fake first-token setting: can a known slowdown be caught? | Fail | +37.48 to +44.66 ms; **fail** |
| Same delayed setting, only two pairs: can we tell when evidence is too thin? | No decision | No finite interval; **inconclusive** |
| Send B's requests to A by mistake: does the tool check the running engine? | Reject measurement | Wrong identity; **invalid measurement** |

All **750/750** requests succeeded across 30 runs; they still led to four different
verdicts. These [CPU results](docs/results/week3-cpu-pairs-2026-11-23.csv) exercise the
decision rules, not vLLM performance. As a separate statistical check,
`python scripts/calibrate_paired.py --seed 20261123 --trials 2000 --e0` uses repeatable
simulated experiments. At the toy regression boundary, ignoring known correlation gave
**135/2,000** false alarms vs **36/2,000** when related pairs were grouped in advance.
That [simulation](docs/paired-gate.md#statistical-sanity-check-not-gpu-calibration)
is not a measured production false-alarm rate.

### 4. How the two real GPU comparisons were run

Each comparison used **six fresh A/B pairs** on one RTX 4090 with vLLM 0.30.0 and
Qwen3-4B-Instruct-2507. A1 was followed by B1, then A2/B2, and so on. Every run had
two warmup requests, then the same **787 arrivals in 30 s**, and drained all responses.
No E0 or other trial's A runs were reused. The primary metric was time to first visible
token (**TTFT**); the decision rule was fixed **before** seeing B.

```bash
# Local preparation only. WORKLOAD is the pinned trace, not the synthetic quickstart file.
python scripts/prepare_small_budget.py prepare --workload "$WORKLOAD" --out runs/budget-trial
python scripts/prepare_eager.py prepare --workload "$WORKLOAD" --out runs/eager-trial
```

`prepare` writes a **new** plan, not results. It needs the [pinned workload](workloads/README.md)
and aiohttp 3.14.5; never re-prepare an already frozen trial. For examples of `slot`
(preview the next run), `begin` (reserve it) and `record` (measure the verified running
server), see the [operator guide](docs/paired-gate.md). `slogate ab report "$TRIAL/ab"`
recomputes the verdict from saved records; it does not contact vLLM. Use a working
copy of the completed trial because it writes `gate.md` and `gate.json`.

#### One GPU decision: 2048 vs 512 tokens per step

**Question:** Can shorter chunks of prompt work reduce pauses for users already receiving
tokens without making new users wait too much longer? **Change:** A used
`--max-num-batched-tokens 2048`; B used `--max-num-batched-tokens 512`. This is the total
work vLLM may schedule **across all requests in one step**, not the length of a prompt or
answer. For example, if 16 ongoing requests each need one decode token this step, up to
**2,032 vs 496** tokens remain for prompt work. A long prompt can span steps; it is not cut off.

**Expectation:** A smaller budget could mean more steps before the first token but fewer
long gaps while streaming. **Result and decision:** B's TTFT p99 was higher in all six
pairs (**+1.04 s** median paired difference); the long streaming gaps got shorter. B
passed the preset limit. Choose based on whether first response or smooth streaming
matters more; the test did not show a win on both. [Trial and limitations](docs/paired-gate.md#first-real-trial-small-budget).

#### Default execution vs forced eager

**Question:** With the token budget kept at **2048 on both sides**, what happens if vLLM
cannot use its usual execution optimizations? **Change:** A used the default settings;
only B added `--enforce-eager`. In [vLLM 0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/vllm.py#L1546-L1552)
this disables **both** PyTorch compilation and CUDA Graph replay. It does not disable
batching or move the model to the CPU.

**Expectation:** The default can reuse work to launch GPU operations more efficiently;
eager could be slower, but the size of any change was unknown. **Result and decision:**
B's TTFT p99 was higher in all six pairs (**+1.79 s** median paired difference), with
no observed improvement in the two streaming metrics. It passed the gate, but there
was **no latency reason to prefer eager** on this workload. No worker trace was taken:
we cannot say how much came from compilation versus graph replay. [Trial and limitations](docs/paired-gate.md#second-trial-eager-execution).

#### GPU results side by side

Positive TTFT differences mean B was slower. Compute each run's p99 first, then subtract
A from B **within** each pair. The TPOT and ITL rows show the **median of six run p99s
per arm**, not pooled token timings. Compare each B with **its own** A.

| Measured result | Smaller budget (Nov 25, 2026 UTC) | Forced eager (Nov 26, 2026 UTC) |
|---|---:|---:|
| TTFT p99, median paired increase | +1,039.38 ms | +1,788.09 ms |
| Two-sided interval for that increase | [+960.42, +1,207.35] ms | [+1,737.37, +1,917.70] ms |
| TPOT: average time per subsequent token within a response, A to B | 53.34 to 50.63 ms/token | 53.46 to 54.11 ms/token |
| ITL: individual client-visible token gaps, A to B | 152.45 to 58.82 ms | 151.90 to 152.54 ms |
| Requests / valid paired runs | 9,444/9,444; 6/6 pairs | 9,444/9,444; 6/6 pairs |
| Performance decision / per-pair evidence | **Pass** ([CSV](docs/results/small-budget-2026-11-25.csv)) | **Pass** ([CSV](docs/results/eager-2026-11-26.csv)) |

**Why can slower pass?** Both interval upper bounds are strictly below **+2,000 ms**;
eager cleared it by only **82.30 ms**. The gate does not promise an answer within two
seconds. Each interval has 96.875% nominal coverage assuming independent pairs; fixed
A-before-B order leaves possible time drift. TPOT and ITL cannot override this verdict.

### 5. Did either change alter the answers?

- **Goal and expectation:** A speed check cannot tell whether the model produced the
  same tokens. We expected identical native output IDs on four fixed prompts, not proof
  of equivalence on every input.
- **Method:** For each trial separately, use dedicated correctness servers and non-streaming
  `POST /v1/completions` with `return_token_ids=true`. Feed the prompts alone and in a
  reversed batch, with greedy decoding and a 24-token cap. This is **not** the timed
  workload. To rescore a complete export offline:

  ```bash
  slogate vllm-exact report "$TRIAL/exact" --spec-sha256 "$SPEC_SHA" --plan-sha256 "$PLAN_SHA"
  ```

  The two hashes identify the independently saved **collection spec** and **A's correctness
  plan** for that trial, not its performance plan. The report checks both arms' receipts;
  a file of matching IDs by itself does not identify the candidate.
- **Observed and conclusion:** **4/4 cases passed in each trial.** All 24 generated IDs
  matched in each case, including the 2,081-token input. Each arm's single and reversed-batch
  outputs also matched. These results cover the controlled suite only. In the separate CPU
  teaching demo, `python scripts/correctness_demo.py --out runs/token-controls` tested six
  known cases: unchanged outputs passed, a changed ID, score drift or a lost answer failed,
  and a missing score was unverified. [Correctness contract and historical receipt checks](docs/correctness.md#offline-native-recheck).

**Not yet measured:** restricting CUDA Graph capture sizes to `[1,2,4,8]` is a **third,
separate** candidate. There is no GPU verdict for it, no held-out production error-rate
estimate, and no validated kernel-level explanation for the two results above.

## What works today

- **Paired comparison.** Freeze one candidate change and a practical margin before collecting
  interleaved A/B runs. Report relative performance, any configured absolute p99 ceiling,
  invalid measurements and insufficient evidence separately.
- **[Output correctness](docs/correctness.md).** Check separate token-ID evidence for
  scheduler changes, or compare same-prefix logprobs and a fixed-answer eval. Missing
  evidence is unverified; these checks are not implied by a performance pass.
- **Open-loop replay.** Every request launches on its scheduled arrival time and never waits
  for earlier responses, so a slow server shows up as latency instead of as reduced load.
- **Per-token timelines.** TTFT, per-token inter-token latency (multi-token chunks are split
  evenly), TPOT, end-to-end latency, and reconciliation against server-reported token usage.
- **Load fidelity.** Client launch lag, client event-loop lag and client GC pauses are
  measured on every run. A run that breaks them is reported as invalid measurement, never as
  a server regression.
- **A/A noise floor.** Repeated runs of an unchanged build give per-metric variation, drift
  and the minimum detectable effect (MDE) for a given number of paired runs.
- **Fake engine.** A CPU-only, OpenAI-compatible streaming server with injectable latency
  regressions, so CI exercises the gate without a GPU.
- **Versioned workloads.** Content-hashed workload files, a synthetic generator, and an
  importer for traces derived from the Azure LLM Inference Dataset 2024.

## Quickstart (CPU only)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
slogate fake-engine --port 8000
```

Leave that terminal open. In a second terminal in the same checkout, run:

```bash
. .venv/bin/activate
slogate workload synth --name smoke --duration-s 20 --rate-rps 8 --out workloads/smoke.json
slogate aa --base-url http://127.0.0.1:8000 --workload workloads/smoke.json \
  --runs 6 --cooldown-s 2 --out runs/aa-smoke
```

Stop the fake engine with Ctrl-C in its terminal when finished. This synthetic quickstart
is not a replay of the published GPU load. `runs/aa-smoke/noise.md` reports the run-to-run variation of each metric and the smallest
regression a paired comparison could detect. Each `run-NNN/` folder holds `run.json`
(metadata, fidelity verdict, metrics) and `records.json.gz` (every request's timeline), so
metrics can be recomputed without rerunning.

Next, [exercise the four gate verdicts](#3-does-the-gate-recognize-a-known-answer);
read `scorecard.json`, then each `gate.md` for its interval and per-pair differences.

## Against vLLM on a GPU

`configs/sut/` pins a system under test (engine version, model revision, serve flags), and
`scripts/gpu/run_aa.sh` runs an A/A measurement on a single-GPU host with a wall-clock cost
cap. See [scripts/gpu/README.md](scripts/gpu/README.md).

Keep worker tracing **off** for E0. A separate `SLOGATE_TRACE=1` run records scheduled work,
observed graph mode and current-stream CUDA events for each supported worker. These
instrumented measurements are diagnostics, not SLO samples.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | `run`: client fidelity passed. `aa`/`noise`: at least three valid runs with no failed requests; check `excluded_runs` for the rest |
| 1 | Usage or input error |
| 3 | Invalid measurement: load fidelity failed, or fewer than three healthy A/A runs |

For `ab report`, exits are **0** performance pass, **2** performance/SLO failure, **3**
invalid measurement, **4** inconclusive, and **1** input error. See the [paired evidence
contract](docs/paired-gate.md) before using the result in CI.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts && ruff format --check src tests scripts
pytest
```

CI runs the same checks on Linux for every push and pull request. Tests use the fake engine
and never need a GPU or network access beyond localhost.

### CPU diagnostic: client pressure

To investigate the 4090 E0 fidelity failures without renting a GPU, run the following in a
Linux checkout with the fake-engine dependencies and the [pinned workload](workloads/README.md).
Choose a new output directory each time:

```bash
PYTHONPATH=src python scripts/cpu_pressure.py --out runs/client-pressure
PYTHONPATH=src python -m slogate.pressure_report runs/client-pressure
```

Six loopback trials keep the arrival schedule and token counts fixed. Only the fake first-token
delay changes (A: 50 ms; B: 23.5 s), to target roughly 650 outstanding requests in B. Each
trial starts fresh server/client processes, leaves automatic GC enabled and keeps the normal
50 ms fidelity limits. `--client-timing` adds wakeup deadlines/lags and GC spans to the replay
diagnostics on one monotonic clock; it is **off by default** for normal SLO runs. The report
includes invalid trials and measures GC overlap only after a wakeup deadline, not during the
intended sleep. Temporal overlap is evidence for investigation, not attribution of all delay.

For a closer CPU diagnostic, supply the six **local** E0 run directories under `SOURCE_AA`.
The fake server joins each request to its saved stream by workload ID and unique prompt/cap,
then emits filler SSE content on absolute deadlines relative to that request's arrival. It
reuses the same schedule for warmup without consuming the measured request. Headers, visible
chunk widths/times and completion times are from client records, not an engine trace. Raw
SSE control events, text, GPU work and unobserved token times are not reconstructed.

```bash
PYTHONPATH=src python scripts/cpu_pressure.py --recorded-aa "$SOURCE_AA" --out "$OUT"
PYTHONPATH=src python -m slogate.timeline_report "$OUT" "$SOURCE_AA"
```

`timeline.md` reports two independent checks: how closely the local timestamps matched, and
whether the replay client passed the normal 50 ms fidelity checks. `timeline.json` retains
per-source SHA-256, paired timing errors, request-seconds, GC overlap and failure reasons.
To test cross-request alignment, add `--shared-clock` to `cpu_pressure.py` and choose a new output
directory. Measured clients pass their same-host monotonic replay epoch to the loopback fixture;
the server targets the original absolute offsets instead of starting each schedule when its HTTP
request arrives. Warmups still run on their own clocks. Both modes use the same predeclared timing
checks, and neither changes ordinary vLLM requests. Never use synthetic timings as SUT SLO samples.

### Qualifying a streaming client change

The measured path uses aiohttp; readiness probes and warmup use HTTPX. Normal runs record the
measured transport/version in `run.json` and leave Python's GC policy unchanged.

To investigate live objects around the reproduced stall, run the allocation profiler against
local source `run-001`. Its snapshot pauses the process; discard **all** profiled timings:

```bash
PYTHONPATH=src python scripts/profile_client_heap.py \
  --source "$SOURCE_AA/run-001" --out "$OUT/heap"
```

For the actual comparison, use the same Linux environment and harness for both arms. Set
`BASELINE_SRC` to a frozen copy of the previous client's `src` directory with shared-clock
support. Run the two commands **serially**, with new output directories. Exit `3` is a
retained qualification failure, not a reason to discard that arm.

```bash
export PYTHONHASHSEED=1
python -B scripts/sustained_replay.py --source "$SOURCE_AA/run-001" \
  --waves 3 --client-src "$BASELINE_SRC" --out "$OUT/before"
python -B scripts/sustained_replay.py --source "$SOURCE_AA/run-001" \
  --waves 3 --client-src ./src --out "$OUT/after"
```

Each arm keeps one client PID across all waves, with no allocation profiler or detailed timing
trace. `report.json` records timing fit, fidelity, stream/usage checks and external RSS samples.
See the [measured results and remaining failure](docs/design.md#normal-gc-streaming-client).

## Data

Traces imported with `slogate workload import-lab` derive from the
[Azure LLM Inference Dataset 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md),
licensed under CC-BY 4.0. Please cite: Stojkovic et al., *DynamoLLM: Designing LLM Inference
Clusters for Performance and Energy Efficiency*, HPCA 2025. Prompt text is synthetic; only
arrival times and token counts come from the trace.

## License

Apache License 2.0.
