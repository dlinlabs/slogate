# slogate design

Version 0.1 (week 3 in progress). This page states what slogate claims, how it will be judged, and what
it deliberately does not do.

## Claim

Given a candidate change to an inference engine, slogate decides within a fixed GPU budget
whether the change violates serving SLOs on a pinned, production-shaped workload, with a
calibrated false-alarm rate, and attributes any regression to a layer of the stack.

## Decisions it supports

- **Upgrade:** should we move from engine version X to Y for this workload?
- **Configuration:** does flag Z (CUDA graph sizes, token budget, attention backend) break SLOs?
- **Patch:** does my change improve p99 ITL without breaking correctness, and at what cost?
- **Regression hunt:** which commit between A and B introduced the regression?

## SLOs

Reference system under test: vLLM 0.30.0, Qwen3-4B-Instruct-2507 at a pinned revision, one
RTX 4090 (`configs/sut/`). For a real deployment, the SLO is a service requirement. E0
measures repeatability; it cannot tell us how long a user is willing to wait. Until a
deployment supplies its SLO, thresholds proposed after E0 are reference targets only.

| SUT metric | Target | Measured as |
|---|---|---|
| TTFT p99 | deployment target not specified; [E0](#first-gpu-baseline-e0) is a reference | client-side, per request |
| ITL p99 | deployment target not specified; E0 reports visible token gaps | client-side, per token |
| Goodput | not computed until per-request TTFT/TPOT bounds are specified | requests per second meeting both configured bounds |

| Property of the gate itself | Target |
|---|---|
| False alarms on A/A comparisons | at most 5% |
| Detection power at the stated MDE | at least 80% |
| Decision cost per candidate | design target: at most 60 GPU-minutes across A and B (GPU count times occupied minutes); check feasibility after E0, not a hard gate |
| Invalid measurements | reported separately; never counted as pass or fail |

## Non-goals

- **Not a load generator.** GuideLLM, inference-perf and AIPerf exist. slogate replays load
  only to control measurement validity, and its metric definitions will be cross-checked
  against one of those tools.
- **Not a dashboard.** vLLM already publishes per-commit benchmark dashboards.
- **Not a router or gateway.** llm-d, the vLLM production stack and Dynamo cover that space.
- **No claims beyond what was run:** no production traffic, no multi-tenant scale, and no
  conclusions about hardware the gate has not measured.

## Architecture

```
workload (versioned, hashed) -> open-loop replay -> request records -> metrics --+
                                      |                                          +-> verdict
                                      +-> load fidelity (launch lag, loop lag, GC) +
SUT config (engine, model revision, flags) + host fingerprint -> GPU executor (cost cap)
```

Planned layers: statistical comparison and correctness checks (weeks 3-4); attribution from
client timing to vLLM `/metrics`, per-step records with CUDA graph mode and CUDA events, and
Nsight Systems kernel diffs (weeks 5-6); bisect over vLLM's per-commit wheels (week 9).

## Measurement validity

- **Open loop.** Arrivals follow the workload schedule regardless of server speed.
- **Fidelity before verdicts.** Launch lag, event-loop lag or GC pauses above 50 ms (the
  limit validated in the predecessor lab) mark a run invalid.
- **Deterministic output lengths.** Performance runs set `ignore_eos`, so builds are compared
  on identical token counts. Natural stopping is reserved for correctness checks.
- **Warmup.** Requests sent before measurement absorb first-use compilation and JIT.
- **Same-host comparisons.** Builds are compared on one host, interleaved. In the predecessor
  lab, differences between two rental hosts could not be attributed to the change under test.

### What can count toward a release decision

The reference is the intended deployment. Pin the serving path, software, hardware,
resource limits and runtime policy before comparing builds. Keep client-to-server
topology and offered traffic the same across arms. The current single-4090 setup is a
reference profile, not evidence from a production fleet.

- **Match the workload to the claim.** Fixed output lengths and synthetic prompts are useful
  for controlled regression tests. Before making a deployment-specific claim, also validate
  against representative arrival bursts, prompt and output lengths, stopping behavior and
  cache settings for that deployment.
- **Keep user-visible delays.** If the serving frontend pauses before flushing a token,
  include that time in client-observed latency. If instead the *load generator* misses
  its dispatch or read deadlines, retain the run as invalid measurement: it cannot
  justify a release or establish an engine regression.
- **No benchmark-only GC policy.** Qualify the generator with automatic cyclic GC on.
  Do not force a collection before each sample, defer collection until drain, lower the
  offered load or relax the 50 ms client limits to get a pass. A GC intervention can be
  a separate diagnostic; do not pool it with baseline runs. A service-side GC change
  intended to ship is itself a candidate to test, not a hidden benchmark setting.
- **Qualify the generator under sustained load.** In the shared-clock CPU reproduction,
  profile client allocation and object lifetimes with normal GC. Compare the proposed
  fix with the unchanged client on the same inputs; verify timing fit, timely launches
  and reads, complete streams and their token counts. Reuse the client process across
  repeated bursts to check that RSS stays within a predeclared budget instead of
  relying on process exit to reclaim memory. Then collect a fresh, unprofiled GPU A/A.
  Passing the CPU diagnostic alone does not qualify the GPU baseline.

## Statistics

The A/A baseline estimates each metric's run-to-run coefficient of variation (CV). The minimum
detectable relative effect for `n` paired runs is
`MDE(n) = (z_{1-alpha/2} + z_{power}) * sqrt(2) * CV / sqrt(n)`.
Comparisons (week 3) use interleaved runs on one host, paired differences and a minimum
practical effect. The predecessor lab found that treating correlated time blocks as
independent inflated nominal false passes to about 14%, so blocks within a run are never
counted as independent samples.

### Week 3 comparison contract

The first implementation covers one run-level metric, `ttft_s.p99`, and one configuration
change. See the [paired gate](paired-gate.md) for the capture format, signed-effect convention,
order-statistic interval and CPU controls. The statistical error targets are not release claims.

1. Before looking at B, freeze the workload digest, hardware/driver, model/tokenizer revision,
   warmup, allowed candidate change, primary metric/direction, SLO, practical regression
   margin and number of pairs. Use E0's MDE to plan the sample size, **not** as the SLO.
2. Alternate A/B on one host. Each adjacent, preassigned pair contributes one run-level
   difference (orient positive as worse); requests and tokens are not replicates. Keep both
   run IDs and record exclusions. A fidelity failure or mismatched input invalidates the
   pair; candidate request failures are outcomes, not an excuse to drop slow requests.
3. Report the paired effect and its uncertainty against the frozen margin, and the absolute
   candidate SLO separately. A bound that straddles the margin or too few healthy pairs
   means *inconclusive*, not *pass*. Check gate-level false alarms on held-out A/A comparisons
   before claiming calibration; do not threshold a collection of unadjusted per-metric tests.

## Correctness

The [offline correctness checker](correctness.md) checks independent evidence against
a frozen manifest. A synthetic pass demonstrates checker behavior, not vLLM output quality.

- **Scheduler-only changes:** run with `VLLM_BATCH_INVARIANT=1` and require exact token matches.
- **Kernel, precision or version changes:** compare logprob drift and a small task evaluation
  against tolerances fixed before the run.

Run these checks separately from the performance replay, with pinned prompts, tokenizer and
greedy settings. Store generated token IDs for the exact check (matching rendered text or
usage counts is insufficient). For drift, score the same reference tokens on the same prefixes.
A mismatch in the exact check fails correctness; missing IDs/logprobs or an unsupported
backend leave it unverified, not passed. First fixture cases: identical A/A, one changed
token and a missing score. [vLLM v0.30.0 batch-invariance documentation](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/batch_invariance.md)
checked 2026-11-18; support for the pinned model/backend pair is to be tested on the SUT.

## Threats to validity

- Rental GPU hosts differ in clocks, thermals, PCIe topology and drivers; every run records a
  host fingerprint, and verdicts compare runs from one host only.
- Client and server share the host CPU; client contention shows up as event-loop lag.
- Windows timers tick at about 15 ms, so local runs are smoke tests; measurements of record
  run on Linux.
- Imported traces preserve arrival times and token counts but use synthetic prompt text, and
  prefix caching is disabled.
- A single run's p99 is noisy; verdicts use repeated runs.

## First GPU baseline (E0)

For the week-by-week account and per-run data, see the [weeks 1-2 report](weeks-01-02.md).

Six A/A runs of `azure-2024-05-12-smoke-30s` (787 requests in 30 s, about 26 requests per
second, workload digest `821ec458...`) against the pinned SUT on one host, 10 s apart. It
yields the first noise floor. Only the **uninstrumented** baseline can be used to propose
SUT targets. With no TTFT/TPOT bounds the client does not calculate goodput; after freezing
those bounds, use the saved request records rather than treating raw completion rate as goodput.
Use [the GPU runbook](../scripts/gpu/README.md) for a separate traced diagnostic run.

**First qualified baseline, checked 2026-11-23.** All six runs on one RTX 4090 were
included, none excluded (exit 0). Each completed 787/787 requests with aiohttp 3.14.5,
CPython 3.12.3, vLLM 0.30.0 and no worker trace. Across the six runs, the largest client
launch lag, loop lag and GC pause were 30.09, 41.55 and 44.09 ms, respectively (limit: 50 ms).
The figures below are computed across six **run-level p99s**, not by pooling requests or tokens.

| Client metric | Mean p99 | Range of run p99s | Run-to-run CV |
|---|---:|---:|---:|
| TTFT | 120.139 s | 119.363-120.345 s | 0.318% |
| ITL (visible chunks, using the documented per-token split) | 160.39 ms | 159.03-161.48 ms | 0.515% |

At the end of the 30 s arrival window, 659 requests per run were still open. One server
sample had 32 running and 627 waiting. The TTFT is a client-observed wait under this load,
not GPU compute time. On 171/4,722 streams the client counted 126 or 127 tokens vs. 128
in the final usage. This did not trip the declared fidelity gate; the ITL statistic is
reproducible from the saved timelines, not an exact timestamp for every generated token.
No request SLOs were configured, so goodput is **unmeasured**, not zero.

For planning three same-host A/B pairs, the noise model estimates an MDE of about
0.87 s for TTFT p99 and 1.89 ms for ITL p99 (alpha 0.05, target power 0.80). These are
estimates, not calibrated false-alarm guarantees, release thresholds or service objectives.

Provenance: capture `e0-5316713-20261123T013233Z-HTGOQ1`, archive SHA-256
`b65c0ee1b1fa812bfa6e2022258a82eaf6fab73b47f23829752c55b10f6ee0cc`. The operator
used release `5316713` **plus** launcher patch SHA-256
`689d963dae3c1c9003ac9453500a77bb6dfc5ecb9edfbf310fae7d7409d74a0f`; both are needed
to reproduce the launch. The experiment watchdog recorded 24.3 minutes, not total Pod uptime.

**Earlier attempt, 2026-11-20:** all 4,722 requests completed, but loop lag and GC
pauses exceeded 50 ms in all six runs. It produced no qualified samples. Keep it as
diagnostic evidence, separate from the new baseline: the host identifiers also differ.

A [CPU loopback diagnostic](../README.md#cpu-diagnostic-client-pressure) then changed only
the fake first-token delay (50 ms to 23.5 s) within each interpreter. It matched E0's peak
outstanding count (662-664), not its full request lifetimes. Short-delay controls passed 3/3
on each interpreter. High-pressure trials passed 3/3 on WSL CPython 3.14.4 and 2/3 on WSL
CPython 3.12.3. The 3.12.3 failure was one 56.4 ms overdue wake with no measured GC overlap;
none had a >50 ms GC pause. The high-pressure fake drained in about 55 s versus 150-151 s
on E0. This does not explain the Pod failures or justify disabling GC or weakening the gate.

**Recorded-stream follow-up (2026-11-20, CPU only).** On WSL CPython 3.12.3 we replayed
each of the six E0 captures once per method, pairing all 604,105 visible content chunks
by request. Both methods peaked at 659 outstanding and drained in 150-151 s. Timing fit
uses the same predeclared limits in both series; it is separate from client fidelity:

| Scheduling clock | Timing fit | Client fidelity | Per-run p99 absolute error vs original chunk times on the workload clock |
|---|---|---|---|
| Start a clock when each HTTP request arrives | 5/6 | 6/6 valid | 31.2-59.6 ms |
| One monotonic epoch for the whole workload | 5/6 | 0/6 valid | 4.0-6.1 ms |

Each shared-clock run had one overdue wake above 50 ms near 23 s, overlapping a recorded
generation-2 client GC pause. Restricting to the five timing-qualified runs, the overlap
was 279.3 of 305.4 ms of overdue time (91.4%); run 004 was excluded from that figure
because its request-relative chunk offset error was 50.7 ms at p99 (limit: 50 ms).
This reproduces the failure signature locally; it does not identify the cause on the Pod:
E0 kept client-observed output times, not a time-aligned client GC trace. No GPU SLO follows.

### Normal-GC streaming client

On 2026-11-21 we profiled the client, not vLLM, on WSL CPython 3.12.3. At 23 s the HTTPX
baseline held 508 suspended instances of each of `aiter_lines`, `aiter_text`, `aiter_bytes`
and `aiter_raw`. The measured replay path now reads bytes from aiohttp and releases parsed
JSON before awaiting the next chunk. The workload, in-flight limit, GC policy and 50 ms
fidelity limits did not change. In separate heap captures, GC-tracked object growth from
start to 23 s was 83,444 before and 32,433 after. These are object counts, not timing results.

**Unprofiled qualification:** each frozen client replayed the same source (`run-001`) three
times in one PID, with warmup and the normal CLI lifecycle on each wave. Both completed
2,361/2,361 streams and reproduced every content-chunk width and terminal usage, including
the source's 30 visible-token/count discrepancies per wave. Only the candidate passed the
predeclared timing-fit, 50 ms client-fidelity and RSS checks jointly on all three waves.

| Worst across three waves | HTTPX baseline | aiohttp candidate |
|---|---:|---:|
| Waves meeting all checks | 0/3 | 3/3 |
| Launch lag / loop lag / GC pause (ms) | 80.6 / 107.2 / 102.2 | 18.5 / 36.7 / 37.9 |
| Peak sampled RSS (limit 192 MiB) | 88.5 MiB | 77.3 MiB |
| Settled RSS growth, last wave vs first (limit 16 MiB) | 2.2 MiB | 1.0 MiB |

**Separate six-schedule diagnostic:** the candidate matched all six captured schedules and
all 604,105 visible chunks; 5/6 passed client fidelity. `run-003` had one 130.9 ms overdue
wake with no recorded GC overlap. It remains invalid; its cause is unknown. These are local
generator results, not a model speedup, an indefinite memory guarantee or a new GPU A/A.

## Milestones

| Weeks | Deliverable |
|---|---|
| 1-2 | Replay, fidelity, fake engine, CPU CI, GPU executor, A/A baseline |
| 3-4 | MVP: statistical comparison, correctness checks, three injected regressions |
| 5-6 | Cross-layer attribution |
| 7-8 | Flagship fix: decode ITL tail under mixed prefill |
| 9-10 | Real upgrade gate with bisect, FP8 candidate, multi-GPU module |
| 11-12 | Nightly operation, runbooks, blind-regression exam, v1.0 |
