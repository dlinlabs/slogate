# Weeks 1-2: measuring vLLM before judging a change

**Result, 2026-11-23:** six unchanged vLLM runs on one RTX 4090 passed the client
measurement checks. All 4,722 requests completed. This is a reference for how much tail
latency moves on one workload, not evidence that vLLM got faster.

## Week 1: can we trust the client?

The [replay client](../src/slogate/replay.py) sends each request at its scheduled arrival,
without waiting for earlier responses. It records time to first token (TTFT), the gaps
between visible output chunks used for inter-token latency (ITL), and server-reported token
usage. A workload digest makes it possible to check that two runs offered the same inputs.

We tested the harness against a CPU fake engine before renting a GPU. In three unchanged
runs, TTFT p99 was 69.1, 67.7 and 74.0 ms. There was no candidate change: the 6.3 ms
difference between the last two runs was not an optimization. This is why the gate starts
with A/A, and why [client fidelity](../src/slogate/fidelity.py) is separate from a serving SLO.

## Week 2: what failed, and what changed?

On the first GPU attempt, **4,722 requests succeeded but 0/6 runs qualified**. In every run,
the load generator's event loop and its garbage collector paused for more than 50 ms. We
kept those results as failed measurements instead of relaxing the limit or blaming vLLM.

Replaying the recorded response timings on CPU reproduced the stalls without a GPU. A heap
profile at 23 s showed 508 live streams each retaining several HTTPX decoding layers. We
changed the measured stream reader to aiohttp and released parsed JSON before waiting for
the next chunk. In an **unprofiled**, three-wave test with one client process per build:

| Check | Previous client | Revised client |
|---|---:|---:|
| Waves passing all checks | 0/3 | 3/3 |
| Worst client event-loop delay | 107.2 ms | 36.7 ms |
| Worst client GC pause | 102.2 ms | 37.9 ms |
| Peak sampled client memory (RSS) | 88.5 MiB | 77.3 MiB |

Automatic GC stayed on. A separate six-schedule CPU diagnostic still found one invalid run:
the loop woke 130.9 ms late with no recorded GC overlap. The change did not fix every source
of client delay. See the [qualification protocol and limitations](design.md#normal-gc-streaming-client).

## The qualified GPU baseline (E0)

One RTX 4090 served Qwen3-4B-Instruct-2507 on vLLM 0.30.0. Each run replayed the same
[787 arrivals over 30 seconds](../workloads/README.md) with `aiohttp 3.14.5`; worker tracing
was off. The original 50 ms client limits applied to all six runs. The figures below use
one p99 per run, not a percentile pooled across all requests.

| Metric | Mean of six p99s | Min-max | Run-to-run CV |
|---|---:|---:|---:|
| TTFT | 120.139 s | 119.363-120.345 s | 0.318% |
| ITL (visible output) | 160.39 ms | 159.03-161.48 ms | 0.515% |

[Per-run measurements (CSV)](results/e0-2026-11-23.csv) ·
[Pinned model, engine and flags](../configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json) ·
[Capture and launcher provenance](design.md#first-gpu-baseline-e0)

**Interpretation:** this workload queues heavily. At the end of the arrival window, 659
requests were still outstanding in each run. Two minutes to first token is a client wait
under that load, not two minutes of GPU computation or an acceptable product SLO. On
171/4,722 streams, visible token counts did not reconcile with final usage; ITL is not a
hardware timestamp for every generated token. No per-request TTFT/TPOT bounds were set,
so SLO goodput was **not measured**. The failed and qualified GPU attempts used different
hosts; they are not an engine A/B comparison.

**Next milestone:** use same-host baseline/candidate pairs and an output correctness check.
Freeze the practical regression margin before observing the candidate, and allow an
*inconclusive* result. At the close of week 2 this was still a design; see the
[week 3 gate](paired-gate.md) for subsequent implementation and remaining validation.
