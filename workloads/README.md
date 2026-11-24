# Workloads

Workload files are inputs, not source code, so they are not committed. Every run records the
workload's SHA-256 digest; regenerate a file and check its digest before comparing runs.

| Name | How to create | Requests | Duration | Digest |
|---|---|---:|---:|---|
| `azure-2024-05-12-smoke-30s` | `slogate workload import-lab <lab>/results/trace-smoke-input/trace.json --name azure-2024-05-12-smoke-30s --out workloads/azure-2024-05-12-smoke-30s.json.gz` | 787 | 30 s | `821ec458b144aa45b4126af56d360f055615aafeb34ac3467c604144664aa13c` |

`<lab>` is a checkout of `dlinlabs/vllm-serving-lab`. Its traces take arrival times and input
sizes from the Azure LLM Inference Dataset 2024 (CC-BY 4.0) and synthesize prompts that match
the Qwen3-4B-Instruct-2507 chat template token for token.

For CPU smoke tests, generate a synthetic workload instead:

```bash
slogate workload synth --name smoke --duration-s 20 --rate-rps 8 --out workloads/smoke.json
```
