"""Command-line entry point: ``slogate <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import httpx

from . import __version__, workload
from .fidelity import FidelityPolicy, FidelityVerdict, assess
from .metrics import SLO, summarize
from .noise import MIN_RUNS, noise_floor, to_markdown
from .replay import FIXTURE_EPOCH_HEADER, ReplayConfig, build_payload, replay
from .report import (
    NOISE_JSON,
    NOISE_MARKDOWN,
    RUN_FILE,
    read_run,
    redact_url,
    write_json,
    write_run,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_MEASUREMENT = 3


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.1f}ms"


def _api_key(args: argparse.Namespace) -> str | None:
    if not args.api_key_env:
        return None
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"environment variable {args.api_key_env} is empty or unset")
    return key


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def wait_ready(base_url: str, api_key: str | None, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "no attempt"
    while True:
        try:
            response = httpx.get(
                f"{base_url}/health", headers=_headers(api_key), timeout=5.0, trust_env=False
            )
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = type(exc).__name__
        if time.monotonic() >= deadline:
            raise SystemExit(f"server at {redact_url(base_url)} is not ready: {last_error}")
        time.sleep(1.0)


def probe_server(base_url: str, api_key: str | None) -> dict[str, Any]:
    """Best-effort record of what the server says it is running."""
    info: dict[str, Any] = {}
    for name, path in (("version", "/version"), ("models", "/v1/models")):
        try:
            response = httpx.get(
                f"{base_url}{path}", headers=_headers(api_key), timeout=5.0, trust_env=False
            )
            info[name] = response.json() if response.status_code == 200 else response.status_code
        except (httpx.HTTPError, ValueError) as exc:
            info[name] = f"unavailable: {type(exc).__name__}"
    return info


async def warmup(load: workload.Workload, config: ReplayConfig, count: int) -> int:
    """Send ``count`` requests sequentially before measurement; returns successes."""
    if count <= 0:
        return 0
    model = config.model or load.model
    successes = 0
    headers = _headers(config.api_key)
    if config.local_fixture_clock:
        headers[FIXTURE_EPOCH_HEADER] = "warmup"
    async with httpx.AsyncClient(
        base_url=config.base_url,
        headers=headers,
        timeout=config.request_timeout_s,
        trust_env=False,
    ) as client:
        for request in load.requests[:count]:
            payload = build_payload(request, model, config)
            async with client.stream("POST", config.path, json=payload) as response:
                async for _ in response.aiter_lines():
                    pass
                successes += response.status_code == 200
    return successes


def _replay_config(args: argparse.Namespace, api_key: str | None) -> ReplayConfig:
    return ReplayConfig(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        api_key=api_key,
        ignore_eos=not args.no_ignore_eos,
        request_timeout_s=args.request_timeout_s,
        drain_timeout_s=args.drain_timeout_s,
        max_in_flight=args.max_in_flight,
        local_fixture_clock=getattr(args, "local_fixture_clock", False),
    )


def _slo(args: argparse.Namespace) -> SLO:
    return SLO(
        ttft_s=None if args.slo_ttft_ms is None else args.slo_ttft_ms / 1000,
        tpot_s=None if args.slo_tpot_ms is None else args.slo_tpot_ms / 1000,
    )


def run_once(
    args: argparse.Namespace, load: workload.Workload, directory: Path, label: str
) -> tuple[dict[str, Any], FidelityVerdict]:
    api_key = _api_key(args)
    config = _replay_config(args, api_key)
    if args.wait_ready_s > 0:
        wait_ready(config.base_url, api_key, args.wait_ready_s)
    server = probe_server(config.base_url, api_key)
    if config.local_fixture_clock:
        version = server["version"]
        if not isinstance(version, dict) or version.get("anchor") != "shared":
            raise SystemExit("expected a recorded stream fixture with a shared clock")
    warmed = asyncio.run(warmup(load, config, args.warmup_requests))
    if warmed < args.warmup_requests:
        raise SystemExit(f"warmup failed: {warmed}/{args.warmup_requests} requests succeeded")
    result = asyncio.run(replay(load, config, client_timing=args.client_timing))
    slo = _slo(args)
    metrics = summarize(result, slo)
    policy = FidelityPolicy(max_launch_lag_s=args.max_launch_lag_ms / 1000)
    verdict = assess(result, policy)
    sut_info = (
        json.loads(Path(args.sut_info).read_text(encoding="utf-8")) if args.sut_info else None
    )
    meta = {
        "label": label,
        "base_url": redact_url(config.base_url),
        "server": server,
        "sut_info": sut_info,
        "workload": load.summary(),
        "replay_config": {
            "transport": {"name": "aiohttp", "version": aiohttp.__version__},
            "model": result.model,
            "ignore_eos": config.ignore_eos,
            "request_timeout_s": config.request_timeout_s,
            "drain_timeout_s": config.drain_timeout_s,
            "max_in_flight": config.max_in_flight,
            "warmup_requests": args.warmup_requests,
            **({"local_fixture_clock": True} if config.local_fixture_clock else {}),
        },
        "slo": slo.to_dict(),
        "fidelity_policy": policy.to_dict(),
    }
    write_run(directory, result, metrics=metrics, fidelity=verdict.to_dict(), meta=meta)
    return metrics, verdict


def _print_run(name: str, metrics: dict[str, Any], verdict: FidelityVerdict) -> None:
    requests = metrics["requests"]
    state = "valid" if verdict.valid else "INVALID (" + "; ".join(verdict.reasons) + ")"
    print(
        f"{name}: {state} ok={requests['succeeded']}/{requests['offered']} "
        f"ttft_p99={_ms(metrics['ttft_s']['p99'])} itl_p99={_ms(metrics['itl_s']['p99'])} "
        f"tpot_p50={_ms(metrics['tpot_s']['p50'])}",
        flush=True,
    )


def cmd_run(args: argparse.Namespace) -> int:
    load = workload.load(args.workload)
    metrics, verdict = run_once(args, load, Path(args.out), args.label or Path(args.out).name)
    _print_run(Path(args.out).name, metrics, verdict)
    return EXIT_OK if verdict.valid else EXIT_INVALID_MEASUREMENT


def aa_exclusion(fidelity: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    """Why a run cannot serve as an A/A noise sample, or None when it can.

    Request failures are server behavior and are kept in the run's metrics, but a noise
    floor must come from healthy runs only.
    """
    if not fidelity["valid"]:
        return "invalid measurement: " + "; ".join(fidelity["reasons"])
    failed = metrics["requests"]["failed"]
    if failed:
        return f"{failed} requests failed"
    return None


def _finish_noise(directory: Path, run_dirs: list[Path]) -> int:
    """Use the persisted run verdicts for both live A/A and offline recomputation."""
    included: list[tuple[str, dict[str, Any]]] = []
    excluded: dict[str, str] = {}
    for path in run_dirs:
        run = read_run(path)
        reason = aa_exclusion(run["fidelity"], run["metrics"])
        if reason is None:
            included.append((path.name, run["metrics"]))
        else:
            excluded[path.name] = reason
    report = noise_floor([metrics for _, metrics in included])
    report["included_runs"] = [name for name, _ in included]
    report["excluded_runs"] = excluded
    write_json(directory / NOISE_JSON, report)
    markdown = to_markdown(report)
    (directory / NOISE_MARKDOWN).write_text(markdown, encoding="utf-8")
    print(markdown)
    return EXIT_OK if len(included) >= MIN_RUNS else EXIT_INVALID_MEASUREMENT


def cmd_aa(args: argparse.Namespace) -> int:
    load = workload.load(args.workload)
    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=False)
    run_dirs = [base / f"run-{index:03d}" for index in range(1, args.runs + 1)]
    for index, path in enumerate(run_dirs, 1):
        metrics, verdict = run_once(args, load, path, f"{args.label or base.name}/{path.name}")
        _print_run(path.name, metrics, verdict)
        if index < args.runs and args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
    return _finish_noise(base, run_dirs)


def cmd_noise(args: argparse.Namespace) -> int:
    base = Path(args.directory)
    return _finish_noise(base, sorted(p.parent for p in base.glob(f"*/{RUN_FILE}")))


def cmd_workload_synth(args: argparse.Namespace) -> int:
    sizes = [int(part) for part in args.input_tokens.split(",") if part.strip()]
    load = workload.synthetic(
        name=args.name,
        model=args.model,
        duration_s=args.duration_s,
        rate_rps=args.rate_rps,
        input_tokens=sizes,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    workload.save(load, args.out)
    print(json.dumps(load.summary(), indent=2))
    return EXIT_OK


def cmd_workload_import_lab(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    load = workload.from_lab_trace(data, name=args.name)
    workload.save(load, args.out)
    print(json.dumps(load.summary(), indent=2))
    return EXIT_OK


def cmd_workload_info(args: argparse.Namespace) -> int:
    print(json.dumps(workload.load(args.path).summary(), indent=2))
    return EXIT_OK


def cmd_fake_engine(args: argparse.Namespace) -> int:
    from .fake_engine import EngineProfile, FakeEngine, serve

    profile = EngineProfile(
        model=args.model,
        ttft_s=args.ttft_ms / 1000,
        ttft_per_input_token_s=args.ttft_per_token_us / 1e6,
        itl_s=args.itl_ms / 1000,
        jitter=args.jitter,
        load_slope=args.load_slope,
        stall_probability=args.stall_probability,
        stall_s=args.stall_ms / 1000,
        extra_itl_s=args.extra_itl_ms / 1000,
        seed=args.seed,
    )
    print(f"fake engine on http://{args.host}:{args.port} {profile.to_dict()}", flush=True)
    serve(FakeEngine(profile), host=args.host, port=args.port)
    return EXIT_OK


def _add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:8000")
    parser.add_argument("--workload", required=True, help="workload .json or .json.gz")
    parser.add_argument("--out", required=True, help="output directory (must not exist)")
    parser.add_argument("--label", help="free-form label stored with the run")
    parser.add_argument("--model", help="override the workload's model name")
    parser.add_argument("--api-key-env", help="name of the environment variable holding the key")
    parser.add_argument("--sut-info", help="JSON file describing the system under test")
    parser.add_argument("--slo-ttft-ms", type=float, help="per-request TTFT bound for goodput")
    parser.add_argument("--slo-tpot-ms", type=float, help="per-request TPOT bound for goodput")
    parser.add_argument("--no-ignore-eos", action="store_true", help="let outputs stop at EOS")
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--wait-ready-s", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--drain-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-in-flight", type=int, default=4096)
    parser.add_argument("--max-launch-lag-ms", type=float, default=50.0)
    parser.add_argument(
        "--client-timing",
        action="store_true",
        help="save client wakeup deadlines/lags and GC spans (diagnostic runs only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slogate", description=__doc__)
    parser.add_argument("--version", action="version", version=f"slogate {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="replay a workload once and record metrics")
    _add_replay_arguments(run)
    run.add_argument(
        "--local-fixture-clock",
        action="store_true",
        help="send replay time zero to a same-host recorded stream fixture (diagnostic only)",
    )
    run.set_defaults(handler=cmd_run)

    aa = commands.add_parser("aa", help="repeat the same run to measure the noise floor")
    _add_replay_arguments(aa)
    aa.add_argument("--runs", type=int, default=6)
    aa.add_argument("--cooldown-s", type=float, default=10.0)
    aa.set_defaults(handler=cmd_aa)

    noise = commands.add_parser("noise", help="recompute the A/A noise report from run folders")
    noise.add_argument("directory")
    noise.set_defaults(handler=cmd_noise)

    load = commands.add_parser("workload", help="create or inspect workloads")
    load_commands = load.add_subparsers(dest="workload_command", required=True)
    synth = load_commands.add_parser("synth", help="Poisson synthetic workload")
    synth.add_argument("--name", required=True)
    synth.add_argument("--model", default="fake-model")
    synth.add_argument("--duration-s", type=float, required=True)
    synth.add_argument("--rate-rps", type=float, required=True)
    synth.add_argument("--input-tokens", default="128,512", help="comma-separated sizes")
    synth.add_argument("--max-tokens", type=int, default=64)
    synth.add_argument("--seed", type=int, default=1)
    synth.add_argument("--out", required=True)
    synth.set_defaults(handler=cmd_workload_synth)
    lab = load_commands.add_parser("import-lab", help="convert a vllm-serving-lab trace")
    lab.add_argument("trace")
    lab.add_argument("--name", required=True)
    lab.add_argument("--out", required=True)
    lab.set_defaults(handler=cmd_workload_import_lab)
    info = load_commands.add_parser("info", help="print a workload summary and digest")
    info.add_argument("path")
    info.set_defaults(handler=cmd_workload_info)

    fake = commands.add_parser("fake-engine", help="serve the CPU fake engine")
    fake.add_argument("--host", default="127.0.0.1")
    fake.add_argument("--port", type=int, default=8000)
    fake.add_argument("--model", default="fake-model")
    fake.add_argument("--ttft-ms", type=float, default=50.0)
    fake.add_argument("--ttft-per-token-us", type=float, default=0.0)
    fake.add_argument("--itl-ms", type=float, default=10.0)
    fake.add_argument("--jitter", type=float, default=0.10)
    fake.add_argument("--load-slope", type=float, default=0.0)
    fake.add_argument("--stall-probability", type=float, default=0.0)
    fake.add_argument("--stall-ms", type=float, default=0.0)
    fake.add_argument("--extra-itl-ms", type=float, default=0.0)
    fake.add_argument("--seed", type=int)
    fake.set_defaults(handler=cmd_fake_engine)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except workload.WorkloadError as exc:
        print(f"workload error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
