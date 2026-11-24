"""Paired TTFT gate; one observation is an adjacent pair of full, same-host replays."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import workload
from .fidelity import FidelityPolicy, assess
from .metrics import distribution
from .paired_host import verify_server
from .paired_plan import (
    digest,
    file_digest,
    freeze,
    load_plan,
    local_url,
    number,
    pin,
    require,
    slots,
)
from .replay import ReplayResult
from .report import RECORDS_FILE, RUN_FILE, read_records, read_run, write_json

EXIT = {"pass": 0, "fail": 2, "invalid_measurement": 3, "inconclusive": 4}
BAD_EVIDENCE = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    EOFError,
    subprocess.SubprocessError,
)


def _rig(plan: dict, arm: str, run: dict) -> str:
    """Return the common rig fingerprint; arm-specific pinned settings are verified separately."""
    meta = run["meta"]
    url = meta["base_url"]
    local_url(url)
    model = plan["workload"]["model"]
    models = meta["server"]["models"]["data"]
    require(any(m["id"] == model for m in models), "pinned model not advertised")
    version = meta["server"]["version"]
    expected = plan["arms"][arm]
    client = run["client"]
    require(
        all(client.get(k) for k in ("host_id", "platform", "python", "slogate_version")),
        "missing client identity",
    )
    cfg = meta["replay_config"]
    require(
        cfg == plan["replay_config"] and cfg["ignore_eos"] is True,
        "model, transport, warmup, stopping, timeouts or load limits changed",
    )
    require(
        meta["fidelity_policy"] == plan["fidelity_policy"] == FidelityPolicy().to_dict(),
        "50 ms load-fidelity policy changed",
    )
    require(
        meta["slo"] == {"ttft_s": None, "tpot_s": None}, "unexpected per-request SLO in capture"
    )
    host = None
    if plan["kind"] == "fake":
        require(
            meta["sut_info"] is None and pin("fake", version) == expected,
            f"wrong fake profile for arm {arm}",
        )
    else:
        sut = meta["sut_info"]
        require(sut["worker_step_trace_requested"] is False, "worker tracing is not a gate sample")
        require(pin("vllm", sut["config"]) == expected, f"SUT manifest differs from arm {arm}")
        engine = expected["engine"]["expect"]
        require(
            all(sut["engine"][k] == v for k, v in engine.items())
            and version["version"] == engine["vllm"],
            "engine pins not observed",
        )
        host = sut["host"]
        devices = host["gpus"]
        require(
            len(devices) == 1 and expected["hardware"]["gpu_name_contains"] in devices[0]["name"],
            "GPU count/model mismatch",
        )
        device_id = devices[0].get("device_id")
        require(
            isinstance(device_id, str)
            and len(device_id) == 64
            and all(c in "0123456789abcdef" for c in device_id)
            and bool(devices[0]["driver_version"]),
            "missing hashed GPU identity or driver; collect fresh SUT evidence",
        )
    # A fake A and B may use different ephemeral ports; real GPU builds use the same endpoint.
    endpoint = (
        url
        if host is not None
        else (urlsplit(url).scheme, urlsplit(url).hostname, urlsplit(url).path)
    )
    return digest([client, cfg, host, endpoint])


def _inspect(root: Path, name: str, arm: str, plan: dict, hash_: str) -> dict[str, Any]:
    directory = root / name
    seal = json.loads((directory / "seal.json").read_bytes())
    observation = seal.pop("vllm_listener", None)
    require(
        seal
        == {
            "plan_sha256": hash_,
            "slot": name,
            "run_sha256": file_digest(directory / RUN_FILE),
            "records_sha256": file_digest(directory / RECORDS_FILE),
        },
        "slot seal or artifact hash mismatch",
    )
    doc = read_run(directory)
    require(
        doc["schema_version"] == 1 and doc["meta"]["label"] == f"{hash_}/{name}",
        "run does not belong to this frozen slot",
    )
    require(doc["meta"]["workload"] == plan["workload"], "workload metadata mismatch")
    rig = _rig(plan, arm, doc)
    if plan["kind"] == "vllm":
        require(isinstance(observation, dict), "missing live vLLM listener evidence")
        before = observation["before"]
        require(
            before == observation["after"]
            and before["pin_sha256"] == digest(plan["arms"][arm])
            and before.get("VLLM_BATCH_INVARIANT") == "0"
            and before.get("worker_trace") == "off"
            and len(before["instance_sha256"]) == 64
            and before["device_id"] == doc["meta"]["sut_info"]["host"]["gpus"][0]["device_id"],
            "vLLM endpoint, normal performance mode, config or device not verified across capture",
        )
    rr = ReplayResult.from_dict(dict(doc["replay"], records=read_records(directory)))
    require(
        rr.workload_digest == plan["workload"]["digest"]
        and rr.model == plan["workload"]["model"]
        and rr.duration_s == plan["workload"]["duration_s"],
        "replayed workload/model mismatch",
    )
    require(
        number(rr.started_unix_s)
        and rr.started_unix_s >= plan["frozen_unix_s"]
        and number(rr.wall_s)
        and rr.wall_s >= rr.duration_s,
        "invalid capture clock or run predates plan",
    )
    di = rr.diagnostics
    if not isinstance(di, dict) or "timing_trace" in di:
        raise ValueError("unqualified diagnostic trace or missing monitors")
    loop, gc = di["event_loop_lag_s"], di["gc"]
    require(
        loop["n"] > 0 and number(loop["max"]) and loop["max"] >= 0, "missing loop-lag observations"
    )
    require(
        type(gc["collections"]) is int
        and gc["collections"] >= 0
        and gc["dropped"] == 0
        and (
            (gc["collections"] == 0 and gc["pause_max"] is None)
            or (gc["collections"] > 0 and number(gc["pause_max"]) and gc["pause_max"] >= 0)
        ),
        "incomplete GC observations",
    )
    ttfts = []
    layout = plan["layout"]
    require(len(rr.records) == len(layout), "request count mismatch")
    for (rid, arrival, cap), r in zip(layout, rr.records, strict=True):
        require(
            r.id == rid and r.scheduled_s == arrival, "request identities/arrival schedule changed"
        )
        require(type(r.success) is bool, "missing request outcome")
        if r.launched_s is not None:
            require(number(r.launched_s) and r.launched_s >= 0, "invalid request launch time")
        if r.success:
            require(
                r.status == 200 and r.output_tokens == cap,
                "success without expected status or final completion-token count",
            )
            value = r.ttft_s
            if value is None or not number(value) or value < 0:
                raise ValueError("missing or invalid first-token time")
            require(
                bool(r.chunk_times_s)
                and r.first_token_s == r.chunk_times_s[0]
                and number(r.finished_s)
                and r.first_token_s is not None
                and r.finished_s is not None
                and r.first_token_s <= r.finished_s <= rr.wall_s,
                "timeline contradicts first-token or completion observation",
            )
            ttfts.append(value)
    failed = len(layout) - len(ttfts)
    metrics = doc["metrics"]
    counts = metrics["requests"]
    require(
        counts["offered"] == len(layout)
        and counts["failed"] == failed
        and counts["succeeded"] == len(ttfts)
        and counts["launched"] == sum(r.launched_s is not None for r in rr.records),
        "summary request counts contradict records",
    )
    p99 = distribution(ttfts)["p99"]
    stored = metrics["ttft_s"]
    require(
        stored["n"] == len(ttfts)
        and (
            (p99 is None and stored["p99"] is None)
            or (
                p99 is not None
                and number(stored["p99"])
                and math.isclose(stored["p99"], p99, rel_tol=1e-12)
            )
        ),
        "summary TTFT contradicts records",
    )
    sanity = assess(rr)
    require(
        doc["fidelity"]["valid"] is sanity.valid, "stored fidelity contradicts measured fidelity"
    )
    return {
        "slot": name,
        "arm": arm,
        "run_sha256": seal["run_sha256"],
        "rig_sha256": rig,
        "start": rr.started_unix_s,
        "end": rr.started_unix_s + rr.wall_s,
        "ttft_p99_ms": None if p99 is None else 1000 * p99,
        "requests_failed": failed,
        "visible_usage_mismatches": sum(
            r.success and r.streamed_tokens != r.output_tokens for r in rr.records
        ),
        "issues": list(sanity.reasons),
    }


def record(
    root: Path,
    arm: str,
    url: str,
    work: Path,
    sut: Path | None,
    key_env: str | None,
    server_pid: int | None = None,
) -> int:
    from .cli import build_parser, run_once

    plan, hash_ = load_plan(root)
    local_url(url)
    w = workload.load(work)
    require(w.digest == plan["workload"]["digest"], "wrong workload for the frozen plan")
    require(
        (sut is not None) == (server_pid is not None) == (plan["kind"] == "vllm"),
        "vLLM requires --sut-info and --server-pid; fake runs must omit both",
    )
    before = None
    if sut is not None:
        info = json.loads(sut.read_bytes())
        require(
            info["worker_step_trace_requested"] is False
            and pin("vllm", info["config"]) == plan["arms"][arm],
            "SUT manifest/trace settings do not match the assigned arm",
        )
        if server_pid is None:
            raise ValueError("missing server PID")
        devices = info["host"]["gpus"]
        require(
            isinstance(devices, list) and len(devices) == 1,
            "single device required in SUT manifest",
        )
        device = devices[0]
        before = verify_server(server_pid, url, plan["arms"][arm], device)
    for name, expected_arm in slots(plan):
        directory = root / name
        if not directory.exists():
            require(arm == expected_arm, f"next slot {name} requires arm {expected_arm}")
            break
        require(
            (directory / "seal.json").is_file(),
            f"unfinished recorded slot {name}; retain this experiment, do not replace it",
        )
    else:
        raise ValueError("all preassigned slots already exist")
    # Exclusive reservation before warmup: a second collector cannot run the next arm concurrently.
    directory.mkdir(exist_ok=False)
    argv = [
        "run",
        "--workload",
        str(work),
        "--base-url",
        url,
        "--out",
        str(directory),
        "--warmup-requests",
        str(plan["warmup_requests"]),
        "--wait-ready-s",
        "15",
    ]
    if sut is not None:
        argv += ["--sut-info", str(sut)]
    if key_env:
        argv += ["--api-key-env", key_env]
    # write_run itself requires a new directory; expose the seal only after both files exist.
    capture = directory / "capture"
    metrics, fidelity = run_once(build_parser().parse_args(argv), w, capture, f"{hash_}/{name}")
    for filename in (RUN_FILE, RECORDS_FILE):
        (capture / filename).rename(directory / filename)
    capture.rmdir()
    seal: dict[str, Any] = {
        "plan_sha256": hash_,
        "slot": name,
        "run_sha256": file_digest(directory / RUN_FILE),
        "records_sha256": file_digest(directory / RECORDS_FILE),
    }
    if before is not None and server_pid is not None:
        after = None
        try:
            after = verify_server(server_pid, url, plan["arms"][arm], device)
        except BAD_EVIDENCE as exc:
            print(f"{name}: final server attestation failed: {exc}", file=sys.stderr)
        seal["vllm_listener"] = {"before": before, "after": after}
    write_json(directory / "seal.json", seal)
    if before is not None and before != seal["vllm_listener"]["after"]:
        return EXIT["invalid_measurement"]
    try:
        _rig(plan, arm, read_run(directory))
    except BAD_EVIDENCE as exc:
        print(f"{name}: INVALID identity: {exc}", file=sys.stderr)
        return EXIT["invalid_measurement"]
    print(
        f"{name}: fidelity={'valid' if fidelity.valid else 'INVALID'}; "
        f"failed requests={metrics['requests']['failed']}"
    )
    if not fidelity.valid or (arm == "A" and metrics["requests"]["failed"]):
        return EXIT["invalid_measurement"]
    return EXIT["fail"] if metrics["requests"]["failed"] else 0


def evaluate(root: Path) -> dict[str, Any]:
    from .paired_stats import paired_interval

    plan, hash_ = load_plan(root)
    observed, missing, issues = [], [], []
    last_end = None
    first_rig = None
    schedule = slots(plan)
    extra = {p.name for p in root.glob("pair-*")} - {name for name, _ in schedule}
    if extra:
        issues.append(f"unplanned slots present: {', '.join(sorted(extra))}")
    for name, arm in schedule:
        if not (root / name).exists():
            missing.append(name)
            continue
        r = None
        try:
            require(not missing, "gap in the preassigned sequence")
            r = _inspect(root, name, arm, plan, hash_)
            if first_rig is None:
                first_rig = r["rig_sha256"]
            require(
                r["rig_sha256"] == first_rig,
                "host, device, client settings or deployment topology changed",
            )
            require(last_end is None or r["start"] >= last_end, "runs overlap or are out of order")
            last_end = r["end"]
            issues.extend(f"{name}: {msg}" for msg in r["issues"])
            if arm == "A" and r["requests_failed"]:
                issues.append(
                    f"{name}: unhealthy baseline ({r['requests_failed']} failed requests)"
                )
        except BAD_EVIDENCE as exc:
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
            if r is None:
                r = {
                    "slot": name,
                    "arm": arm,
                    "ttft_p99_ms": None,
                    "requests_failed": None,
                    "issues": [],
                }
            r["issues"].append(str(exc))
        observed.append(r)
    absolute = {
        "ceiling_ms": plan["candidate_ceiling_ms"],
        "status": "not_configured" if plan["candidate_ceiling_ms"] is None else "unverified",
    }
    relative: dict[str, Any] = {"verdict": "not_evaluated"}
    failures = [r["requests_failed"] for r in observed if r["arm"] == "B"]
    n_failed = sum(n for n in failures if n is not None)
    known_counts = None not in failures and len(failures) == plan["pairs"]
    deltas = []
    if issues:
        verdict, reason = (
            "invalid_measurement",
            "invalid or noncomparable evidence; no pairs were silently dropped",
        )
    elif n_failed:
        verdict, reason = "fail", "candidate request failures; latency of survivors is not a pass"
    elif missing:
        verdict, reason = "inconclusive", "not all preassigned pairs were collected"
    else:
        deltas = [
            observed[i + 1]["ttft_p99_ms"] - observed[i]["ttft_p99_ms"]
            for i in range(0, len(observed), 2)
        ]
        relative = paired_interval(
            deltas, margin_ms=plan["margin_ms"], block_pairs=plan["block_pairs"]
        )
        if plan["candidate_ceiling_ms"] is not None:
            absolute["status"] = (
                "fail"
                if any(
                    r["ttft_p99_ms"] > plan["candidate_ceiling_ms"]
                    for r in observed
                    if r["arm"] == "B"
                )
                else "pass"
            )
        if absolute["status"] == "fail":
            verdict, reason = (
                "fail",
                "at least one candidate run exceeded the frozen absolute p99 ceiling",
            )
        else:
            verdict = "fail" if relative["verdict"] == "regression" else relative["verdict"]
            reason = relative["reason"]
    return {
        "scope": "performance_only",
        "verdict": verdict,
        "reason": reason,
        "plan_sha256": hash_,
        "metric": plan["metric"],
        "margin_ms": plan["margin_ms"],
        "planned_pairs": plan["pairs"],
        "runs": observed,
        "missing_slots": missing,
        "invalid_reasons": issues,
        "candidate_failed_requests": n_failed if known_counts else None,
        "paired_deltas_ms": deltas,
        "relative": relative,
        "candidate_absolute_slo": absolute,
        "correctness": "not_evaluated_here",
    }


def markdown(result: dict[str, Any]) -> str:
    r = result["relative"]
    absolute = result["candidate_absolute_slo"]
    ceiling = (
        "not configured"
        if absolute["ceiling_ms"] is None
        else f"{absolute['ceiling_ms']} ms ({absolute['status']})"
    )
    lines = [
        f"# Performance only: {result['verdict']}",
        "",
        str(result["reason"]),
        "",
        f"TTFT p99 margin: {result['margin_ms']} ms (B minus A); "
        f"candidate absolute ceiling: {ceiling}.",
        f"Relative verdict: {r['verdict']}. Output correctness is checked separately.",
    ]
    if r.get("lower_ms") is not None:
        lines.append(f"Paired interval: [{r['lower_ms']:.3f}, {r['upper_ms']:.3f}] ms.")
    if result["paired_deltas_ms"]:
        deltas = ", ".join(f"{d:+.3f}" for d in result["paired_deltas_ms"])
        lines.append(f"Per-pair B - A (ms): {deltas}.")
    lines += ["", "| Slot | TTFT p99 (ms) | Failed requests |", "|---|---:|---:|"]
    for run in result["runs"]:
        value = run["ttft_p99_ms"]
        formatted = "n/a" if value is None else f"{value:.3f}"
        failed = run["requests_failed"] if run["requests_failed"] is not None else "unknown"
        lines.append(f"| {run['slot']} | {formatted} | {failed} |")
    lines.extend(f"- {why}" for why in result["invalid_reasons"])
    if result["missing_slots"]:
        lines.append("Missing: " + ", ".join(result["missing_slots"]))
    return "\n".join(lines) + "\n"


def command(args: argparse.Namespace) -> int:
    try:
        if args.operation == "init":
            key = freeze(
                args.out,
                workload.load(args.workload),
                kind=args.kind,
                baseline=json.loads(args.baseline.read_bytes()),
                candidate=json.loads(args.candidate.read_bytes()),
                pairs=args.pairs,
                block_pairs=args.block_pairs,
                margin_ms=args.margin_ms,
                ceiling_ms=args.candidate_ceiling_ms,
                warmup_requests=args.warmup_requests,
            )
            print(f"Frozen {args.pairs} A/B pairs: {key}")
            return 0
        if args.operation == "record":
            return record(
                args.directory,
                args.arm,
                args.base_url,
                args.workload,
                args.sut_info,
                args.api_key_env,
                args.server_pid,
            )
        result = evaluate(args.directory)
        write_json(args.directory / "gate.json", result)
        text = markdown(result)
        (args.directory / "gate.md").write_text(text, encoding="utf-8")
        print(text)
        return EXIT[result["verdict"]]
    except BAD_EVIDENCE as exc:
        print(f"paired experiment error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def add_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("ab", help="freeze, record and judge same-host TTFT pairs")
    parser.set_defaults(handler=command)
    ops = parser.add_subparsers(dest="operation", required=True)
    init = ops.add_parser("init", help="freeze a new experiment before measuring A or B")
    init.add_argument("--workload", type=Path, required=True)
    init.add_argument("--kind", choices=["fake", "vllm"], required=True)
    init.add_argument(
        "--baseline", type=Path, required=True, help="expected /version JSON (fake) or SUT config"
    )
    init.add_argument(
        "--candidate", type=Path, required=True, help="expected candidate identity JSON"
    )
    init.add_argument("--pairs", type=int, required=True)
    init.add_argument("--block-pairs", type=int, default=1)
    init.add_argument("--margin-ms", type=float, required=True)
    init.add_argument(
        "--candidate-ceiling-ms",
        type=float,
        help="optional absolute per-run TTFT p99 ceiling, not the relative margin",
    )
    init.add_argument("--warmup-requests", type=int, default=2)
    init.add_argument("--out", type=Path, required=True)
    take = ops.add_parser("record", help="measure exactly the next preassigned arm")
    take.add_argument("directory", type=Path)
    take.add_argument("--arm", choices=["A", "B"], required=True)
    take.add_argument("--workload", type=Path, required=True)
    take.add_argument("--base-url", required=True)
    take.add_argument("--sut-info", type=Path)
    take.add_argument("--server-pid", type=int, help="live API-server PID owned by this vLLM arm")
    take.add_argument("--api-key-env")
    ops.add_parser(
        "report", help="verify the sealed runs and calculate a performance verdict"
    ).add_argument("directory", type=Path)
