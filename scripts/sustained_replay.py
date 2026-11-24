"""Qualify one client snapshot on repeated E0 waves (Linux/WSL, loopback only).

Run the same script twice, serially, with different --client-src and fresh --out directories.
Exit 0: all waves qualified; 3: qualification/budget failure; 1: experiment error.
Finite external RSS samples do not bound unsampled peaks or prove unlimited service life.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

SCRIPT = Path(__file__).resolve()
WORKER = SCRIPT.with_name("_sustained_client.py")
ROOT = SCRIPT.parents[1]
WORKLOAD = ROOT / "workloads/azure-2024-05-12-smoke-30s.json.gz"
NAME = "run-001"
SOURCE_SHA = "7b461fbb5e870edeac3628be447d7b10d24b405ce7d20a463794c64c509da391"
LIMITS = {
    "ttft_p99_abs_ms": 50,
    "chunk_offset_p99_abs_ms": 50,
    "global_chunk_p99_abs_ms": 50,
    "chunk_gap_p99_abs_ms": 10,
    "wall_abs_s": 2,
    "peak_rel": 0.05,
    "request_seconds_rel": 0.02,
}
POLICY = {
    "max_launch_lag_s": 0.05,
    "max_event_loop_lag_s": 0.05,
    "max_gc_pause_s": 0.05,
    "require_all_launched": True,
}
COUNTS = {
    "offered": 787,
    "launched": 787,
    "succeeded": 787,
    "failed": 0,
    "usage_token_mismatch": 30,
}
PEAK_MIB, GROWTH_MIB, WAVE_S = 192, 16, 240
POLL_S, SETTLE_S, SETTLE_N = 0.2, 2, 5


class BudgetExceeded(Exception):
    """Stop this arm, retaining all observed evidence."""


def client_package(explicit: Path | None) -> Path:
    paths = (
        [explicit]
        if explicit
        else [Path(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        or [ROOT / "src"]
    )
    for path in paths:
        pkg = path if path.name == "slogate" and (path / "cli.py").is_file() else path / "slogate"
        if (pkg / "__init__.py").is_file() and (pkg / "cli.py").is_file():
            return pkg.resolve()
    raise ValueError(f"no slogate package in {paths}; set --client-src explicitly")


def observe(proc, url, wave, row, first_settled):
    from cpu_pressure import rss_kib

    start = time.monotonic()
    done_at = None
    settled = []
    receipt = wave / "done.json"
    with (wave / "rss.tsv").open("x", buffering=1, encoding="utf-8") as log:
        print("unix_s\tphase\trss_kib", file=log)
        print(int(url.rsplit(":", 1)[1]), file=proc.stdin, flush=True)
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"client exited {proc.returncode}; see client.log")
            kib = rss_kib(proc.pid)
            if not kib:
                raise RuntimeError(f"missing VmRSS for client {proc.pid}")
            now = time.monotonic()
            if done_at is None and receipt.is_file():
                row.update(json.loads(receipt.read_text(encoding="utf-8")))
                if row["pid"] != proc.pid or row["cli_exit"] not in (0, 3):
                    raise RuntimeError(f"unexpected client receipt: {row}")
                done_at = now
            phase = "run" if done_at is None else "drained"
            if done_at is not None and now >= done_at + SETTLE_S:
                phase = "settled"
            print(f"{time.time():.6f}\t{phase}\t{kib}", file=log)
            row["elapsed_s"] = now - start
            row["peak_mib"] = max(row.get("peak_mib", 0), kib / 1024)
            if kib > PEAK_MIB * 1024:
                raise BudgetExceeded(f"{wave.name}: sampled {kib / 1024:.2f} > {PEAK_MIB} MiB")
            if now - start >= WAVE_S:
                raise BudgetExceeded(f"{wave.name}: exceeded {WAVE_S}s command/settle deadline")
            if phase == "settled":
                settled.append(kib / 1024)
                if len(settled) == SETTLE_N:
                    row["settled_mib"] = max(settled)
                    if first_settled is not None and max(settled) > first_settled + GROWTH_MIB:
                        raise BudgetExceeded(f"{wave.name}: settled RSS grew > {GROWTH_MIB} MiB")
                    return
            time.sleep(POLL_S)


def measure(args, pkg, fixture, profile, report):
    from cpu_pressure import local_engine

    cmd = [
        sys.executable,
        "-B",
        "-u",
        str(WORKER),
        str(args.out),
        str(args.workload),
        str(pkg),
        NAME,
    ]
    rows = report["waves"]
    with (
        (args.out / "client.log").open("x") as log,
        subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as proc,
    ):
        report["client_pid"] = proc.pid
        try:
            for i in range(1, args.waves + 1):
                wave = args.out / f"wave-{i:03}"
                wave.mkdir()
                row = {"wave": wave.name, "pid": proc.pid, "within_budget": False}
                rows.append(row)
                print(f"{wave.name}/{args.waves}: client PID {proc.pid}", flush=True)
                with local_engine(profile, fixture) as url:
                    try:
                        observe(proc, url, wave, row, rows[0].get("settled_mib") if i > 1 else None)
                        row["within_budget"] = True
                    except BaseException:
                        proc.kill()
                        raise
            assert proc.stdin is not None
            proc.stdin.close()
            report["client_exit"] = proc.wait(timeout=5)
            if report["client_exit"] != max(r["cli_exit"] for r in rows):
                raise RuntimeError(f"unexpected client exit {proc.returncode}; see client.log")
        finally:
            if proc.poll() is None:
                proc.kill()


def assess_wave(wave, original, src_wall, digest, row):
    from slogate.pressure_report import occupancy
    from slogate.report import read_records, read_run
    from slogate.timeline_report import timing_errors

    # compare_run/summarize_run require timing_trace. Use their real trace-free primitives.
    actual = read_records(wave / NAME)
    run = read_run(wave / NAME)
    replay, meta, fidelity = run["replay"], run["meta"], run["fidelity"]
    diag, req = replay["diagnostics"], run["metrics"]["requests"]
    fit = timing_errors(original, actual)
    peak, _, area = occupancy(actual, [])
    src_peak, _, src_area = occupancy(original, [])
    fit.update(
        source_peak=src_peak,
        source_request_seconds=src_area,
        source_wall_s=src_wall,
        peak_rel=abs(peak - src_peak) / src_peak,
        request_seconds_rel=abs(area - src_area) / src_area,
        wall_abs_s=abs(replay["wall_s"] - src_wall),
    )
    missed = [k for k, limit in LIMITS.items() if fit[k] is None or not fit[k] <= limit]
    server = meta["server"]["version"]
    maxima = {
        "launch": fidelity["stats"].get("launch_lag_s", {}).get("max"),
        "loop": diag["event_loop_lag_s"]["max"],
        "gc": diag["gc"]["pause_max"],
    }
    checks = {
        "identity": (
            (server["source"], server["source_records_sha256"], server["anchor"])
            == (NAME, SOURCE_SHA, "shared")
            and meta["replay_config"]["local_fixture_clock"]
            and meta["workload"]["digest"] == digest
        ),
        "ordered_ids": [r["id"] for r in original] == [r["id"] for r in actual],
        "streams_and_usage": not fit["unmatched_ids"],
        "counts": all(req.get(k) == v for k, v in COUNTS.items()),
        "timing_fit": not missed,
        "policy": meta["fidelity_policy"] == POLICY,
        "summaries_only": (
            "timing_trace" not in diag
            and diag["event_loop_lag_s"]["n"] > 0
            and maxima["launch"] is not None
            and maxima["loop"] is not None
            and (not diag["gc"]["collections"] or maxima["gc"] is not None)
        ),
        "fidelity": fidelity["valid"] and row["cli_exit"] == 0,
        "resources": row["within_budget"],
    }
    ms = {k: v * 1000 if v is not None else None for k, v in maxima.items()}
    row.update(
        requests=req,
        checks=checks,
        reasons=fidelity["reasons"],
        maxima_ms=ms,
        timing=fit,
        missed_limits=missed,
        peak_open=peak,
        request_seconds=area,
        wall_s=replay["wall_s"],
        transport=meta["replay_config"].get("transport"),
    )
    print(
        f"{wave.name}: CLI {row['cli_exit']}, ok={req['succeeded']}/{req['offered']}, "
        f"mismatches={req['usage_token_mismatch']}, max ms={ms}, "
        f"peak/settled MiB={row.get('peak_mib')}/{row.get('settled_mib')}, "
        f"failed={[k for k in checks if not checks[k]]} {missed} {fidelity['reasons']}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="local E0 run-001 directory")
    p.add_argument("--out", type=Path, required=True, help="new directory for this arm")
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--client-src", type=Path, help="slogate package or parent (else PYTHONPATH)")
    p.add_argument("--workload", type=Path, default=WORKLOAD)
    args = p.parse_args()
    if sys.platform != "linux" or args.waves < 3:
        p.error("requires Linux procfs (e.g. WSL) and --waves >= 3")
    args.out, args.workload = args.out.resolve(), args.workload.resolve()
    source = args.source.resolve()
    # Fixture and scorer always come from the harness tree, never from the selected client.
    sys.path[:0] = [str(ROOT / "src"), str(SCRIPT.parent)]
    from _sustained_client import FLAGS
    from cpu_pressure import DIGEST, EngineProfile, RecordedStreams, load

    from slogate.report import read_records, read_run, write_json

    args.out.mkdir(parents=True, exist_ok=False)
    report = {"waves": [], "errors": []}
    rc = 0
    try:
        pkg = client_package(args.client_src)
        workload = load(args.workload)
        fixture = replace(RecordedStreams.load(source, workload), shared_clock=True)
        if (workload.digest, fixture.source, fixture.records_sha256) != (DIGEST, NAME, SOURCE_SHA):
            raise ValueError("expected the pinned E0 workload and run-001 records")
        original, src_wall = read_records(source), read_run(source)["replay"]["wall_s"]
        profile = EngineProfile(itl_s=0.010, jitter=0, seed=1)
        write_json(
            args.out / "protocol.json",
            {
                "source": str(source),
                "source_sha256": SOURCE_SHA,
                "workload_digest": DIGEST,
                "anchor": "shared",
                "waves": args.waves,
                "client_package": str(pkg),
                "policy": POLICY,
                "match_limits": LIMITS,
                "expected_requests": COUNTS,
                "profile": profile.to_dict(),
                "workload": str(args.workload),
                "cli_flags": FLAGS,
                "wave_timeout_s": WAVE_S,
                "max_sampled_mib": PEAK_MIB,
                "max_growth_mib": GROWTH_MIB,
                "sample_s": POLL_S,
                "settled_delay_s": SETTLE_S,
                "settled_samples": SETTLE_N,
                "settled_rule": "maximum of settled samples; compare every later wave to wave 1",
                "scope": "command through settled; sampled RSS is not an unlimited-lifetime proof",
                "harness_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCRIPT, WORKER)
                },
            },
        )
        try:
            measure(args, pkg, fixture, profile, report)
        finally:
            for row in report["waves"]:
                if "cli_exit" not in row:
                    continue
                try:
                    assess_wave(args.out / row["wave"], original, src_wall, DIGEST, row)
                    if not all(row["checks"].values()):
                        rc = rc or 3
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    report["errors"].append(f"{row['wave']}: {type(exc).__name__}: {exc}")
                    rc = 1
    except (
        KeyboardInterrupt,
        OSError,
        RuntimeError,
        ValueError,
        BudgetExceeded,
        subprocess.SubprocessError,
    ) as exc:
        rc = (rc or 3) if isinstance(exc, BudgetExceeded) else 1
        if isinstance(exc, KeyboardInterrupt):
            rc = 130
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    manifest = args.out / "client.json"
    if manifest.is_file():
        report["client"] = json.loads(manifest.read_text(encoding="utf-8"))
    report.update(qualified=rc == 0, exit_code=rc)
    write_json(args.out / "report.json", report)
    print(f"{args.out}: qualified={rc == 0}, exit={rc}", *report["errors"], sep="\n")
    return rc


if __name__ == "__main__":
    sys.dont_write_bytecode = True

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
