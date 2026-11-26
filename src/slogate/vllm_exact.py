"""Collect native token-ID evidence for the pinned small-budget and eager reference trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from ._cli_args import UsageParser
from .correctness import EXIT, check
from .native_tokens import MODEL, REVISION, TOKENIZER, json_reply, observed_cases, suite_cases
from .paired_host import process_instance, verify_server
from .paired_plan import digest, file_digest, load_plan, require
from .report import write_json

EXTRA = {
    "--attention-backend": ["FLASH_ATTN"],
    "--dtype": ["bfloat16"],
    "--generation-config": ["vllm"],
}
ENGINE = {"vllm": "0.30.0", "torch": "2.13.0+cu130", "cuda_runtime": "13.0"}
ERRORS = (OSError, ValueError, KeyError, TypeError, httpx.HTTPError, subprocess.SubprocessError)


def _api_key() -> str:
    key = os.environ.get("VLLM_API_KEY", "")
    require(
        bool(key) and all(0x21 <= ord(char) <= 0x7E for char in key),
        "a nonempty ASCII VLLM_API_KEY without whitespace is required in the environment",
    )
    return key


def prepare(ab: Path, suite_path: Path, out: Path) -> str:
    plan, plan_hash = load_plan(ab)
    suite = json.loads(suite_path.read_bytes())
    cases = suite_cases(suite)
    base_flags = {
        "--max-num-batched-tokens": ["2048"],
        "--max-model-len": ["8192"],
        "--max-num-seqs": ["32"],
        "--tensor-parallel-size": ["1"],
        "--no-enable-prefix-caching": [],
    }
    overrides = {"--max-num-batched-tokens": ["512"], "--enforce-eager": []}
    changes = plan["change"]
    require(
        plan["kind"] == "vllm" and len(changes) == 1 and changes[0] in overrides,
        "expected a pinned small-budget or eager candidate",
    )
    changed = changes[0]
    arms = {}
    for arm, flags in (("A", base_flags), ("B", {**base_flags, changed: overrides[changed]})):
        p = plan["arms"][arm]
        require(
            p["model"] == {"id": MODEL, "revision": REVISION}
            and p["engine"]["expect"] == ENGINE
            and p["hardware"] == {"gpu_count": 1, "gpu_name_contains": "RTX 4090"}
            and p["flags"] == flags,
            "unexpected reference model, runtime or configuration",
        )
        arms[arm] = {**p, "flags": {**p["flags"], **EXTRA}}
    spec = {
        "schema": "vllm-exact-collection-v1",
        "ab_plan_sha256": plan_hash,
        "arms": arms,
        "identity": {
            "model_sha256": digest({"repo": MODEL, "revision": REVISION}),
            "tokenizer_sha256": digest(
                {"repo": MODEL, "revision": REVISION, "tokenizer.json": TOKENIZER}
            ),
            "suite_sha256": digest(cases),
        },
        "protocol": {
            "decoding": "greedy",
            "temperature": 0,
            "ignore_eos": False,
            "max_new_tokens": suite["max_new_tokens"],
        },
        "cases": cases,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "spec.json", spec)
    return file_digest(out / "spec.json")


def _read_spec(root: Path, expected: str) -> dict:
    data = (root / "spec.json").read_bytes()
    require(
        hashlib.sha256(data).hexdigest() == expected,
        "collection spec checksum differs from trusted pin",
    )
    spec = json.loads(data)
    require(
        spec["schema"] == "vllm-exact-collection-v1"
        and spec["identity"]["suite_sha256"] == digest(spec["cases"]),
        "invalid collection spec",
    )
    return spec


def _plan(spec: dict, ref_hash: str) -> dict:
    return {
        "schema": "correctness-plan-v1",
        "mode": "exact",
        "backend": "vllm",
        "reference_sha256": ref_hash,
        **{key: spec[key] for key in ("identity", "protocol", "cases")},
    }


def _reference(root: Path, spec: dict, spec_hash: str, plan_hash: str | None) -> tuple[Path, Path]:
    r, p = root / "reference" / "evidence.json", root / "reference" / "plan.json"
    require(
        plan_hash is not None and file_digest(p) == plan_hash, "trusted reference plan pin required"
    )
    receipt = json.loads((r.parent / "receipt.json").read_bytes())
    require(
        receipt["status"] == "collected"
        and receipt["role"] == "reference"
        and receipt["ab_plan_sha256"] == spec["ab_plan_sha256"]
        and receipt["spec_sha256"] == spec_hash
        and receipt["arm_pin_sha256"] == digest(_pin(spec, "reference"))
        and receipt["plan_sha256"] == plan_hash
        and receipt["evidence_sha256"] == file_digest(r)
        and json.loads(p.read_bytes()) == _plan(spec, file_digest(r)),
        "reference binding mismatch",
    )
    return r, p


def _pin(spec: dict, role: str) -> dict:
    return spec["arms"][{"reference": "A", "candidate": "B"}[role]]


def server_argv(expected: dict, port: int, python: Path) -> list[str]:
    args = [
        str(python.absolute()),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    for flag, values in expected["flags"].items():
        args.extend([flag, *values])
    return args


def _engine_build(python: str) -> dict:
    probe = (
        "import json,sys,torch; from importlib.metadata import version; "
        "print(json.dumps(dict(python=sys.version.split()[0], vllm=version('vllm'), "
        "torch=str(torch.__version__), cuda_runtime=torch.version.cuda)))"
    )
    raw = subprocess.run(
        [python, "-I", "-B", "-c", probe], check=True, capture_output=True, text=True, timeout=60
    ).stdout
    build = json.loads(raw)
    require(build == {"python": "3.12.3", **ENGINE}, "incorrect engine interpreter/build")
    return build


def _runtime_stamp(pid: int, port: int, expected: dict, log: Path) -> dict:
    gpu = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        .stdout.strip()
        .splitlines()
    )
    require(len(gpu) == 1, "exact check requires one visible GPU")
    uuid, name, driver, capability = [part.strip() for part in gpu[0].split(",")]
    require(
        uuid.startswith("GPU-")
        and "RTX 4090" in name
        and int(driver.split(".")[0]) >= 580
        and float(capability) >= 8.0,
        "unexpected GPU or unsupported compute capability",
    )
    device = {"device_id": hashlib.sha256(uuid.encode()).hexdigest(), "driver_version": driver}
    live = verify_server(pid, f"http://127.0.0.1:{port}", expected, device, batch_invariant=True)
    launch = json.loads(log.with_suffix(".json").read_bytes())
    require(
        launch
        == {
            "instance_sha256": live["instance_sha256"],
            "pin_sha256": live["pin_sha256"],
            "build": {"python": "3.12.3", **ENGINE},
        },
        "correctness engine build receipt does not match this live process",
    )
    # Do not trust a log from an old/other process, or infer the runtime from /models alone.
    require(Path(f"/proc/{pid}/fd/1").samefile(log), "server stdout does not match the arm log")
    with log.open("rb") as handle:
        startup = re.sub(rb"\x1b\[[0-9;]*m", b"", handle.read(2 * 1024 * 1024))
    require(
        re.search(rb"\bdtype=torch\.bfloat16\b", startup) is not None
        and re.search(
            rb"\bUsing (?:AttentionBackendEnum\.)?FLASH_ATTN(?: attention)? backend\b", startup
        )
        is not None,
        "resolved BF16/FLASH_ATTN startup evidence missing",
    )
    return {**live, "hardware": digest({**device, "compute_cap": capability})}


def collect(
    root: Path,
    spec: dict,
    spec_hash: str,
    role: str,
    pid: int,
    port: int,
    key: str,
    plan_hash: str | None,
) -> tuple[int, dict]:
    arm = root / role
    require(not arm.exists(), "attempt directory already exists; retain it, do not overwrite")
    baseline = _reference(root, spec, spec_hash, plan_hash) if role == "candidate" else None
    arm.mkdir()
    receipt: dict[str, Any] = {
        "spec_sha256": spec_hash,
        "ab_plan_sha256": spec["ab_plan_sha256"],
        "role": role,
    }
    try:
        expected, log = _pin(spec, role), root / f"{role}.server.log"
        before = _runtime_stamp(pid, port, expected, log)
        if baseline is not None:
            require(
                json.loads(baseline[0].read_bytes())["provenance"]["hardware"]
                == before["hardware"],
                "correctness arms are not on the same GPU and driver",
            )
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=120,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            require(
                json_reply(client.get("/version")).get("version") == "0.30.0",
                "wrong live vLLM version",
            )
            models = json_reply(client.get("/v1/models")).get("data")
            require(
                isinstance(models, list)
                and any(
                    isinstance(m, dict) and m.get("id") == MODEL and m.get("max_model_len") == 8192
                    for m in models
                ),
                "wrong live model or context length",
            )
            cases = observed_cases(
                client,
                spec["cases"],
                spec["protocol"]["max_new_tokens"],
                save=lambda name, rows: write_json(arm / f"{name}.json", rows),
            )
        after = _runtime_stamp(pid, port, expected, log)
        require(before == after, "server identity changed during correctness collection")
        provenance = {
            "backend": "vllm",
            "version": "0.30.0",
            "hardware": before["hardware"],
            "attention_backend": "FLASH_ATTN",
            "dtype": "bfloat16",
            "VLLM_BATCH_INVARIANT": "1",
            "batch_invariance_supported": True,
        }
        evidence = {
            "schema": "correctness-evidence-v1",
            "role": role,
            "provenance": provenance,
            "identity": spec["identity"],
            "protocol": spec["protocol"],
            "cases": cases,
        }
        e = arm / "evidence.json"
        write_json(e, evidence)
        sha = file_digest(e)
        receipt.update(
            status="collected",
            engine_instance=before["instance_sha256"],
            arm_pin_sha256=before["pin_sha256"],
            evidence_sha256=sha,
            singleton_vs_batch="matched_on_this_suite",
            cases=len(cases),
        )
        rc = 0
        if baseline is None:
            p = arm / "plan.json"
            write_json(p, _plan(spec, sha))
            receipt["plan_sha256"] = file_digest(p)
        else:
            ref, frozen_plan = _reference(root, spec, spec_hash, plan_hash)
            verdict = check(frozen_plan, ref, e, plan_sha256=plan_hash)
            write_json(arm / "report.json", verdict)
            receipt["correctness"] = verdict["status"]
            rc = EXIT[verdict["status"]]
    except httpx.HTTPError as exc:
        # Transport exceptions can include header values. Store only the exception class.
        receipt.update(status="unverified", reason=f"HTTP transport error: {type(exc).__name__}")
        rc = 4
    except ERRORS as exc:
        receipt.update(status="unverified", reason=str(exc))
        rc = 4
    write_json(arm / "receipt.json", receipt)
    return rc, receipt


def _configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="operation", required=True)
    init = sub.add_parser("prepare", help="local only: expand frozen token suite and bind A/B plan")
    for arg in ("ab", "suite", "out"):
        init.add_argument(f"--{arg}", type=Path, required=True)
    for name in ("server", "capture"):
        cmd = sub.add_parser(
            name,
            help="print correctness-only server argv"
            if name == "server"
            else "capture true token IDs and check controls",
        )
        cmd.add_argument("directory", type=Path)
        cmd.add_argument("--role", choices=["reference", "candidate"], required=True)
        cmd.add_argument("--spec-sha256", required=True)
        cmd.add_argument(
            "--plan-sha256", help="trusted digest saved after the reference (required for B)"
        )
        cmd.add_argument("--port", type=int, default=8000)
        if name == "server":
            cmd.add_argument("--engine-python", type=Path, required=True)
            cmd.add_argument(
                "--execute",
                action="store_true",
                help="actually run a dedicated engine on an authorized Linux host",
            )
        else:
            cmd.add_argument("--server-pid", type=int, required=True)
    parser.set_defaults(handler=_command)


def _command(args: argparse.Namespace) -> int:
    try:
        if args.operation == "prepare":
            sha = prepare(args.ab, args.suite, args.out)
            print(f"Prepared correctness spec (no model run): {sha}")
            return 0
        require(1 <= args.port <= 65535, "invalid port")
        root = args.directory
        spec = _read_spec(root, args.spec_sha256)
        if args.operation == "server":
            expected = _pin(spec, args.role)
            argv_ = server_argv(expected, args.port, args.engine_python)
            if not args.execute:
                print(
                    json.dumps(
                        {"argv": argv_, "env": {"VLLM_BATCH_INVARIANT": "1"}, "preview_only": True}
                    )
                )
                return 0
            require(sys.platform == "linux", "Linux is required for execution")
            _api_key()
            require(
                not os.environ.get("SLOGATE_TRACE_DIR") and not (root / args.role).exists(),
                "trace enabled or previous arm attempt exists",
            )
            if args.role == "candidate":
                _reference(root, spec, args.spec_sha256, args.plan_sha256)
            require(
                args.engine_python.is_file() and os.access(args.engine_python, os.X_OK),
                "engine interpreter unavailable",
            )
            logfile = root / f"{args.role}.server.log"
            build = _engine_build(argv_[0])
            with logfile.open("xb") as log:
                with logfile.with_suffix(".json").open("x", encoding="utf-8") as receipt:
                    json.dump(
                        {
                            "instance_sha256": process_instance(os.getpid()),
                            "pin_sha256": digest(expected),
                            "build": build,
                        },
                        receipt,
                    )
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
            os.execve(argv_[0], argv_, {**os.environ, "VLLM_BATCH_INVARIANT": "1"})
        key = _api_key()
        rc, receipt = collect(
            root,
            spec,
            args.spec_sha256,
            args.role,
            args.server_pid,
            args.port,
            key,
            args.plan_sha256,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return rc
    except ERRORS as exc:
        print(f"{args.operation}: {exc}", file=sys.stderr)
        return 4 if args.operation == "capture" else 1


def add_parser(commands: argparse._SubParsersAction) -> None:
    _configure(commands.add_parser("vllm-exact", help="prepare or collect native vLLM token IDs"))


def main(argv: list[str] | None = None) -> int:
    parser = UsageParser(description=__doc__)
    _configure(parser)
    return _command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
