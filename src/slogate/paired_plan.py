"""Freeze one same-host, configuration-only TTFT experiment before either arm runs."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .fidelity import FidelityPolicy
from .replay import ReplayConfig
from .report import write_json
from .workload import Workload, canonical_json


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_url(url: str) -> None:
    parts = urlsplit(url)
    require(
        parts.scheme in {"http", "https"}
        and parts.hostname in {"127.0.0.1", "::1"}
        and parts.username is None
        and parts.password is None
        and not parts.query
        and not parts.fragment,
        "record on the SUT host using its literal loopback URL, without credentials or query",
    )


def serve_flags(argv: list[str]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {}
    key = ""
    for arg in argv:
        if arg.startswith("--"):
            key, sep, value = arg.partition("=")
            require(key not in flags, f"duplicate serve flag: {key}")
            flags[key] = [value] if sep else []
        else:
            require(bool(key), "expected a long serve flag first")
            flags[key].append(arg)
    return flags


def pin(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    if kind == "fake":
        require(
            isinstance(data["version"], str) and data["version"].startswith("slogate-fake-engine "),
            "expected fake /version identity",
        )
        require(
            isinstance(data["profile"]["model"], str) and bool(data["profile"]["model"]),
            "fake model missing",
        )
        return {"version": data["version"], "profile": data["profile"]}
    require(kind == "vllm" and data["engine"]["name"] == "vllm", "expected pinned vLLM config")
    for field in ("vllm", "torch", "cuda_runtime"):
        require(bool(data["engine"]["expect"][field]), f"missing engine pin: {field}")
    require(bool(data["model"]["id"]) and bool(data["model"]["revision"]), "model/revision missing")
    require(
        data["hardware"]["gpu_count"] == 1 and bool(data["hardware"]["gpu_name_contains"]),
        "single-GPU SUT required",
    )
    args = data["serve_args"]
    require(isinstance(args, list) and all(isinstance(x, str) for x in args), "invalid serve_args")
    flags = serve_flags(args)
    require(
        not flags.keys() & {"--host", "--port", "--model", "--revision", "--tokenizer-revision"},
        "model, endpoint and revisions must not be overridden in serve_args",
    )
    return {
        "engine": data["engine"],
        "model": data["model"],
        "hardware": data["hardware"],
        "flags": flags,
    }


def freeze(
    out: Path,
    workload: Workload,
    *,
    kind: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    pairs: int,
    block_pairs: int,
    margin_ms: float,
    ceiling_ms: float | None,
    warmup_requests: int,
) -> str:
    require(type(pairs) is int and 1 <= pairs <= 128, "pairs must be 1..128")
    require(
        type(block_pairs) is int and 1 <= block_pairs <= pairs and pairs % block_pairs == 0,
        "block_pairs must divide the preassigned pairs",
    )
    require(number(margin_ms) and margin_ms > 0, "explicit positive margin in ms required")
    require(
        ceiling_ms is None or (number(ceiling_ms) and ceiling_ms > 0),
        "invalid optional candidate p99 ceiling",
    )
    require(
        type(warmup_requests) is int and 0 <= warmup_requests <= len(workload.requests),
        "invalid warmup count",
    )
    a, b = pin(kind, baseline), pin(kind, candidate)
    settings_key = "profile" if kind == "fake" else "flags"
    require(
        {k: v for k, v in a.items() if k != settings_key}
        == {k: v for k, v in b.items() if k != settings_key},
        "model, hardware and engine must stay fixed",
    )
    for side in (a, b):
        model = side["profile"]["model"] if kind == "fake" else side["model"]["id"]
        require(model == workload.model, "workload and pinned model disagree")
    sa, sb = a[settings_key], b[settings_key]
    changed = sorted(
        k for k in sa.keys() | sb.keys() if k not in sa or k not in sb or sa[k] != sb[k]
    )
    require(len(changed) <= 1, "only one configuration change (or none for A/A) is supported")
    replay = ReplayConfig(base_url="http://127.0.0.1")
    plan = {
        "schema": 1,
        "id": uuid.uuid4().hex,
        "frozen_unix_s": time.time(),
        "kind": kind,
        "metric": "ttft_s.p99",
        "method": "paired-block-median-sign-interval-95-v1",
        "order": "AB",
        "pairs": pairs,
        "block_pairs": block_pairs,
        "margin_ms": margin_ms,
        "candidate_ceiling_ms": ceiling_ms,
        "warmup_requests": warmup_requests,
        "fidelity_policy": FidelityPolicy().to_dict(),
        "replay_config": {
            "model": workload.model,
            "ignore_eos": True,
            "warmup_requests": warmup_requests,
            "request_timeout_s": replay.request_timeout_s,
            "drain_timeout_s": replay.drain_timeout_s,
            "max_in_flight": replay.max_in_flight,
            "transport": {"name": "aiohttp", "version": version("aiohttp")},
        },
        "change": changed,
        "arms": {"A": a, "B": b},
        "workload": workload.summary(),
        "layout": [[r.id, r.arrival_s, r.max_tokens] for r in workload.requests],
    }
    hash_ = digest(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", {"sha256": hash_, "plan": plan})
    return hash_


def load_plan(root: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads((root / "plan.json").read_bytes())
    plan = envelope["plan"]
    hash_ = digest(plan)
    require(
        hash_ == envelope["sha256"]
        and plan["schema"] == 1
        and plan["order"] == "AB"
        and plan["metric"] == "ttft_s.p99"
        and plan["method"] == "paired-block-median-sign-interval-95-v1",
        "frozen plan digest/schema/order mismatch",
    )
    return plan, hash_


def slots(plan: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"pair-{i:03d}-{arm}", arm) for i in range(1, plan["pairs"] + 1) for arm in ("A", "B")]
