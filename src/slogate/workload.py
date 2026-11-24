"""Versioned replay workloads: validation, content digests and builders.

A workload is the frozen input of a gate run: arrival offsets, prompts and output caps.
It never carries historical output lengths, so a replay cannot leak an oracle into the
system under test.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_REQUESTS = 200_000
_REQUEST_KEYS = frozenset({"id", "arrival_s", "input_tokens", "max_tokens", "prompt"})
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "name", "model", "duration_s", "provenance", "requests"}
)
_FILLER_WORDS = ("alpha", "bravo", "delta", "echo", "golf", "hotel", "india", "kilo", "lima")

AZURE_2024_SOURCE = {
    "dataset": "Azure LLM Inference Dataset 2024",
    "url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
    "license": "CC-BY-4.0",
    "citation": (
        "Stojkovic et al., DynamoLLM: Designing LLM Inference Clusters for Performance and "
        "Energy Efficiency, HPCA 2025 (arXiv:2408.00741)"
    ),
}


class WorkloadError(ValueError):
    """Raised when a workload does not satisfy the schema."""


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    arrival_s: float
    input_tokens: int
    max_tokens: int
    prompt: str


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    model: str
    duration_s: float
    requests: tuple[Request, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "model": self.model,
            "duration_s": self.duration_s,
            "provenance": self.provenance,
            "requests": [
                {
                    "id": r.id,
                    "arrival_s": r.arrival_s,
                    "input_tokens": r.input_tokens,
                    "max_tokens": r.max_tokens,
                    "prompt": r.prompt,
                }
                for r in self.requests
            ],
        }

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical JSON encoding, independent of file formatting."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        inputs = sorted(r.input_tokens for r in self.requests)
        return {
            "name": self.name,
            "model": self.model,
            "digest": self.digest,
            "requests": len(self.requests),
            "duration_s": self.duration_s,
            "offered_rps": len(self.requests) / self.duration_s,
            "input_tokens": {
                "min": inputs[0],
                "p50": inputs[len(inputs) // 2],
                "max": inputs[-1],
            },
            "max_output_tokens_total": sum(r.max_tokens for r in self.requests),
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"{name} must be a number")
    if not math.isfinite(value):
        raise WorkloadError(f"{name} must be finite")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkloadError(f"{name} must be a positive integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadError(f"{name} must be a non-empty string")
    return value


def from_dict(data: Any) -> Workload:
    """Validate a decoded workload document and return an immutable workload."""
    if not isinstance(data, dict):
        raise WorkloadError("workload must be a JSON object")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise WorkloadError(f"unknown top-level fields: {sorted(unknown)}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise WorkloadError(f"unsupported schema_version {data.get('schema_version')!r}")
    name = _text(data.get("name"), "name")
    model = _text(data.get("model"), "model")
    duration = _finite(data.get("duration_s"), "duration_s")
    if duration <= 0:
        raise WorkloadError("duration_s must be positive")
    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise WorkloadError("provenance must be an object")
    raw_requests = data.get("requests")
    if not isinstance(raw_requests, list) or not 0 < len(raw_requests) <= MAX_REQUESTS:
        raise WorkloadError(f"requests must be a list of 1..{MAX_REQUESTS} entries")

    seen: set[str] = set()
    previous = 0.0
    requests = []
    for index, raw in enumerate(raw_requests):
        where = f"requests[{index}]"
        if not isinstance(raw, dict) or set(raw) != _REQUEST_KEYS:
            raise WorkloadError(f"{where} must have exactly the fields {sorted(_REQUEST_KEYS)}")
        identifier = _text(raw["id"], f"{where}.id")
        if identifier in seen:
            raise WorkloadError(f"{where}.id {identifier!r} is duplicated")
        seen.add(identifier)
        arrival = _finite(raw["arrival_s"], f"{where}.arrival_s")
        if not 0 <= arrival < duration:
            raise WorkloadError(f"{where}.arrival_s must lie in [0, duration_s)")
        if arrival < previous:
            raise WorkloadError(f"{where}.arrival_s is earlier than the previous request")
        previous = arrival
        requests.append(
            Request(
                id=identifier,
                arrival_s=arrival,
                input_tokens=_positive_int(raw["input_tokens"], f"{where}.input_tokens"),
                max_tokens=_positive_int(raw["max_tokens"], f"{where}.max_tokens"),
                prompt=_text(raw["prompt"], f"{where}.prompt"),
            )
        )
    return Workload(name, model, duration, tuple(requests), provenance)


def load(path: str | Path) -> Workload:
    """Load a workload from ``.json`` or gzip-compressed ``.json.gz``."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkloadError(f"{path}: invalid JSON: {exc}") from exc
    return from_dict(data)


def save(workload: Workload, path: str | Path) -> Path:
    """Write atomically; gzip output is byte-for-byte reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        workload.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def synthetic(
    *,
    name: str,
    duration_s: float,
    rate_rps: float,
    input_tokens: Sequence[int],
    max_tokens: int,
    seed: int,
    model: str = "fake-model",
) -> Workload:
    """Poisson arrivals with prompt sizes drawn from ``input_tokens``.

    Prompts are filler words whose count equals the requested input size. Real tokenizers
    will not map them one-to-one; synthetic workloads are for the fake engine and smoke tests.
    """
    if rate_rps <= 0 or duration_s <= 0:
        raise WorkloadError("rate_rps and duration_s must be positive")
    if not input_tokens:
        raise WorkloadError("input_tokens must not be empty")
    sizes = [_positive_int(n, "input_tokens") for n in input_tokens]
    rng = random.Random(seed)
    arrivals = []
    now = 0.0
    while True:
        now += rng.expovariate(rate_rps)
        if now >= duration_s:
            break
        arrivals.append(round(now, 6))
    if not arrivals:
        raise WorkloadError("no arrivals were generated; increase rate_rps or duration_s")
    requests = []
    for index, arrival in enumerate(arrivals):
        size = rng.choice(sizes)
        words = [_FILLER_WORDS[rng.randrange(len(_FILLER_WORDS))] for _ in range(size)]
        requests.append(
            {
                "id": f"{name}-{index:06d}",
                "arrival_s": arrival,
                "input_tokens": size,
                "max_tokens": max_tokens,
                "prompt": " ".join(words),
            }
        )
    return from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "model": model,
            "duration_s": float(duration_s),
            "provenance": {
                "kind": "synthetic",
                "generator": "slogate.workload.synthetic",
                "rate_rps": rate_rps,
                "input_tokens": sizes,
                "seed": seed,
            },
            "requests": requests,
        }
    )


def from_lab_trace(data: Any, *, name: str) -> Workload:
    """Convert a vllm-serving-lab trace (schema 1) into a slogate workload.

    Lab traces hold Azure LLM Inference 2024 arrivals and input sizes with synthetic prompts
    that were verified against the model's chat template, so prompt sizes are exact.
    """
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise WorkloadError("expected a vllm-serving-lab trace with schema_version 1")
    try:
        requests = [
            {
                "id": row["trace_id"],
                "arrival_s": row["arrival_seconds"],
                "input_tokens": row["input_tokens"],
                "max_tokens": row["max_tokens"],
                "prompt": row["prompt"],
            }
            for row in data["requests"]
        ]
        duration = data["duration_seconds"]
        model = data["model"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(f"lab trace is missing a required field: {exc}") from exc
    provenance = {
        "kind": "lab-trace",
        "source": AZURE_2024_SOURCE,
        "lab_provenance": data.get("provenance", {}),
        "max_model_len": data.get("max_model_len"),
        "output_cap": data.get("output_cap"),
    }
    return from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "model": model,
            "duration_s": float(duration),
            "provenance": provenance,
            "requests": requests,
        }
    )
