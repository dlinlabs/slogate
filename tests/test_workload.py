from __future__ import annotations

import copy
import json

import pytest

from slogate import workload
from slogate.workload import WorkloadError


def make(**overrides: object) -> workload.Workload:
    options: dict[str, object] = {
        "name": "unit",
        "duration_s": 5.0,
        "rate_rps": 4.0,
        "input_tokens": [32, 64],
        "max_tokens": 8,
        "seed": 7,
    }
    options.update(overrides)
    return workload.synthetic(**options)  # type: ignore[arg-type]


def test_synthetic_is_deterministic_and_valid() -> None:
    first, second = make(), make()
    assert first.digest == second.digest
    assert make(seed=8).digest != first.digest
    arrivals = [r.arrival_s for r in first.requests]
    assert arrivals == sorted(arrivals)
    assert all(0 <= a < first.duration_s for a in arrivals)
    assert {r.input_tokens for r in first.requests} <= {32, 64}
    assert all(len(r.prompt.split()) == r.input_tokens for r in first.requests)


def test_digest_survives_round_trip_in_both_formats(tmp_path) -> None:
    original = make()
    plain = workload.save(original, tmp_path / "w.json")
    packed = workload.save(original, tmp_path / "w.json.gz")
    assert workload.load(plain).digest == original.digest
    assert workload.load(packed).digest == original.digest


def test_gzip_output_is_byte_reproducible(tmp_path) -> None:
    original = make()
    first = workload.save(original, tmp_path / "a.json.gz").read_bytes()
    second = workload.save(original, tmp_path / "b.json.gz").read_bytes()
    assert first == second


def test_digest_ignores_key_order() -> None:
    original = make()
    shuffled = json.loads(json.dumps(original.to_dict(), sort_keys=False))
    shuffled = dict(reversed(list(shuffled.items())))
    assert workload.from_dict(shuffled).digest == original.digest


def _valid_document() -> dict:
    return make().to_dict()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(extra=1), "unknown top-level"),
        (lambda d: d["requests"][1].update(arrival_s=-0.1), "arrival_s"),
        (lambda d: d["requests"][0].update(arrival_s=99.0), "arrival_s"),
        (lambda d: d["requests"][1].update(id=d["requests"][0]["id"]), "duplicated"),
        (lambda d: d["requests"][0].update(input_tokens=0), "input_tokens"),
        (lambda d: d["requests"][0].update(max_tokens=True), "max_tokens"),
        (lambda d: d["requests"][0].update(output_tokens=5), "exactly the fields"),
        (lambda d: d.update(duration_s=float("nan")), "finite"),
        (lambda d: d.update(requests=[]), "requests"),
    ],
)
def test_validation_rejects_bad_documents(mutate, message) -> None:
    document = copy.deepcopy(_valid_document())
    mutate(document)
    with pytest.raises(WorkloadError, match=message):
        workload.from_dict(document)


def test_unsorted_arrivals_are_rejected() -> None:
    document = _valid_document()
    first, second = document["requests"][0], document["requests"][1]
    first["arrival_s"], second["arrival_s"] = second["arrival_s"], first["arrival_s"]
    with pytest.raises(WorkloadError, match="earlier than the previous"):
        workload.from_dict(document)


def test_lab_trace_import_keeps_provenance() -> None:
    lab = {
        "schema_version": 1,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "max_model_len": 8192,
        "duration_seconds": 30,
        "output_cap": 128,
        "provenance": {"source_sha256": "abc", "seed": 1},
        "requests": [
            {
                "trace_id": "t:1",
                "arrival_seconds": 0.0,
                "input_tokens": 100,
                "max_tokens": 128,
                "prompt": "a b c",
            },
            {
                "trace_id": "t:2",
                "arrival_seconds": 1.5,
                "input_tokens": 200,
                "max_tokens": 128,
                "prompt": "d e f",
            },
        ],
    }
    imported = workload.from_lab_trace(lab, name="azure-smoke")
    assert [r.id for r in imported.requests] == ["t:1", "t:2"]
    assert imported.duration_s == 30.0
    assert imported.provenance["source"]["license"] == "CC-BY-4.0"
    assert imported.provenance["lab_provenance"]["source_sha256"] == "abc"


def test_lab_trace_import_reports_missing_fields() -> None:
    with pytest.raises(WorkloadError, match="missing"):
        workload.from_lab_trace({"schema_version": 1, "requests": [{}]}, name="x")
