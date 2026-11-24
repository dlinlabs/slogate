from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from slogate import workload
from slogate.cli import main
from slogate.fidelity import FidelityPolicy, assess
from slogate.metrics import summarize
from slogate.paired_gate import evaluate, record
from slogate.paired_plan import digest, file_digest, freeze, load_plan, local_url, pin, slots
from slogate.replay import ReplayResult, RequestRecord
from slogate.report import RECORDS_FILE, RUN_FILE, read_run, redact_url, write_json, write_run


def load():
    return workload.Workload("unit", "fake", 1.0, (workload.Request("q", 0.0, 1, 2, "hi"),))


def identity(ttft: float):
    return {"version": "slogate-fake-engine test", "profile": {"model": "fake", "ttft_s": ttft}}


def experiment(path, **overrides):
    opts = dict(
        kind="fake",
        baseline=identity(0.02),
        candidate=identity(0.08),
        pairs=6,
        block_pairs=1,
        margin_ms=10,
        ceiling_ms=None,
        warmup_requests=1,
    )
    freeze(path, load(), **(opts | overrides))
    return path


def seal(root, name, plan_hash):
    d = root / name
    write_json(
        d / "seal.json",
        {
            "slot": name,
            "plan_sha256": plan_hash,
            "run_sha256": file_digest(d / RUN_FILE),
            "records_sha256": file_digest(d / RECORDS_FILE),
        },
    )


def populate(root, differences, *, bad_fidelity=None, failure=None):
    plan, hash_ = load_plan(root)
    w = load()
    for i, (name, arm) in enumerate(slots(plan)[: 2 * len(differences)]):
        ttft = 0.020 if arm == "A" else 0.020 + differences[i // 2] / 1000
        err = "HTTP503" if failure == name else None
        rec = RequestRecord(
            "q",
            0,
            launched_s=0.001,
            headers_s=0.002,
            first_token_s=None if err else 0.001 + ttft,
            finished_s=0.5,
            status=503 if err else 200,
            output_tokens=None if err else 2,
            chunk_times_s=[] if err else [0.001 + ttft, 0.49],
            chunk_tokens=[] if err else [1, 1],
            success=err is None,
            error=err,
        )
        rr = ReplayResult(
            w.name,
            w.digest,
            w.model,
            w.duration_s,
            plan["frozen_unix_s"] + 1 + 2 * i,
            1.001,
            [rec],
            False,
            {
                "event_loop_lag_s": {"n": 50, "max": 0.06 if name == bad_fidelity else 0.002},
                "gc": {"collections": 0, "dropped": 0, "pause_max": None},
            },
        )
        meta = {
            "label": f"{hash_}/{name}",
            "base_url": "http://127.0.0.1:8000",
            "workload": w.summary(),
            "sut_info": None,
            "server": {"models": {"data": [{"id": "fake"}]}, "version": plan["arms"][arm]},
            "replay_config": plan["replay_config"],
            "fidelity_policy": FidelityPolicy().to_dict(),
            "slo": {"ttft_s": None, "tpot_s": None},
        }
        write_run(root / name, rr, metrics=summarize(rr), fidelity=assess(rr).to_dict(), meta=meta)
        seal(root, name, hash_)


@pytest.mark.parametrize(
    ("delta", "status"), [([1] * 6, "pass"), ([30] * 6, "fail"), ([0, 30] * 3, "inconclusive")]
)
def test_frozen_paired_controls_and_recompute(tmp_path, delta, status):
    root = experiment(tmp_path / "experiment")
    populate(root, delta)
    r = evaluate(root)
    assert r["verdict"] == status and r["paired_deltas_ms"] == pytest.approx(delta)
    assert r["scope"] == "performance_only" and r["correctness"] == "not_evaluated_here"
    assert r["candidate_absolute_slo"]["status"] == "not_configured"
    code = main(["ab", "report", str(root)])
    assert code == {"pass": 0, "fail": 2, "inconclusive": 4}[status]
    assert json.loads((root / "gate.json").read_bytes()) == r
    assert status in (root / "gate.md").read_text(encoding="utf-8")


def test_no_retroactive_pairing_optional_stopping_or_flag_changes(tmp_path):
    root = experiment(tmp_path / "e")
    with pytest.raises(FileExistsError):
        experiment(root)
    populate(root, [3] * 5)
    r = evaluate(root)
    assert r["verdict"] == "inconclusive" and r["relative"]["verdict"] == "not_evaluated"
    assert r["missing_slots"] == ["pair-006-A", "pair-006-B"]
    envelope = json.loads((root / "plan.json").read_bytes())
    envelope["plan"]["margin_ms"] = 500
    write_json(root / "plan.json", envelope)
    assert main(["ab", "report", str(root)]) == 1


@pytest.mark.parametrize(
    ("failure", "bad", "expect"),
    [
        ("pair-003-B", None, "fail"),
        ("pair-003-A", None, "invalid_measurement"),
        (None, "pair-002-B", "invalid_measurement"),
    ],
)
def test_failed_requests_and_fidelity_are_not_omitted(tmp_path, failure, bad, expect):
    root = experiment(tmp_path / "e")
    populate(root, [1] * 6, bad_fidelity=bad, failure=failure)
    r = evaluate(root)
    assert r["verdict"] == expect and r["relative"]["verdict"] == "not_evaluated"
    assert len(r["runs"]) == 12
    if failure == "pair-003-B":
        assert r["candidate_failed_requests"] == 1 and not r["paired_deltas_ms"]
    else:
        assert r["invalid_reasons"]


def test_absolute_ceiling_is_independent_of_relative_effect(tmp_path):
    root = experiment(tmp_path / "e", ceiling_ms=15)
    populate(root, [0] * 6)
    r = evaluate(root)
    assert r["relative"]["verdict"] == "pass"
    assert r["candidate_absolute_slo"]["status"] == "fail" and r["verdict"] == "fail"


def test_extra_unplanned_slots_are_not_silently_ignored(tmp_path):
    root = experiment(tmp_path / "e", pairs=1)
    populate(root, [1])
    (root / "pair-002-A").mkdir()
    result = evaluate(root)
    assert result["verdict"] == "invalid_measurement"
    assert result["invalid_reasons"] == ["unplanned slots present: pair-002-A"]


def test_cli_records_and_seals_real_loopback_streams(tmp_path, fake_engine):
    from slogate import __version__
    from slogate.fake_engine import EngineProfile

    w = workload.save(load(), tmp_path / "load.json")
    profile = EngineProfile(model="fake", ttft_s=0.02, jitter=0)
    url = fake_engine(**profile.to_dict())
    expected = tmp_path / "pin.json"
    write_json(
        expected, {"version": f"slogate-fake-engine {__version__}", "profile": profile.to_dict()}
    )
    root = tmp_path / "ab"
    assert (
        main(
            [
                "ab",
                "init",
                "--out",
                str(root),
                "--kind",
                "fake",
                "--baseline",
                str(expected),
                "--candidate",
                str(expected),
                "--workload",
                str(w),
                "--margin-ms",
                "10",
                "--pairs",
                "1",
                "--warmup-requests",
                "1",
            ]
        )
        == 0
    )
    for arm in ("A", "B"):
        rc = main(
            ["ab", "record", str(root), "--arm", arm, "--base-url", url, "--workload", str(w)]
        )
        run_dir = root / f"pair-001-{arm}"
        doc = read_run(run_dir)
        assert rc == (0 if doc["fidelity"]["valid"] else 3)
        assert doc["metrics"]["requests"]["succeeded"] == 1
        assert doc["meta"]["label"] == f"{load_plan(root)[1]}/{run_dir.name}"
        seal_doc = json.loads((run_dir / "seal.json").read_bytes())
        assert seal_doc["records_sha256"] == file_digest(run_dir / RECORDS_FILE)


@pytest.mark.parametrize(
    "flaw",
    [
        "tamper",
        "host",
        "profile",
        "clock",
        "workload",
        "summary",
        "diagnostics",
        "limit",
        "record_artifact",
    ],
)
def test_invalid_or_noncomparable_evidence_cannot_pass(tmp_path, flaw):
    root = experiment(tmp_path / "e", pairs=2)
    populate(root, [1, 1])
    name = "pair-002-B"
    d = root / name
    doc = read_run(d)
    if flaw == "record_artifact":
        (d / RECORDS_FILE).unlink()
    else:
        if flaw == "tamper":
            doc["meta"]["label"] = "different experiment"
        if flaw == "host":
            doc["client"]["host_id"] = "another host"
        if flaw == "profile":
            doc["meta"]["server"]["version"] = identity(0.99)
        if flaw == "clock":
            doc["replay"]["started_unix_s"] -= 2
        if flaw == "workload":
            doc["meta"]["workload"]["digest"] = "0" * 64
        if flaw == "summary":
            doc["metrics"]["ttft_s"]["p99"] = 0.01
        if flaw == "diagnostics":
            del doc["replay"]["diagnostics"]["event_loop_lag_s"]
        if flaw == "limit":
            doc["meta"]["fidelity_policy"]["max_launch_lag_s"] = 0.99
        write_json(d / RUN_FILE, doc)
        if flaw != "tamper":
            seal(root, name, load_plan(root)[1])  # Producer error, not a broken hash.
    assert evaluate(root)["verdict"] == "invalid_measurement"
    assert main(["ab", "report", str(root)]) == 3


def test_record_refuses_wrong_arm_or_unfinished_slot_without_contacting_server(tmp_path):
    root = experiment(tmp_path / "e")
    w = workload.save(load(), tmp_path / "work.json")
    with pytest.raises(ValueError, match="requires arm A"):
        record(root, "B", "http://127.0.0.1", w, None, None)
    (root / "pair-001-A").mkdir()
    with pytest.raises(ValueError, match="unfinished"):
        record(root, "B", "http://127.0.0.1", w, None, None)
    assert evaluate(root)["verdict"] == "invalid_measurement"
    with pytest.raises(ValueError, match="literal loopback"):
        local_url("https://outside.invalid/v1")
    ipv6 = redact_url("http://[::1]:8000")
    assert ipv6 == "http://[::1]:8000"
    local_url(ipv6)


def test_vllm_plan_only_allows_one_same_model_configuration_change(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "sut"
        / "vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json"
    )
    base = json.loads(path.read_bytes())
    w = workload.Workload("w", base["model"]["id"], 1, load().requests)
    opts = dict(
        kind="vllm",
        baseline=base,
        pairs=6,
        block_pairs=1,
        margin_ms=10,
        ceiling_ms=None,
        warmup_requests=1,
    )
    candidate = dict(base, serve_args=base["serve_args"] + ["--enforce-eager"])
    freeze(tmp_path / "one", w, candidate=candidate, **opts)
    assert load_plan(tmp_path / "one")[0]["change"] == ["--enforce-eager"]
    candidate["serve_args"] = candidate["serve_args"] + ["--other=1"]
    with pytest.raises(ValueError, match="one configuration"):
        freeze(tmp_path / "two", w, candidate=candidate, **opts)
    with pytest.raises(ValueError, match="duplicate"):
        pin("vllm", dict(base, serve_args=["--foo=1", "--foo", "2"]))


def test_gpu_trial_recipes_are_one_change_from_the_pinned_baseline():
    script = Path(__file__).resolve().parents[1] / "scripts" / "week3_candidates.py"
    recipe = runpy.run_path(str(script))
    base = json.loads(recipe["BASE"].read_bytes())
    original = pin("vllm", base)
    variants = recipe["candidates"](base)
    assert set(variants) == {"eager", "small-graphs", "small-budget"}
    checks = {
        "eager": ("--enforce-eager", []),
        "small-graphs": ("--compilation-config", ['{"cudagraph_capture_sizes":[1,2,4,8]}']),
        "small-budget": ("--max-num-batched-tokens", ["512"]),
    }
    for name, (changed, value) in checks.items():
        new = pin("vllm", variants[name])
        assert new == {**original, "flags": {**original["flags"], changed: value}}
    assert original["flags"]["--max-num-batched-tokens"] == ["2048"]
    assert pin("vllm", base) == original  # Materialization doesn't mutate the control.


@pytest.mark.parametrize(
    "flaw", [None, "different_gpu", "missing_device", "changed_server", "tracing"]
)
def test_offline_vllm_report_requires_attested_same_device(tmp_path, flaw):
    sut_dir = Path(__file__).resolve().parents[1] / "configs" / "sut"
    base = json.loads(next(sut_dir.glob("*.json")).read_bytes())
    base["model"]["id"] = load().model
    root = experiment(tmp_path / "e", kind="vllm", baseline=base, candidate=base)
    populate(root, [1] * 6)
    plan, hash_ = load_plan(root)
    for name, arm in slots(plan):
        bad = name == "pair-003-B"
        device = {
            "device_id": ("1" if bad and flaw == "different_gpu" else "a") * 64,
            "name": "RTX 4090",
            "driver_version": "unit-test",
        }
        doc = read_run(root / name)
        doc["meta"]["sut_info"] = {
            "config": base,
            "engine": base["engine"]["expect"],
            "host": {"gpus": [device]},
            "worker_step_trace_requested": bool(bad and flaw == "tracing"),
        }
        doc["meta"]["server"]["version"] = {"version": "0.30.0"}
        live = {
            "instance_sha256": "b" * 64,
            "device_id": device["device_id"],
            "pin_sha256": digest(plan["arms"][arm]),
        }
        if bad and flaw == "missing_device":
            del device["device_id"]
        write_json(root / name / RUN_FILE, doc)
        seal(root, name, hash_)
        p = root / name / "seal.json"
        evidence = json.loads(p.read_bytes())
        evidence["vllm_listener"] = {
            "before": live,
            "after": None if bad and flaw == "changed_server" else live,
        }
        write_json(p, evidence)
    assert evaluate(root)["verdict"] == ("pass" if flaw is None else "invalid_measurement")
