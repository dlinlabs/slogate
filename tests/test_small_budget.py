"""CPU-only: real frozen inputs, read-only previews, and fail-closed host/collector doubles."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest

from slogate import paired_plan, workload
from slogate.cli import build_parser, main
from slogate.paired_gate import evaluate
from slogate.paired_plan import digest, file_digest, load_plan
from slogate.report import RECORDS_FILE, RUN_FILE, write_json

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "workloads/azure-2024-05-12-smoke-30s.json.gz"


@pytest.fixture
def sb(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("prepare_small_budget")


@pytest.fixture
def small_input(tmp_path):
    # A clean git checkout need not contain the real, untracked trace. Never send a request.
    w = workload.Workload(
        "fixture",
        "Qwen/Qwen3-4B-Instruct-2507",
        30.0,
        tuple(workload.Request(str(i), i / 30, 1, 1, "x") for i in range(787)),
    )
    return workload.save(w, tmp_path / "input.json"), w.digest


@pytest.fixture
def trial(sb, small_input, tmp_path, monkeypatch):
    path, sha = small_input
    monkeypatch.setattr(sb, "WORK_DIGEST", sha)
    for module in (paired_plan, sb):
        monkeypatch.setattr(module, "version", lambda name: {"aiohttp": "3.14.5"}[name])
    root = tmp_path / "portable trial"
    return root, sb.prepare(root, path)["receipt_sha256"]


def command(trial, action="slot", slot="pair-001-A"):
    root, sha = trial
    return [action, str(root), "--receipt-sha256", sha, "--slot", slot]


@pytest.mark.skipif(not REAL.exists(), reason="untracked E0 workload not available")
def test_real_direct_script_prepares_a_relocatable_unmeasured_trial(sb, tmp_path):
    root = tmp_path / "new"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/prepare_small_budget.py"),
            "--out",
            str(root),
            "--workload",
            str(REAL),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if version("aiohttp") != "3.14.5":
        assert proc.returncode == 1 and "aiohttp==3.14.5" in proc.stderr and not root.exists()
        return
    assert proc.returncode == 0, proc.stderr
    pins = json.loads(proc.stdout)
    moved = root.rename(tmp_path / "moved trial")
    assert {p.relative_to(moved).as_posix() for p in moved.rglob("*") if p.is_file()} == {
        "receipt.json",
        "baseline.json",
        "candidate.json",
        "workload.json.gz",
        "ab/plan.json",
    }
    doc, (plan, sha) = sb.read(moved / "receipt.json"), load_plan(moved / "ab")
    assert pins == {"ab_plan_sha256": sha, "receipt_sha256": file_digest(moved / "receipt.json")}
    assert file_digest(moved / sb.WORK) == file_digest(REAL)
    assert doc["workload_sha256"] == plan["workload"]["digest"] == sb.WORK_DIGEST
    assert (plan["workload"]["requests"], plan["workload"]["duration_s"]) == (787, 30)
    assert plan["change"] == [sb.FLAG]
    for a, n in [("A", "2048"), ("B", "512")]:
        assert plan["arms"][a]["flags"][sb.FLAG] == [n]
    assert (plan["margin_ms"], plan["candidate_ceiling_ms"], plan["warmup_requests"]) == (
        2000,
        None,
        2,
    )
    assert plan["pairs"] == 6 and plan["block_pairs"] == 1
    assert plan["fidelity_policy"] == sb.FIDELITY
    assert evaluate(moved / "ab")["missing_slots"] == doc["slots"]
    assert evaluate(moved / "ab")["verdict"] == "inconclusive"
    assert sb.main(command((moved, pins["receipt_sha256"]))) == 0
    assert not (moved / "ab/operations").exists()


def test_rejects_a_lookalike_trace_and_unqualified_client(sb, tmp_path, small_input, monkeypatch):
    path, _ = small_input
    monkeypatch.setattr(sb, "version", lambda _: "3.14.5")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        sb.prepare(tmp_path / "absent", path)
    monkeypatch.setattr(sb, "version", lambda _: "wrong")
    with pytest.raises(ValueError, match=r"aiohttp==3\.14\.5"):
        sb.prepare(tmp_path / "absent", path)
    assert not (tmp_path / "absent").exists()


def test_no_overwriting_or_optional_weaker_trial(sb, trial, small_input):
    root, sha = trial
    with pytest.raises(FileExistsError):
        sb.prepare(root, small_input[0])
    with pytest.raises(SystemExit):
        sb.main(["--out", str(root), "--workload", str(small_input[0]), "--pairs", "2"])
    assert file_digest(root / "receipt.json") == sha
    cost = sb.read(root / "receipt.json")["cost_estimate_only"]
    assert cost["replay_gpu_min_if_b_like_e0"] == 30.6
    assert cost["startup_gpu_min_if_all_12_like_one_cold_e0"] == 42.4
    assert cost["fit_guaranteed"] is False and "separate correctness" in cost["excludes"]


@pytest.mark.parametrize(
    "rel", ["receipt.json", "ab/plan.json", "candidate.json", "workload.json.gz"]
)
def test_mutating_any_frozen_input_refuses_a_recipe_without_creating_a_slot(sb, trial, rel):
    root, _ = trial
    with (root / rel).open("ab") as f:
        f.write(b" ")
    assert sb.main(command(trial)) == 1
    assert not list((root / "ab").glob("pair-*")) and not (root / "ab/operations").exists()


@pytest.mark.parametrize(
    ("key", "changed"),
    [
        ("margin_ms", 10),
        ("candidate_ceiling_ms", 2000),
        ("warmup_requests", 1),
        ("layout", []),
        ("replay_config", {"ignore_eos": False}),
        ("fidelity_policy", {"max_launch_lag_s": 0.051}),
    ],
)
def test_even_a_resealed_generic_plan_must_satisfy_the_preregistration(sb, trial, key, changed):
    root, _ = trial
    p = root / "ab/plan.json"
    envelope = sb.read(p)
    envelope["plan"][key] = changed
    envelope["sha256"] = digest(envelope["plan"])
    write_json(p, envelope)
    r = root / "receipt.json"
    doc = sb.read(r)
    doc["file_sha256"]["ab/plan.json"] = file_digest(p)
    doc["ab_plan_sha256"] = envelope["sha256"]
    write_json(r, doc)
    assert sb.main(command((root, file_digest(r)))) == 1


@pytest.mark.parametrize("slot", ["pair-001-B", "pair-002-A", "../elsewhere", "A"])
def test_refuses_a_nonassigned_slot_including_a_later_same_arm(sb, trial, slot):
    assert sb.main(command(trial, slot=slot)) == 1


def test_offline_preview_is_deterministic_and_only_uses_the_arm_pin(sb, trial, capsys):
    root, _ = trial
    assert sb.main(command(trial)) == 0
    first = capsys.readouterr().out
    assert sb.main(command(trial)) == 0 and capsys.readouterr().out == first
    revision = sb.read(root / "baseline.json")["model"]["revision"]
    assert f"--revision {revision} --tokenizer-revision {revision}" in first
    assert "--host 127.0.0.1 --port 8000" in first and f"{sb.FLAG} 2048" in first
    assert "--api-key " not in first and "--attention-backend" not in first
    assert 'CFG="$TRIAL/baseline.json"' in first and "VLLM_BATCH_INVARIANT=0" in first
    assert str(root) not in first and "$!" in first and "pair-001-B" not in first
    assert not (root / "ab/operations").exists()
    (root / "ab/pair-999-Z").mkdir()
    assert sb.main(command(trial)) == 1


@pytest.fixture
def host(sb, monkeypatch):
    # No process inspection/launch. Parse forwarded argv with the *real* ab parser.
    state = {"stamp": "a" * 64, "pids": [], "listeners": "", "exit": 0, "commands": []}
    monkeypatch.setattr(sb, "client_stamp", lambda _: state["stamp"])
    monkeypatch.setattr(sb, "gpu_pids", lambda: state["pids"])
    monkeypatch.setenv("ENGINE_PYTHON", "test-engine")
    monkeypatch.setattr(
        sb, "query", lambda *args: "3.12" if args[0] == "test-engine" else state["listeners"]
    )
    monkeypatch.setattr(sb, "check_engine", lambda *_: None)

    def collect(argv, *, check):
        assert not check and argv[:3] == [sys.executable, "-m", "slogate"]
        assert argv[3:5] == ["ab", "record"]
        args = build_parser().parse_args(argv[3:])
        state["commands"].append(args)
        assert args.api_key_env == "VLLM_API_KEY" and args.server_pid == 1234
        assert args.base_url == "http://127.0.0.1:8000"
        assert args.workload == args.directory.parent / "workload.json.gz"
        slot = args.sut_info.parent.name
        assert args.sut_info == args.directory / "operations" / slot / "sut-info.json"
        assert slot.endswith(args.arm)
        if state["exit"] in (0, 2):
            d = args.directory / slot
            d.mkdir()
            # Test doubles for the *envelope*, deliberately not replay/latency samples.
            (d / RUN_FILE).write_bytes(b"test")
            (d / RECORDS_FILE).write_bytes(b"test records")
            write_json(
                d / "seal.json",
                {
                    "slot": slot,
                    "plan_sha256": load_plan(args.directory)[1],
                    "run_sha256": file_digest(d / RUN_FILE),
                    "records_sha256": file_digest(d / RECORDS_FILE),
                },
            )
        return SimpleNamespace(returncode=state["exit"])

    monkeypatch.setattr(subprocess, "run", collect)
    return state


def probed(sb, trial, slot, *, candidate="small-budget"):
    root, _ = trial
    assert sb.main(command(trial, "begin", slot), candidate=candidate) == 0
    op = root / "ab/operations" / slot
    cfg = sb.read(root / sb.CONFIG[slot[-1]])
    write_json(op / "engine.json", cfg["engine"]["expect"])
    write_json(
        op / "sut-info.json",
        {"config": cfg, "engine": cfg["engine"]["expect"], "worker_step_trace_requested": False},
    )
    return op


def take(sb, trial, slot):
    return sb.main([*command(trial, "record", slot), "--server-pid", "1234"])


def test_eager_protocol_cannot_mix_trials_or_skip_a_and_only_adds_eager_to_b(
    sb, host, trial, small_input, tmp_path, capsys
):
    eager_cli = importlib.import_module("prepare_eager").main
    root = tmp_path / "eager"
    assert eager_cli(["prepare", "--out", str(root), "--workload", str(small_input[0])]) == 0
    pins = json.loads(capsys.readouterr().out)
    eager = (root, pins["receipt_sha256"])
    doc = sb.read(root / "receipt.json")
    plan, sha = load_plan(root / "ab")
    assert doc["schema"] == "eager-performance-v1" and sha == pins["ab_plan_sha256"]
    a, b = (plan["arms"][arm]["flags"] for arm in "AB")
    assert a[sb.FLAG] == ["2048"] and b == {**a, "--enforce-eager": []}
    assert plan["change"] == ["--enforce-eager"]
    assert (plan["pairs"], plan["margin_ms"], plan["candidate_ceiling_ms"]) == (6, 2000, None)
    assert evaluate(root / "ab")["missing_slots"] == doc["slots"]
    assert sb.main(command(eager)) == 1 and eager_cli(command(trial)) == 1
    assert eager_cli(command(eager, slot="pair-001-B")) == 1
    assert not (root / "ab/operations").exists()
    assert eager_cli(command(eager)) == 0
    first = capsys.readouterr().out
    assert "scripts/prepare_eager.py begin" in first and "--enforce-eager" not in first
    probed(sb, eager, "pair-001-A", candidate="eager")
    assert eager_cli([*command(eager, "record"), "--server-pid", "1234"]) == 0
    assert eager_cli(command(eager, slot="pair-001-B")) == 0
    second = capsys.readouterr().out
    assert "scripts/prepare_eager.py record" in second and f"{sb.FLAG} 2048" in second
    assert "--enforce-eager" in second and 'CFG="$TRIAL/candidate.json"' in second


def test_one_slot_at_a_time_retain_even_a_candidate_failure(sb, host, trial, capsys):
    a, b = "pair-001-A", "pair-001-B"
    probed(sb, trial, a)
    assert sb.main(command(trial, "begin", a)) == 1
    assert sb.main(command(trial, slot=b)) == 1
    assert take(sb, trial, a) == 0
    assert take(sb, trial, a) == 1
    capsys.readouterr()
    assert sb.main(command(trial, slot=b)) == 0
    preview = capsys.readouterr().out
    assert f"{sb.FLAG} 512" in preview and 'CFG="$TRIAL/candidate.json"' in preview
    host["stamp"] = "b" * 64
    assert sb.main(command(trial, "begin", b)) == 1
    host["stamp"] = "a" * 64
    d = probed(sb, trial, b)
    host["exit"] = 2
    assert take(sb, trial, b) == 2 and sb.read(d / "result.json")["exit"] == 2
    assert sb.main(command(trial, slot="pair-002-A")) == 0
    assert sb.main(command(trial, slot=b)) == 1
    assert [cmd.arm for cmd in host["commands"]] == ["A", "B"]
    (trial[0] / "ab" / b / RECORDS_FILE).write_bytes(b"replaced")
    assert sb.main(command(trial, slot="pair-002-A")) == 1


@pytest.mark.parametrize("fault", ["wrong-manifest", "invalid-collector"])
def test_failed_attempt_is_not_a_new_slot(sb, trial, host, fault):
    slot = "pair-001-A"
    ab = trial[0] / "ab"
    op = probed(sb, trial, slot)
    if fault == "wrong-manifest":
        info = sb.read(op / "sut-info.json")
        info["config"] = sb.read(trial[0] / "candidate.json")
        write_json(op / "sut-info.json", info)
        assert take(sb, trial, slot) == 1 and not host["commands"]
        assert not (op / "result.json").exists()
    else:
        host["exit"] = 3
        assert take(sb, trial, slot) == 3 and sb.read(op / "result.json")["exit"] == 3
    assert (op / "record.started").exists()
    assert not (ab / slot).exists()
    assert main(["ab", "report", str(ab)]) == 3
    report = sb.read(ab / "gate.json")
    assert report["verdict"] == "invalid_measurement"
    assert report["missing_slots"] == sb.read(trial[0] / "receipt.json")["slots"][1:]
    assert any(slot in reason and "seal" in reason for reason in report["invalid_reasons"])
    for cmd in ("slot", "begin"):
        assert sb.main(command(trial, cmd)) == 1
    assert take(sb, trial, slot) == 1 and sb.main(command(trial, slot="pair-001-B")) == 1


def test_preview_is_unattempted_but_begin_reserves_a_slot(sb, trial, host):
    root, _ = trial
    ab = root / "ab"
    assert sb.main(command(trial)) == 0
    unattempted = evaluate(ab)
    assert unattempted["verdict"] == "inconclusive"
    assert unattempted["missing_slots"] == sb.read(root / "receipt.json")["slots"]

    (ab / "operations").mkdir()  # Only a slot's own operation directory reserves it.
    assert evaluate(ab) == unattempted
    assert sb.main(command(trial, "begin")) == 0 and not host["commands"]
    report = evaluate(ab)
    assert report["verdict"] == "invalid_measurement"
    assert report["missing_slots"] == unattempted["missing_slots"][1:]


@pytest.mark.parametrize(("key", "value"), [("pids", [17]), ("listeners", "LISTEN")])
def test_an_occupied_host_is_not_a_launch_permission(sb, trial, host, key, value):
    host[key] = value
    assert sb.main(command(trial, "begin")) == 1
    assert not (trial[0] / "ab/operations").exists() and not host["commands"]


@pytest.fixture
def env(sb, tmp_path):
    return {
        **sb.ENV,
        "ENGINE_PYTHON": str(tmp_path / "engine"),
        "CLIENT_PYTHON": str(tmp_path / "client"),
        "TRIAL": str(tmp_path),
        "HF_HOME": str(tmp_path / "cache"),
        sb.KEY: "private-test-key",
    }


@pytest.mark.parametrize(
    ("key", "v"),
    [
        ("VLLM_API_KEY", ""),
        ("SLOGATE_TRACE_DIR", "trace"),
        ("PYTHONTRACEMALLOC", "1"),
        ("PYTHON_GC_THRESHOLD0", "0"),
        ("VLLM_BATCH_INVARIANT", "1"),
        ("LD_PRELOAD", "lib.so"),
        ("HF_HUB_OFFLINE", "0"),
    ],
)
def test_no_profiler_gc_tuning_correctness_mode_or_auth_fallback(sb, env, key, v):
    with pytest.raises(ValueError):
        sb.environment({**env, key: v})


def test_pinned_cuda_build_tool_paths_are_part_of_the_environment_fingerprint(sb, env):
    paths = {
        "CUDA_HOME": "/opt/cuda13",
        "CUDA_PATH": "/opt/cuda13",
        "CPATH": "/opt/cuda13/include/cccl",
        "LIBRARY_PATH": "/opt/cuda13/lib64",
    }
    frozen = sb.environment({**env, **paths})
    assert frozen != sb.environment(env)
    for key in paths:
        assert sb.environment({**env, **paths, key: "/other"}) != frozen


def test_client_constraints_include_transitive_packages(sb, env, monkeypatch):
    monkeypatch.setattr(sb, "os", SimpleNamespace(environ=env))
    monkeypatch.setattr(
        sb,
        "sys",
        SimpleNamespace(
            platform="linux",
            version_info=(3, 12, 3),
            executable=env["CLIENT_PYTHON"],
            getprofile=lambda: None,
            gettrace=lambda: None,
        ),
    )
    monkeypatch.setattr(
        sb, "gc", SimpleNamespace(isenabled=lambda: True, get_threshold=lambda: (700, 10, 10))
    )
    pins = dict(
        s.split("==")
        for s in (ROOT / "configs/client/e0-py312-linux.txt").read_text().splitlines()
        if s and not s.startswith("#")
    )
    monkeypatch.setattr(sb, "version", pins.__getitem__)
    assert sb.client_stamp(Path(env["TRIAL"])) == sb.environment(env)
    pins["frozenlist"] = "not-E0"
    with pytest.raises(ValueError, match="constraint mismatch: frozenlist"):
        sb.client_stamp(Path(env["TRIAL"]))


def test_read_only_live_checks_reject_another_gpu_job_binary_or_key(sb, env, tmp_path, monkeypatch):
    # Construct Linux kernel records under the fixture directory; no live PID/GPU or sockets.
    proc = tmp_path / "proc"
    api, worker = proc / "100", proc / "101"
    worker.mkdir(parents=True)
    api.mkdir()
    exe = Path(env["ENGINE_PYTHON"])
    exe.write_bytes(b"test")
    (api / "exe").hardlink_to(exe)
    cfg = sb.read(sb.BASE)
    argv = [str(exe), *sb.engine_args(cfg)]
    (api / "cmdline").write_bytes(b"\0".join(map(os.fsencode, argv)) + b"\0")
    (api / "environ").write_bytes(b"\0".join(os.fsencode(f"{k}={v}") for k, v in env.items()))
    (worker / "stat").write_bytes(b"101 (test) child) S 100 0")
    monkeypatch.setattr(sb, "PROC", proc)
    monkeypatch.setenv("ENGINE_PYTHON", str(exe))
    monkeypatch.setenv(sb.KEY, env[sb.KEY])
    monkeypatch.setattr(sb, "gpu_pids", lambda: [101])
    stamp = sb.environment(env)
    sb.check_engine(100, cfg, stamp)
    with pytest.raises(ValueError, match="wrong live engine"):
        sb.check_engine(100, sb.candidates(cfg)["small-budget"], stamp)
    (worker / "stat").write_bytes(b"101 (other) S 1 0")
    with pytest.raises(ValueError, match="unrelated"):
        sb.check_engine(100, cfg, stamp)
    monkeypatch.setenv(sb.KEY, "different")
    with pytest.raises(ValueError, match="env/auth mismatch"):
        sb.check_engine(100, cfg, stamp)
