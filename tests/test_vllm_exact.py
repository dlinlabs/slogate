"""Integration of token collector, immutable reference and scorer, with no real SUT."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from slogate import vllm_exact as exact
from slogate.cli import main as cli
from slogate.native_tokens import MODEL
from slogate.paired_plan import digest, file_digest, freeze
from slogate.workload import Request, Workload

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json"
SUITE = ROOT / "configs/correctness/qwen3-small-budget.json"


def configured(tmp_path):
    a = json.loads(BASE.read_bytes())
    argv = list(a["serve_args"])
    argv[argv.index("--max-num-batched-tokens") + 1] = "512"
    w = Workload("tiny", MODEL, 1.0, (Request("x", 0.0, 1, 4, "hi"),))
    p = tmp_path / "ab"
    freeze(
        p,
        w,
        kind="vllm",
        baseline=a,
        candidate={**a, "serve_args": argv},
        pairs=6,
        block_pairs=1,
        margin_ms=2000,
        ceiling_ms=None,
        warmup_requests=1,
    )
    out = tmp_path / "exact"
    assert (
        cli(["vllm-exact", "prepare", "--ab", str(p), "--suite", str(SUITE), "--out", str(out)])
        == 0
    )
    return out, file_digest(out / "spec.json")


def wire(monkeypatch, state):
    real = httpx.Client

    def responder(req):
        assert req.url.host == "127.0.0.1" and req.headers["authorization"].endswith("unit-test")
        state["calls"] += 1
        if req.url.path == "/version":
            return httpx.Response(200, json={"version": state.get("version", "0.30.0")})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL, "max_model_len": 8192}]})
        assert req.url.path == "/v1/completions"
        b = json.loads(req.content)
        choices = [
            {
                "index": i,
                "text": "never used in evidence",
                "prompt_token_ids": p,
                "token_ids": [state.get("out", 11), 151645],
                "finish_reason": "stop",
            }
            for i, p in enumerate(b["prompt"])
        ]
        if state.get("missing"):
            del choices[0]["token_ids"]
        n = sum(len(p) for p in b["prompt"])
        total = 2 * len(choices)
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "object": "text_completion",
                "choices": choices,
                "usage": {
                    "prompt_tokens": n,
                    "completion_tokens": total,
                    "total_tokens": n + total,
                },
            },
        )

    def make(**kw):
        assert kw["trust_env"] is False and kw["follow_redirects"] is False
        return real(**kw, transport=httpx.MockTransport(responder))

    monkeypatch.setattr(exact.httpx, "Client", make)

    def stamp(pid, port, pin, log):
        assert pid == 54321 and port == 8000 and log.name.endswith(".server.log")
        state["stamps"] += 1
        if state.get("restart") and state["stamps"] == 2:
            raise ValueError("changed server")
        return {
            "instance_sha256": digest(pid),
            "pin_sha256": digest(pin),
            "hardware": state.get("hw", "a" * 64),
        }

    monkeypatch.setattr(exact, "_runtime_stamp", stamp)
    monkeypatch.setenv("VLLM_API_KEY", "unit-test")


def take(root, sha, role, pin=None):
    cmd = ["capture", str(root), "--spec-sha256", sha, "--role", role, "--server-pid", "54321"]
    return exact.main(cmd if pin is None else [*cmd, "--plan-sha256", pin])


@pytest.mark.parametrize("delta, expected", [(0, 0), (1, 2)])
def test_collected_native_outputs_feed_the_offline_gate(tmp_path, monkeypatch, delta, expected):
    root, sha = configured(tmp_path)
    state = {"stamps": 0, "calls": 0}
    wire(monkeypatch, state)
    assert take(root, sha, "candidate") == 4 and state["calls"] == 0
    assert take(root, sha, "reference") == 0 and state["stamps"] == 2
    ref = root / "reference"
    assert json.loads((ref / "singleton-ids.json").read_bytes()) == json.loads(
        (ref / "mixed-batch-ids.json").read_bytes()
    )
    plan_sha = file_digest(ref / "plan.json")
    assert take(root, sha, "reference") == 4  # No overwrite or repeat after observing outputs.
    assert take(root, sha, "candidate", "b" * 64) == 4 and not (root / "candidate").exists()
    state["out"] = 11 + delta
    assert take(root, sha, "candidate", plan_sha) == expected
    b = json.loads((root / "candidate" / "report.json").read_bytes())
    assert b["status"] == ("pass" if delta == 0 else "fail")
    if delta:
        assert b["cases"][0]["first_difference"] == {
            "index": 0,
            "reference_token_id": 11,
            "candidate_token_id": 12,
        }
    assert (root / "candidate" / "receipt.json").is_file() and state["stamps"] == 4
    assert "unit-test" not in (ref / "receipt.json").read_text()


@pytest.mark.parametrize(
    "fault", ["version", "missing", "restart", "wrong-hardware", "tamper-ref", "tamper-spec"]
)
def test_failed_or_incomplete_attempt_cannot_greenlight_a_candidate(tmp_path, monkeypatch, fault):
    root, sha = configured(tmp_path)
    state = {"stamps": 0, "calls": 0}
    wire(monkeypatch, state)
    if fault in ("wrong-hardware", "tamper-ref"):
        assert take(root, sha, "reference") == 0
        plan_sha = file_digest(root / "reference" / "plan.json")
        if fault == "wrong-hardware":
            state["hw"] = "other"
        else:
            (root / "reference" / "evidence.json").write_bytes(b"{}")
        assert (
            take(root, sha, "candidate", plan_sha) == 4
            and not (root / "candidate" / "evidence.json").exists()
        )
    else:
        if fault == "version":
            state["version"] = "other"
        elif fault == "tamper-spec":
            (root / "spec.json").write_bytes(b"{}")
        else:
            state[fault] = True
        assert take(root, sha, "reference") == 4 and not (root / "reference" / "plan.json").exists()
        if fault != "tamper-spec":
            assert (
                json.loads((root / "reference" / "receipt.json").read_bytes())["status"]
                == "unverified"
            )
        else:
            assert state["calls"] == 0


def test_server_preview_never_invokes_model_or_needs_credentials(tmp_path, monkeypatch, capsys):
    root, sha = configured(tmp_path)
    capsys.readouterr()
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr(exact.os, "execve", lambda *_: pytest.fail("preview started an engine"))
    for role, budget in (("reference", "2048"), ("candidate", "512")):
        cmd = [
            "server",
            str(root),
            "--spec-sha256",
            sha,
            "--role",
            role,
            "--engine-python",
            sys.executable,
        ]
        assert exact.main(cmd) == 0
        manifest = json.loads(capsys.readouterr().out)
        assert manifest["preview_only"] and manifest["env"] == {"VLLM_BATCH_INVARIANT": "1"}
        argv = manifest["argv"]
        assert argv[argv.index("--max-num-batched-tokens") + 1] == budget
        assert argv[argv.index("--generation-config") + 1] == "vllm"
        assert argv[argv.index("--attention-backend") + 1] == "FLASH_ATTN"
        assert not (root / f"{role}.server.log").exists()
        assert exact.main([*cmd, "--execute"]) == 1 and not (root / f"{role}.server.log").exists()


def test_engine_build_probe_isolated_from_client_and_checks_every_pin(monkeypatch):
    actual = {"python": "3.12.3", **exact.ENGINE}

    def run(args, **kwargs):
        assert args[:4] == ["/chosen/engine/python", "-I", "-B", "-c"]
        assert kwargs["check"] and kwargs["timeout"] > 0
        return SimpleNamespace(stdout=json.dumps(actual))

    monkeypatch.setattr(exact.subprocess, "run", run)
    assert exact._engine_build("/chosen/engine/python") == actual
    actual["torch"] = "other"
    with pytest.raises(ValueError, match="engine interpreter/build"):
        exact._engine_build("/chosen/engine/python")


def test_bad_key_or_transport_message_is_never_saved(tmp_path, monkeypatch, capsys):
    root, sha = configured(tmp_path)
    state = {"stamps": 0, "calls": 0}
    wire(monkeypatch, state)
    monkeypatch.setenv("VLLM_API_KEY", "test-secret\n")
    assert take(root, sha, "reference") == 4
    assert "test-secret" not in capsys.readouterr().err
    assert not (root / "reference").exists()
    monkeypatch.setenv("VLLM_API_KEY", "test-secret")

    def broken_client(**kwargs):
        raise httpx.LocalProtocolError("invalid header containing test-secret")

    monkeypatch.setattr(exact.httpx, "Client", broken_client)
    assert take(root, sha, "reference") == 4
    receipt = (root / "reference" / "receipt.json").read_text()
    assert "LocalProtocolError" in receipt and "test-secret" not in receipt
    assert not (root / "reference" / "evidence.json").exists()


def test_public_capture_cli_over_real_loopback_with_stubbed_gpu_identity(tmp_path, monkeypatch):
    """Only the HTTP round trip is real; outputs/identity are test doubles in tmp_path."""
    root, sha = configured(tmp_path)
    monkeypatch.setenv("VLLM_API_KEY", "loopback-test")
    monkeypatch.setattr(
        exact,
        "_runtime_stamp",
        lambda pid, port, pin, log: {
            "instance_sha256": digest(pid),
            "pin_sha256": digest(pin),
            "hardware": "a" * 64,
        },
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def answer(self, data):
            assert self.headers["Authorization"] == "Bearer loopback-test"
            b = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/version":
                self.answer({"version": "0.30.0"})
            else:
                assert self.path == "/v1/models"
                self.answer({"data": [{"id": MODEL, "max_model_len": 8192}]})

        def do_POST(self):
            assert self.path == "/v1/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["return_token_ids"] and body["add_special_tokens"] is False
            prompts = body["prompt"]
            n = sum(map(len, prompts))
            k = len(prompts)
            self.answer(
                {
                    "model": MODEL,
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": i,
                            "prompt_token_ids": p,
                            "token_ids": [151645],
                            "text": "",
                            "finish_reason": "stop",
                        }
                        for i, p in reversed(list(enumerate(prompts)))
                    ],
                    "usage": {"prompt_tokens": n, "completion_tokens": k, "total_tokens": n + k},
                }
            )

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            args = [
                "vllm-exact",
                "capture",
                str(root),
                "--spec-sha256",
                sha,
                "--server-pid",
                "4321",
                "--port",
                str(server.server_port),
            ]
            assert cli([*args, "--role", "reference"]) == 0
            plan_hash = file_digest(root / "reference" / "plan.json")
            assert cli([*args, "--role", "candidate", "--plan-sha256", plan_hash]) == 0
        finally:
            server.shutdown()
            worker.join(timeout=3)
    b = json.loads((root / "candidate" / "evidence.json").read_bytes())
    assert len(b["cases"]) == 4 and all(row["output_ids"] == [151645] for row in b["cases"])


@pytest.mark.skipif(sys.platform != "linux", reason="same live stdout inode requires Linux procfs")
def test_runtime_requires_resolved_startup_evidence_bound_to_stdout(tmp_path, monkeypatch):
    import subprocess

    log = tmp_path / "server.log"
    with (
        log.open("w") as handle,
        subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            stdout=handle,
        ) as child,
    ):
        try:
            monkeypatch.setattr(
                exact.subprocess,
                "run",
                lambda *a, **kw: SimpleNamespace(
                    stdout="GPU-test, NVIDIA GeForce RTX 4090, 595.1, 8.9\n"
                ),
            )

            def listener(pid, url, conf, device, *, batch_invariant=False):
                assert batch_invariant and child.pid == pid
                return {
                    "instance_sha256": "x",
                    "pin_sha256": digest(conf),
                    "device_id": device["device_id"],
                }

            monkeypatch.setattr(exact, "verify_server", listener)
            build = {
                "instance_sha256": "x",
                "pin_sha256": digest({}),
                "build": {"python": "3.12.3", **exact.ENGINE},
            }
            log.with_suffix(".json").write_text(json.dumps(build))
            with pytest.raises(ValueError, match="startup evidence"):
                exact._runtime_stamp(child.pid, 8000, {}, log)
            handle.write("kv_cache_dtype=torch.bfloat16\nUsing FLASH_ATTN backend.\n")
            handle.flush()
            with pytest.raises(ValueError, match="startup evidence"):
                exact._runtime_stamp(child.pid, 8000, {}, log)
            handle.write("dtype=torch.bfloat16\nUsing AttentionBackendEnum.FLASH_ATTN backend.\n")
            handle.flush()
            assert len(exact._runtime_stamp(child.pid, 8000, {}, log)["hardware"]) == 64
            other = tmp_path / "old.log"
            other.write_text("dtype=torch.bfloat16\nUsing FLASH_ATTN backend.\n")
            other.with_suffix(".json").write_text(json.dumps(build))
            with pytest.raises(ValueError, match="stdout"):
                exact._runtime_stamp(child.pid, 8000, {}, other)
            log.with_suffix(".json").write_text(json.dumps({**build, "instance_sha256": "stale"}))
            with pytest.raises(ValueError, match="build receipt"):
                exact._runtime_stamp(child.pid, 8000, {}, log)
        finally:
            child.communicate(timeout=5)
