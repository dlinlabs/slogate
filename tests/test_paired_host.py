from __future__ import annotations

import hashlib
import os
import runpy
import socket
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from slogate.paired_host import _listener_flags, verify_server
from slogate.paired_plan import digest


def test_collect_host_emits_fingerprint_not_raw_gpu_uuid(monkeypatch):
    collect = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/gpu/collect_host.py")
    )
    row = "GPU-unit-id,RTX 4090,595,24564,2600,10000,450,4,16"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=row))
    (device,) = collect["gpus"]()
    assert device["name"] == "RTX 4090"
    assert device["device_id"] == hashlib.sha256(b"GPU-unit-id").hexdigest()
    assert "GPU-unit-id" not in device.values()


@pytest.mark.parametrize(
    ("switches", "invariant", "accepted"),
    [
        ([], False, True),
        (["-I", "-B"], True, True),
        ([], True, False),
        (["-B"], True, False),
        (["-I", "-B"], False, False),
    ],
)
def test_listener_requires_the_launcher_for_this_mode(switches, invariant, accepted):
    argv = [
        "engine-python",
        *switches,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--port",
        "8000",
    ]
    if accepted:
        assert _listener_flags(argv, batch_invariant=invariant) == {"--port": ["8000"]}
    else:
        with pytest.raises(ValueError, match="module launcher"):
            _listener_flags(argv, batch_invariant=invariant)


@pytest.mark.skipif(sys.platform != "linux", reason="reference launcher uses Linux procfs")
@pytest.mark.parametrize("invariant", [False, True])
@pytest.mark.parametrize("wrong_env", [False, True])
def test_live_listener_argv_and_device(tmp_path, monkeypatch, invariant, wrong_env):
    # A test double owns a real loopback listener with the reference launcher's argv.
    python, lib = sys.executable, tmp_path
    if invariant:
        # Isolated Python deliberately cannot load the stub from cwd/PYTHONPATH.
        env = tmp_path / "engine"
        # Keep relocatable CPython's shared library next to the original executable.
        venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
        python = str(env / "bin/python")
        lib = Path(sysconfig.get_path("purelib", "venv", {"base": str(env), "platbase": str(env)}))
    package = lib / "vllm" / "entrypoints" / "openai"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").touch()
    (package / "api_server.py").write_text(
        "import sys\nprint('ready', flush=True)\nsys.stdin.read()\n", encoding="utf-8"
    )
    uuid = "GPU-test"
    device = {"device_id": hashlib.sha256(uuid.encode()).hexdigest(), "driver_version": "600"}
    expected = {"model": {"id": "m", "revision": "r"}, "flags": {"--enforce-eager": []}}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"{uuid}, 600\n"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(
            [
                python,
                *(["-I", "-B"] if invariant else []),
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                "m",
                "--revision",
                "r",
                "--tokenizer-revision",
                "r",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--enforce-eager",
            ],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            pass_fds=(sock.fileno(),),
            env={
                **os.environ,
                "PYTHONPATH": str(tmp_path),
                "SLOGATE_TRACE_DIR": "",
                "VLLM_BATCH_INVARIANT": "1" if invariant != wrong_env else "0",
            },
        )
        try:
            assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"

            def verify(pin=expected, gpu=device):
                return verify_server(proc.pid, url, pin, gpu, batch_invariant=invariant)

            if wrong_env:
                with pytest.raises(ValueError, match="batch-invariance mode"):
                    verify()
                return
            assert verify()["pin_sha256"] == digest(expected)
            with pytest.raises(ValueError, match="module launcher"):
                verify_server(proc.pid, url, expected, device, batch_invariant=not invariant)
            with pytest.raises(ValueError, match="launch options"):
                verify(dict(expected, flags={}))
            with pytest.raises(ValueError, match="GPU or driver"):
                verify(gpu=dict(device, driver_version="other"))
            with monkeypatch.context() as m:
                m.setattr(Path, "samefile", lambda *_: False)
                with pytest.raises(ValueError, match="network namespace"):
                    verify()
        finally:
            proc.communicate(timeout=5)
    assert not Path(f"/proc/{proc.pid}").exists()
