"""Read-only attestation for the Linux, single-GPU, localhost vLLM reference launcher."""

from __future__ import annotations

import hashlib
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .paired_plan import digest, local_url, require, serve_flags


def listener_inodes(process: Path, address: str, port: int) -> set[str]:
    ipv6 = ":" in address
    packed = socket.inet_pton(socket.AF_INET6 if ipv6 else socket.AF_INET, address)
    words = struct.unpack("=4I" if ipv6 else "=I", packed)
    endpoint = "".join(f"{w:08X}" for w in words) + f":{port:04X}"
    sockets = set()
    for row in (process / "net" / ("tcp6" if ipv6 else "tcp")).read_text().splitlines()[1:]:
        fields = row.split()
        if len(fields) >= 10 and fields[1].upper() == endpoint and fields[3] == "0A":
            sockets.add(f"socket:[{fields[9]}]")
    return sockets


def verify_server(pid: int, url: str, expected: dict, declared_device: dict) -> dict[str, str]:
    local_url(url)
    require(
        sys.platform == "linux" and type(pid) is int and pid > 0,
        "vLLM needs its live Linux API server PID",
    )
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port
    if parts.scheme != "http" or host is None or port is None:
        raise ValueError("vLLM requires the explicit HTTP loopback port")
    proc = Path(f"/proc/{pid}")
    require(
        (proc / "ns/net").samefile("/proc/self/ns/net"),
        "API server and client must share the measured loopback network namespace",
    )
    # PID + kernel start tick detects replacement even if the numeric PID gets reused.
    started = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    listeners = listener_inodes(proc, host, port)
    found = False
    for fd in (proc / "fd").iterdir():
        try:
            if os.readlink(fd) in listeners:
                found = True
                break
        except FileNotFoundError:
            continue  # A transient request socket can close during enumeration.
    require(found, "declared process does not own the measured listener")
    argv = [os.fsdecode(v) for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
    require(
        len(argv) >= 3 and argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"],
        "unsupported vLLM argv: cannot verify the reference module launcher",
    )
    model, rev = expected["model"]["id"], expected["model"]["revision"]
    fixed = {
        "--model": [model],
        "--revision": [rev],
        "--tokenizer-revision": [rev],
        "--host": [host],
        "--port": [str(port)],
    }
    require(
        serve_flags(argv[3:]) == expected["flags"] | fixed,
        "live vLLM launch options differ from pinned arm",
    )
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        .stdout.strip()
        .splitlines()
    )
    require(len(gpu) == 1, "expected one visible GPU on the SUT host")
    uuid, sep, driver = gpu[0].partition(",")
    device = hashlib.sha256(uuid.strip().encode()).hexdigest()
    require(
        bool(sep)
        and uuid.startswith("GPU-")
        and declared_device["device_id"] == device
        and declared_device["driver_version"] == driver.strip(),
        "live GPU or driver differs from SUT manifest",
    )
    require(
        (proc / "stat").read_text().rsplit(")", 1)[1].split()[19] == started,
        "API server PID changed",
    )
    return {
        "instance_sha256": digest([pid, started]),
        "pin_sha256": digest(expected),
        "device_id": device,
    }
