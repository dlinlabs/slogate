"""Write the SUT description stored with every run: config, engine versions, host facts.

Fails when the host does not match the config (wrong GPU model or count), so a run on the
wrong hardware never produces numbers that look comparable.
"""

import json
import os
import platform
import subprocess
import sys

GPU_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.limit",
    "pcie.link.gen.max",
    "pcie.link.width.max",
)


def run(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def gpus() -> list[dict[str, str]]:
    output = run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output.startswith("unavailable"):
        return []
    return [
        dict(zip(GPU_FIELDS, (part.strip() for part in line.split(",")), strict=True))
        for line in output.splitlines()
    ]


def cpu_model() -> str:
    for line in run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
    with open(sys.argv[2], encoding="utf-8") as handle:
        engine = json.load(handle)
    devices = gpus()
    hardware = config.get("hardware", {})
    expected_name = hardware.get("gpu_name_contains", "")
    expected_count = hardware.get("gpu_count")
    if expected_count is not None and len(devices) != expected_count:
        print(f"expected {expected_count} GPUs, found {len(devices)}", file=sys.stderr)
        return 1
    if expected_name and not all(expected_name in d["name"] for d in devices):
        print(f"expected GPUs named *{expected_name}*, found {devices}", file=sys.stderr)
        return 1
    info = {
        "config": config,
        "engine": engine,
        "worker_step_trace_requested": bool(os.environ.get("SLOGATE_TRACE_DIR")),
        "host": {
            "gpus": devices,
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "kernel": platform.release(),
            "runpod_pod_id_present": bool(os.environ.get("RUNPOD_POD_ID")),
        },
    }
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
