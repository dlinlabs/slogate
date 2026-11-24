"""Run inside the engine venv: report installed versions and fail on any mismatch."""

import importlib.metadata
import json
import sys


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as handle:
        expect = json.load(handle)["engine"].get("expect", {})
    import torch

    if not torch.cuda.is_available():
        print("CUDA is unavailable: check the host driver and GPU passthrough", file=sys.stderr)
        return 1
    a = torch.ones((32, 32), device="cuda")
    product = a @ a
    torch.cuda.synchronize()
    if product[0, 0].item() != 32:
        print("CUDA compute check failed", file=sys.stderr)
        return 1
    found = {
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    print(json.dumps(found, indent=2))
    mismatches = {k: (v, found.get(k)) for k, v in expect.items() if found.get(k) != v}
    if mismatches:
        print(f"engine does not match the SUT config: {mismatches}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
