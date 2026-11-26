"""Materialize the three reference vLLM 0.30.0 configuration trials; never start a GPU."""

import argparse
import json
from pathlib import Path

from slogate.paired_plan import pin
from slogate.report import write_json

BASE = (
    Path(__file__).resolve().parents[1]
    / "configs/sut/vllm-0.30.0_qwen3-4b-instruct-2507_rtx4090.json"
)


def candidates(base: dict) -> dict[str, dict]:
    frozen = pin("vllm", base)
    flags = frozen["flags"]
    if (
        frozen["engine"]["expect"]["vllm"] != "0.30.0"
        or flags.get("--max-num-batched-tokens") != ["2048"]
        or "--enforce-eager" in flags
        or "--compilation-config" in flags
    ):
        raise ValueError("expected the E0 baseline flags, not a previously modified candidate")
    original = base["serve_args"]
    budget = original.copy()
    budget[budget.index("--max-num-batched-tokens") + 1] = "512"
    variants = {
        "eager": [*original, "--enforce-eager"],
        "small-graphs": [
            *original,
            "--compilation-config",
            '{"cudagraph_capture_sizes":[1,2,4,8]}',
        ],
        "small-budget": budget,
    }
    return {
        name: {**base, "name": f"{base['name']}-{name}", "serve_args": argv}
        for name, argv in variants.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    variants = candidates(json.loads(BASE.read_bytes()))
    out.mkdir(parents=True, exist_ok=False)
    for name, candidate in variants.items():
        write_json(out / f"{name}.json", candidate)
    print(f"Wrote {len(variants)} candidate configs to {out}. No measurements or verdicts.")


if __name__ == "__main__":
    main()
