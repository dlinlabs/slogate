"""Smoke check for the optional diagnostic rerun; never used to set a serving SLO."""

import json
import math
import sys
from pathlib import Path


def coverage(path: Path) -> tuple[int, int]:
    steps = timed = 0
    pending: set[int] = set()
    with path.open(encoding="utf-8") as records:
        header = json.loads(records.readline())
        if header["kind"] != "header" or header["schema_version"] != 1:
            raise ValueError(f"{path}: unsupported trace header")
        for line in records:
            row = json.loads(line)
            step_id = row["step_id"]
            if row["kind"] == "step" and row["ok"]:
                steps += 1
                if row["cuda_status"] == "pending":
                    pending.add(step_id)
            elif row["kind"] == "cuda" and step_id in pending:
                pending.remove(step_id)
                if row["status"] == "ready":
                    seconds = row["cuda_elapsed_s"]
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError(f"{path}: invalid CUDA duration on step {step_id}")
                    timed += 1
    return steps, timed


def main(directory: Path) -> int:
    paths = sorted(directory.glob("steps-*.jsonl"))
    if not paths:
        print("No worker step files; check plugin discovery in host/server.log", file=sys.stderr)
        return 1
    all_have_timings = True
    for path in paths:
        try:
            steps, timed = coverage(path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"Invalid trace {path}: {exc}", file=sys.stderr)
            return 1
        print(f"{path.name}: {timed}/{steps} successful steps have current-stream CUDA timing")
        all_have_timings &= timed > 0
    return 0 if all_have_timings else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
