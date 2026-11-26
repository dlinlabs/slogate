"""E0 vs --enforce-eager. Reuse the operational checks, never a small-budget receipt."""

import sys
from pathlib import Path

if __package__:
    from .prepare_small_budget import main as _run_trial
else:
    if sys.flags.safe_path:  # Also support a direct, isolated `python -I` invocation.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_small_budget import main as _run_trial


def main(argv: list[str] | None = None) -> int:
    return _run_trial(argv, candidate="eager")


if __name__ == "__main__":
    raise SystemExit(main())
