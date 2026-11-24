"""Exercise only the worker protocol: no HTTP, subprocesses or files written."""

import os
import runpy
from pathlib import Path
from unittest.mock import Mock


def test_cli_invalid_wave_does_not_skip_next_command_or_erase_exit_three():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    harness = runpy.run_path(str(scripts / "sustained_replay.py"))
    worker = runpy.run_path(str(harness["WORKER"]))
    out, workload = Path("arm"), Path("workload")
    cli = Mock(side_effect=[0, 3, 0])
    receipts = {}
    ports = ["9001\n", "9002\n", "9003\n"]
    name = harness["NAME"]
    assert worker["run_commands"](ports, out, workload, name, cli, receipts.__setitem__) == 3
    for i, rc in enumerate([0, 3, 0], 1):
        wave = out / f"wave-{i:03}"
        assert receipts[wave / "done.json"] == {"pid": os.getpid(), "cli_exit": rc}
        (argv,) = cli.call_args_list[i - 1].args
        assert argv[0] == "run" and argv[argv.index("--out") + 1] == str(wave / name)
        assert argv[argv.index("--base-url") + 1] == f"http://127.0.0.1:{9000 + i}"
        assert "--local-fixture-clock" in argv and "--client-timing" not in argv
