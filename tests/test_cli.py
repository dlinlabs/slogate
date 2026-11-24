from __future__ import annotations

import json

from slogate import report, workload
from slogate.cli import EXIT_INVALID_MEASUREMENT, EXIT_OK, aa_exclusion, main


def test_aa_exclusion_keeps_only_healthy_valid_runs() -> None:
    healthy = {"requests": {"failed": 0}}
    assert aa_exclusion({"valid": True, "reasons": []}, healthy) is None
    reason = aa_exclusion({"valid": False, "reasons": ["launch lag max 80.0 ms"]}, healthy)
    assert reason == "invalid measurement: launch lag max 80.0 ms"
    assert aa_exclusion({"valid": True, "reasons": []}, {"requests": {"failed": 2}}) == (
        "2 requests failed"
    )


def test_workload_commands(tmp_path, capsys) -> None:
    out = tmp_path / "w.json.gz"
    code = main(
        [
            "workload",
            "synth",
            "--name",
            "cli",
            "--duration-s",
            "2",
            "--rate-rps",
            "5",
            "--input-tokens",
            "8,16",
            "--max-tokens",
            "4",
            "--seed",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    assert summary["digest"] == workload.load(out).digest
    assert main(["workload", "info", str(out)]) == EXIT_OK


def test_invalid_workload_returns_error(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 9}), encoding="utf-8")
    assert main(["workload", "info", str(bad)]) == 1
    assert "workload error" in capsys.readouterr().err


def test_aa_writes_runs_and_noise_report(fake_engine, tmp_path) -> None:
    url = fake_engine(ttft_s=0.02, itl_s=0.01, jitter=0.2)
    load_path = workload.save(
        workload.synthetic(
            name="aa", duration_s=1.0, rate_rps=6.0, input_tokens=[8], max_tokens=4, seed=4
        ),
        tmp_path / "aa.json",
    )
    out = tmp_path / "aa-run"
    code = main(
        [
            "aa",
            "--base-url",
            url,
            "--workload",
            str(load_path),
            "--out",
            str(out),
            "--runs",
            "3",
            "--cooldown-s",
            "0",
            "--warmup-requests",
            "1",
            "--slo-ttft-ms",
            "500",
            "--slo-tpot-ms",
            "100",
        ]
    )
    assert code == EXIT_OK
    noise = json.loads((out / report.NOISE_JSON).read_text(encoding="utf-8"))
    assert noise["included_runs"] == ["run-001", "run-002", "run-003"]
    assert noise["excluded_runs"] == {}
    first = report.read_run(out / "run-001")
    assert first["fidelity"]["valid"]
    assert "timing_trace" not in first["replay"]["diagnostics"]
    assert first["meta"]["workload"]["digest"] == workload.load(load_path).digest
    assert first["meta"]["server"]["models"]["data"][0]["id"] == "fake-model"
    assert first["metrics"]["goodput"]["attainment"] == 1.0
    assert len(report.read_records(out / "run-001")) == first["metrics"]["requests"]["offered"]
    markdown = (out / report.NOISE_MARKDOWN).read_bytes()
    (out / report.NOISE_JSON).unlink()
    assert main(["noise", str(out)]) == EXIT_OK
    assert json.loads((out / report.NOISE_JSON).read_bytes()) == noise
    assert (out / report.NOISE_MARKDOWN).read_bytes() == markdown


def test_noise_excludes_failed_requests_and_invalid_measurements(tmp_path) -> None:
    for i in range(1, 6):
        directory = tmp_path / f"run-{i:03d}"
        directory.mkdir()
        report.write_json(
            directory / report.RUN_FILE,
            {
                "fidelity": {"valid": i != 2, "reasons": ["client lag"] if i == 2 else []},
                "metrics": {"requests": {"failed": int(i == 4)}},
            },
        )
    assert main(["noise", str(tmp_path)]) == EXIT_OK
    noise = json.loads((tmp_path / report.NOISE_JSON).read_bytes())
    assert noise["included_runs"] == ["run-001", "run-003", "run-005"]
    assert noise["excluded_runs"] == {
        "run-002": "invalid measurement: client lag",
        "run-004": "1 requests failed",
    }
    (tmp_path / "run-005" / report.RUN_FILE).unlink()
    assert main(["noise", str(tmp_path)]) == EXIT_INVALID_MEASUREMENT
    assert json.loads((tmp_path / report.NOISE_JSON).read_bytes())["included_runs"] == [
        "run-001",
        "run-003",
    ]
