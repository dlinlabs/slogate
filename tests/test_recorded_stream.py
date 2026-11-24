from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import replace

import httpx
import pytest

from slogate.cli import main
from slogate.fake_engine import EngineProfile, StreamingChunks, serve_in_thread
from slogate.recorded_stream import RecordedStreamEngine, RecordedStreams
from slogate.replay import FIXTURE_EPOCH_HEADER, ReplayConfig, build_payload
from slogate.report import read_run
from slogate.timeline_report import timing_errors
from slogate.workload import Request, Workload, save


def example():
    w = Workload(
        "tiny", "fake", 2, (Request("x", 0.1, 4, 4, "one"), Request("y", 0.2, 4, 4, "two"))
    )
    rows = [
        {
            "id": q.id,
            "scheduled_s": q.arrival_s,
            "success": True,
            "launched_s": 1.0,
            "headers_s": 1.005,
            "first_token_s": 1.010,
            "chunk_times_s": [1.010, 1.025],
            "chunk_tokens": [1, 2],
            "finished_s": 1.035,
            "prompt_tokens": 4,
            "output_tokens": 4,
            "finish_reason": "length",
        }
        for q in w.requests
    ]
    return w, rows


def test_fixture_joins_by_id_and_payload_not_http_arrival_order():
    w, rows = example()
    rows[1]["chunk_tokens"] = [2, 2]
    f = RecordedStreams.from_rows("arm", "hash", w, list(reversed(rows)))
    assert f.by_payload[("one", 4)].content_s == pytest.approx([0.010, 0.025])
    assert f.by_payload[("one", 4)].header_s == pytest.approx(0.005)
    assert f.by_payload[("one", 4)].end_s == pytest.approx(0.035)
    assert f.by_payload[("one", 4)].widths == (1, 2)
    assert f.by_payload[("two", 4)].widths == (2, 2)


def test_fixture_rejects_ambiguous_malformed_or_unmatched_inputs():
    w, rows = example()
    with pytest.raises(ValueError, match="identity"):
        RecordedStreams.from_rows("r", "h", w, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="payload"):
        duplicate = replace(w, requests=(w.requests[0], replace(w.requests[1], prompt="one")))
        RecordedStreams.from_rows("r", "h", duplicate, rows)
    for field, value in (("chunk_tokens", [1]), ("finished_s", 0.9), ("launched_s", float("nan"))):
        with pytest.raises(ValueError):
            RecordedStreams.from_rows("r", "h", w, [dict(rows[0], **{field: value}), rows[1]])


def test_sse_preserves_observed_widths_separate_usage_and_warmup_identity():
    w, rows = example()
    f = RecordedStreams.from_rows("r", "h", w, rows)
    app = RecordedStreamEngine(EngineProfile(), f).app()

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            path = "/v1/chat/completions"
            body = build_payload(w.requests[0], "fake", ReplayConfig("http://local"))
            for _ in range(2):  # Warmup must not consume the measured schedule.
                r = await client.post(path, json=body)
                assert r.status_code == 200 and r.text.endswith("data: [DONE]\n\n")
                events = [
                    json.loads(line[6:])
                    for line in r.text.splitlines()
                    if line.startswith("data: {")
                ]
                assert [e["usage"]["completion_tokens"] for e in events] == [0, 1, 3, 4, 4]
                contents = [e["choices"][0]["delta"].get("content") for e in events if e["choices"]]
                assert contents == ["", " tok", " tok tok", None]
            body["messages"][0]["content"] = "unknown"
            assert (await client.post(path, json=body)).status_code == 400

    asyncio.run(run())


def test_content_and_end_use_absolute_deadlines_without_accumulating_send_time(monkeypatch):
    w, rows = example()
    fixture = RecordedStreams.from_rows("arm", "h", w, rows)
    engine = RecordedStreamEngine(EngineProfile(), fixture)
    due = []

    async def record(deadline):
        due.append(deadline)

    monkeypatch.setattr("slogate.recorded_stream._until", record)

    async def consume():
        wire = StreamingChunks("fake", 4, {})
        return [v async for v in engine._emit(100, fixture.by_payload[("one", 4)], wire)]

    assert asyncio.run(consume())[-1] == b"data: [DONE]\n\n"
    assert due == pytest.approx([100.010, 100.025, 100.035])


def test_pairwise_report_separates_arrival_shift_from_relative_gap_error():
    _, s = example()
    actual = copy.deepcopy(list(reversed(s)))
    for r in actual:
        r["launched_s"] += 0.2
        r["finished_s"] += 0.22
        r["chunk_times_s"] = [r["chunk_times_s"][0] + 0.21, r["chunk_times_s"][1] + 0.22]
    e = timing_errors(s, actual)
    assert not e["unmatched_ids"] and e["matched_content_chunks"] == 4
    assert e["ttft_p99_abs_ms"] == pytest.approx(10)
    assert e["chunk_gap_p99_abs_ms"] == pytest.approx(10)
    assert e["chunk_offset_p99_abs_ms"] == pytest.approx(20)
    assert e["global_chunk_p99_abs_ms"] == pytest.approx(220)
    actual[0]["chunk_tokens"] = [2, 1]
    assert timing_errors(s, actual)["unmatched_ids"] == ["y"]


def test_shared_epoch_targets_source_offsets_even_after_warmup_and_reordering(monkeypatch):
    w, rows = example()
    for k in ("launched_s", "headers_s", "first_token_s", "finished_s"):
        rows[1][k] += 2
    rows[1]["chunk_times_s"] = [t + 2 for t in rows[1]["chunk_times_s"]]
    fixture = replace(RecordedStreams.from_rows("r", "h", w, rows), shared_clock=True)
    app = RecordedStreamEngine(EngineProfile(), fixture).app()
    calls = []

    async def record(deadline):
        calls.append(deadline)

    monkeypatch.setattr("slogate.recorded_stream._until", record)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            path = "/v1/chat/completions"
            body = build_payload(w.requests[0], "fake", ReplayConfig("http://local"))
            assert (await client.post(path, json=body)).status_code == 400
            header = {FIXTURE_EPOCH_HEADER: "warmup"}
            assert (await client.post(path, json=body, headers=header)).status_code == 200
            calls.clear()
            epoch = time.perf_counter() - 0.3
            header = {FIXTURE_EPOCH_HEADER: repr(epoch)}
            for q in reversed(w.requests):
                payload = build_payload(q, "fake", ReplayConfig("http://local"))
                assert (await client.post(path, json=payload, headers=header)).status_code == 200
            assert [t - epoch for t in calls] == pytest.approx(
                [3.005, 3.010, 3.025, 3.035, 1.005, 1.010, 1.025, 1.035], abs=1e-7
            )
            for invalid in ("warmup", repr(epoch - 1), "nan"):
                response = await client.post(
                    path, json=body, headers={FIXTURE_EPOCH_HEADER: invalid}
                )
                assert response.status_code == 400

    asyncio.run(check())


def test_shared_clock_only_used_with_verified_loopback_fixture(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        ReplayConfig("http://outside.invalid", local_fixture_clock=True)
    w, rows = example()
    fixture = replace(RecordedStreams.from_rows("source", "digest", w, rows), shared_clock=True)
    out = tmp_path / "run"
    with serve_in_thread(RecordedStreamEngine(EngineProfile(), fixture)) as url:
        code = main(
            [
                "run",
                "--base-url",
                url,
                "--workload",
                str(save(w, tmp_path / "workload.json")),
                "--out",
                str(out),
                "--client-timing",
                "--local-fixture-clock",
                "--warmup-requests",
                "2",
            ]
        )
    run = read_run(out)
    assert code == (0 if run["fidelity"]["valid"] else 3)
    assert run["metrics"]["requests"]["succeeded"] == len(w.requests)
    assert run["meta"]["replay_config"]["local_fixture_clock"] is True
    assert run["meta"]["server"]["version"]["anchor"] == "shared"
