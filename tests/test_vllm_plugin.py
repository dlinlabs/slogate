from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from slogate import step_trace
from slogate import vllm_plugin as plugin
from slogate.check_trace import main as check_trace


class Cuda:
    def __init__(self) -> None:
        self.ready = False
        self.stream = object()
        self.records: list[object] = []
        self.elapsed_ms = 8.5

    def current_stream(self) -> object:
        return self.stream

    def Event(self, *, enable_timing: bool) -> SimpleNamespace:
        assert enable_timing

        def elapsed(end: Any) -> float:
            assert end.query(), "elapsed_time must not wait on an unfinished event"
            return self.elapsed_ms

        return SimpleNamespace(
            record=self.records.append, query=lambda: self.ready, elapsed_time=elapsed
        )


@pytest.fixture
def worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Cuda, dict[str, type], Callable[[], None]]]:
    cuda = Cuda()
    runners: dict[str, type] = {}
    callbacks: list[Callable[[], None]] = []

    def load(name: str) -> SimpleNamespace:
        if name == "torch":
            return SimpleNamespace(cuda=cuda)
        if name in runners:
            return SimpleNamespace(GPUModelRunner=runners[name])
        raise ModuleNotFoundError(name)

    def close() -> None:
        for callback in callbacks:
            callback()

    monkeypatch.setenv("SLOGATE_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(plugin, "import_module", load)
    monkeypatch.setattr(plugin, "atexit", SimpleNamespace(register=callbacks.append))
    yield cuda, runners, close
    close()
    plugin._worker_trace.cache_clear()


def rows(path: Path) -> list[dict[str, Any]]:
    (trace,) = path.glob("steps-*.jsonl")
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def test_discovery_is_inert_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLOGATE_TRACE_DIR", raising=False)
    monkeypatch.setattr(plugin, "import_module", lambda name: pytest.fail(f"imported {name}"))
    (entry,) = entry_points(group="vllm.general_plugins", name="slogate")
    assert entry.load() is plugin.register
    entry.load()()


@pytest.mark.parametrize("adapter", ["v1", "v2"])
def test_step_graph_key_and_deferred_cuda(adapter: str, worker, tmp_path: Path) -> None:
    cuda, runners, close = worker
    mode = SimpleNamespace(name="FULL")
    desc = SimpleNamespace(cg_mode=mode, num_tokens=8)
    output = SimpleNamespace(
        num_scheduled_tokens={"private-new": 4, "private-cached": 1},
        total_num_scheduled_tokens=5,
        scheduled_new_reqs=[SimpleNamespace(req_id="private-new", num_computed_tokens=0)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["private-cached"], num_computed_tokens=[19]),
    )
    result = object()

    class FlashAttentionBackend:
        pass

    class V1:
        attn_groups = ((SimpleNamespace(backend=FlashAttentionBackend),),)

        def _determine_batch_execution_and_padding(self):
            return mode, desc, False, None, None

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            assert scheduler_output is output
            cuda.stream = object()
            assert self._determine_batch_execution_and_padding()[1] is desc
            return result

    class V2:
        attn_groups = V1.attn_groups

        def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
            return batch_desc

        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
            cuda.stream = object()
            assert self.prepare_inputs(scheduler_output, None, batch_desc=desc) is desc
            return result

    cls = V1 if adapter == "v1" else V2
    runners[plugin._RUNNERS[adapter]] = cls
    plugin.register()
    wrapped = cls.execute_model
    plugin.register()
    assert cls.execute_model is wrapped
    runner = cls()
    if isinstance(runner, V2):
        assert runner.execute_model(output, None, True) is result  # positional dummy
        assert not list(tmp_path.iterdir())
    stream_a = cuda.stream
    assert runner.execute_model(scheduler_output=output) is result
    stream_b = cuda.stream
    assert stream_a is not stream_b
    assert runner.execute_model(output, None) is result
    assert cuda.records == [stream_a, stream_a, stream_b, stream_b]
    assert [r["kind"] for r in rows(tmp_path)] == ["header", "step", "step"]
    cuda.ready = True
    assert runner.execute_model(output) is result
    close()
    records = rows(tmp_path)
    steps = [r for r in records if r["kind"] == "step"]
    timings = [r for r in records if r["kind"] == "cuda"]
    key = hashlib.sha256(b'[["private-cached",19,1],["private-new",0,4]]').hexdigest()
    assert [r["step_id"] for r in steps] == [1, 2, 3]
    for row in steps:
        assert row["ok"] and row["runner"] == adapter
        assert row["schedule_key"] == key
        assert (row["scheduled_requests"], row["scheduled_tokens"], row["padded_tokens"]) == (
            2,
            5,
            8,
        )
        assert (row["graph_mode"], row["attention_backends"]) == ("FULL", ["FlashAttentionBackend"])
        assert row["host_end_ns"] >= row["host_start_ns"]
    assert [r["step_id"] for r in timings] == [1, 2, 3]
    assert all(r["cuda_elapsed_s"] == pytest.approx(0.0085) for r in timings)
    assert "private-" not in str(records)
    assert check_trace(tmp_path) == 0


def test_layout_drift_and_engine_error(worker, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    _, runners, close = worker
    failure = RuntimeError("model failure")
    calls = []

    class Changed:
        def execute_model(self, renamed_output):
            return renamed_output

    class Runner:
        def execute_model(self, scheduler_output):
            calls.append(scheduler_output)
            if getattr(scheduler_output, "fail", False):
                raise failure
            return scheduler_output

    runners.update(dict.fromkeys(plugin._RUNNERS.values(), Runner))
    runners[plugin._RUNNERS["v2"]] = Changed
    original = Changed.execute_model
    plugin.register()
    assert Changed.execute_model is original
    assert "unsupported execute_model signature" in caplog.text
    runner = Runner()
    for skipped in (SimpleNamespace(), SimpleNamespace(num_scheduled_tokens={})):
        assert runner.execute_model(skipped) is skipped
    assert not list(tmp_path.iterdir())
    output = SimpleNamespace(num_scheduled_tokens={"a": 1}, fail=True)
    with pytest.raises(RuntimeError) as raised:
        runner.execute_model(output)
    assert raised.value is failure
    assert len(calls) == 3 and plugin._ACTIVE.get() is None
    close()
    _, step, cuda_row = rows(tmp_path)
    assert step["ok"] is False
    assert step["schedule_key"] is step["graph_mode"] is step["attention_backends"] is None
    assert cuda_row["status"] == "not_ready_at_exit" and cuda_row["cuda_elapsed_s"] is None
    assert check_trace(tmp_path) != 0


def test_cuda_backpressure_and_invalid_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(step_trace, "MAX_PENDING", 1)
    cuda = Cuda()
    trace = step_trace.StepTrace(tmp_path, cuda)
    for _ in range(2):
        row = {}
        pair = trace.begin(row)
        trace.finish(row, pair, True)
    assert len(cuda.records) == 2  # only one start/end pair, no wait to make room
    cuda.ready = True
    cuda.elapsed_ms = float("nan")
    trace.close()
    _, step1, step2, timing = rows(tmp_path)
    assert (step1["cuda_status"], step2["cuda_status"]) == ("pending", "queue_full")
    assert timing["status"] == "unavailable:ValueError" and timing["cuda_elapsed_s"] is None
    assert check_trace(tmp_path) != 0


def test_unwritable_trace_does_not_break_engine(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _, runners, _ = worker
    path = tmp_path / "not-a-directory"
    path.touch()
    monkeypatch.setenv("SLOGATE_TRACE_DIR", str(path))

    class Runner:
        def execute_model(self, scheduler_output):
            return scheduler_output

    runners[plugin._RUNNERS["v1"]] = Runner
    plugin.register()
    output = SimpleNamespace(num_scheduled_tokens={"a": 1})
    assert Runner().execute_model(output) is output
    assert "Cannot open worker trace" in caplog.text
    assert check_trace(tmp_path) != 0
