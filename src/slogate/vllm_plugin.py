"""Opt-in vLLM general plugin. Only the serving worker opens a trace or touches CUDA."""

from __future__ import annotations

import atexit
import hashlib
import inspect
import json
import logging
import os
from contextvars import ContextVar
from functools import lru_cache, wraps
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import Any

from .step_trace import StepTrace

_LOG = logging.getLogger(__name__)
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("slogate_step", default=None)
_RUNNERS = {
    "v1": "vllm.v1.worker.gpu_model_runner",
    "v2": "vllm.v1.worker.gpu.model_runner",
}


@lru_cache(maxsize=1)
def _worker_trace(directory: str, pid: int) -> StepTrace | None:
    # Key by PID: general plugins may be registered before vLLM forks its workers.
    try:
        cuda = import_module("torch").cuda
    except (ImportError, AttributeError) as exc:
        _LOG.warning("CUDA step timing unavailable in pid %s: %s", pid, exc)
        cuda = None
    try:
        trace = StepTrace(Path(directory), cuda)
    except OSError as exc:
        _LOG.error("Cannot open worker trace in %s: %s", directory, exc)
        return None
    atexit.register(trace.close)
    return trace


def _backends(runner: Any) -> list[str] | None:
    groups = getattr(runner, "attn_groups", None)
    if not isinstance(groups, (list, tuple)):
        return None
    names = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            return None
        for item in group:
            name = getattr(getattr(item, "backend", None), "__name__", None)
            if not isinstance(name, str):
                return None
            names.add(name)
    return sorted(names) or None


def _schedule_key(output: Any, counts: dict[str, int]) -> str | None:
    """Same join key as the lab, from pre-execution positions, not submission order."""
    try:
        cached = output.scheduled_cached_reqs
        positions = {}
        for rid, pos in chain(
            ((req.req_id, req.num_computed_tokens) for req in output.scheduled_new_reqs),
            zip(cached.req_ids, cached.num_computed_tokens, strict=True),
        ):
            if not isinstance(rid, str) or type(pos) is not int or pos < 0 or rid in positions:
                return None
            positions[rid] = pos
        if sum(counts.values()) != output.total_num_scheduled_tokens:
            return None
        work = sorted((rid, positions[rid], count) for rid, count in counts.items())
    except (AttributeError, KeyError, TypeError, ValueError):
        return None  # Unknown layout: keep timing, but don't join this step to a scheduler row.
    return hashlib.sha256(json.dumps(work, separators=(",", ":")).encode()).hexdigest()


def _signature(method: Any) -> inspect.Signature | None:
    if not inspect.isfunction(method) or any(
        check(method)
        for check in (
            inspect.iscoroutinefunction,
            inspect.isgeneratorfunction,
            inspect.isasyncgenfunction,
        )
    ):
        return None
    try:
        return inspect.signature(method)
    except (TypeError, ValueError):
        return None


def _install_graph(cls: type, adapter: str) -> None:
    v2 = adapter == "v2"
    name = "prepare_inputs" if v2 else "_determine_batch_execution_and_padding"
    original = getattr(cls, name, None)
    sig = _signature(original)
    if sig is None or (v2 and "batch_desc" not in sig.parameters):
        _LOG.warning("No compatible %s.%s; graph mode/padding will be null", cls.__name__, name)
        return

    @wraps(original)
    def observed(runner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(runner, *args, **kwargs)
        row = _ACTIVE.get()
        if row is not None:
            if v2:
                try:
                    desc = sig.bind(runner, *args, **kwargs).arguments.get("batch_desc")
                except TypeError:
                    return result  # A decorator accepts more calls than its published signature.
                mode = getattr(desc, "cg_mode", None)
            elif isinstance(result, tuple) and len(result) >= 2:
                mode, desc = result[:2]
            else:
                return result
            mode_name = getattr(mode, "name", None)
            tokens = getattr(desc, "num_tokens", None)
            row["graph_mode"] = mode_name if isinstance(mode_name, str) else None
            row["padded_tokens"] = tokens if type(tokens) is int else None
        return result

    setattr(cls, name, observed)


def _install(cls: type, adapter: str, directory: str) -> None:
    original = getattr(cls, "execute_model", None)
    if getattr(original, "_slogate_step_hook", False):
        return
    sig = _signature(original)
    if sig is None or "scheduler_output" not in sig.parameters:
        _LOG.warning("Skipping %s: unsupported execute_model signature", adapter)
        return
    warned_counts = False

    @wraps(original)
    def execute(runner: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal warned_counts
        try:
            bound = sig.bind(runner, *args, **kwargs).arguments
        except TypeError:
            return original(runner, *args, **kwargs)  # Preserve vLLM's own argument handling.
        if bound.get("dummy_run") or bound.get("is_profile"):
            return original(runner, *args, **kwargs)
        output = bound.get("scheduler_output")
        counts = getattr(output, "num_scheduled_tokens", None)
        if not isinstance(counts, dict) or not all(
            type(n) is int and n > 0 for n in counts.values()
        ):
            if not warned_counts:
                _LOG.warning("Skipping steps with unsupported %s scheduled token counts", adapter)
                warned_counts = True
            return original(runner, *args, **kwargs)
        if not counts:
            return original(runner, *args, **kwargs)
        trace = _worker_trace(directory, os.getpid())
        if trace is None or trace.file.closed:
            return original(runner, *args, **kwargs)
        row = {
            "runner": adapter,
            "scheduled_requests": len(counts),
            "scheduled_tokens": sum(counts.values()),
            "schedule_key": _schedule_key(output, counts),
            "attention_backends": _backends(runner),
            "graph_mode": None,
            "padded_tokens": None,
        }
        pair = trace.begin(row)
        token = _ACTIVE.set(row)
        ok = False
        try:
            result = original(runner, *args, **kwargs)
            ok = True
            return result
        finally:
            _ACTIVE.reset(token)
            trace.finish(row, pair, ok)

    execute._slogate_step_hook = True
    cls.execute_model = execute
    _install_graph(cls, adapter)


def register() -> None:
    directory = os.environ.get("SLOGATE_TRACE_DIR")
    if not directory:
        return
    for adapter, module in _RUNNERS.items():
        try:
            cls = import_module(module).GPUModelRunner
        except (ImportError, AttributeError, OSError) as exc:
            _LOG.warning("Skipping %s step hook (%s): %s", adapter, module, exc)
            continue
        _install(cls, adapter, directory)
