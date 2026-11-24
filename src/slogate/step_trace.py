"""Worker-local JSONL step records and deferred, current-stream CUDA event timing."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

_LOG = logging.getLogger(__name__)
_EVENT_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
MAX_PENDING = 256


class EventPair(NamedTuple):
    start: Any
    end: Any
    stream: Any


class StepTrace:
    def __init__(self, directory: Path, cuda: Any) -> None:
        self.pid = os.getpid()
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / f"steps-{self.pid}.jsonl").open("x", encoding="utf-8", buffering=1)
        self.cuda = cuda
        self.sequence = 0
        self.pending: deque[tuple[int, EventPair]] = deque()
        self._write(
            kind="header",
            schema_version=1,
            pid=self.pid,
            monotonic_ns=time.monotonic_ns(),
            unix_ns=time.time_ns(),
        )

    def _write(self, **record: object) -> None:
        try:
            self.file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            _LOG.error("Step trace write failed; disabling capture: %s", exc)
            self._close_file()

    def _close_file(self) -> None:
        self.pending.clear()
        try:
            self.file.close()
        except OSError as exc:
            _LOG.error("Step trace close failed: %s", exc)

    def _drain(self, *, closing: bool = False) -> None:
        while self.pending:
            step_id, pair = self.pending[0]
            seconds = None
            try:
                if pair.end.query():
                    seconds = pair.start.elapsed_time(pair.end) / 1000
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError("invalid event duration")
                    status = "ready"
                elif closing:
                    status = "not_ready_at_exit"
                else:
                    break
            except _EVENT_ERRORS as exc:
                seconds, status = None, f"unavailable:{type(exc).__name__}"
            self.pending.popleft()
            self._write(kind="cuda", step_id=step_id, status=status, cuda_elapsed_s=seconds)

    def begin(self, row: dict[str, Any]) -> EventPair | None:
        self._drain()
        self.sequence += 1
        row["step_id"] = self.sequence
        pair = None
        if self.cuda is None:
            row["cuda_status"] = "unavailable"
        elif len(self.pending) >= MAX_PENDING:
            row["cuda_status"] = "queue_full"
        elif not self.file.closed:
            try:
                stream = self.cuda.current_stream()
                start = self.cuda.Event(enable_timing=True)
                end = self.cuda.Event(enable_timing=True)
                start.record(stream)
                pair = EventPair(start, end, stream)
            except _EVENT_ERRORS as exc:
                row["cuda_status"] = f"unavailable:{type(exc).__name__}"
        row["host_start_ns"] = time.monotonic_ns()
        return pair

    def finish(self, row: dict[str, Any], pair: EventPair | None, ok: bool) -> None:
        row["host_end_ns"] = time.monotonic_ns()
        if self.file.closed:
            return
        if pair is not None:
            try:
                pair.end.record(pair.stream)
                self.pending.append((row["step_id"], pair))
                row["cuda_status"] = "pending"
            except _EVENT_ERRORS as exc:
                row["cuda_status"] = f"unavailable:{type(exc).__name__}"
        self._write(kind="step", ok=ok, **row)

    def close(self) -> None:
        if self.pid == os.getpid() and not self.file.closed:
            self._drain(closing=True)
            self._close_file()
