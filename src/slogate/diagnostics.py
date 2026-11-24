"""Client-side diagnostics that decide whether a replay measured the server or the client.

Two failure modes corrupt open-loop latency measurements without any server change:
the client's event loop falls behind its arrival schedule, or the client process pauses for
garbage collection while streams are in flight. Both are recorded here.
"""

from __future__ import annotations

import asyncio
import gc
import time
from array import array
from dataclasses import dataclass
from typing import Any

import numpy as np

LOOP_INTERVAL_S = 0.010


def _distribution(values: list[float] | array) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"n": 0, "p50": None, "p99": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "n": int(data.size),
        "p50": float(np.percentile(data, 50)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


class LoopLagMonitor:
    """Samples how late the event loop wakes up relative to a fixed cadence."""

    def __init__(self, interval_s: float = LOOP_INTERVAL_S, *, timing_trace: bool = False) -> None:
        self.interval_s = interval_s
        self.lags_s: list[float] = []
        self.due_offsets_s = array("d") if timing_trace else None
        self._origin = 0.0
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def _run(self) -> None:
        clock = time.perf_counter
        while self._running:
            due = clock() + self.interval_s
            await asyncio.sleep(self.interval_s)
            self.lags_s.append(max(0.0, clock() - due))
            if self.due_offsets_s is not None:
                self.due_offsets_s.append(due - self._origin)

    def start(self, origin: float | None = None) -> None:
        self._origin = time.perf_counter() if origin is None else origin
        self._running = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    def summary(self) -> dict[str, Any]:
        return {"interval_s": self.interval_s, **_distribution(self.lags_s)}


class GCMonitor:
    """Records garbage-collection pauses using preallocated storage.

    The callback runs inside the collector, so it only writes into fixed-size arrays.
    """

    def __init__(self, capacity: int = 65_536) -> None:
        self.capacity = capacity
        self._start_at = array("d", [0.0]) * capacity
        self._duration = array("d", [0.0]) * capacity
        self._generation = array("b", [0]) * capacity
        self._count = 0
        self.dropped = 0
        self._pause_started = 0.0
        self._origin = time.perf_counter()

    def _callback(self, phase: str, info: dict[str, int]) -> None:
        now = time.perf_counter()
        if phase == "start":
            self._pause_started = now
            return
        if self._count >= self.capacity:
            self.dropped += 1
            return
        index = self._count
        self._start_at[index] = self._pause_started - self._origin
        self._duration[index] = now - self._pause_started
        self._generation[index] = info.get("generation", -1)
        self._count = index + 1

    def start(self, origin: float | None = None) -> None:
        if origin is not None:
            self._origin = origin
        gc.callbacks.append(self._callback)

    def stop(self) -> None:
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)

    def pauses(self) -> list[tuple[float, int, float]]:
        """(start offset, generation, duration) for each recorded collection."""
        return [
            (self._start_at[i], self._generation[i], self._duration[i]) for i in range(self._count)
        ]

    def summary(self) -> dict[str, Any]:
        durations = self._duration[: self._count]
        by_generation: dict[str, int] = {}
        for i in range(self._count):
            key = str(self._generation[i])
            by_generation[key] = by_generation.get(key, 0) + 1
        return {
            "collections": self._count,
            "dropped": self.dropped,
            "by_generation": by_generation,
            "total_pause_s": float(sum(durations)),
            **{f"pause_{k}": v for k, v in _distribution(durations).items() if k != "n"},
        }


@dataclass
class ClientDiagnostics:
    """Runs the loop-lag and GC monitors for the duration of one replay."""

    loop: LoopLagMonitor
    gc: GCMonitor

    @classmethod
    def create(cls, *, timing_trace: bool = False) -> ClientDiagnostics:
        return cls(loop=LoopLagMonitor(timing_trace=timing_trace), gc=GCMonitor())

    def start(self, origin: float) -> None:
        self.gc.start(origin)
        self.loop.start(origin)

    async def stop(self) -> None:
        await self.loop.stop()
        self.gc.stop()

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"event_loop_lag_s": self.loop.summary(), "gc": self.gc.summary()}
        if self.loop.due_offsets_s is not None:
            result["timing_trace"] = {
                "clock": "perf_counter seconds relative to replay start",
                "loop_due_s": self.loop.due_offsets_s.tolist(),
                "loop_lag_s": self.loop.lags_s,
                "gc_start_gen_duration": self.gc.pauses(),
            }
        return result
