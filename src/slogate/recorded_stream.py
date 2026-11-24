"""Loopback diagnostic fixture: replay *client-observed* timings, not GPU computation."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .fake_engine import EngineProfile, FakeEngine, StreamingChunks
from .replay import FIXTURE_EPOCH_HEADER
from .report import RECORDS_FILE, read_records, read_run
from .workload import Workload


@dataclass(frozen=True)
class Schedule:
    source_launch_s: float
    header_s: float
    content_s: tuple[float, ...]
    widths: tuple[int, ...]
    end_s: float
    prompt_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass
class RecordedStreams:
    source: str
    records_sha256: str
    by_payload: dict[tuple[str, int], Schedule]
    shared_clock: bool = False

    @classmethod
    def load(cls, folder: Path, workload: Workload) -> RecordedStreams:
        if read_run(folder)["meta"]["workload"]["digest"] != workload.digest:
            raise ValueError(f"{folder.name}: workload digest mismatch")
        with (folder / RECORDS_FILE).open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return cls.from_rows(folder.name, digest, workload, read_records(folder))

    @classmethod
    def from_rows(
        cls, source: str, digest: str, workload: Workload, rows: Sequence[dict[str, Any]]
    ) -> RecordedStreams:
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows) or by_id.keys() != {r.id for r in workload.requests}:
            raise ValueError(f"{source}: duplicate or missing request identity")
        schedules = {}
        for req in workload.requests:
            r = by_id[req.id]
            key = (req.prompt, req.max_tokens)
            if key in schedules:
                raise ValueError(f"{source}: ambiguous payload identity")
            times = r["chunk_times_s"]
            widths = r["chunk_tokens"]
            launch, headers, end = r["launched_s"], r["headers_s"], r["finished_s"]
            if (
                not r["success"]
                or r["scheduled_s"] != req.arrival_s
                or not times
                or r["first_token_s"] != times[0]
                or len(times) != len(widths)
                or not all(type(n) is int and n > 0 for n in widths)
                or r["output_tokens"] != req.max_tokens
                or sum(widths) > req.max_tokens
                or type(r["prompt_tokens"]) is not int
                or r["prompt_tokens"] < 0
                or not isinstance(r["finish_reason"], str)
            ):
                raise ValueError(f"{source}: incomplete or inconsistent stream for {req.id}")
            stamps = [launch, headers, *times, end]
            if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in stamps) or any(
                b < a for a, b in pairwise(stamps)
            ):
                raise ValueError(f"{source}: nonfinite or unordered clock for {req.id}")
            schedules[key] = Schedule(
                launch,
                headers - launch,
                tuple(t - launch for t in times),
                tuple(widths),
                end - launch,
                r["prompt_tokens"],
                r["output_tokens"],
                r["finish_reason"],
            )
        return cls(source, digest, schedules)


async def _until(due: float) -> None:
    delay = due - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)


class RecordedStreamEngine(FakeEngine):
    def __init__(self, profile: EngineProfile, fixture: RecordedStreams) -> None:
        super().__init__(profile)
        self.fixture = fixture
        self._epoch: float | None = None

    async def version(self, request: Request) -> Response:
        return JSONResponse(
            {
                "version": "slogate-recorded-content-timing-v1",
                "source": self.fixture.source,
                "source_records_sha256": self.fixture.records_sha256,
                "anchor": "shared" if self.fixture.shared_clock else "request",
            }
        )

    async def chat_completions(self, request: Request) -> Response:
        origin = time.perf_counter()
        body = await request.json()
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
        ):
            return JSONResponse({"error": "one user prompt required"}, status_code=400)
        prompt = messages[0].get("content")
        cap = body.get("max_tokens")
        if (
            not isinstance(prompt, str)
            or type(cap) is not int
            or messages[0].get("role") != "user"
            or not body.get("stream")
            or not body.get("ignore_eos")
        ):
            return JSONResponse({"error": "recorded streaming workload required"}, status_code=400)
        plan = self.fixture.by_payload.get((prompt, cap))
        if plan is None:
            return JSONResponse({"error": "no matching recorded request"}, status_code=400)
        stamp = request.headers.get(FIXTURE_EPOCH_HEADER)
        if self.fixture.shared_clock:
            if not (stamp == "warmup" and self._epoch is None):
                try:
                    epoch = float(stamp) if stamp is not None else math.nan
                except ValueError:
                    epoch = math.nan
                if not 0 < epoch <= origin or not math.isfinite(epoch):
                    return JSONResponse({"error": "invalid local replay epoch"}, status_code=400)
                if self._epoch is not None and self._epoch != epoch:
                    return JSONResponse({"error": "inconsistent replay epoch"}, status_code=400)
                self._epoch = epoch
                origin = epoch + plan.source_launch_s
        elif stamp is not None:
            return JSONResponse({"error": "fixture uses a request-relative clock"}, status_code=400)
        wire = StreamingChunks(
            self.profile.model, plan.prompt_tokens, body.get("stream_options") or {}
        )
        await _until(origin + plan.header_s)
        return StreamingResponse(self._emit(origin, plan, wire), media_type="text/event-stream")

    async def _emit(
        self, origin: float, plan: Schedule, wire: StreamingChunks
    ) -> AsyncIterator[bytes]:
        yield wire.chunk({"role": "assistant", "content": ""}, None, 0)
        completed = 0
        for offset, width in zip(plan.content_s, plan.widths, strict=True):
            await _until(origin + offset)
            completed += width
            yield wire.chunk({"content": " tok" * width}, None, completed)
        await _until(origin + plan.end_s)
        # Final usage can exceed the visible content count. Do not invent missing token times.
        for event in wire.finish(plan.finish_reason, plan.output_tokens):
            yield event
