"""Open-loop replay of a workload against an OpenAI-compatible streaming endpoint.

Requests are launched on the workload's arrival schedule and never wait for earlier
responses, so server slowdowns show up as latency instead of silently reducing offered load.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .diagnostics import ClientDiagnostics
from .workload import Request, Workload

START_LEAD_S = 0.5
FIXTURE_EPOCH_HEADER = "X-Slogate-Fixture-Epoch"
_wire_json = partial(json.dumps, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(slots=True)
class RequestRecord:
    """Client-observed timeline of one request; times are seconds since replay start."""

    id: str
    scheduled_s: float
    launched_s: float | None = None
    headers_s: float | None = None
    first_token_s: float | None = None
    finished_s: float | None = None
    status: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    chunk_times_s: list[float] = field(default_factory=list)
    chunk_tokens: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    error_phase: str | None = None
    success: bool = False

    @property
    def launch_lag_s(self) -> float | None:
        return None if self.launched_s is None else self.launched_s - self.scheduled_s

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None or self.launched_s is None:
            return None
        return self.first_token_s - self.launched_s

    @property
    def e2e_s(self) -> float | None:
        if self.finished_s is None or self.launched_s is None:
            return None
        return self.finished_s - self.launched_s

    @property
    def streamed_tokens(self) -> int:
        return sum(self.chunk_tokens)

    def token_gaps_s(self) -> list[float]:
        """Per-token inter-token latencies.

        A chunk that carries ``k`` tokens after a gap ``g`` contributes ``k`` samples of
        ``g / k``. With one token per chunk this is the plain gap between chunks.
        """
        gaps: list[float] = []
        for i in range(1, len(self.chunk_times_s)):
            gap = self.chunk_times_s[i] - self.chunk_times_s[i - 1]
            tokens = max(1, self.chunk_tokens[i])
            gaps.extend([gap / tokens] * tokens)
        return gaps

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first one."""
        tokens = self.output_tokens if self.output_tokens is not None else self.streamed_tokens
        if self.first_token_s is None or not self.chunk_times_s or tokens < 2:
            return None
        return (self.chunk_times_s[-1] - self.first_token_s) / (tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scheduled_s": self.scheduled_s,
            "launched_s": self.launched_s,
            "headers_s": self.headers_s,
            "first_token_s": self.first_token_s,
            "finished_s": self.finished_s,
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "chunk_times_s": self.chunk_times_s,
            "chunk_tokens": self.chunk_tokens,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "error_phase": self.error_phase,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RequestRecord:
        return cls(**{name: data[name] for name in cls.__slots__ if name in data})


@dataclass(frozen=True)
class ReplayConfig:
    base_url: str
    model: str | None = None
    api_key: str | None = None
    path: str = "/v1/chat/completions"
    temperature: float = 0.0
    ignore_eos: bool = True
    request_timeout_s: float = 300.0
    drain_timeout_s: float = 300.0
    max_in_flight: int = 4096
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    local_fixture_clock: bool = False

    def __post_init__(self) -> None:
        if self.local_fixture_clock:
            url = urlsplit(self.base_url)
            if url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"}:
                raise ValueError("shared fixture clock requires an HTTP loopback address")


@dataclass
class ReplayResult:
    workload_name: str
    workload_digest: str
    model: str
    duration_s: float
    started_unix_s: float
    wall_s: float
    records: list[RequestRecord]
    in_flight_limit_hit: bool
    diagnostics: dict[str, Any] | None

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "workload_name": self.workload_name,
            "workload_digest": self.workload_digest,
            "model": self.model,
            "duration_s": self.duration_s,
            "started_unix_s": self.started_unix_s,
            "wall_s": self.wall_s,
            "in_flight_limit_hit": self.in_flight_limit_hit,
            "diagnostics": self.diagnostics,
        }
        if include_records:
            data["records"] = [record.to_dict() for record in self.records]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayResult:
        return cls(
            workload_name=data["workload_name"],
            workload_digest=data["workload_digest"],
            model=data["model"],
            duration_s=data["duration_s"],
            started_unix_s=data["started_unix_s"],
            wall_s=data["wall_s"],
            records=[RequestRecord.from_dict(r) for r in data.get("records", [])],
            in_flight_limit_hit=data["in_flight_limit_hit"],
            diagnostics=data.get("diagnostics"),
        )


def build_payload(request: Request, model: str, config: ReplayConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "temperature": config.temperature,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "ignore_eos": config.ignore_eos,
    }
    payload.update(config.extra_body)
    return payload


async def _sse_data(reader: aiohttp.StreamReader) -> AsyncIterator[bytes]:
    """OpenAI streams use one JSON data line per event. Ignore blank and comment lines."""
    pending = bytearray()
    async for block in reader.iter_any():
        for part in block.splitlines(keepends=True):
            if not part.endswith((b"\n", b"\r")):
                pending.extend(part)
                continue
            if pending:
                pending.extend(part)
                part = bytes(pending)
                pending.clear()
            if part.startswith(b"data:"):
                yield part[5:].strip()
    if pending.startswith(b"data:"):
        yield bytes(pending[5:]).strip()


def _record_chunk(record: RequestRecord, data: bytes, now: float) -> None:
    # Keep the decoded JSON tree out of the suspended per-stream reader frame.
    event = json.loads(data)
    if not isinstance(event, dict):
        raise ValueError("expected an SSE JSON object")
    choices = event.get("choices")
    if choices is not None and not isinstance(choices, list):
        raise ValueError("invalid SSE choices")
    has_content = False
    for choice in choices or ():
        if not isinstance(choice, dict):
            raise ValueError("invalid SSE choice")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ValueError("invalid SSE delta")
        has_content |= bool(delta.get("content"))
        if choice.get("finish_reason"):
            record.finish_reason = choice["finish_reason"]
    usage = event.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("invalid SSE usage")
    reported = usage.get("completion_tokens") if usage else None
    if reported is not None and (type(reported) is not int or reported < 0):
        raise ValueError("invalid SSE token count")
    if has_content:
        previous = record.output_tokens or 0
        tokens = reported - previous if reported is not None and reported > previous else 1
        record.chunk_times_s.append(now)
        record.chunk_tokens.append(tokens)
        if record.first_token_s is None:
            record.first_token_s = now
    if usage:
        record.prompt_tokens = usage.get("prompt_tokens", record.prompt_tokens)
        if reported is not None:
            record.output_tokens = reported


async def _send(
    client: aiohttp.ClientSession,
    request: Request,
    record: RequestRecord,
    model: str,
    config: ReplayConfig,
    origin: float,
) -> None:
    clock = time.perf_counter
    record.launched_s = clock() - origin
    try:
        async with client.post(
            config.path.lstrip("/"),
            json=build_payload(request, model, config),
            allow_redirects=False,
        ) as response:
            record.headers_s = clock() - origin
            record.status = response.status
            if response.status != 200:
                async for _ in response.content.iter_chunked(65_536):
                    pass
                record.error, record.error_phase = f"HTTP{response.status}", "status"
                return
            async for data in _sse_data(response.content):
                if data == b"[DONE]":
                    if record.first_token_s is None:
                        record.error, record.error_phase = "EmptyStream", "body"
                    else:
                        record.success = True
                    return
                try:
                    _record_chunk(record, data, clock() - origin)
                except ValueError:  # Malformed JSON, UTF-8 or OpenAI event shape.
                    record.error, record.error_phase = "InvalidSSE", "body"
                    return
            record.error, record.error_phase = "IncompleteStream", "body"
    except TimeoutError:
        record.error = "Timeout"
        record.error_phase = "headers" if record.headers_s is None else "body"
    except aiohttp.ClientError as exc:
        record.error = type(exc).__name__
        record.error_phase = "headers" if record.headers_s is None else "body"
    except asyncio.CancelledError:
        record.error = "Cancelled"
        record.error_phase = "headers" if record.headers_s is None else "body"
        raise
    finally:
        record.finished_s = clock() - origin


async def replay(
    workload: Workload,
    config: ReplayConfig,
    *,
    diagnostics: bool = True,
    client_timing: bool = False,
) -> ReplayResult:
    """Offer every request at its scheduled time and collect client-side timelines."""
    model = config.model or workload.model
    records = [RequestRecord(id=r.id, scheduled_s=r.arrival_s) for r in workload.requests]
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    timeout = aiohttp.ClientTimeout(
        total=None, sock_connect=30.0, sock_read=config.request_timeout_s
    )
    if client_timing and not diagnostics:
        raise ValueError("client timing capture requires diagnostics")
    monitors = ClientDiagnostics.create(timing_trace=client_timing) if diagnostics else None
    clock = time.perf_counter
    active: set[asyncio.Task[None]] = set()
    in_flight_limit_hit = False

    # The replay scheduler owns the in-flight limit. A second connector queue
    # would silently pace the workload and turn client waiting time into apparent engine latency.
    async with aiohttp.ClientSession(
        base_url=config.base_url.rstrip("/") + "/",
        headers=headers,
        connector=aiohttp.TCPConnector(limit=0),
        timeout=timeout,
        trust_env=False,
        json_serialize=_wire_json,
    ) as client:
        started_unix = time.time() + START_LEAD_S
        origin = clock() + START_LEAD_S
        if config.local_fixture_clock:
            client.headers[FIXTURE_EPOCH_HEADER] = repr(origin)
        if monitors is not None:
            monitors.start(origin)
        try:
            for request, record in zip(workload.requests, records, strict=True):
                delay = origin + request.arrival_s - clock()
                if delay > 0:
                    await asyncio.sleep(delay)
                if len(active) >= config.max_in_flight:
                    in_flight_limit_hit = True
                    record.error, record.error_phase = "NotLaunched", "client"
                    continue
                task = asyncio.create_task(_send(client, request, record, model, config, origin))
                active.add(task)
                task.add_done_callback(active.discard)
            await asyncio.sleep(max(0.0, origin + workload.duration_s - clock()))
            if active:
                _, pending = await asyncio.wait(set(active), timeout=config.drain_timeout_s)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            remaining = list(active)
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
            if monitors is not None:
                await monitors.stop()
        wall = clock() - origin

    return ReplayResult(
        workload_name=workload.name,
        workload_digest=workload.digest,
        model=model,
        duration_s=workload.duration_s,
        started_unix_s=started_unix,
        wall_s=wall,
        records=records,
        in_flight_limit_hit=in_flight_limit_hit,
        diagnostics=monitors.summary() if monitors is not None else None,
    )
