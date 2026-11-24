"""A CPU-only, OpenAI-compatible streaming engine with controllable latency.

It models time, not language: responses are filler tokens emitted on a configurable
schedule. Tests and CI use it to exercise the gate, and regressions can be injected
deterministically (for example ``extra_itl_s``) to check that the gate detects them.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from . import __version__

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError("the fake engine needs: pip install 'slogate[fake-engine]'") from exc


@dataclass(frozen=True)
class EngineProfile:
    """Latency model. Times are seconds; jitter is the sigma of a log-normal multiplier."""

    model: str = "fake-model"
    ttft_s: float = 0.050
    ttft_per_input_token_s: float = 0.0
    itl_s: float = 0.010
    jitter: float = 0.10
    load_slope: float = 0.0
    stall_probability: float = 0.0
    stall_s: float = 0.0
    extra_itl_s: float = 0.0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingChunks:
    """Encode the same OpenAI-style SSE envelopes for synthetic and recorded pacing."""

    def __init__(self, model: str, prompt_tokens: int, options: dict[str, Any]) -> None:
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.include_usage = bool(options.get("include_usage"))
        self.continuous = self.include_usage and bool(options.get("continuous_usage_stats"))

    def usage(self, completed: int) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completed,
            "total_tokens": self.prompt_tokens + completed,
        }

    def event(self, choices: list[dict[str, Any]], completed: int, include_usage: bool) -> bytes:
        data: dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }
        if include_usage:
            data["usage"] = self.usage(completed)
        return f"data: {json.dumps(data)}\n\n".encode()

    def chunk(self, delta: dict[str, Any], reason: str | None, completed: int) -> bytes:
        choice = {"index": 0, "delta": delta, "finish_reason": reason}
        return self.event([choice], completed, self.continuous)

    def finish(self, reason: str, completed: int) -> Iterator[bytes]:
        yield self.chunk({}, reason, completed)
        if self.include_usage:
            yield self.event([], completed, True)
        yield b"data: [DONE]\n\n"


class FakeEngine:
    def __init__(self, profile: EngineProfile | None = None) -> None:
        self.profile = profile or EngineProfile()
        self.active_streams = 0

    def _rng(self, body: dict[str, Any]) -> random.Random:
        if self.profile.seed is None:
            return random.Random()
        material = f"{self.profile.seed}:{json.dumps(body, sort_keys=True)}"
        return random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16], 16))

    def _multiplier(self, rng: random.Random) -> float:
        if self.profile.jitter <= 0:
            return 1.0
        sigma = self.profile.jitter
        return rng.lognormvariate(-0.5 * sigma * sigma, sigma)

    def _gap(self, rng: random.Random) -> float:
        p = self.profile
        load = 1.0 + p.load_slope * max(0, self.active_streams - 1)
        gap = p.itl_s * load * self._multiplier(rng) + p.extra_itl_s
        if p.stall_probability > 0 and rng.random() < p.stall_probability:
            gap += p.stall_s
        return gap

    async def _events(
        self, body: dict[str, Any], prompt_tokens: int, output_tokens: int, finish: str
    ) -> AsyncIterator[bytes]:
        p = self.profile
        rng = self._rng(body)
        wire = StreamingChunks(p.model, prompt_tokens, body.get("stream_options") or {})
        self.active_streams += 1
        try:
            yield wire.chunk({"role": "assistant", "content": ""}, None, 0)
            ttft = p.ttft_s + p.ttft_per_input_token_s * prompt_tokens
            await asyncio.sleep(max(0.0, ttft * self._multiplier(rng)))
            for index in range(1, output_tokens + 1):
                if index > 1:
                    await asyncio.sleep(max(0.0, self._gap(rng)))
                yield wire.chunk({"content": f" tok{index}"}, None, index)
            for event in wire.finish(finish, output_tokens):
                yield event
        finally:
            self.active_streams -= 1

    async def chat_completions(self, request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        messages = body.get("messages")
        max_tokens = body.get("max_tokens")
        if not isinstance(messages, list) or not messages:
            return JSONResponse({"error": {"message": "messages are required"}}, status_code=400)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            return JSONResponse({"error": {"message": "max_tokens must be >= 1"}}, status_code=400)
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        prompt_tokens = max(1, len(text.split()))
        if body.get("ignore_eos"):
            output_tokens, finish = max_tokens, "length"
        else:
            output_tokens = self._rng(body).randint(1, max_tokens)
            finish = "length" if output_tokens == max_tokens else "stop"
        if not body.get("stream"):
            return JSONResponse(
                {"error": {"message": "the fake engine only serves streaming requests"}},
                status_code=400,
            )
        return StreamingResponse(
            self._events(body, prompt_tokens, output_tokens, finish),
            media_type="text/event-stream",
        )

    async def models(self, request: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": self.profile.model, "object": "model", "owned_by": "slogate"}],
            }
        )

    async def health(self, request: Request) -> Response:
        return Response(status_code=200)

    async def version(self, request: Request) -> Response:
        return JSONResponse(
            {"version": f"slogate-fake-engine {__version__}", "profile": self.profile.to_dict()}
        )

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat_completions, methods=["POST"]),
                Route("/v1/models", self.models, methods=["GET"]),
                Route("/health", self.health, methods=["GET"]),
                Route("/version", self.version, methods=["GET"]),
            ]
        )


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def serve_in_thread(
    engine: FakeEngine, host: str = "127.0.0.1", port: int = 0, startup_timeout_s: float = 15.0
) -> Iterator[str]:
    """Run the engine on a background thread and yield its base URL."""
    import uvicorn

    port = port or free_port(host)
    config = uvicorn.Config(engine.app(), host=host, port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="slogate-fake-engine", daemon=True)
    thread.start()
    deadline = time.monotonic() + startup_timeout_s
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("fake engine exited during startup")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise TimeoutError("fake engine did not start in time")
        time.sleep(0.01)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)


def serve(engine: FakeEngine, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(engine.app(), host=host, port=port, log_level="warning", lifespan="off")
