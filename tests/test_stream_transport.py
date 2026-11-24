from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from slogate.replay import ReplayConfig, RequestRecord, _record_chunk, replay
from slogate.workload import Request, Workload


def one() -> Workload:
    return Workload("transport", "m", 0.001, (Request("one", 0.0, 1, 4, "hello"),))


async def against(handler, **options):
    app = web.Application()
    app.router.add_post("/prefix/v1/chat/completions", handler)
    async with TestServer(app) as server:
        config = ReplayConfig(str(server.make_url("/prefix")), api_key="unit-test", **options)
        result = await replay(one(), config)
    return result.records[0]


def test_split_utf8_crlf_usage_only_and_nonroot_base():
    async def handler(request):
        assert request.headers["Authorization"] == "Bearer unit-test"
        assert (await request.json())["messages"] == [{"role": "user", "content": "hello"}]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        # One event is fragmented inside UTF-8; others are coalesced in one HTTP read.
        await resp.write(b': hello\r\ndata: {"choices":[{"delta":{"content":"\xe2')
        await asyncio.sleep(0.01)
        await resp.write(
            b'\x98\x83"}}],"usage":{"completion_tokens":2}}\r\n\r\n'
            b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}],'
            b'"usage":{"completion_tokens":3}}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":4,"prompt_tokens":1}}\r'
            b"data: [DONE]"
        )
        return resp

    r = asyncio.run(against(handler))
    assert r.success and r.finish_reason == "length" and r.status == 200
    assert r.chunk_tokens == [2, 1] and r.output_tokens == 4
    assert r.prompt_tokens == 1 and r.first_token_s == r.chunk_times_s[0]
    assert r.chunk_times_s[0] <= r.chunk_times_s[1] <= r.finished_s


@pytest.mark.parametrize(
    ("body", "status", "error", "phase"),
    [
        (b"data: []\n", 200, "InvalidSSE", "body"),
        (b'data: {"choices":4}\n', 200, "InvalidSSE", "body"),
        (b'data: {"choices":[{"delta":true}]}\n', 200, "InvalidSSE", "body"),
        (b"data: {broken}\n", 200, "InvalidSSE", "body"),
        (b"data: [DONE]\n", 200, "EmptyStream", "body"),
        (b'data: {"choices":[{"delta":{"content":"one"}}]}\n', 200, "IncompleteStream", "body"),
        (b"unavailable", 503, "HTTP503", "status"),
        (b"redirect", 307, "HTTP307", "status"),
    ],
)
def test_failures_never_become_success(body, status, error, phase):
    async def handler(request):
        return web.Response(body=body, status=status, headers={"Location": "/do-not-follow"})

    r = asyncio.run(against(handler))
    assert not r.success and (r.error, r.error_phase) == (error, phase)


def test_stalled_stream_keeps_read_timeout():
    async def handler(request):
        resp = web.StreamResponse()
        await resp.prepare(request)
        await resp.write(b": ready\n")
        await asyncio.sleep(0.2)
        return resp

    r = asyncio.run(against(handler, request_timeout_s=0.05))
    assert not r.success and (r.error, r.error_phase) == ("Timeout", "body")


def test_no_usage_falls_back_to_one_observed_token_per_content_chunk():
    r = RequestRecord("x", 0)
    for i in range(2):
        _record_chunk(r, json.dumps({"choices": [{"delta": {"content": "x"}}]}).encode(), float(i))
    assert r.output_tokens is None and r.chunk_tokens == [1, 1] and r.first_token_s == 0.0
