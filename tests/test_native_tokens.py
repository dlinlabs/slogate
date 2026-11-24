"""Pinned vLLM completion wire contract, tested with invented HTTP responses (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from slogate.native_tokens import EOS_IDS, MODEL, observed_cases, read_batch, suite_cases

SUITE = Path(__file__).resolve().parents[1] / "configs/correctness/qwen3-small-budget.json"
CASES = [{"id": "first", "prompt_ids": [10, 11]}, {"id": "second", "prompt_ids": [10, 12, 13]}]


def reply(body: dict, *, delta: int = 0) -> dict:
    """Deliberately misleading rendered text: IDs must come directly from each choice."""
    assert body["return_token_ids"] is True and body["add_special_tokens"] is False
    assert body["n"] == 1 and not body["ignore_eos"] and not body["stream"]
    assert body["temperature"] == 0 and body["echo"] is False and body["model"] == MODEL
    choices = [
        {
            "index": i,
            "prompt_token_ids": p,
            "token_ids": [p[-1] + delta, EOS_IDS[0]],
            "text": "not a source of token IDs",
            "finish_reason": "stop",
            "stop_reason": None,
        }
        for i, p in enumerate(body["prompt"])
    ]
    prompt = sum(len(c["prompt_token_ids"]) for c in choices)
    completion = sum(len(c["token_ids"]) for c in choices)
    return {
        "model": MODEL,
        "object": "text_completion",
        "choices": list(reversed(choices)),
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def test_frozen_inputs_span_both_scheduler_budgets():
    raw = json.loads(SUITE.read_bytes())
    assert [len(c["prompt_ids"]) for c in suite_cases(raw)] == [10, 34, 552, 2081]
    assert raw["max_new_tokens"] == 24
    for key, bad in (
        ("revision", "main"),
        ("tokenizer_json_sha256", "latest"),
        ("max_new_tokens", True),
    ):
        with pytest.raises(ValueError):
            suite_cases({**raw, key: bad})
    raw["cases"][0]["repeat"] = [False]
    with pytest.raises(ValueError):
        suite_cases(raw)


def test_singletons_and_reordered_multi_prompt_use_native_ids():
    batches = []

    def endpoint(req):
        assert req.method == "POST" and req.url.path == "/v1/completions"
        body = json.loads(req.content)
        batches.append(body["prompt"])
        return httpx.Response(200, json=reply(body))

    with httpx.Client(
        transport=httpx.MockTransport(endpoint), base_url="http://127.0.0.1"
    ) as client:
        result = observed_cases(client, CASES, 4)
    assert batches == [[[10, 11]], [[10, 12, 13]], [[10, 12, 13], [10, 11]]]
    assert [r["output_ids"] for r in result] == [[11, 151645], [13, 151645]]
    assert all(r["complete"] and "text" not in r for r in result)


@pytest.mark.parametrize(
    "bad",
    [
        "no_ids",
        "text_only",
        "wrong_prompt",
        "duplicate_index",
        "boolean_id",
        "over_cap",
        "early_length",
        "bad_eos",
        "custom_stop",
        "missing_choice",
        "usage",
        "wrong_model",
        "bad_http",
        "out_of_vocab",
        "no_prompt_echo",
        "early_eos",
    ],
)
def test_bad_native_observations_never_pass(bad):
    def endpoint(req):
        data = reply(json.loads(req.content))
        c = data["choices"][0]
        if bad == "no_ids":
            del c["token_ids"]
        if bad == "text_only":
            c["token_ids"] = ["tok1"]
        if bad == "wrong_prompt":
            c["prompt_token_ids"] = [5]
        if bad == "duplicate_index":
            c["index"] = data["choices"][1]["index"]
        if bad == "boolean_id":
            c["token_ids"][0] = True
        if bad == "over_cap":
            c["token_ids"] = [1] * 5
        if bad == "early_length":
            c["finish_reason"] = "length"
        if bad == "bad_eos":
            c["token_ids"][-1] = 9
        if bad == "custom_stop":
            c["stop_reason"] = "stop text"
        if bad == "missing_choice":
            data["choices"].pop()
        if bad == "usage":
            data["usage"]["completion_tokens"] += 1
        if bad == "wrong_model":
            data["model"] = "another"
        if bad == "out_of_vocab":
            c["token_ids"][0] = 151936
        if bad == "no_prompt_echo":
            del c["prompt_token_ids"]
        if bad == "early_eos":
            c["token_ids"][0] = 151645
        return httpx.Response(500 if bad == "bad_http" else 200, json=data)

    with (
        httpx.Client(
            transport=httpx.MockTransport(endpoint), base_url="http://127.0.0.1"
        ) as client,
        pytest.raises(ValueError),
    ):
        read_batch(client, CASES, 4)


def test_cap_length_and_batch_invariance_precondition():
    def endpoint(req):
        body = json.loads(req.content)
        many = len(body["prompt"]) > 1
        data = reply(body, delta=int(many))
        for c in data["choices"]:
            c["token_ids"][-1] = 7
            c["finish_reason"] = "length"
        return httpx.Response(200, json=data)

    with httpx.Client(
        transport=httpx.MockTransport(endpoint), base_url="http://127.0.0.1"
    ) as client:
        assert read_batch(client, CASES, 2)[0]["output_ids"] == [12, 7]
        with pytest.raises(ValueError, match="singleton and mixed-batch"):
            observed_cases(client, CASES, 2)
