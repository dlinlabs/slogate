"""Strict reader of vLLM 0.30.0 nonstreaming completion choices. No text retokenization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .paired_plan import require

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKENIZER = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
VOCAB_SIZE = 151936
EOS_IDS = (151645, 151643)


def ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in value)
    )


def suite_cases(suite: dict) -> list[dict]:
    require(
        suite["schema"] == "qwen3-token-suite-v1"
        and suite["model"] == MODEL
        and suite["revision"] == REVISION
        and suite["tokenizer_json_sha256"] == TOKENIZER,
        "wrong token-suite schema, model or tokenizer revision",
    )
    cap = suite["max_new_tokens"]
    require(type(cap) is int and 1 <= cap <= 128, "invalid exact-check output cap")
    rows = []
    names: set[str] = set()
    for c in suite["cases"]:
        name = c["id"]
        require(
            isinstance(name, str)
            and name.isascii()
            and name.replace("-", "").isalnum()
            and name not in names,
            "invalid or duplicate case ID",
        )
        names.add(name)
        n = c["times"]
        require(
            all(ids(c[k]) for k in ("prefix", "repeat", "suffix"))
            and type(n) is int
            and 1 <= n <= 8192,
            "invalid token input or repeat count",
        )
        length = len(c["prefix"]) + len(c["repeat"]) * n + len(c["suffix"])
        require(length + cap <= 8192, "case does not fit the frozen context length")
        rows.append({"id": name, "prompt_ids": c["prefix"] + c["repeat"] * n + c["suffix"]})
    require(2 <= len(rows) <= 8, "expected a small multi-case suite")
    return rows


def json_reply(response: httpx.Response) -> dict:
    require(response.status_code == 200, f"vLLM request returned HTTP {response.status_code}")
    data = response.json()
    require(isinstance(data, dict), "expected a JSON object from vLLM")
    return data


def read_batch(client: httpx.Client, cases: list[dict], cap: int) -> list[dict]:
    prompts = [c["prompt_ids"] for c in cases]
    out = json_reply(
        client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompts,
                "n": 1,
                "temperature": 0,
                "max_tokens": cap,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "min_tokens": 0,
                "seed": 0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": False,
                "stream": False,
                "return_token_ids": True,
                "add_special_tokens": False,
                "echo": False,
                "watermarking": False,
            },
        )
    )
    choices = out.get("choices")
    require(
        out.get("object") == "text_completion"
        and out.get("model") == MODEL
        and isinstance(choices, list)
        and len(choices) == len(cases),
        "wrong model/choice count",
    )
    indexed = {}
    for choice in choices:
        require(isinstance(choice, dict), "invalid completion choice")
        i = choice.get("index")
        require(type(i) is int and 0 <= i < len(cases) and i not in indexed, "invalid choice index")
        p, t = choice.get("prompt_token_ids"), choice.get("token_ids")
        require(
            ids(p) and p == prompts[i], "native prompt IDs missing or differ from the frozen input"
        )
        require(ids(t) and 1 <= len(t) <= cap, "native generated token IDs missing or invalid")
        require(
            not any(token in EOS_IDS for token in t[:-1]),
            "output continues after a termination token",
        )
        reason = choice.get("finish_reason")
        require(
            (reason == "length" and len(t) == cap and choice.get("stop_reason") is None)
            or (
                reason == "stop"
                and t[-1] in EOS_IDS
                and (
                    choice.get("stop_reason") is None
                    or (type(choice["stop_reason"]) is int and choice["stop_reason"] == t[-1])
                )
            ),
            "incomplete generation or unexpected stopping condition",
        )
        indexed[i] = {"id": cases[i]["id"], "prompt_ids": p, "output_ids": t, "complete": True}
    usage = out.get("usage")
    counts = {
        "prompt_tokens": sum(map(len, prompts)),
        "completion_tokens": sum(len(row["output_ids"]) for row in indexed.values()),
    }
    counts["total_tokens"] = sum(counts.values())
    require(
        isinstance(usage, dict)
        and all(type(usage.get(k)) is int and usage[k] == v for k, v in counts.items()),
        "native ID lengths disagree with final usage",
    )
    return [indexed[i] for i in range(len(cases))]


def observed_cases(
    client: httpx.Client,
    cases: list[dict],
    cap: int,
    *,
    save: Callable[[str, list[dict]], None] | None = None,
) -> list[dict]:
    # A live singleton-vs-batched control supplements the runtime/driver checks; this is
    # evidence for these inputs only, not a claim about all Qwen3 models or configurations.
    singles = []
    for c in cases:
        singles.extend(read_batch(client, [c], cap))
        if save is not None:
            save("singleton-ids", singles)
    mixed = list(reversed(read_batch(client, list(reversed(cases)), cap)))
    if save is not None:
        save("mixed-batch-ids", mixed)
    for one, many in zip(singles, mixed, strict=True):
        require(one == many, f"{one['id']}: singleton and mixed-batch outputs differ")
    return mixed
