"""The model client against a scripted HTTP transport: no network, no key.

httpx's own `MockTransport` plays the provider, so every case goes through the real client code:
request building, the bounded retry, the wall, structured-output fallback and error mapping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from changeorder import llm
from changeorder.config import LLMSettings
from changeorder.llm import (
    LLMError,
    LLMRateLimitedError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAICompatibleModel,
    Reply,
    ScriptedModel,
)

KEY = "sk-test-0123456789abcdef"
MESSAGES = [
    {"role": "system", "content": "rules"},
    {"role": "user", "content": "a confidential document"},
]
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def completion(text: str = '{"ok": true}', finish: str | None = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model-2026-01",
            "provider": "TestCloud",
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "cost": 0.0012},
        },
    )


class Provider:
    """A scripted provider: one response (or exception) per request, and a record of requests."""

    def __init__(
        self, *steps: httpx.Response | Exception | Callable[[dict[str, Any]], httpx.Response]
    ) -> None:
        self.steps = list(steps)
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step(body) if callable(step) else step


def model_for(
    provider: Provider, sleeps: list[float] | None = None, **settings: Any
) -> OpenAICompatibleModel:
    defaults: dict[str, Any] = {
        "api_key": KEY,
        "base_url": "https://llm.test/v1/",
        "model": "test-model",
    }
    return OpenAICompatibleModel(
        LLMSettings(**{**defaults, **settings}),
        transport=httpx.MockTransport(provider),
        sleep=(sleeps if sleeps is not None else []).append,
    )


# ---- The request and the reply ------------------------------------------------------------------


def test_one_call_sends_the_schema_strictly_and_returns_the_reply_with_its_usage():
    provider = Provider(completion())
    model = model_for(provider)

    reply = model.complete(MESSAGES, schema=SCHEMA)

    assert reply == Reply(
        text='{"ok": true}',
        finish_reason="stop",
        model="test-model-2026-01",
        provider="TestCloud",
        response_format="json_schema",
        prompt_tokens=120,
        completion_tokens=30,
        cost=0.0012,
    )
    (body,) = provider.bodies
    assert body["model"] == "test-model" and body["messages"] == MESSAGES
    assert body["max_tokens"] == 4000
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "temperature" not in body  # not sent unless configured: some models refuse it
    assert provider.headers[0]["authorization"] == f"Bearer {KEY}"
    assert model.describe()["http_requests"] == 1
    assert KEY not in json.dumps(model.describe())


def test_optional_settings_reach_the_request():
    provider = Provider(completion())
    model_for(provider, temperature=0.0, max_output_tokens=900, structured_output="none").complete(
        MESSAGES, schema=SCHEMA
    )

    (body,) = provider.bodies
    assert body["temperature"] == 0.0 and body["max_tokens"] == 900
    assert "response_format" not in body


def test_content_given_in_parts_is_joined_and_missing_content_is_empty():
    parts = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"a":'},
                            {"type": "text", "text": " 1}"},
                        ]
                    }
                }
            ]
        },
    )
    nothing = httpx.Response(
        200, json={"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
    )

    assert model_for(Provider(parts)).complete(MESSAGES).text == '{"a": 1}'
    refused = model_for(Provider(nothing)).complete(MESSAGES)
    assert refused.text == "" and refused.finish_reason == "content_filter"


# ---- Structured output that degrades in the open --------------------------------------------------


def rejects(*formats: str) -> Callable[[dict[str, Any]], httpx.Response]:
    """A provider that answers 400 to the given response formats and completes anything else."""

    def respond(body: dict[str, Any]) -> httpx.Response:
        kind = body.get("response_format", {}).get("type", "none")
        if kind in formats:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "response_format is not supported",
                        "code": "unsupported_parameter",
                    }
                },
            )
        return completion()

    return respond


@pytest.mark.parametrize(
    ("unsupported", "used", "requests"),
    [
        ((), "json_schema", 1),
        (("json_schema",), "json_object", 2),
        (("json_schema", "json_object"), "none", 3),
    ],
)
def test_a_provider_without_strict_schemas_is_asked_again_with_a_weaker_format(
    unsupported, used, requests, caplog
):
    provider = Provider(rejects(*unsupported))

    reply = model_for(provider).complete(MESSAGES, schema=SCHEMA)

    assert reply.response_format == used  # recorded, so a run says which mode produced it
    assert len(provider.bodies) == requests
    assert len([r for r in caplog.records if "response format" in r.getMessage()]) == requests - 1


def test_the_fallback_is_bounded_and_can_be_switched_off():
    always_400 = Provider(httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}))
    with pytest.raises(LLMError, match="HTTP 400, error code context_length_exceeded"):
        model_for(always_400).complete(MESSAGES, schema=SCHEMA)
    assert len(always_400.bodies) == 3  # one per format, never more

    pinned = Provider(rejects("json_schema"))
    with pytest.raises(LLMError, match="HTTP 400"):
        model_for(pinned, structured_output="json_schema").complete(MESSAGES, schema=SCHEMA)
    assert len(pinned.bodies) == 1


# ---- Failures a retry can fix, and failures it cannot -----------------------------------------------


@pytest.mark.parametrize(
    ("first", "wait"),
    [
        (httpx.Response(429, headers={"retry-after": "3"}), 3.0),
        (httpx.Response(429, headers={"retry-after-ms": "250"}), 0.25),
        (httpx.Response(503), 2.0),  # no wait named: a short default
        (httpx.Response(500), 2.0),
        (httpx.ConnectError("connection refused"), 1.0),
        (httpx.RemoteProtocolError("server disconnected"), 1.0),
    ],
)
def test_a_transient_failure_gets_exactly_one_retry_after_the_wait_asked_for(first, wait):
    provider, sleeps = Provider(first, completion()), []

    reply = model_for(provider, sleeps).complete(MESSAGES)

    assert reply.text == '{"ok": true}'
    assert sleeps == [wait] and len(provider.bodies) == 2


@pytest.mark.parametrize(
    ("response", "error", "requests"),
    [
        (httpx.Response(429), LLMRateLimitedError, 2),
        (httpx.Response(503), LLMRateLimitedError, 2),
        (httpx.Response(500), LLMError, 2),
        (
            httpx.Response(429, headers={"retry-after": "600"}),
            LLMRateLimitedError,
            1,
        ),  # too long to wait
        (httpx.Response(401), LLMUnavailableError, 1),
        (httpx.Response(403), LLMUnavailableError, 1),
        (httpx.Response(402), LLMUnavailableError, 1),
        (
            httpx.Response(429, json={"error": {"code": "insufficient_quota"}}),
            LLMUnavailableError,
            2,
        ),
        (httpx.Response(404), LLMUnavailableError, 1),
        (httpx.Response(400), LLMError, 1),
    ],
)
def test_a_failure_that_stays_is_raised_as_what_it_is(response, error, requests):
    provider = Provider(response)

    with pytest.raises(error):
        model_for(provider, structured_output="none").complete(MESSAGES)

    assert len(provider.bodies) == requests


def test_a_timeout_is_raised_at_once_and_never_retried():
    provider, sleeps = Provider(httpx.ReadTimeout("no bytes")), []

    with pytest.raises(LLMTimeoutError, match="LLM_TIMEOUT_SECONDS"):
        model_for(provider, sleeps, timeout=5).complete(MESSAGES)

    assert len(provider.bodies) == 1 and sleeps == []


def test_two_dropped_connections_end_the_call():
    provider = Provider(httpx.ConnectError("https://llm.test/v1/chat/completions refused"))

    with pytest.raises(LLMError) as raised:
        model_for(provider).complete(MESSAGES)

    assert len(provider.bodies) == 2
    assert "ConnectError" in str(raised.value) and "llm.test" not in str(raised.value)


def test_no_retry_is_made_that_would_end_after_the_wall():
    # The clock says 28 of the 30 seconds are gone when the 429 arrives; the wait asked for is 5.
    times = iter([0.0, 0.0, 28.0, 28.0, 28.0])
    provider, sleeps = Provider(httpx.Response(429, headers={"retry-after": "5"})), []
    model = OpenAICompatibleModel(
        LLMSettings(
            api_key=KEY,
            base_url="https://llm.test/v1",
            model="m",
            timeout=30,
            structured_output="none",
        ),
        transport=httpx.MockTransport(provider),
        sleep=sleeps.append,
        clock=lambda: next(times, 28.0),
    )

    with pytest.raises(LLMRateLimitedError):
        model.complete(MESSAGES)

    assert sleeps == [] and len(provider.bodies) == 1


def test_a_call_that_starts_after_its_wall_makes_no_request():
    times = iter([0.0, 31.0])
    provider = Provider(completion())
    model = OpenAICompatibleModel(
        LLMSettings(api_key=KEY, base_url="https://llm.test/v1", model="m", timeout=30),
        transport=httpx.MockTransport(provider),
        clock=lambda: next(times, 31.0),
    )

    with pytest.raises(LLMTimeoutError):
        model.complete(MESSAGES)
    assert provider.bodies == []


# ---- Replies that are not replies -----------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>gateway error</html>"),
        httpx.Response(
            200, json={"error": {"message": "upstream overloaded", "code": "overloaded"}}
        ),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
def test_a_200_that_is_not_a_completion_is_an_error(response):
    with pytest.raises(LLMResponseError):
        model_for(Provider(response)).complete(MESSAGES)


def test_an_oversized_response_is_refused(monkeypatch):
    monkeypatch.setattr(llm, "_MAX_RESPONSE_BYTES", 64)

    with pytest.raises(LLMResponseError, match="oversized"):
        model_for(Provider(completion("x" * 500))).complete(MESSAGES)


def test_no_error_message_carries_the_key_the_request_or_the_providers_text():
    # A provider can echo the request, even the key, in an error body.
    echo = {
        "error": {
            "message": f"Invalid key {KEY}. You sent: a confidential document",
            "code": f"{KEY}",
        }
    }
    for status in (401, 400, 429, 500):
        provider = Provider(httpx.Response(status, json=echo))
        with pytest.raises(LLMError) as raised:
            model_for(provider, structured_output="none").complete(MESSAGES)
        message = str(raised.value)
        assert KEY not in message and "confidential" not in message and "Invalid key" not in message
        assert raised.value.__cause__ is None  # nothing of the response rides along in a traceback


def test_a_client_without_a_key_is_refused_before_any_call():
    with pytest.raises(LLMUnavailableError):
        OpenAICompatibleModel(LLMSettings(api_key="", base_url="https://llm.test/v1", model="m"))


# ---- The scripted model the tests and --replay use --------------------------------------------------


def test_the_scripted_model_answers_in_order_records_what_it_was_shown_and_fails_when_it_runs_out():
    model = ScriptedModel(
        ["first", Reply(text="second", finish_reason="length"), LLMTimeoutError("slow")]
    )

    assert model.complete(MESSAGES, schema=SCHEMA).text == "first"
    assert model.complete(MESSAGES).finish_reason == "length"
    with pytest.raises(LLMTimeoutError):
        model.complete(MESSAGES)
    with pytest.raises(LLMError, match="no reply left"):
        model.complete(MESSAGES)

    assert model.calls[0] == MESSAGES and model.schemas[0] == SCHEMA
    assert len(model.calls) == 4
