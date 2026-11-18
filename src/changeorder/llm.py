"""The language model as the pipeline sees it: messages in, one reply out, or a named failure.

`OpenAICompatibleModel` speaks the chat-completions format that OpenAI, OpenRouter, vLLM, Ollama
and most model servers accept. Everything specific to that format is in this file.

One request has one wall: `LLMSettings.timeout` seconds for everything, retries included. The
response is read as a stream and the clock is checked between chunks, so a server that trickles
bytes cannot hold the call open past the wall. Within it there is at most one retry, and only for
a failure a retry can fix: a dropped connection, or a 429 or 5xx, after the wait the server asks
for (20 seconds at most). A refused key, a missing model and a timeout are raised at once.

Structured output degrades in the open. With `structured_output="auto"` the request first carries
a strict `json_schema` response format. A provider that rejects the request (400, 404, 415, 422,
501) is asked again with `json_object`, then with no response format at all; the prompt always
describes the JSON, so the reply can be read in every mode. Each step down is logged and the mode
that produced the reply is recorded with it. A rejected request generates nothing, so the steps
cost no tokens. Whatever the mode, the reply is validated by code afterwards.

No error message ever holds the key, the request or the response body: a provider can quote the
document back in an error, and a log line should not carry it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from .config import LLMSettings

__all__ = [
    "ChatModel",
    "LLMError",
    "LLMRateLimitedError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "OpenAICompatibleModel",
    "Reply",
    "ScriptedModel",
]

log = logging.getLogger("changeorder.llm")

Message = dict[str, str]

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# A request the provider rejects outright. With a response format on it, the format is the first
# suspect, so the next weaker one is tried.
_DOWNGRADE_STATUSES = frozenset({400, 404, 415, 422, 501})
_QUOTA_CODES = frozenset({"insufficient_quota", "quota_exceeded", "insufficient_credits"})
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_RETRY_WAIT = 20.0
# A provider's error code: a word or a short number. Anything else may be free text and is dropped.
_CODE_SHAPED = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}|[0-9]{1,5}")
_NAME_SHAPED = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:()/&-]{0,95}")


class LLMError(Exception):
    """The model gave no usable answer. The message is safe to log."""


class LLMUnavailableError(LLMError):
    """Nothing will work until the configuration changes: a refused key, no credit, no model."""


class LLMRateLimitedError(LLMError):
    """The service is limiting or overloaded and stayed so through the one retry."""


class LLMTimeoutError(LLMError):
    """The call did not finish inside its wall."""


class LLMResponseError(LLMError):
    """The service answered, but not with a reply that can be used."""


@dataclass(frozen=True)
class Reply:
    text: str
    finish_reason: str | None = None  # "stop"; "length" when cut at the output limit
    model: str | None = None  # as the provider reports it
    provider: str | None = None  # the upstream that served it, when a router names one
    response_format: str = "none"  # the format the request carried: json_schema, json_object, none
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None  # in the provider's unit, when it reports one

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Reply:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


class ChatModel(Protocol):
    def complete(
        self, messages: Sequence[Message], *, schema: Mapping[str, Any] | None = None
    ) -> Reply:
        """One reply to the messages, or an `LLMError`."""
        ...

    def describe(self) -> dict[str, object]:
        """Non-secret settings worth recording with a run."""
        ...


class OpenAICompatibleModel:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not settings.api_key:
            raise LLMUnavailableError("LLM_API_KEY is not set")
        self._settings = settings
        self._url = settings.base_url.rstrip("/") + "/chat/completions"
        self._http = httpx.Client(
            transport=transport, headers={"Authorization": f"Bearer {settings.api_key}"}
        )
        self._sleep = sleep
        self._clock = clock
        self.requests = 0  # HTTP requests made, rejected and retried ones included

    def describe(self) -> dict[str, object]:
        return {**self._settings.describe(), "http_requests": self.requests}

    def close(self) -> None:
        self._http.close()

    def complete(
        self, messages: Sequence[Message], *, schema: Mapping[str, Any] | None = None
    ) -> Reply:
        deadline = self._clock() + self._settings.timeout
        formats = self._formats(schema)
        for position, mode in enumerate(formats):
            status, payload = self._send(self._body(messages, mode, schema), deadline)
            if status == httpx.codes.OK:
                return _reply(payload, mode)
            weaker = formats[position + 1] if position + 1 < len(formats) else None
            if weaker is not None and status in _DOWNGRADE_STATUSES:
                log.warning(
                    "the provider answered HTTP %s to a %s response format; asking again with %s",
                    status,
                    mode,
                    weaker,
                )
                continue
            raise _error_for(status, payload)
        raise AssertionError("every format either returns or raises")  # pragma: no cover

    def _formats(self, schema: Mapping[str, Any] | None) -> tuple[str, ...]:
        mode = self._settings.structured_output
        if schema is None or mode == "none":
            return ("none",)
        if mode == "auto":
            return ("json_schema", "json_object", "none")
        return (mode,)

    def _body(
        self, messages: Sequence[Message], mode: str, schema: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": list(messages),
            "max_tokens": self._settings.max_output_tokens,
        }
        if self._settings.temperature is not None:
            body["temperature"] = self._settings.temperature
        if mode == "json_schema" and schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "change_order_facts",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def _send(self, body: dict[str, Any], deadline: float) -> tuple[int, bytes]:
        """The response to one request, after at most one retry of a failure a retry can fix."""
        for attempt in (1, 2):
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise LLMTimeoutError(self._timeout_message())
            self.requests += 1
            try:
                status, payload, headers = self._post(body, remaining, deadline)
            except httpx.TimeoutException:
                raise LLMTimeoutError(self._timeout_message()) from None
            except httpx.TransportError as exc:
                # The class name only: the message can hold the URL.
                failure = LLMError(f"the model service could not be reached ({type(exc).__name__})")
                if attempt == 2 or self._clock() + 1.0 >= deadline:
                    raise failure from None
                log.warning("the connection to the model service dropped; one retry")
                self._sleep(1.0)
                continue
            if status not in _RETRY_STATUSES or attempt == 2:
                return status, payload
            wait = _retry_after_seconds(headers)
            wait = 2.0 if wait is None else wait
            if wait > _MAX_RETRY_WAIT or self._clock() + wait >= deadline:
                return status, payload
            log.warning("the model service answered HTTP %s; one retry in %.1f s", status, wait)
            self._sleep(wait)
        raise AssertionError("the loop returns or raises on its second pass")  # pragma: no cover

    def _post(
        self, body: dict[str, Any], remaining: float, deadline: float
    ) -> tuple[int, bytes, httpx.Headers]:
        timeout = httpx.Timeout(remaining, connect=min(remaining, 15.0))
        with self._http.stream("POST", self._url, json=body, timeout=timeout) as response:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > _MAX_RESPONSE_BYTES:
                    raise LLMResponseError("the model service sent an oversized response")
                if self._clock() > deadline:
                    raise httpx.ReadTimeout("the response outlasted the wall")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks), response.headers

    def _timeout_message(self) -> str:
        return (
            f"the model call did not finish within {self._settings.timeout:g} seconds "
            "(LLM_TIMEOUT_SECONDS)"
        )


class ScriptedModel:
    """A model that answers from a list and never touches the network.

    Tests script it with replies and errors; `--replay` feeds it the replies recorded from a live
    run, so a committed output can be regenerated without a key. A script that runs out raises
    `LLMError`: asking for more replies than were recorded means the conversation changed.
    """

    def __init__(
        self,
        replies: Iterable[Reply | str | Exception],
        *,
        name: str = "scripted",
        description: Mapping[str, object] | None = None,
    ) -> None:
        self._replies = deque(replies)
        self.name = name
        self._description = dict(description) if description is not None else None
        self.calls: list[list[Message]] = []
        self.schemas: list[Mapping[str, Any] | None] = []

    def describe(self) -> dict[str, object]:
        if self._description is not None:  # replaying a live run: say what made the replies
            return dict(self._description)
        return {"model": self.name, "scripted": True}

    def complete(
        self, messages: Sequence[Message], *, schema: Mapping[str, Any] | None = None
    ) -> Reply:
        self.calls.append([dict(message) for message in messages])
        self.schemas.append(schema)
        if not self._replies:
            raise LLMError("the scripted model has no reply left for this request")
        item = self._replies.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Reply):
            return item
        return Reply(text=item, finish_reason="stop", model=self.name)


def _reply(payload: bytes, mode: str) -> Reply:
    try:
        data = json.loads(payload)
    except ValueError:
        raise LLMResponseError("the model service sent a response that is not JSON") from None
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        # A 200 can carry an error object where the choices should be.
        code = _error_code(data)
        raise LLMResponseError(
            "the model service sent no completion" + (f" (error code {code})" if code else "")
        )
    choice = choices[0]
    message = choice.get("message")
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    finish = choice.get("finish_reason")
    return Reply(
        text=_content_text(message.get("content") if isinstance(message, dict) else None),
        finish_reason=finish if isinstance(finish, str) else None,
        model=_name(data.get("model")),
        provider=_name(data.get("provider")),
        response_format=mode,
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        cost=_cost(usage.get("cost")),
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: keep the text ones
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _error_for(status: int, payload: bytes) -> LLMError:
    try:
        code = _error_code(json.loads(payload))
    except ValueError:
        code = None
    suffix = f", error code {code}" if code else ""
    if status in (401, 403):
        return LLMUnavailableError(
            f"the model service refused the credentials (HTTP {status}{suffix}); check LLM_API_KEY"
        )
    if status == 402 or (status == 429 and code in _QUOTA_CODES):
        return LLMUnavailableError(
            f"the model account has no credit or quota left (HTTP {status}{suffix})"
        )
    if status == 404:
        return LLMUnavailableError(
            f"the model service answered HTTP 404{suffix}; check LLM_BASE_URL and LLM_MODEL"
        )
    if status in (429, 503):
        return LLMRateLimitedError(
            f"the model service is rate-limited or overloaded (HTTP {status}{suffix})"
        )
    return LLMError(f"the model service answered HTTP {status}{suffix}")


def _error_code(data: Any) -> str | None:
    """The provider's error code or type, only when it is shaped like one."""
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None
    for key in ("code", "type"):
        value = error.get(key)
        if isinstance(value, bool) or not isinstance(value, str | int):
            continue
        text = str(value)
        if _CODE_SHAPED.fullmatch(text) and not text.lower().startswith(("sk-", "pk-", "rk-")):
            return text
    return None


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    """The wait a response asks for, or None when it names none that can be read."""
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        try:
            return max(0.0, float(milliseconds) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - datetime.now(UTC)).total_seconds())


def _name(value: Any) -> str | None:
    return value if isinstance(value, str) and _NAME_SHAPED.fullmatch(value) else None


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return 0
    return int(value)


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0 else None
