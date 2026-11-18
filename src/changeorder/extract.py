"""Asking the model for candidates, and accepting nothing that is not exactly the asked shape.

One call, validated here whatever the provider promised about structured output. A reply that
does not parse gets exactly one corrective attempt that names what was wrong with its shape. A
reply cut off at the output limit is not retried (the same request would be cut again), and a
second bad reply is an error. There is no fallback value: a failed extraction raises, and the
caller exits unsuccessfully instead of printing something that looks like a result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .llm import ChatModel, LLMResponseError, Reply
from .prompt import Extraction, build_messages, correction_message, response_schema

__all__ = ["ExtractionRun", "extract_candidates", "first_json_object"]

log = logging.getLogger("changeorder.extract")


@dataclass(frozen=True)
class ExtractionRun:
    extraction: Extraction
    replies: tuple[Reply, ...]  # every reply received; the last one is the one that was accepted

    @property
    def corrected(self) -> bool:
        return len(self.replies) > 1


def extract_candidates(document_text: str, model: ChatModel) -> ExtractionRun:
    """The model's candidates for one document, or an `LLMError`."""
    messages = build_messages(document_text)
    schema = response_schema()
    reply = model.complete(messages, schema=schema)
    extraction, problems = _parse(reply)
    if extraction is not None:
        return ExtractionRun(extraction, (reply,))
    if reply.finish_reason == "length":
        raise LLMResponseError(
            "the model's reply was cut off at the output limit, so nothing was extracted; "
            "raise LLM_MAX_OUTPUT_TOKENS"
        )
    log.warning(
        "the model's reply did not have the required shape (%s); one corrective attempt",
        "; ".join(problems[:3]),
    )
    retry = [
        *messages,
        {"role": "assistant", "content": reply.text or "(empty reply)"},
        correction_message(problems),
    ]
    second = model.complete(retry, schema=schema)
    extraction, problems = _parse(second)
    if extraction is None:
        cut = " and was cut off at the output limit" if second.finish_reason == "length" else ""
        raise LLMResponseError(
            "the model's reply still did not have the required shape after one correction"
            f"{cut}: " + "; ".join(problems[:5])
        )
    return ExtractionRun(extraction, (reply, second))


def _parse(reply: Reply) -> tuple[Extraction | None, list[str]]:
    """The reply as an `Extraction`, or what was wrong with it: field paths and rules only,
    never the reply's own values."""
    found = first_json_object(reply.text)
    if found is None:
        return None, ["the reply holds no JSON object"]
    try:
        return Extraction.model_validate(found), []
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(part) for part in error['loc']) or 'the object'}: {error['msg']}"
            for error in exc.errors()
        ]


def first_json_object(text: str) -> dict[str, Any] | None:
    """The first complete JSON object in the text, ignoring anything around it, such as a code
    fence or a sentence before it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            found, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(found, dict):
            return found
        start = text.find("{", start + 1)
    return None
