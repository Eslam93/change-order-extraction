"""Reading the model's reply: exactly the asked shape, one correction, never a made-up result."""

from __future__ import annotations

import json

import pytest

from changeorder.extract import extract_candidates, first_json_object
from changeorder.llm import (
    LLMRateLimitedError,
    LLMResponseError,
    LLMTimeoutError,
    Reply,
    ScriptedModel,
)
from changeorder.pipeline import process
from helpers import BASE, base_fields, model_reply

GOOD = model_reply(**base_fields())


def test_a_reply_in_the_asked_shape_is_accepted_in_one_call():
    model = ScriptedModel([GOOD])

    run = extract_candidates(BASE, model)

    assert run.extraction.change_order_number.value == "7"
    assert run.corrected is False and len(model.calls) == 1
    assert model.schemas[0] is not None  # the schema went out with the request


def test_json_wrapped_in_a_code_fence_or_a_sentence_is_still_read():
    wrapped = f"Here is the result:\n```json\n{GOOD}\n```\nLet me know if you need anything else."

    assert (
        extract_candidates(BASE, ScriptedModel([wrapped])).extraction.amount.value == "$12,450.00"
    )
    assert first_json_object('noise {"a": {"b": [1, 2]}} trailing {"c": 3}') == {"a": {"b": [1, 2]}}
    assert first_json_object("no object here [1, 2]") is None


@pytest.mark.parametrize(
    ("bad", "named"),
    [
        ("I could not find a change order in this document.", "no JSON object"),
        ("", "no JSON object"),
        ('{"project_name": {"status": "stated"', "no JSON object"),  # not closed
        (json.dumps({"project_name": "Maple Court Apartments"}), "project_name"),  # the wrong shape
        (
            GOOD.replace('"status": "stated"', '"status": "certain"', 1),
            "status",
        ),  # not an allowed word
        (
            GOOD.replace('"document_notes"', '"confidence": 0.99, "document_notes"'),
            "confidence",
        ),  # an extra key
    ],
)
def test_a_reply_in_the_wrong_shape_gets_one_correction_that_names_the_problem(bad, named):
    model = ScriptedModel([bad, GOOD])

    run = extract_candidates(BASE, model)

    assert run.corrected is True and len(run.replies) == 2
    first, second = model.calls
    assert second[: len(first)] == first  # the same conversation, continued
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "user"
    assert named in second[-1]["content"]


def test_a_second_bad_reply_is_an_error_and_there_is_no_third_call():
    wrong = json.dumps({"project_name": "Maple Court Apartments", "amount": 12450})
    model = ScriptedModel([wrong, wrong, GOOD])

    with pytest.raises(LLMResponseError) as raised:
        extract_candidates(BASE, model)

    assert len(model.calls) == 2
    message = str(raised.value)
    assert "after one correction" in message
    assert "Maple Court" not in message and "12450" not in message  # paths and rules, not values


def test_a_reply_cut_off_at_the_output_limit_is_an_error_and_is_not_asked_for_again():
    cut = Reply(text=GOOD[: len(GOOD) // 2], finish_reason="length")
    model = ScriptedModel([cut, GOOD])

    with pytest.raises(LLMResponseError, match="LLM_MAX_OUTPUT_TOKENS"):
        extract_candidates(BASE, model)

    assert len(model.calls) == 1  # the same request would be cut again


@pytest.mark.parametrize(
    "failure", [LLMTimeoutError("slow"), LLMRateLimitedError("busy"), LLMResponseError("odd")]
)
def test_a_model_failure_ends_processing_with_no_result(tmp_path, failure):
    path = tmp_path / "co.txt"
    path.write_text(BASE, encoding="utf-8")

    with pytest.raises(type(failure)):
        process(path, ScriptedModel([failure]))
