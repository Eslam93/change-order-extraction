"""Text built to be slow, and input built to crash.

The patterns in this project read text that someone else wrote. An earlier version had two that
took a minute on a rule of twenty thousand hyphens or a run of colons, and a check that compared
every date with every other. Each shape below goes through the whole pipeline and has to finish
within `LIMIT` seconds. It takes a fraction of one; the limit is wide because the failure it
guards against is a hundred times slower, not ten percent. Measured on the commit before the
fix, on the machine this was written on: 110 seconds for the rule of hyphens, 66 for the
hyphens with line breaks and 18 for the colons followed by a word.

The second half is input that used to end in a traceback: every one now ends in a refusal with
a reason, or is simply read.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from changeorder.document import read_document
from changeorder.extract import extract_candidates, first_json_object
from changeorder.llm import LLMResponseError, ScriptedModel
from changeorder.normalize import find_dates, find_days, find_money, status_cues
from changeorder.pipeline import process_document
from helpers import model_reply, stated

LIMIT = 10.0
# Ordinary words, so that a document made mostly of punctuation still passes as text.
PAD = "\n" + ("lorem ipsum dolor sit amet consectetur " * 4 + "\n") * 40
SHAPES = {
    # A run that ends in something else is the slow case: a pattern that can start anywhere in
    # the run and reads to its end before failing reads the run once per character.
    "a rule of hyphens": "Contract Time: see below\n" + "-" * 20000 + " x" + PAD * 8,
    "hyphens before a unit": "Contract Time: " + "-" * 20000 + " days" + PAD * 8,
    "hyphens and line breaks": "Contract Time: see below\n" + "-\n" * 9000 + "x" + PAD * 8,
    "colons after a label, then a word": "Approved by" + ":" * 20000 + "x" + PAD * 8,
    "colons after a label": "Approved by" + ":" * 30000 + PAD * 12,
    "lines of dashes": "Status: Approved\n" + "--\n" * 15000 + PAD * 12,
    "one line of status words": "approved " * 7000,
    "one line of negated labels": "not approved by: " * 4000,
    "one line of dates": "Date: " + "01/02/2026 " * 6000,
    "dates on their own lines": "Date: 01/02/2026\n" * 4000,
    "signature blocks": "By: A. Person\nDate: 01/02/2026\n" * 2200,
    "one line of figures": "Amount: " + "$1,000.00 " * 6000,
    "figures on their own lines": "Amount of this change: $1,000.00\n" * 2000,
    "one line of durations": "Contract Time: " + "5 days " * 7000,
    "durations on their own lines": "Contract Time extended by 5 days\n" * 2000,
    "tick boxes": "Approved [ ] Rejected [X] " * 2700,
    "open parentheses": "Amount: $1,000.00 " + "(" * 40000 + PAD * 14,
    "a long number before a unit": "Contract Time: " + "9" * 40000 + " days" + PAD * 4,
    "figures and commas": "Amount: " + "1," * 20000 + PAD * 8,
    "a label on every line": "a: b\n" * 14000,
    "one line of no change": "Contract Time: " + "no change to time. " * 3500,
    "no change on every line": "Time: no change\n" * 4500,
    "a long table": "Item  Description  Amount  Days\n" + "1  thing  $1,000.00  5 days\n" * 2500,
    "a number on every line": "Change Order No. 7\n" * 3500,
    "a gap inside a label": "change" + " " * 20000 + "order" + PAD * 8,
    "an empty label on every line": "By: \n" * 14000,
}


def replies(text: str) -> list[str]:
    """A model that states every field from the first line, and one that reports no change."""
    head = text[:400].split("\n")[0]
    return [
        model_reply(
            project_name=stated("lorem ipsum", head),
            contract_number=stated("1", head),
            change_order_number=stated("7", head),
            date=stated("01/02/2026", head),
            requested_by=stated("lorem", head),
            description=stated(head),
            amount=stated("$1,000.00", head, direction="increase"),
            schedule_impact_days=stated("5 days", head, direction="increase"),
            approval_status=stated("approved", head),
            contract_sum_before=stated("$1,000.00", head),
            contract_sum_after=stated("$1,000.00", head),
        ),
        model_reply(
            amount=stated("0", head, direction="no_change"),
            schedule_impact_days=stated("0", head, direction="no_change"),
            approval_status=stated("rejected", head),
            date=stated("01/02/2026", head),
        ),
    ]


@pytest.mark.parametrize("shape", SHAPES)
def test_text_built_to_be_slow_is_processed_in_bounded_time(tmp_path, shape):
    path = tmp_path / "hostile.txt"
    path.write_text(SHAPES[shape], encoding="utf-8")

    began = time.perf_counter()
    document = read_document(path)
    for reply in replies(document.text):
        result = process_document(document, ScriptedModel([reply]))
        assert result.output.to_dict()  # a result, whatever it holds
    elapsed = time.perf_counter() - began

    assert elapsed < LIMIT, f"{shape}: {elapsed:.1f}s for {len(document.text):,} characters"


# ---- Input that used to end in a traceback ------------------------------------------------------


def test_a_month_written_with_a_long_s_is_read_not_crashed_on():
    # The pattern is case-insensitive, and under case folding a long s is an "s".
    (token,) = find_dates("1 \u017feptember 2026")

    assert (token.first, token.second, token.year) == (1, 9, 2026)


@pytest.mark.parametrize("digits", [4_400, 20_000])
def test_a_number_thousands_of_digits_long_is_not_a_duration_or_an_amount(digits):
    number = "9" * digits

    assert find_days(f"Contract Time: {number} days") == []
    assert find_money(f"Amount: ${number}.00") == []


def test_a_line_with_no_end_is_read_without_looking_past_what_matters():
    line = "approved " * 20000

    began = time.perf_counter()
    assert status_cues(line) == {"approved": "explicit"}
    assert time.perf_counter() - began < LIMIT


@pytest.mark.parametrize(
    "reply",
    [
        '{"amount": ' + "[" * 20000,  # never closed
        '{"amount": '
        + "[" * 20000
        + "]" * 20000
        + "}",  # closed, and far too deep to mean anything
        "{" * 20000,
    ],
)
def test_a_reply_nested_thousands_deep_is_a_bad_reply_not_a_crash(reply):
    assert first_json_object(reply) is None

    model = ScriptedModel([reply, reply])
    with pytest.raises(LLMResponseError):
        extract_candidates("CHANGE ORDER No. 7", model)
    assert len(model.calls) == 2  # one correction was asked for, and no third call


def test_a_file_that_cannot_be_opened_is_a_document_error(tmp_path, monkeypatch):
    from changeorder.document import DocumentError

    path = tmp_path / "locked.txt"
    path.write_text("CHANGE ORDER No. 7", encoding="utf-8")

    def locked(self: Path) -> bytes:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", locked)
    with pytest.raises(DocumentError) as raised:
        read_document(path)

    assert raised.value.code == "not_readable"
    assert "Permission denied" in raised.value.message
