"""Scoring against labels, the report's numbers, the replay check, and the stress test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changeorder.evaluation import (
    Document,
    Scored,
    confidence_metrics,
    field_metrics,
    outcome,
    rebuild,
    replay,
    review_metrics,
    wilson,
)
from changeorder.llm import ScriptedModel
from changeorder.pipeline import process
from changeorder.stress import ABSENT_KIND, inject, summarise
from helpers import BASE, base_fields, model_reply

DESCRIPTION = "Replace the specified maple strip flooring with a resilient athletic floor system."


@pytest.mark.parametrize(
    ("field", "published", "label", "also", "expected"),
    [
        ("amount", None, None, (), "correct_null"),
        ("amount", 0, None, (), "invented"),  # a zero where the document says nothing
        ("amount", None, 0, (), "missed"),  # and nothing where the document says zero
        ("amount", 12450, 12450.00, (), "correct_value"),
        ("amount", -12450.0, 12450.00, (), "wrong_value"),  # the sign is part of the value
        ("amount", 12450.004, 12450.00, (), "correct_value"),  # compared in cents
        ("schedule_impact_days", 14, 14, (), "correct_value"),
        ("schedule_impact_days", 14, -14, (), "wrong_value"),
        ("date", "2026-03-04", "2026-04-03", (), "wrong_value"),
        ("approval_status", "approved", "pending", (), "wrong_value"),
        # Names: case, punctuation and the kind of dash are not differences.
        ("project_name", "RIVERBEND LOGISTICS CENTER - BUILDING C", "Riverbend Logistics Center – Building C", (), "correct_value"),
        ("project_name", "Riverbend Logistics Center", "Riverbend Logistics Center – Building C", (), "wrong_value"),
        ("requested_by", "J. Ortega", "Hallmark General Contractors", ("J. Ortega",), "correct_value"),
        # Identifiers: spacing and punctuation are not differences; digits are.
        ("change_order_number", "CO 011", "CO-011", (), "correct_value"),
        ("change_order_number", "11", "CO-011", ("011", "11"), "correct_value"),
        ("change_order_number", "12", "CO-011", ("011", "11"), "wrong_value"),
        ("contract_number", "PAUSD-2025-0117", "PAUSD-2025-0171", (), "wrong_value"),
        # A description may keep a list number or drop a trailing clause, and no more.
        ("description", "1. " + DESCRIPTION, DESCRIPTION, (), "correct_value"),
        ("description", "Replace the flooring.", DESCRIPTION, (), "wrong_value"),
        ("description", "Install twelve data drops in Classrooms 201 to 206.", DESCRIPTION, (), "wrong_value"),
    ],
)  # fmt: skip
def test_a_published_value_is_scored_against_its_label(field, published, label, also, expected):
    assert outcome(field, published, label, also) == expected


def scored(
    result: str, confidence: float = 0.9, field: str = "amount", document: str = "a"
) -> Scored:
    values = {
        "correct_value": (1, 1),
        "correct_null": (None, None),
        "wrong_value": (2, 1),
        "invented": (2, None),
        "missed": (None, 1),
    }
    published, label = values[result]
    shown = confidence if published is not None else 0.0
    return Scored(document, field, result, published, label, shown, "")


def test_the_headline_rates_separate_wrong_values_from_abstentions():
    fields = [
        *[scored("correct_value")] * 6,
        *[scored("correct_null")] * 2,
        scored("wrong_value"),
        scored("invented"),
        *[scored("missed")] * 2,
    ]

    metrics = field_metrics(fields)

    assert metrics["fields"] == 12
    assert metrics["accuracy"] == round(8 / 12, 4)
    assert metrics["published"] == 8  # six right, one wrong, one invented
    assert metrics["precision_on_published_values"] == 0.75
    assert metrics["stated_in_document"] == 9  # six right, one wrong, two missed
    assert metrics["recall_of_stated_values"] == round(6 / 9, 4)
    assert metrics["abstention_rate_on_stated_values"] == round(2 / 9, 4)
    assert metrics["not_stated_in_document"] == 3
    assert metrics["null_detection_rate"] == round(2 / 3, 4)
    assert metrics["false_extraction_rate"] == round(1 / 3, 4)


def test_accuracy_is_reported_by_band_and_at_each_threshold_over_published_values_only():
    fields = [
        *[scored("correct_value", 0.97)] * 4,
        *[scored("correct_value", 0.9)] * 8,
        *[scored("wrong_value", 0.9)] * 2,
        scored("correct_value", 0.75),
        scored("invented", 0.75),
        *[scored("missed")] * 3,  # nulls have no confidence to assess
    ]

    metrics = confidence_metrics(fields)

    assert [(b["confidence"], b["published"], b["correct"]) for b in metrics["bands"]] == [
        (0.97, 4, 4),
        (0.9, 10, 8),
        (0.75, 2, 1),
    ]
    assert [(t["at_or_above"], t["published"], t["errors"]) for t in metrics["thresholds"]] == [
        (0.75, 16, 3),
        (0.9, 14, 2),
        (0.97, 4, 0),
    ]
    assert metrics["thresholds"][1]["coverage_of_published"] == 0.875
    gap = (4 * abs(1.0 - 0.97) + 10 * abs(0.8 - 0.9) + 2 * abs(0.5 - 0.75)) / 16
    assert metrics["expected_calibration_error"] == round(gap, 4)


def test_the_interval_is_wide_when_the_sample_is_small():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(5, 5)
    assert 0.56 < low < 0.57 and high == 1.0  # five out of five proves little
    low, high = wilson(200, 200)
    assert low > 0.98
    low, high = wilson(5, 10)
    assert 0.23 < low < 0.24 and 0.76 < high < 0.77


def test_review_is_compared_with_the_label_and_with_where_the_errors_are():
    def document(name: str, flagged: bool, expected: bool, *results: str) -> Document:
        fields = tuple(scored(result, document=name) for result in results)
        return Document(name, flagged, expected, fields, (0, 0, 0.0))

    metrics = review_metrics(
        [
            document("clean", False, False, "correct_value", "correct_value", "correct_null"),
            document("slipped", False, True, "correct_value", "wrong_value", "missed"),
            document("caught", True, True, "correct_value", "invented", "missed"),
            document("cautious", True, False, "correct_value", "correct_value", "missed"),
        ]
    )

    assert metrics["review_rate"] == 0.5
    assert metrics["flagged_and_expected"] == 1 and metrics["flagged_not_expected"] == 1
    assert metrics["passed_but_review_expected"] == 1
    passed = metrics["passed_without_review"]
    assert passed["published_values"] == 4 and passed["wrong_or_invented_values"] == 1
    assert passed["error_rate"] == 0.25 and passed["values_missed_silently"] == 1
    assert metrics["flagged"]["wrong_or_invented_values"] == 1
    assert metrics["flagged"]["values_missed"] == 2


# ---- Committed outputs cannot drift from what the pipeline produces --------------------------------


def recorded_set(root: Path) -> Path:
    """A one-document set as `run` leaves it: input, label, output and diagnostics."""
    for folder in ("inputs", "truth", "outputs", "diagnostics"):
        (root / folder).mkdir(parents=True)
    document = root / "inputs" / "co.txt"
    document.write_text(BASE, encoding="utf-8")
    result = process(document, ScriptedModel([model_reply(**base_fields())]))
    dump = lambda data: json.dumps(data, indent=2, ensure_ascii=False) + "\n"  # noqa: E731
    (root / "outputs" / "co.json").write_text(
        dump(result.output.to_dict()), encoding="utf-8", newline="\n"
    )
    (root / "diagnostics" / "co.json").write_text(
        dump(result.diagnostics()), encoding="utf-8", newline="\n"
    )
    truth = {
        "document": "co.txt",
        "fields": {
            name: field["value"]
            for name, field in result.output.to_dict().items()
            if name != "needs_review"
        },
        "also_accept": {},
        "review_expected": False,
    }
    (root / "truth" / "co.json").write_text(json.dumps(truth), encoding="utf-8")
    return root


def test_replay_passes_untouched_outputs_and_catches_an_edited_one(tmp_path, capsys):
    root = recorded_set(tmp_path / "set")
    assert replay(root) == 0

    output = root / "outputs" / "co.json"
    output.write_text(
        output.read_text(encoding="utf-8").replace("12450", "12540"), encoding="utf-8", newline="\n"
    )

    assert replay(root) == 1
    assert "the committed output differs from the rebuilt one" in capsys.readouterr().err

    assert rebuild(root) == 0  # derives it again from the recorded replies
    assert replay(root) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["amount"]["value"] == 12450


def test_replay_refuses_a_recording_made_with_another_prompt(tmp_path, capsys):
    root = recorded_set(tmp_path / "set")
    recorded = root / "diagnostics" / "co.json"
    data = json.loads(recorded.read_text(encoding="utf-8"))
    data["run"]["prompt_fingerprint"] = "0" * 64
    recorded.write_text(json.dumps(data), encoding="utf-8")

    assert replay(root) == 1
    assert "recorded with another prompt" in capsys.readouterr().err


# ---- The stress test ---------------------------------------------------------------------------------


def test_each_injected_mistake_on_a_clean_document_is_refused_or_flagged(tmp_path):
    root = recorded_set(tmp_path / "set")

    injections = inject(root)

    by_kind = {(item.field, item.kind): item.outcome for item in injections}
    assert by_kind[("amount", "the amount's direction reversed")] == "rejected"
    assert by_kind[("amount", "a figure that is not in the document")] == "rejected"
    assert by_kind[("schedule_impact_days", "the schedule's direction reversed")] == "rejected"
    assert (
        by_kind[("change_order_number", "the contract number taken for the change order number")]
        == "rejected"
    )
    assert by_kind[("project_name", "the requester taken for the project name")] == "rejected"
    assert {
        outcome
        for (field, kind), outcome in by_kind.items()
        if kind == "another status claimed on the same quotes"
    } == {"rejected"}
    # A value reported absent does not pass unnoticed.
    assert {outcome for (_, kind), outcome in by_kind.items() if kind == ABSENT_KIND} == {"flagged"}
    assert summarise(injections)["published_wrong"] == 0


def test_without_the_label_checks_the_same_mistakes_get_through(tmp_path):
    root = recorded_set(tmp_path / "set")

    checked = summarise(inject(root))
    unchecked = summarise(inject(root, label_checks=False))

    assert checked["published_wrong"] == 0
    assert unchecked["published_wrong"] >= 1
    wrong = {(row["field"], row["mistake"]) for row in unchecked["wrong_values"]}
    # The requester's name is in the document, in the quote, and readable: nothing but its
    # label says it is not the project.
    assert ("project_name", "the requester taken for the project name") in wrong
