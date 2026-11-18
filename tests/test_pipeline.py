"""What the pipeline publishes, given a document and whatever a model says about it.

Each test plays one model reply against one small document and checks the three things the
contract promises: the value (or null), its confidence band, and whether a person must look.
The model is scripted, so a test can make it right, wrong or dishonest on purpose.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from changeorder.pipeline import contract_problems
from helpers import (
    AMOUNT_LINE,
    DATE_LINE,
    STATUS_LINE,
    TIME_LINE,
    absent,
    codes,
    run_base,
    stated,
    unclear,
)

# ---- The straightforward case -----------------------------------------------------------------


def test_a_straightforward_document_is_extracted_and_needs_no_review(tmp_path):
    result = run_base(tmp_path)

    assert result.output.to_dict() == {
        "project_name": {"value": "Maple Court Apartments", "confidence": 0.9},
        "contract_number": {"value": "MC-2025-014", "confidence": 0.9},
        "change_order_number": {"value": "7", "confidence": 0.9},
        "date": {"value": "2026-01-15", "confidence": 0.9},
        "requested_by": {"value": "Maple Court Housing LLC", "confidence": 0.9},
        "description": {
            "value": "Add a second coat of intumescent paint to the steel columns at Level 1.",
            "confidence": 0.9,
        },
        "amount": {"value": 12450, "confidence": 0.9},
        "schedule_impact_days": {"value": 3, "confidence": 0.9},
        "approval_status": {"value": "approved", "confidence": 0.9},
        "needs_review": False,
    }
    assert contract_problems(result.output.to_dict()) == []
    assert result.findings == ()


def test_the_published_value_is_the_documents_wording_not_the_models(tmp_path):
    # The model tidied the name's case and the dashes; the document's own characters go out.
    result = run_base(
        tmp_path,
        ("Project: Maple Court Apartments", "Project: Maple Court Apartments – Phase II"),
        project_name=stated(
            "MAPLE COURT APARTMENTS - PHASE II", "project: maple court apartments - phase ii"
        ),
    )

    assert result.output.project_name.value == "Maple Court Apartments – Phase II"
    assert result.output.project_name.confidence == 0.9
    assert result.decisions["project_name"].evidence[0].kind == "normalized"


# ---- Missing is null; null is not zero ----------------------------------------------------------


def test_fields_a_change_order_may_lack_are_null_without_review(tmp_path):
    result = run_base(
        tmp_path,
        ("Requested by: Maple Court Housing LLC\n", ""),
        (TIME_LINE, ""),
        (DATE_LINE, ""),
        requested_by=absent(),
        schedule_impact_days=absent(),
        date=absent(),
    )

    out = result.output.to_dict()
    for name in ("requested_by", "schedule_impact_days", "date"):
        assert out[name] == {"value": None, "confidence": 0.0}
        assert result.decisions[name].state == "absent"
    assert out["needs_review"] is False


@pytest.mark.parametrize(
    ("removed_line", "field"),
    [
        (AMOUNT_LINE, "amount"),
        (STATUS_LINE, "approval_status"),
        ("Change Order No.: 7\n", "change_order_number"),
    ],
)
def test_a_missing_critical_field_is_null_and_sends_the_document_to_a_person(
    tmp_path, removed_line, field
):
    result = run_base(tmp_path, (removed_line, ""), **{field: absent()})

    assert result.output.to_dict()[field] == {"value": None, "confidence": 0.0}
    assert result.output.needs_review is True
    assert codes(result, field) == {"critical_field_missing"}


def test_a_change_order_tied_to_no_contract_needs_review(tmp_path):
    result = run_base(
        tmp_path,
        ("Project: Maple Court Apartments\n", ""),
        ("Contract No.: MC-2025-014\n", ""),
        project_name=absent(),
        contract_number=absent(),
    )

    assert "no_contract_reference" in codes(result)
    # One of the two is enough: a contract number alone does not need a project name.
    assert (
        run_base(
            tmp_path, ("Project: Maple Court Apartments\n", ""), project_name=absent()
        ).output.needs_review
        is False
    )


@pytest.mark.parametrize(
    ("time_line", "candidate", "value", "confidence", "review"),
    [
        # Explicitly unchanged is 0.
        (
            "Contract Time: No change to the Contract Time.\n",
            stated("No change to the Contract Time", "Contract Time: No change to the Contract Time.", direction="no_change"),
            0, 0.9, False,
        ),
        ("Contract Time: 0 days.\n", stated("0 days", "Contract Time: 0 days.", direction="no_change"), 0, 0.9, False),
        # "N/A" is read as no change, one band lower: it says less than "no change" does.
        ("Contract Time: N/A\n", stated("N/A", "Contract Time: N/A", direction="no_change"), 0, 0.75, False),
        # Not mentioned is null, and that is fine.
        ("", absent(), None, 0.0, False),
        # Put off is null and unresolved: "to be determined" is not "no change".
        (
            "Contract Time: To be determined.\n",
            unclear("the time is to be determined", "Contract Time: To be determined."),
            None, 0.0, True,
        ),
        # A model that reports "no change" beside a real duration is not believed.
        (TIME_LINE, stated("3 calendar days", TIME_LINE.strip(), direction="no_change"), None, 0.0, True),
        # Nor one whose quote does not say so.
        (
            "Contract Time: see the attached programme.\n",
            stated("see the attached programme", "Contract Time: see the attached programme.", direction="no_change"),
            None, 0.0, True,
        ),
    ],
)  # fmt: skip
def test_schedule_zero_is_stated_not_assumed(
    tmp_path, time_line, candidate, value, confidence, review
):
    result = run_base(tmp_path, (TIME_LINE, time_line), schedule_impact_days=candidate)

    assert result.output.schedule_impact_days.value == value
    assert result.output.schedule_impact_days.value is not False  # 0, never a boolean or ""
    assert result.output.schedule_impact_days.confidence == confidence
    assert result.output.needs_review is review


@pytest.mark.parametrize(
    ("amount_line", "candidate"),
    [
        ("Amount of this change order: $0.00\n", stated("$0.00", "Amount of this change order: $0.00", direction="no_change")),
        (
            "This change is made at no cost to the Owner.\n",
            stated("no cost", "This change is made at no cost to the Owner.", direction="no_change"),
        ),
    ],
)  # fmt: skip
def test_an_explicitly_zero_cost_change_is_zero(tmp_path, amount_line, candidate):
    result = run_base(tmp_path, (AMOUNT_LINE, amount_line), amount=candidate)

    assert result.output.amount.value == Decimal(0)
    assert result.output.to_dict()["amount"] == {"value": 0, "confidence": 0.9}
    assert result.output.needs_review is False


# ---- The sign of the amount ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount_line", "direction", "value"),
    [
        ("Amount of this change order (add): $12,450.00\n", "increase", Decimal("12450.00")),
        ("Amount of this change order: $12,450.00\n", "increase", Decimal("12450.00")),
        ("Amount of this change order: ($5,000.00)\n", "decrease", Decimal("-5000.00")),
        ("Amount of this change order: -$5,000.00\n", "decrease", Decimal("-5000.00")),
        ("Deduct from the Contract Sum: $5,000.00\n", "decrease", Decimal("-5000.00")),
        ("Credit to the Owner for deleted flooring: $5,000.00\n", "decrease", Decimal("-5000.00")),
    ],
)
def test_the_sign_follows_the_document(tmp_path, amount_line, direction, value):
    figure = amount_line.rsplit(": ", 1)[1].strip()
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, amount_line),
        amount=stated(figure, amount_line.strip(), direction=direction),
    )

    assert result.output.amount.value == value
    assert result.output.amount.confidence == 0.9
    assert result.output.needs_review is False


@pytest.mark.parametrize(
    ("amount_line", "direction", "reason"),
    [
        # The figure is written as a credit; the model calls it an increase.
        ("Amount of this change order: ($5,000.00)\n", "increase", "direction_contradicts_figure"),
        # The wording says deduct; the model calls it an increase.
        ("Deduct from the Contract Sum: $5,000.00\n", "increase", "direction_contradicts_wording"),
        # The wording says add; the model calls it a decrease.
        ("Amount of this change order (add): $5,000.00\n", "decrease", "direction_contradicts_wording"),
        # Nothing in the document shows a decrease; the model's word is not enough.
        ("Amount of this change order: $5,000.00\n", "decrease", "decrease_not_shown_by_document"),
        # A blank form: both words are printed, and nothing says which applies.
        (
            "The Contract Sum will be (increased) (decreased) by this Change Order in the amount of: $5,000.00\n",
            "increase",
            "wording_gives_both_directions",
        ),
        ("Amount of this change order: $5,000.00\n", "unclear", "direction_not_established"),
    ],
)  # fmt: skip
def test_a_sign_the_document_does_not_support_is_not_published(
    tmp_path, amount_line, direction, reason
):
    figure = amount_line.rsplit(": ", 1)[1].strip()
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, amount_line),
        amount=stated(figure, amount_line.strip(), direction=direction),
    )

    assert result.output.to_dict()["amount"] == {"value": None, "confidence": 0.0}
    assert result.decisions["amount"].state == "unresolved"
    assert codes(result, "amount") == {reason}
    assert result.output.needs_review is True


# ---- Which figure is the change -----------------------------------------------------------------

SUMS = (
    "Subtotal: $11,000.00\n"
    "Overhead and profit: $1,450.00\n"
    "Amount of this change order (add): $12,450.00\n"
    "Contract Sum prior to this Change Order: $100,000.00\n"
    "New Contract Sum including this Change Order: $112,450.00\n"
)
BEFORE = stated("$100,000.00", "Contract Sum prior to this Change Order: $100,000.00")
AFTER = stated("$112,450.00", "New Contract Sum including this Change Order: $112,450.00")


def test_contract_sums_that_differ_by_the_amount_corroborate_it(tmp_path):
    result = run_base(
        tmp_path, (AMOUNT_LINE, SUMS), contract_sum_before=BEFORE, contract_sum_after=AFTER
    )

    assert result.output.to_dict()["amount"] == {"value": 12450, "confidence": 0.97}
    assert result.decisions["amount"].corroborated is True
    assert result.output.needs_review is False


@pytest.mark.parametrize(
    ("figure", "quote", "reason"),
    [
        # The largest, most prominent figure on the page is the revised total. It is not the change.
        ("$112,450.00", "New Contract Sum including this Change Order: $112,450.00", "figure_is_a_contract_total"),
        ("$100,000.00", "Contract Sum prior to this Change Order: $100,000.00", "figure_is_a_contract_total"),
        ("$11,000.00", "Subtotal: $11,000.00", "figure_is_a_part_of_the_change"),
        ("$1,450.00", "Overhead and profit: $1,450.00", "figure_is_a_part_of_the_change"),
    ],
)  # fmt: skip
def test_a_total_or_a_part_is_not_accepted_as_the_change(tmp_path, figure, quote, reason):
    result = run_base(
        tmp_path, (AMOUNT_LINE, SUMS), amount=stated(figure, quote, direction="increase")
    )

    assert result.output.amount.value is None
    assert codes(result, "amount") == {reason}
    assert result.output.needs_review is True


def test_a_figure_the_contract_sums_contradict_is_not_published(tmp_path):
    # The model took the subtotal. The sums before and after say the change is something else.
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, SUMS),
        amount=stated("$11,000.00", "Subtotal: $11,000.00", direction="increase"),
        contract_sum_before=BEFORE,
        contract_sum_after=AFTER,
    )

    assert result.output.amount.value is None
    assert codes(result, "amount") == {"figure_contradicts_contract_sums"}


def test_contract_sums_settle_the_sign_a_blank_form_leaves_open(tmp_path):
    form = (
        "The Contract Sum will be (increased) (decreased) by this Change Order in the amount of: $5,000.00\n"
        "Contract Sum prior to this Change Order: $100,000.00\n"
        "New Contract Sum including this Change Order: $95,000.00\n"
    )
    quote = "The Contract Sum will be (increased) (decreased) by this Change Order in the amount of: $5,000.00"
    fields: dict[str, Any] = {
        "contract_sum_before": BEFORE,
        "contract_sum_after": stated(
            "$95,000.00", "New Contract Sum including this Change Order: $95,000.00"
        ),
    }

    credit = run_base(
        tmp_path,
        (AMOUNT_LINE, form),
        amount=stated("$5,000.00", quote, direction="decrease"),
        **fields,
    )
    assert credit.output.to_dict()["amount"] == {"value": -5000, "confidence": 0.97}

    wrong = run_base(
        tmp_path,
        (AMOUNT_LINE, form),
        amount=stated("$5,000.00", quote, direction="increase"),
        **fields,
    )
    assert wrong.output.amount.value is None
    assert codes(wrong, "amount") == {"direction_contradicts_contract_sums"}


def test_an_original_sum_alone_cannot_contradict_the_amount(tmp_path):
    # Original 100,000 and revised 150,000, with a change of 12,450: change orders the document
    # does not list can explain the gap, so it neither corroborates nor contradicts.
    sums = AMOUNT_LINE + "Original Contract Sum: $100,000.00\nRevised Contract Sum: $150,000.00\n"
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, sums),
        original_contract_sum=stated("$100,000.00", "Original Contract Sum: $100,000.00"),
        contract_sum_after=stated("$150,000.00", "Revised Contract Sum: $150,000.00"),
    )

    assert result.output.to_dict()["amount"] == {"value": 12450, "confidence": 0.9}

    with_earlier = AMOUNT_LINE + (
        "Original Contract Sum: $100,000.00\n"
        "Net change by previous Change Orders: $37,550.00\n"
        "Revised Contract Sum: $150,000.00\n"
    )
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, with_earlier),
        original_contract_sum=stated("$100,000.00", "Original Contract Sum: $100,000.00"),
        previous_changes=stated("$37,550.00", "Net change by previous Change Orders: $37,550.00"),
        contract_sum_after=stated("$150,000.00", "Revised Contract Sum: $150,000.00"),
    )
    assert result.output.to_dict()["amount"] == {"value": 12450, "confidence": 0.97}


@pytest.mark.parametrize(
    ("amount_line", "extra", "value"),
    [
        ("Amount of this change order: EUR 1.234,56\n", "", Decimal("1234.56")),
        ("Amount of this change order: $12,450.5\n", "", Decimal("12450.5")),
        # Where the document's other figures prove a decimal point, "1.500" is not 1500.
        ("Amount of this change order: $1.500\n", "Retainage held to date: $4,120.50\n", None),
        ("Amount of this change order: $12,45,0.00\n", "", None),
    ],
)
def test_number_formats_are_read_by_rule_or_refused(tmp_path, amount_line, extra, value):
    figure = amount_line.rsplit(": ", 1)[1].strip()
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, amount_line + extra),
        amount=stated(figure, amount_line.strip(), direction="increase"),
    )

    assert result.output.amount.value == value
    assert result.output.needs_review is (value is None)


# ---- Dates --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("date_line", "written", "value", "confidence", "reason"),
    [
        ("Date: 2026-03-04\n", "2026-03-04", "2026-03-04", 0.9, None),
        ("Date: March 4, 2026\n", "March 4, 2026", "2026-03-04", 0.9, None),
        ("Date: 4-Mar-2026\n", "4-Mar-2026", "2026-03-04", 0.9, None),
        ("Date: 25/03/2026\n", "25/03/2026", "2026-03-25", 0.9, None),  # 25 cannot be a month
        ("Date: 04/04/2026\n", "04/04/2026", "2026-04-04", 0.9, None),  # the same either way
        ("Date: 4 Mar 26\n", "4 Mar 26", "2026-03-04", 0.75, None),  # the century is assumed
        # 3 April or 4 March: nothing in the document says which.
        ("Date: 03/04/2026\n", "03/04/2026", None, 0.0, "day_and_month_ambiguous"),
        ("Date: 31/04/2026\n", "31/04/2026", None, 0.0, "not_a_calendar_date"),
    ],
)
def test_dates_are_normalised_only_when_they_read_one_way(
    tmp_path, date_line, written, value, confidence, reason
):
    result = run_base(tmp_path, (DATE_LINE, date_line), date=stated(written, date_line.strip()))

    assert result.output.to_dict()["date"] == {"value": value, "confidence": confidence}
    assert codes(result, "date") == ({reason} if reason else set())
    assert result.output.needs_review is (reason is not None)


def test_the_documents_other_dates_can_settle_the_order_of_day_and_month(tmp_path):
    # "Received: 25/03/2026" can only be day first, so this document writes day first.
    result = run_base(
        tmp_path,
        (DATE_LINE, "Date: 03/04/2026\nReceived: 25/03/2026\n"),
        date=stated("03/04/2026", "Date: 03/04/2026"),
    )

    assert result.output.to_dict()["date"] == {"value": "2026-04-03", "confidence": 0.75}
    assert result.decisions["date"].caveats == ["day_month_order_from_other_dates"]
    assert result.output.needs_review is False


def test_a_model_cannot_settle_an_ambiguous_date_by_rewriting_it(tmp_path):
    # The model normalised 03/04/2026 itself. Code still reads the document's token.
    result = run_base(
        tmp_path,
        (DATE_LINE, "Date: 03/04/2026\n"),
        date=stated("2026-03-04", "Date: 03/04/2026"),
    )

    assert result.output.date.value is None
    assert codes(result, "date") == {"day_and_month_ambiguous"}


@pytest.mark.parametrize(
    "other_line",
    [
        "Contract Date: 5 June 2025",
        "Date of Substantial Completion: 30 September 2026",
        "Owner   By: H. Tran   Date: 20 January 2026",
    ],
)
def test_the_date_of_another_event_is_not_the_change_order_date(tmp_path, other_line):
    written = other_line.rsplit(": ", 1)[1]
    result = run_base(tmp_path, (DATE_LINE, other_line + "\n"), date=stated(written, other_line))

    assert result.output.date.value is None
    assert codes(result, "date") == {"date_belongs_to_another_event"}


def test_a_change_order_dated_before_its_contract_keeps_its_value_and_needs_review(tmp_path):
    result = run_base(tmp_path, (DATE_LINE, DATE_LINE + "Contract Date: 1 March 2026\n"))

    # The value is what the document says, at full confidence, and a rule still objects.
    assert result.output.to_dict()["date"] == {"value": "2026-01-15", "confidence": 0.9}
    assert codes(result) == {"date_before_contract_date"}
    assert result.output.needs_review is True


# ---- Schedule -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_line", "written", "direction", "value", "confidence"),
    [
        ("Contract Time: extended by 3 calendar days.\n", "3 calendar days", "increase", 3, 0.9),
        ("Contract Time: reduced by 5 calendar days.\n", "5 calendar days", "decrease", -5, 0.9),
        ("Completion is accelerated by 5 working days.\n", "5 working days", "decrease", -5, 0.9),
        # Written twice, in words and in figures, and they agree.
        ("Contract Time: extended by ten (10) calendar days.\n", "ten (10) calendar days", "increase", 10, 0.97),
        ("Contract Time: extended by 2 weeks.\n", "2 weeks", "increase", 14, 0.75),
        ("Contract Time adjustment: 14 calendar days\n", "14 calendar days", "increase", 14, 0.9),
        # The model wrote the number alone; the document's token is still what is read.
        ("Contract Time: extended by 3 calendar days.\n", "3", "increase", 3, 0.9),
    ],
)  # fmt: skip
def test_schedule_impact_is_read_in_days_with_its_sign(
    tmp_path, time_line, written, direction, value, confidence
):
    result = run_base(
        tmp_path,
        (TIME_LINE, time_line),
        schedule_impact_days=stated(written, time_line.strip(), direction=direction),
    )

    assert result.output.to_dict()["schedule_impact_days"] == {
        "value": value,
        "confidence": confidence,
    }
    assert result.output.needs_review is False


@pytest.mark.parametrize(
    ("time_line", "written", "direction", "reason"),
    [
        ("Contract Time: extended by ten (12) calendar days.\n", "ten (12) calendar days", "increase",
         "duration_not_readable:words_and_figures_disagree"),
        ("Contract Time: extended by 2 months.\n", "2 months", "increase",
         "duration_not_readable:months_have_no_fixed_length"),
        ("Contract Time: extended by 2.5 days.\n", "2.5 days", "increase", "duration_not_readable:not_whole_days"),
        ("Contract Time: extended by 3 calendar days.\n", "3 calendar days", "decrease", "direction_contradicts_wording"),
        ("Contract Time adjustment: 14 calendar days\n", "14 calendar days", "decrease", "decrease_not_shown_by_document"),
    ],
)  # fmt: skip
def test_a_schedule_impact_that_cannot_be_read_one_way_is_not_published(
    tmp_path, time_line, written, direction, reason
):
    result = run_base(
        tmp_path,
        (TIME_LINE, time_line),
        schedule_impact_days=stated(written, time_line.strip(), direction=direction),
    )

    assert result.output.to_dict()["schedule_impact_days"] == {"value": None, "confidence": 0.0}
    assert codes(result, "schedule_impact_days") == {reason}


# ---- Conflicting evidence -----------------------------------------------------------------------


def test_two_change_order_numbers_in_one_document_give_none(tmp_path):
    result = run_base(
        tmp_path,
        (DATE_LINE, DATE_LINE + "This Change Order No. 9 revises the fire protection scope.\n"),
    )

    assert result.output.to_dict()["change_order_number"] == {"value": None, "confidence": 0.0}
    assert codes(result, "change_order_number") == {"conflicting_change_order_numbers"}
    assert result.output.needs_review is True


def test_a_reference_to_another_change_order_is_not_a_conflict(tmp_path):
    result = run_base(
        tmp_path, (DATE_LINE, DATE_LINE + "This supersedes Change Order No. 5 in full.\n")
    )

    assert result.output.change_order_number.value == "7"
    assert result.output.needs_review is False


def test_a_number_the_document_labels_as_something_else_is_not_the_change_order_number(tmp_path):
    result = run_base(
        tmp_path,
        (DATE_LINE, DATE_LINE + "Job No.: 2214\n"),
        change_order_number=stated("2214", "Job No.: 2214"),
    )

    assert result.output.change_order_number.value is None
    assert codes(result, "change_order_number") == {"labelled_as_something_else"}

    # Under a label that says nothing either way, the number the document does label decides.
    result = run_base(
        tmp_path,
        (DATE_LINE, DATE_LINE + "Ref 2214\n"),
        change_order_number=stated("2214", "Ref 2214"),
    )
    assert result.output.change_order_number.value is None
    assert codes(result, "change_order_number") == {
        "number_differs_from_labelled_change_order_number"
    }


@pytest.mark.parametrize(
    ("field", "value", "quote"),
    [
        # Each value is in the document, word for word, under a label that makes it something else.
        ("project_name", "Maple Court Housing LLC", "Requested by: Maple Court Housing LLC"),
        ("requested_by", "Maple Court Apartments", "Project: Maple Court Apartments"),
        ("change_order_number", "MC-2025-014", "Contract No.: MC-2025-014"),
        ("contract_number", "7", "Change Order No.: 7"),
    ],
)
def test_a_real_value_under_another_fields_label_is_not_accepted(tmp_path, field, value, quote):
    result = run_base(tmp_path, **{field: stated(value, quote)})

    assert result.output.to_dict()[field] == {"value": None, "confidence": 0.0}
    assert "labelled_as_something_else" in codes(result, field)
    assert result.output.needs_review is True


@pytest.mark.parametrize(
    ("extra", "figure", "quote", "reason"),
    [
        ("Paint, 2,400 sq ft at Level 1\n", "2,400", "Paint, 2,400 sq ft at Level 1", "figure_is_a_quantity_or_a_rate"),
        ("Painter, 16 hrs @ $85.00\n", "$85.00", "Painter, 16 hrs @ $85.00", "figure_is_a_quantity_or_a_rate"),
        ("Not to exceed $20,000.00 without approval\n", "$20,000.00", "Not to exceed $20,000.00 without approval", "figure_is_a_ceiling_or_an_estimate"),
        ("Scaffold hire  $1,900.00\n", "$1,900.00", "Scaffold hire  $1,900.00", "another_figure_is_labelled_as_the_change"),
    ],
)  # fmt: skip
def test_a_figure_that_is_not_the_change_is_refused_by_its_label(
    tmp_path, extra, figure, quote, reason
):
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, extra + AMOUNT_LINE),
        amount=stated(figure, quote, direction="increase"),
    )

    assert result.output.amount.value is None
    assert codes(result, "amount") == {reason}


def test_what_was_requested_is_not_published_when_something_else_was_granted(tmp_path):
    lines = (
        "Amount requested by Contractor: $14,000.00\n"
        "Amount approved by Owner: $12,450.00\n"
        "Time extension requested: 6 working days. Time extension granted: 3 working days.\n"
    )
    asked = run_base(
        tmp_path,
        (AMOUNT_LINE, lines),
        (TIME_LINE, ""),
        amount=stated(
            "$14,000.00", "Amount requested by Contractor: $14,000.00", direction="increase"
        ),
        schedule_impact_days=stated(
            "6 working days", "Time extension requested: 6 working days.", direction="increase"
        ),
    )
    assert asked.output.amount.value is None
    assert asked.output.schedule_impact_days.value is None
    assert codes(asked, "amount") == {"a_different_figure_was_approved"}
    assert codes(asked, "schedule_impact_days") == {"a_different_duration_was_granted"}

    granted = run_base(
        tmp_path,
        (AMOUNT_LINE, lines),
        (TIME_LINE, ""),
        amount=stated("$12,450.00", "Amount approved by Owner: $12,450.00", direction="increase"),
        schedule_impact_days=stated(
            "3 working days", "Time extension granted: 3 working days.", direction="increase"
        ),
    )
    assert granted.output.to_dict()["amount"] == {"value": 12450, "confidence": 0.9}
    assert granted.output.to_dict()["schedule_impact_days"] == {"value": 3, "confidence": 0.9}
    assert granted.output.needs_review is False


def test_two_differing_candidates_under_the_same_kind_of_label_give_no_value(tmp_path):
    # Each value below is real, quoted and readable. The document simply says two things.
    amounts = run_base(
        tmp_path,
        (AMOUNT_LINE, AMOUNT_LINE + "Total change order amount: $12,950.00\n"),
    )
    assert amounts.output.amount.value is None
    assert codes(amounts, "amount") == {"two_figures_are_labelled_as_the_change"}

    dates = run_base(tmp_path, (DATE_LINE, DATE_LINE + "Date issued: 22 January 2026\n"))
    assert dates.output.date.value is None
    assert codes(dates, "date") == {"two_dates_are_labelled_as_the_documents_own"}

    durations = run_base(
        tmp_path, (TIME_LINE, TIME_LINE + "Time extension for this change: 9 calendar days\n")
    )
    assert durations.output.schedule_impact_days.value is None
    assert codes(durations, "schedule_impact_days") == {"two_durations_are_given_for_the_schedule"}

    # The same value stated twice is not a conflict.
    repeated = run_base(
        tmp_path,
        (AMOUNT_LINE, AMOUNT_LINE + "Total change order amount: $12,450.00\n"),
        (DATE_LINE, DATE_LINE + "Date issued: 15 January 2026\n"),
    )
    assert repeated.findings == ()


@pytest.mark.parametrize(
    ("line", "written"),
    [
        ("This proposal is valid for 15 days.", "15 days"),
        ("Crane hire - 2 days", "2 days"),
    ],
)
def test_a_duration_that_is_not_about_the_contract_time_is_not_the_schedule_impact(
    tmp_path, line, written
):
    result = run_base(
        tmp_path,
        (TIME_LINE, line + "\n"),
        schedule_impact_days=stated(written, line, direction="increase"),
    )

    assert result.output.schedule_impact_days.value is None
    assert codes(result, "schedule_impact_days") == {"duration_is_not_about_the_schedule"}


def test_a_date_in_running_text_is_not_taken_for_the_documents_date(tmp_path):
    line = "The crew mobilised to site 12 January 2026 and began the paint preparation."
    result = run_base(tmp_path, (DATE_LINE, line + "\n"), date=stated("12 January 2026", line))

    assert result.output.date.value is None
    assert codes(result, "date") == {"date_not_labelled_as_the_documents_own"}


def test_a_field_the_model_calls_unclear_is_null_and_keeps_its_note(tmp_path):
    result = run_base(
        tmp_path,
        requested_by=unclear(
            "the header names the Owner and the body names the Architect",
            "Requested by: Maple Court Housing LLC",
        ),
    )

    assert result.output.requested_by.value is None
    assert result.decisions["requested_by"].state == "unresolved"
    assert codes(result, "requested_by") == {"model_reported_unclear"}
    assert "Architect" in result.decisions["requested_by"].note


# ---- Approval -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_line", "status", "confidence"),
    [
        ("Status: Approved\n", "approved", 0.9),
        ("[X] Approved   [ ] Rejected   [ ] Resubmit\n", "approved", 0.9),
        ("[ ] Approved   [X] Rejected   [ ] Resubmit\n", "rejected", 0.9),
        ("Status: Rejected by the Owner on 20 January 2026\n", "rejected", 0.9),
        ("Status: Pending Owner review\n", "pending", 0.9),
        ("Submitted for approval; not yet signed by the Owner.\n", "pending", 0.9),
        ("*** DRAFT - NOT FOR EXECUTION ***\n", "draft", 0.9),
        ("Status: VOID - superseded by Change Order No. 8\n", "void", 0.9),
        ("Approved by: Dana Whitlock, Owner's Representative\n", "approved", 0.9),
        # A completed signature and nothing else: accepted, one band lower.
        ("Owner: /s/ D. Whitlock   20 January 2026\n", "approved", 0.75),
    ],
)
def test_a_status_the_quoted_text_shows_is_accepted(tmp_path, status_line, status, confidence):
    result = run_base(
        tmp_path, (STATUS_LINE, status_line), approval_status=stated(status, status_line.strip())
    )

    assert result.output.to_dict()["approval_status"] == {"value": status, "confidence": confidence}
    assert result.output.needs_review is False


@pytest.mark.parametrize(
    "status_line",
    [
        # Each of these has an approval section or the word "approved", and shows no approval.
        "This Change Order is not valid until approved by the Owner.\n",
        "Approved by (Owner): ______________________   Date: ____________\n",
        "APPROVED BY:\n",
        "Upon approval, the Contractor shall proceed with the work.\n",
        "[ ] Approved   [ ] Rejected   [ ] Resubmit\n",
        "Authorization: signatures of the Owner and the Contractor appear on page 2.\n",
        "This Change Order must be signed by the Owner before work proceeds.\n",
        "The Owner has not approved this Change Order.\n",
    ],
)
def test_an_approval_section_is_not_an_approval(tmp_path, status_line):
    result = run_base(
        tmp_path,
        (STATUS_LINE, status_line),
        approval_status=stated("approved", status_line.strip()),
    )

    assert result.output.to_dict()["approval_status"] == {"value": None, "confidence": 0.0}
    assert "status_not_shown_by_quoted_text" in codes(result, "approval_status")
    assert result.output.needs_review is True


def test_an_approval_beside_a_rejection_gives_no_status(tmp_path):
    result = run_base(
        tmp_path,
        (STATUS_LINE, "APPROVED\nStatus: Rejected by the Owner, see letter of 22 January 2026\n"),
        approval_status=stated("approved", "APPROVED"),
    )

    assert result.output.approval_status.value is None
    assert codes(result, "approval_status") == {"conflicting_approval_evidence"}


def test_a_qualified_approval_is_approved_and_still_goes_to_a_person(tmp_path):
    result = run_base(
        tmp_path,
        (STATUS_LINE, "Status: Approved as noted\n"),
        approval_status=stated("approved", "Status: Approved as noted"),
    )

    assert result.output.to_dict()["approval_status"] == {"value": "approved", "confidence": 0.75}
    assert codes(result) == {"approval_is_qualified"}
    assert result.output.needs_review is True


def test_approval_never_contradicts_a_negative_or_pending_amount(tmp_path):
    # An approved credit and a pending request with a stated amount are both legitimate.
    credit = run_base(
        tmp_path,
        (AMOUNT_LINE, "Amount of this change order: ($5,000.00)\n"),
        amount=stated(
            "($5,000.00)", "Amount of this change order: ($5,000.00)", direction="decrease"
        ),
    )
    pending = run_base(
        tmp_path,
        (STATUS_LINE, "Status: Pending Owner review\n"),
        approval_status=stated("pending", "Status: Pending Owner review"),
    )

    assert credit.findings == () and pending.findings == ()


# ---- Unsupported and hallucinated values ----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "candidate", "reason"),
    [
        # A quote that is not in the document supports nothing.
        ("amount", stated("$99,999.00", "Total of this change order: $99,999.00", direction="increase"), "quote_not_found_in_document"),
        # A real quote that does not hold the value supports nothing either.
        ("change_order_number", stated("8", "Change Order No.: 7"), "value_not_in_quoted_text"),
        ("contract_number", stated("MC-2025-041", "Contract No.: MC-2025-014"), "value_not_in_quoted_text"),
        ("amount", stated("$21,450.00", AMOUNT_LINE.strip(), direction="increase"), "value_not_in_quoted_text"),
        ("requested_by", stated("Maple Court Housing LLC"), "no_quote_given"),
        ("project_name", stated("", "Project: Maple Court Apartments"), "stated_without_a_value"),
        # A paraphrase adds wording the document does not have.
        ("description", stated("Apply extra fireproofing to all structural steel on every level."), "wording_not_in_document"),
        # A schedule impact the document never mentions.
        ("schedule_impact_days", stated("10 calendar days", "Contract Time: extended by 10 calendar days.", direction="increase"), "quote_not_found_in_document"),
    ],
)  # fmt: skip
def test_a_value_the_document_does_not_support_is_never_published(
    tmp_path, field, candidate, reason
):
    edits = [(TIME_LINE, "")] if field == "schedule_impact_days" else []
    result = run_base(tmp_path, *edits, **{field: candidate})

    assert result.output.to_dict()[field] == {"value": None, "confidence": 0.0}
    assert result.decisions[field].state == "unresolved"
    assert reason in codes(result, field)
    assert result.output.needs_review is True


def test_an_absence_the_document_contradicts_is_not_taken_on_the_models_word(tmp_path):
    # The model says there is no amount, no schedule impact and no status. The document has all three.
    result = run_base(
        tmp_path, amount=absent(), schedule_impact_days=absent(), approval_status=absent()
    )

    out = result.output.to_dict()
    assert out["amount"]["value"] is None and out["schedule_impact_days"]["value"] is None
    assert "possibly_missed_value" in codes(result, "amount")
    assert "possibly_missed_value" in codes(result, "schedule_impact_days")
    assert "possibly_missed_value" in codes(result, "approval_status")
    assert out["needs_review"] is True


# ---- Review policy ------------------------------------------------------------------------------


def test_two_caveats_put_a_value_under_the_review_threshold(tmp_path):
    result = run_base(
        tmp_path,
        (DATE_LINE, "Date: 03/04/26\nReceived: 25/03/26\n"),
        date=stated("03/04/26", "Date: 03/04/26"),
    )

    assert result.output.to_dict()["date"] == {"value": "2026-04-03", "confidence": 0.5}
    assert set(result.decisions["date"].caveats) == {
        "day_month_order_from_other_dates",
        "two_digit_year",
    }
    assert codes(result) == {"low_confidence"}
    assert result.output.needs_review is True


def test_a_quote_found_only_approximately_lowers_the_band(tmp_path):
    # The model dropped a word from the line it quoted.
    result = run_base(
        tmp_path,
        requested_by=stated("Maple Court Housing LLC", "Requested: Maple Court Housing LLC"),
    )

    assert result.decisions["requested_by"].evidence[0].kind == "fuzzy"
    assert result.output.to_dict()["requested_by"] == {
        "value": "Maple Court Housing LLC",
        "confidence": 0.75,
    }
    assert result.output.needs_review is False
