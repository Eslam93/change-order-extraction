"""Wrong values that a mistaken model could once get published.

Every refusal here was found by an independent review of the first version of the checks: a
document and a model reply for which the pipeline published a wrong value, at 0.90 or 0.97,
without asking for review. Each was reproduced, fixed in the deterministic stages and pinned
here, next to the neighbouring case that must still be accepted.

The model is scripted. What is tested is what the pipeline lets through when the model is
wrong: a refusal is null, confidence 0.0, a reason code, and `needs_review`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from changeorder.pipeline import Result
from helpers import AMOUNT_LINE, DATE_LINE, STATUS_LINE, TIME_LINE, codes, run_base, stated


def refused(result: Result, field: str, reason: str) -> None:
    assert result.output.to_dict()[field] == {"value": None, "confidence": 0.0}
    assert result.decisions[field].state == "unresolved"
    assert reason in codes(result, field)
    assert result.output.needs_review is True


def no_change(value: str, quote: str) -> dict[str, Any]:
    return stated(value, quote, direction="no_change")


# ---- An explicit zero is read from the document, not from the model ---------------------------


@pytest.mark.parametrize(
    ("edits", "field", "candidate", "reason"),
    [
        # The model writes "$0.00" and quotes the line that says $12,450.00.
        (
            [],
            "amount",
            no_change("$0.00", AMOUNT_LINE.strip()),
            "no_change_not_shown_by_document",
        ),
        ([], "schedule_impact_days", no_change("0", TIME_LINE.strip()), "no_change_not_shown_by_document"),
        # "At this time" leaves it open, and so does a right to claim reserved in the next line.
        (
            [(TIME_LINE, "No time extension is requested at this time.\n")],
            "schedule_impact_days",
            no_change("No time extension is requested", "No time extension is requested at this time."),
            "no_change_is_hedged",
        ),
        (
            [(TIME_LINE, "No time extension is requested.\nContractor reserves the right to claim additional time.\n")],
            "schedule_impact_days",
            no_change("No time extension is requested", "No time extension is requested."),
            "no_change_is_hedged",
        ),
        (
            [(TIME_LINE, "Contract Time: no change, subject to the Engineer's review.\n")],
            "schedule_impact_days",
            no_change("no change", "Contract Time: no change, subject to the Engineer's review."),
            "no_change_is_hedged",
        ),
        # A zero that is the total of earlier change orders is not this change.
        (
            [(AMOUNT_LINE, AMOUNT_LINE + "Net change by previous Change Orders: $0.00\n")],
            "amount",
            no_change("$0.00", "Net change by previous Change Orders: $0.00"),
            "figure_is_a_contract_total",
        ),
        # "No change" said about the time is not said about the price, and the reverse.
        (
            [(TIME_LINE, "Contract Time: no change.\n")],
            "amount",
            no_change("no change", "Contract Time: no change."),
            "no_change_is_about_something_else",
        ),
        (
            [(TIME_LINE, "No change to the Contract Sum.\n")],
            "schedule_impact_days",
            no_change("No change", "No change to the Contract Sum."),
            "no_change_is_about_something_else",
        ),
        # Two labelled statements on one line: "none" belongs to the time, not to the cost.
        (
            [(AMOUNT_LINE, ""), (TIME_LINE, "Cost: see the attached breakdown   Time: none\n")],
            "amount",
            no_change("none", "Cost: see the attached breakdown   Time: none"),
            "no_change_is_about_something_else",
        ),
        # The document says the sum is unchanged and also labels a figure as the change.
        (
            [(AMOUNT_LINE, "The Contract Sum is unchanged.\nAmount of this change order: $4,500.00\n")],
            "amount",
            no_change("unchanged", "The Contract Sum is unchanged."),
            "another_figure_is_labelled_as_the_change",
        ),
        (
            [(TIME_LINE, "Time: no change\nThe Contract Time is extended by 14 calendar days.\n")],
            "schedule_impact_days",
            no_change("no change", "Time: no change"),
            "another_duration_is_given_for_the_schedule",
        ),
        # "No change" beside the figure it is supposed to replace.
        (
            [(AMOUNT_LINE, "Amount of this change order: $4,500.00 (no change to scope or cost)\n")],
            "amount",
            no_change("no change", "Amount of this change order: $4,500.00 (no change to scope or cost)"),
            "no_change_reported_beside_a_figure",
        ),
        # A bare zero under a label that is not about time.
        (
            [(TIME_LINE, "Open items: 0\n")],
            "schedule_impact_days",
            no_change("0", "Open items: 0"),
            "duration_is_not_about_the_schedule",
        ),
    ],
)  # fmt: skip
def test_no_change_is_not_taken_on_the_models_word(tmp_path, edits, field, candidate, reason):
    refused(run_base(tmp_path, *edits, **{field: candidate}), field, reason)


@pytest.mark.parametrize(
    ("edits", "field", "candidate", "confidence"),
    [
        # The model normalised the value; the document's own words carry the zero.
        (
            [(TIME_LINE, "The Contract Time is unchanged.\n")],
            "schedule_impact_days",
            no_change("0", "The Contract Time is unchanged."),
            0.9,
        ),
        (
            [(TIME_LINE, "The Contract Time will be increased by Zero (0) days.\n")],
            "schedule_impact_days",
            no_change("Zero (0) days", "The Contract Time will be increased by Zero (0) days."),
            0.9,
        ),
        ([(TIME_LINE, "Days added: 0\n")], "schedule_impact_days", no_change("0", "Days added: 0"), 0.9),
        # A sentence that wraps: the subject is on the second line.
        (
            [(TIME_LINE, "There is no change to the\nContract Time under this Change Order.\n")],
            "schedule_impact_days",
            no_change("no change", "There is no change to the Contract Time under this Change Order."),
            0.9,
        ),
        (
            [(AMOUNT_LINE, "The Contract Sum is unchanged by this Change Order.\n")],
            "amount",
            no_change("unchanged", "The Contract Sum is unchanged by this Change Order."),
            0.9,
        ),
        # Two labelled statements on one line, and this time "none" is the cost's.
        (
            [(AMOUNT_LINE, ""), (TIME_LINE, "Cost: none   Time: 3 calendar days added\n")],
            "amount",
            no_change("none", "Cost: none   Time: 3 calendar days added"),
            0.9,
        ),
        # What was asked for and what was granted are not two answers.
        (
            [(TIME_LINE, "Time extension requested: 10 days\nTime extension granted: none\n")],
            "schedule_impact_days",
            no_change("none", "Time extension granted: none"),
            0.9,
        ),
    ],
)  # fmt: skip
def test_no_change_the_document_states_about_the_field_is_zero(
    tmp_path, edits, field, candidate, confidence
):
    result = run_base(tmp_path, *edits, **{field: candidate})

    assert result.output.to_dict()[field] == {"value": 0, "confidence": confidence}
    assert codes(result, field) == set()


def test_a_zero_in_a_table_is_read_under_its_own_column(tmp_path):
    table = "No.  Description  Amount  Days  Status\n7  Paint columns  $12,450.00  0  Approved\n"
    row = "7  Paint columns  $12,450.00  0  Approved"
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, ""),
        (TIME_LINE, table),
        amount=stated("$12,450.00", row, direction="increase"),
        schedule_impact_days=no_change("0", row),
    )
    assert result.output.to_dict()["schedule_impact_days"] == {"value": 0, "confidence": 0.9}

    # The same zero offered as the amount: its column is headed "Days".
    wrong = run_base(
        tmp_path,
        (AMOUNT_LINE, ""),
        (TIME_LINE, table),
        amount=no_change("0", row),
    )
    refused(wrong, "amount", "figure_is_not_labelled_as_a_price")


# ---- A figure is read whole, or not at all ------------------------------------------------------


@pytest.mark.parametrize(
    ("figure", "problem"),
    [
        ("$1.2 million", "figure_is_scaled_by_a_word"),
        ("$350 thousand", "figure_is_scaled_by_a_word"),
        ("USD 2.75 MM", "figure_is_scaled_by_a_word"),
        ("$40 K", "figure_is_scaled_by_a_word"),
        # 950,250, or 950.25 in a currency of three decimals: nothing in the document says.
        ("KWD 950.250", "point_before_three_digits_is_ambiguous"),
    ],
)
def test_a_figure_that_is_not_the_whole_amount_is_not_read_as_one(tmp_path, figure, problem):
    line = f"Amount of this change order (add): {figure}"
    result = run_base(
        tmp_path, (AMOUNT_LINE, line + "\n"), amount=stated(figure, line, direction="increase")
    )

    refused(result, "amount", f"figure_not_readable:{problem}")


def test_a_model_cannot_scale_the_figure_itself(tmp_path):
    line = "Amount of this change order (add): $1.2 million"
    for value in ("$1,200,000.00", "1.2"):
        result = run_base(
            tmp_path, (AMOUNT_LINE, line + "\n"), amount=stated(value, line, direction="increase")
        )
        refused(result, "amount", "value_not_in_quoted_text")


def test_a_cost_code_does_not_settle_which_number_is_the_day(tmp_path):
    # "26-05-19" is a cost code. Read as a date it would "prove" day-month-year order and turn
    # 03/04/2026 into 3 April.
    result = run_base(
        tmp_path,
        (DATE_LINE, "Date: 03/04/2026\nCost code: 26-05-19\n"),
        date=stated("03/04/2026", "Date: 03/04/2026"),
    )

    refused(result, "date", "day_and_month_ambiguous")


# ---- Contract sums vouch for a figure only in the roles the document gives them -----------------

PRIOR = "Contract Sum prior to this Change Order: $500,000.00"
NEW = "New Contract Sum including this Change Order: $512,450.00"
SUMS = f"{PRIOR}\n{NEW}\n"


def test_swapping_the_contract_sums_does_not_reverse_the_sign(tmp_path):
    plain = "Amount of this change order: $12,450.00"
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, f"{plain}\n{SUMS}"),
        amount=stated("$12,450.00", plain, direction="decrease"),
        contract_sum_before=stated("$512,450.00", NEW),
        contract_sum_after=stated("$500,000.00", PRIOR),
    )

    refused(result, "amount", "decrease_not_shown_by_document")


def test_a_contract_sum_is_never_the_change_whatever_arithmetic_is_offered(tmp_path):
    # The original sum and the revised sum differ by 100,000, and so the original sum "is" the
    # change. The earlier change orders the model left out explain the difference.
    sums = (
        "Original Contract Sum: $100,000.00\n"
        "Net change by previous Change Orders: $20,000.00\n"
        "Revised Contract Sum: $200,000.00\n"
    )
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, "Amount of this change order (add): $80,000.00\n" + sums),
        amount=stated("$100,000.00", "Original Contract Sum: $100,000.00", direction="increase"),
        original_contract_sum=stated("$100,000.00", "Original Contract Sum: $100,000.00"),
        contract_sum_after=stated("$200,000.00", "Revised Contract Sum: $200,000.00"),
    )

    refused(result, "amount", "figure_is_a_contract_total")


def test_lines_of_a_breakdown_are_not_contract_sums(tmp_path):
    # Subtotal and total differ by the labour line, which does not make labour the change.
    lines = "Labor: $2,000.00\nSubtotal: $10,000.00\nTotal: $12,000.00\n"
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, lines),
        amount=stated("$2,000.00", "Labor: $2,000.00", direction="increase"),
        contract_sum_before=stated("$10,000.00", "Subtotal: $10,000.00"),
        contract_sum_after=stated("$12,000.00", "Total: $12,000.00"),
    )

    refused(result, "amount", "figure_is_a_part_of_the_change")


def test_contract_sums_in_their_own_roles_still_corroborate(tmp_path):
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, AMOUNT_LINE + SUMS),
        contract_sum_before=stated("$500,000.00", PRIOR),
        contract_sum_after=stated("$512,450.00", NEW),
    )

    assert result.output.to_dict()["amount"] == {"value": 12450, "confidence": 0.97}
    assert result.output.needs_review is False


# ---- The sign --------------------------------------------------------------------------------

IN_WORDS = (
    "Amount of this change order (add): Twelve Thousand Four Hundred Fifty Dollars ($12,450.00)"
)
CREDIT = "Amount of this change order: $12,450.00 (credit to Owner)"
DEDUCT_COLUMN = "Item  Description  Deduct\n1  Delete canopy  $12,450.00\n"
BOTH_COLUMNS = "Item  Description  Add  Deduct\n1  Delete canopy  $12,450.00\n"
ROW = "1  Delete canopy  $12,450.00"
PLAIN = "Amount of this change order: $12,450.00"
DASHED = "Amount of this change order: \u2013$12,450.00"
TITLE = ("CHANGE ORDER\n", "DEDUCTIVE CHANGE ORDER\n")


@pytest.mark.parametrize(
    ("edits", "quote", "direction", "value"),
    [
        # Parentheses that restate an amount in words are not an accountant's minus.
        ([(AMOUNT_LINE, IN_WORDS + "\n")], IN_WORDS, "increase", Decimal("12450.00")),
        ([(AMOUNT_LINE, CREDIT + "\n")], CREDIT, "decrease", Decimal("-12450.00")),
        ([(AMOUNT_LINE, DEDUCT_COLUMN)], ROW, "decrease", Decimal("-12450.00")),
        ([TITLE, (AMOUNT_LINE, PLAIN + "\n")], PLAIN, "decrease", Decimal("-12450.00")),
        ([(AMOUNT_LINE, DASHED + "\n")], DASHED, "decrease", Decimal("-12450.00")),
    ],
)
def test_the_sign_is_read_where_the_document_puts_it(tmp_path, edits, quote, direction, value):
    result = run_base(tmp_path, *edits, amount=stated("$12,450.00", quote, direction=direction))

    assert result.output.amount.value == value
    assert result.output.amount.confidence == 0.9
    assert result.output.needs_review is False


@pytest.mark.parametrize(
    ("edits", "quote", "direction", "reason"),
    [
        # A remark after the figure that opens with the direction.
        ([(AMOUNT_LINE, CREDIT + "\n")], CREDIT, "increase", "direction_contradicts_wording"),
        # The column's heading, when the figure stands in a table.
        ([(AMOUNT_LINE, DEDUCT_COLUMN)], ROW, "increase", "direction_contradicts_wording"),
        # Two columns and one figure: which column it stands in cannot be told.
        ([(AMOUNT_LINE, BOTH_COLUMNS)], ROW, "increase", "wording_gives_both_directions"),
        ([(AMOUNT_LINE, BOTH_COLUMNS)], ROW, "decrease", "wording_gives_both_directions"),
        # The document's title, which the model did not quote.
        ([TITLE, (AMOUNT_LINE, PLAIN + "\n")], PLAIN, "increase", "direction_contradicts_wording"),
        # A dash set right against the figure is a minus sign.
        ([(AMOUNT_LINE, DASHED + "\n")], DASHED, "increase", "direction_contradicts_figure"),
    ],
)
def test_a_sign_the_document_contradicts_is_refused(tmp_path, edits, quote, direction, reason):
    result = run_base(tmp_path, *edits, amount=stated("$12,450.00", quote, direction=direction))

    refused(result, "amount", reason)


# ---- Which figure is the change -----------------------------------------------------------------

PRICES = "Proposed price: $14,000.00\nAgreed price: $13,100.00\n"
TWO_ON_A_ROW = "Revise door hardware  $4,250.00  $5,100.00"


@pytest.mark.parametrize(
    ("lines", "value", "quote", "reason"),
    [
        (
            "Amount of this change order: $22,000.00 (not to exceed)\n",
            "$22,000.00",
            "Amount of this change order: $22,000.00 (not to exceed)",
            "figure_is_a_ceiling_or_an_estimate",
        ),
        (
            "Amount of this change order: $22,000.00 estimated\n",
            "$22,000.00",
            "Amount of this change order: $22,000.00 estimated",
            "figure_is_a_ceiling_or_an_estimate",
        ),
        (PRICES, "$14,000.00", "Proposed price: $14,000.00", "a_different_figure_was_approved"),
        (TWO_ON_A_ROW + "\n", "$5,100.00", TWO_ON_A_ROW, "another_figure_shares_the_line"),
        (TWO_ON_A_ROW + "\n", "$4,250.00", TWO_ON_A_ROW, "another_figure_shares_the_line"),
    ],
)
def test_a_figure_its_surroundings_do_not_make_the_change_is_refused(
    tmp_path, lines, value, quote, reason
):
    result = run_base(
        tmp_path, (AMOUNT_LINE, lines), amount=stated(value, quote, direction="increase")
    )

    refused(result, "amount", reason)


def test_the_price_that_was_agreed_is_the_amount(tmp_path):
    result = run_base(
        tmp_path,
        (AMOUNT_LINE, PRICES),
        amount=stated("$13,100.00", "Agreed price: $13,100.00", direction="increase"),
    )

    assert result.output.to_dict()["amount"] == {"value": 13100, "confidence": 0.9}


# ---- The schedule -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "value", "reason"),
    [
        ("Contract Time: extended by 10-14 days.", "14 days", "duration_not_readable:duration_is_a_range"),
        ("Contract Time: extended by 10 to 14 days.", "14 days", "duration_not_readable:duration_is_a_range"),
        ("Contract Time: extended by up to 10 days.", "10 days", "duration_is_a_ceiling_or_an_estimate"),
        (
            "Contract Time: extended by 10 days, to be confirmed.",
            "10 days",
            "duration_is_a_ceiling_or_an_estimate",
        ),
        ("Estimated schedule impact: 5 days", "5 days", "duration_is_a_ceiling_or_an_estimate"),
        # The contract time itself is not the change in it.
        ("Revised Contract Time: 386 calendar days", "386 calendar days", "duration_is_a_contract_time_total"),
        (
            "The Contract Time prior to this Change Order was 365 days.",
            "365 days",
            "duration_is_a_contract_time_total",
        ),
        # A term for something else, on a line that happens to mention completion.
        (
            "Payment is due within 30 days of substantial completion.",
            "30 days",
            "duration_is_not_about_the_schedule",
        ),
        ("Lead time: 14 days", "14 days", "duration_is_not_about_the_schedule"),
        # What a neighbouring sentence says about time is not said about this number.
        (
            "No change to the Contract Time. Mobilization 5 days.",
            "5 days",
            "duration_is_not_about_the_schedule",
        ),
        # The day of a date is not a number of days.
        ("Completion date: 30 June 2026", "30", "value_not_in_quoted_text"),
        # "one hundred twenty days" is not twenty days.
        (
            "Contract Time: extended by one hundred twenty days.",
            "120 days",
            "duration_not_readable:number_in_words_not_read",
        ),
    ],
)  # fmt: skip
def test_a_duration_that_is_not_the_change_in_contract_time_is_refused(
    tmp_path, line, value, reason
):
    result = run_base(
        tmp_path,
        (TIME_LINE, line + "\n"),
        schedule_impact_days=stated(value, line, direction="increase"),
    )

    refused(result, "schedule_impact_days", reason)


@pytest.mark.parametrize(
    ("line", "value", "days", "confidence"),
    [
        ("Days added: 14", "14", 14, 0.75),  # no unit: one caveat
        ("Add 5 working days to the schedule.", "5 working days", 5, 0.9),  # wording after
        ("Contract Time: extended by one hundred twenty (120) days.", "120 days", 120, 0.9),
        ("Contract Time: extended by (14) calendar days.", "14 calendar days", 14, 0.9),
    ],
)
def test_the_neighbouring_durations_are_still_read(tmp_path, line, value, days, confidence):
    result = run_base(
        tmp_path,
        (TIME_LINE, line + "\n"),
        schedule_impact_days=stated(value, line, direction="increase"),
    )

    assert result.output.to_dict()["schedule_impact_days"] == {
        "value": days,
        "confidence": confidence,
    }


# ---- Approval -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "claimed", "published"),
    [
        # Boxes after their options: the ticked one is "Rejected".
        ("Approved [ ]   Rejected [X]", "approved", None),
        ("Approved [ ]   Rejected [X]", "rejected", "rejected"),
        ("Approved: [ ]   Rejected: [X]", "approved", None),
        # Boxes before their options, as before.
        ("[ ] Approved   [X] Rejected", "approved", None),
        ("[X] Approved   [ ] Rejected", "approved", "approved"),
        # A label with nothing after it, or only something to write on.
        ("Approved by Owner:", "approved", None),
        ("Approved by Owner: -----", "approved", None),
        ("Approved by Owner: . . . . .", "approved", None),
        ("Approved by Owner: ____________   Date: ________", "approved", None),
    ],
)
def test_approval_is_read_from_the_mark_not_from_the_word(tmp_path, line, claimed, published):
    result = run_base(tmp_path, (STATUS_LINE, line + "\n"), approval_status=stated(claimed, line))

    assert result.output.approval_status.value == published
    if published is None:
        refused(result, "approval_status", "status_not_shown_by_quoted_text")


def test_one_signature_is_not_an_approval_while_another_line_is_unsigned(tmp_path):
    block = "Contractor: /s/ J. Smith\nOwner: ______________\n"
    result = run_base(
        tmp_path,
        (STATUS_LINE, block),
        approval_status=stated("approved", "Contractor: /s/ J. Smith"),
    )
    refused(result, "approval_status", "signature_block_is_incomplete")

    signed = run_base(
        tmp_path,
        (STATUS_LINE, "Accepted:\nContractor: /s/ J. Smith\nOwner: /s/ R. Haddad\n"),
        approval_status=stated("approved", "Contractor: /s/ J. Smith", "Owner: /s/ R. Haddad"),
    )
    # A heading over two signed lines is not an unsigned line. Signatures alone: one band lower.
    assert signed.output.to_dict()["approval_status"] == {"value": "approved", "confidence": 0.75}


@pytest.mark.parametrize(
    "line",
    [
        "Status: Approved, subject to the Engineer's review of shop drawings",
        "Status: Approved - as noted",
        "Status: Approved (with conditions)",
    ],
)
def test_an_approval_with_a_condition_goes_to_a_person_however_it_is_punctuated(tmp_path, line):
    result = run_base(
        tmp_path, (STATUS_LINE, line + "\n"), approval_status=stated("approved", line)
    )

    assert result.output.to_dict()["approval_status"] == {"value": "approved", "confidence": 0.75}
    assert codes(result, "approval_status") == {"approval_is_qualified"}
    assert result.output.needs_review is True


# ---- What is published is the document's text ---------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "field", "value"),
    [
        # Both halves are on the quoted line, with another label between them.
        (
            "Project: Maple Court Apartments",
            "Project: Maple Court   Owner: Housing LLC",
            "project_name",
            "Maple Court LLC",
        ),
        (
            "Contract No.: MC-2025-014",
            "Contract No.: MC-2025 (rev 2)",
            "contract_number",
            "MC-2025-2",
        ),
    ],
)
def test_a_value_the_model_assembled_from_pieces_is_not_published(tmp_path, old, new, field, value):
    result = run_base(tmp_path, (old, new), **{field: stated(value, new)})

    refused(result, field, "value_not_in_quoted_text")


# ---- A value under the wrong label --------------------------------------------------------------


@pytest.mark.parametrize("label", ["Invoice Date", "Start Date", "Meeting Date", "Delivery Date"])
def test_the_date_of_something_else_is_not_the_documents_date(tmp_path, label):
    line = f"{label}: 15 January 2026"
    result = run_base(tmp_path, (DATE_LINE, line + "\n"), date=stated("15 January 2026", line))

    refused(result, "date", "date_belongs_to_another_event")


@pytest.mark.parametrize(
    "label",
    ["Date", "Dated", "Change Order Date", "Date Prepared", "Date of Issue", "Issued", "Date of this Change Order"],
)  # fmt: skip
def test_the_documents_own_date_is_still_accepted(tmp_path, label):
    line = f"{label}: 15 January 2026"
    result = run_base(tmp_path, (DATE_LINE, line + "\n"), date=stated("15 January 2026", line))

    assert result.output.to_dict()["date"] == {"value": "2026-01-15", "confidence": 0.9}


def test_a_date_further_down_a_signature_block_is_the_signatures(tmp_path):
    block = "\nOWNER\nBy: R. Haddad\nName: R. Haddad\nTitle: Director\nDate: 21 January 2026\n"
    result = run_base(
        tmp_path,
        (DATE_LINE, ""),
        (STATUS_LINE, STATUS_LINE + block),
        date=stated("21 January 2026", "Date: 21 January 2026"),
    )

    refused(result, "date", "date_belongs_to_another_event")


def test_part_of_a_date_is_not_a_change_order_number(tmp_path):
    result = run_base(
        tmp_path,
        ("Change Order No.: 7\n", ""),
        change_order_number=stated("2026", DATE_LINE.strip()),
    )

    refused(result, "change_order_number", "value_is_part_of_a_date")


def test_a_contract_sum_is_not_a_contract_number(tmp_path):
    result = run_base(
        tmp_path,
        ("Contract No.: MC-2025-014\n", "Contract Sum: 850000\n"),
        contract_number=stated("850000", "Contract Sum: 850000"),
    )

    refused(result, "contract_number", "labelled_as_something_else")
