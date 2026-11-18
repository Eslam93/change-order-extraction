"""The fixed rules that read money, durations and dates, and the lexicons beside them."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from changeorder.normalize import (
    amount_cues,
    amount_label,
    date_convention,
    date_label,
    decimal_mark_of,
    find_dates,
    find_days,
    find_money,
    is_granted,
    is_requested,
    is_schedule_label,
    label_before,
    label_says_otherwise,
    nearest_label,
    no_change_cue,
    schedule_cues,
    status_cues,
)


@pytest.mark.parametrize(
    ("text", "value", "negative"),
    [
        ("$12,450.00", "12450.00", False),
        ("$ 4,860,000.00", "4860000.00", False),
        ("12450", "12450", False),
        ("USD 1,250", "1250", False),
        ("1.234,56 EUR", "1234.56", False),  # a point groups, a comma is the decimal mark
        ("1 234,56", "1234.56", False),  # spaces group when a decimal comma follows
        ("12,50", "12.50", False),
        # One mark before exactly three digits groups thousands in money, which has two places...
        ("1,500", "1500", False),
        ("1.500", "1500", False),
        ("($110,983.75)", "110983.75", True),  # parentheses: an accountant's minus
        ("(5,000.00)", "5000.00", True),
        ("-$5,000.00", "5000.00", True),
        ("$-5,000.00", "5000.00", True),
        ("5,000.00 CR", "5000.00", True),
    ],
)
def test_one_figure_is_read_with_its_size_and_whether_it_is_written_negative(text, value, negative):
    (money,) = find_money(text)

    assert money.value == Decimal(value)
    assert money.negative is negative
    assert money.problem is None


@pytest.mark.parametrize(
    ("text", "mark"),
    [
        # ...unless the document's own figures prove that it writes a decimal point. Then
        # "1.500" is neither 1500 nor a way anyone writes one and a half: it is refused.
        ("1.500", "."),
        ("12.505", "."),
        ("1,500", ","),
    ],
)
def test_a_figure_with_three_decimals_is_refused(text, mark):
    (money,) = find_money(text, mark)

    assert money.value is None
    assert money.problem in {"more_than_two_decimal_places", "too_precise"}


def test_groups_that_are_not_thousands_are_not_a_figure():
    assert find_money("1,23,456.00") == []
    assert find_money("12,45,0.00") == []


def test_what_is_not_an_amount_is_not_read_as_one():
    text = "Crane - 1 day, RTU-3, (8.25% of 45,660.00), 400A panel, 03/04/2026, CO-011"

    found = {money.raw: money for money in find_money(text)}

    assert "8.25" not in found and "400" not in found  # a percentage, a rating
    assert all(not money.negative for money in found.values())  # no dash here is a minus
    assert found["45,660.00"].value == Decimal("45660.00")
    assert not {"03", "04", "011", "3"} & set(found)  # parts of a date and of codes


def test_the_documents_own_figures_prove_its_decimal_mark():
    assert decimal_mark_of("Total 4,120.50 and 12,450.00") == "."
    assert decimal_mark_of("Summe 4.120,50") == ","
    assert decimal_mark_of("Total 1,500 and 2,000") is None  # proves nothing
    assert decimal_mark_of("4,120.50 and 4.120,50") is None  # they disagree


@pytest.mark.parametrize(
    ("text", "value", "words_agree", "in_weeks"),
    [
        ("14 calendar days", 14, False, False),
        ("fourteen (14) calendar days", 14, True, False),
        ("ten working days", 10, False, False),
        ("twenty-one (21) days", 21, True, False),
        ("4 working days", 4, False, False),
        ("1 day", 1, False, False),
        ("0 days", 0, False, False),
        ("2 weeks", 14, False, True),
        ("three (3) weeks", 21, True, True),
    ],
)
def test_a_duration_is_read_in_days(text, value, words_agree, in_weeks):
    (days,) = find_days(text)

    assert (days.value, days.words_agree, days.in_weeks, days.problem) == (
        value,
        words_agree,
        in_weeks,
        None,
    )


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("ten (12) calendar days", "words_and_figures_disagree"),
        ("2.5 days", "not_whole_days"),
        ("2 months", "months_have_no_fixed_length"),
    ],
)
def test_a_duration_that_is_not_a_whole_number_of_days_is_refused(text, problem):
    (days,) = find_days(text)

    assert days.value is None
    assert days.problem == problem


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-04", datetime.date(2026, 3, 4)),
        ("2026/03/04", datetime.date(2026, 3, 4)),
        ("March 4, 2026", datetime.date(2026, 3, 4)),
        ("Mar. 4th, 2026", datetime.date(2026, 3, 4)),
        ("4 March 2026", datetime.date(2026, 3, 4)),
        ("12-Jan-2026", datetime.date(2026, 1, 12)),
        ("04 MAR 26", datetime.date(2026, 3, 4)),
        ("25/03/2026", datetime.date(2026, 3, 25)),  # 25 is not a month: day first
        ("03/25/2026", datetime.date(2026, 3, 25)),  # 25 is not a month: month first
        ("25.03.2026", datetime.date(2026, 3, 25)),
        ("04/04/2026", datetime.date(2026, 4, 4)),
    ],
)
def test_a_date_that_reads_one_way_is_normalised(text, expected):
    (token,) = find_dates(text)

    assert token.resolve(None) == (expected, None, False)


def test_a_date_that_reads_two_ways_needs_the_documents_convention():
    (token,) = find_dates("03/04/2026")

    assert token.resolve(None) == (None, "day_and_month_ambiguous", False)
    assert token.resolve("mdy") == (datetime.date(2026, 3, 4), None, True)
    assert token.resolve("dmy") == (datetime.date(2026, 4, 3), None, True)


def test_the_convention_comes_only_from_dates_that_prove_one():
    assert date_convention(find_dates("03/04/2026 and 03/09/2026")) is None
    assert date_convention(find_dates("03/04/2026 and 25/03/2026")) == "dmy"
    assert date_convention(find_dates("03/04/2026 and 03/25/2026")) == "mdy"
    assert date_convention(find_dates("25/03/2026 and 03/25/2026")) is None  # they disagree
    assert date_convention(find_dates("March 4, 2026")) is None  # a month name proves no order


def test_an_impossible_date_and_a_date_without_a_year_are_not_dates():
    (token,) = find_dates("30/02/2026")
    assert token.resolve(None) == (None, "not_a_calendar_date", False)
    assert find_dates("per your email of 2/3 and drawing 14/A") == []


@pytest.mark.parametrize(
    ("text", "cues"),
    [
        ("The Contract Sum will be increased by this Change Order", {"increase"}),
        ("Change Order total (add)", {"increase"}),
        ("NET CHANGE - DEDUCT", {"decrease"}),
        ("Credit for deleted flooring", {"decrease"}),
        ("The Contract Sum will be (increased) (decreased) by", {"increase", "decrease"}),
        ("AMOUNT APPROVED:", set()),
        ("Furnish twelve additional data drops", set()),  # scope wording says nothing about price
    ],
)
def test_the_wording_gives_the_direction_of_a_price_change(text, cues):
    assert amount_cues(text) == cues


@pytest.mark.parametrize(
    ("text", "cues"),
    [
        ("The Contract Time will be increased by fourteen (14) calendar days", {"increase"}),
        ("Time extension granted: 4 working days", {"increase"}),
        ("the Contract Time is decreased by ten (10) calendar days", {"decrease"}),
        ("Completion is accelerated by 5 working days", {"decrease"}),
        ("Contract Time adjustment: 14 calendar days", set()),
    ],
)
def test_the_wording_gives_the_direction_of_a_schedule_change(text, cues):
    assert schedule_cues(text) == cues


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("No change to the contract completion date.", "explicit"),
        ("The Contract Time is unchanged.", "explicit"),
        ("no additional days", "explicit"),
        ("This change is made at no cost to the Owner.", "explicit"),
        ("Schedule impact: None", "explicit"),
        ("Schedule impact: N/A", "not_applicable"),
        ("To be determined.", None),
        ("Work will be performed concurrently with roofing.", None),
    ],
)
def test_no_change_is_recognised_only_when_it_is_said(text, cue):
    assert no_change_cue(text) == cue


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("The original Contract Sum was", "total"),
        ("Net change by previously authorized Change Orders", "total"),
        ("The Contract Sum prior to this Change Order was", "total"),
        ("The new Contract Sum including this Change Order will be", "total"),
        ("Revised subcontract value:", "total"),
        ("Contract price after this Change Order", "total"),
        ("The Contract Sum will be increased by this Change Order in the amount of", "change"),
        ("NET CHANGE - DEDUCT", "change"),
        ("Change Order total (add)", "change"),
        ("TOTAL THIS CHANGE", "change"),
        ("AMOUNT APPROVED:", "change"),
        ("TOTAL PROPOSED", "change"),
        ("Subtotal", "part"),
        ("Overhead & profit 12%", "part"),
        ("Sales tax on material", "part"),
        ("Description: Add a second coat of paint at Level", None),  # a verb, not a label
        ("Job:", None),
    ],
)
def test_the_words_beside_a_figure_say_what_it_is(label, kind):
    assert amount_label(label) == kind


def test_a_label_is_the_text_between_the_previous_figure_and_the_value():
    line = "Contract Date: 5 June 2025   Change Order Date: 18 March 2026"

    assert label_before(line, line.index("18 March")).strip() == "Change Order Date:"
    assert nearest_label(line, line.index("18 March")) == "Change Order Date:"
    assert nearest_label(line, line.index("5 June")) == "Contract Date:"
    # A code is part of a label, not a figure that ends it.
    total = "TOTAL COR-031   12,792.71"
    assert label_before(total, total.index("12,792")).strip() == "TOTAL COR-031"
    # A line item's own label is everything before its last figure.
    item = "Labor - 28 hrs @ 92.00   2,576.00"
    assert label_before(item, item.index("2,576")).startswith("Labor")
    # A form that puts the label on the line above the value.
    stacked = "AMOUNT APPROVED:\n$81,640.55"
    assert "AMOUNT APPROVED" in label_before(stacked, stacked.index("$"))
    # The nearest label is the last column before the value.
    row = "Project: Estuary Point Seawall Repair   CO No.: 6"
    assert nearest_label(row, row.index("6")) == "CO No.:"
    assert nearest_label("January 22, 2026", 0) == ""


@pytest.mark.parametrize(
    ("before", "after", "kind"),
    [
        ("(approx. ", " sq ft), including primer", "measure"),  # a quantity
        ("Labor (64 hrs @ ", ")", "measure"),  # a rate
        ("reduced to ", "/LF per subcontract unit rates", "measure"),
        ("Final amount TBD - not to exceed ", " without further authorization", "limit"),
        ("Budget estimate: ", "", "limit"),
        ("TOTAL COR-031  ", "", "change"),  # a plain total, when nothing more specific is said
        ("Amount: ", " (add)", "change"),  # the word straight after the figure
        ("Contract Sum remains ", ".", "total"),
        ("Piping and insulation  ", "", None),
    ],
)
def test_the_words_around_a_figure_can_make_it_a_quantity_a_ceiling_or_the_change(
    before, after, kind
):
    assert amount_label(before, after) == kind


def test_what_was_asked_for_is_told_from_what_was_granted():
    assert is_requested("Amount requested by Subcontractor:")
    assert is_requested("Time extension requested:")
    assert not is_requested("Amount approved by Contractor:")
    assert is_granted("Time extension granted:") and is_granted("Amount approved by Contractor:")
    assert not is_granted("TOTAL PROPOSED")


@pytest.mark.parametrize(
    ("line", "about_schedule"),
    [
        ("The Contract Time will be increased by", True),
        ("Time extension granted:", True),
        ("Adjustment to completion date ........", True),
        ("Days added:", True),
        ("This request is valid for", False),
        ("Crane -", False),
        ("General conditions (", False),
    ],
)
def test_a_duration_is_the_schedule_impact_only_under_schedule_wording(line, about_schedule):
    assert is_schedule_label(line) is about_schedule


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("Date: 15 January 2026", "15 January", "own"),
        ("Change Order No.: 004   DATE:  March 18, 2026", "March 18", "own"),
        ("Date Prepared:  2026-04-22", "2026-04-22", "own"),
        ("Owner: Stonebridge LP   Issued: 7 May 2026", "7 May", "own"),
        ("Sent: Tuesday, February 10, 2026 4:52 PM", "February 10", "own"),
        ("January 22, 2026", "January 22", "own"),  # a letter's date, alone at the head of a line
        ("Contract Date: 5 June 2025", "5 June", "other"),
        # The nearest words are a name: no label says whose date this is, so it is not "own".
        ("Approved by Owner:  L. Brandt  15 September 2025", "15 September", "unlabelled"),
        ("Status: Approved by Owner 2026-02-24.", "2026-02-24", "other"),
        ("Executed by the City on 20 November 2025", "20 November", "other"),
        ("Owner   By: H. Tran   Date: 20 January 2026", "20 January", "other"),  # a signature line
        ("By: Dana Whitlock\nDate: 03/19/2026", "03/19/2026", "other"),  # under a signature line
        ("Substantial Completion: 30 September 2026", "30 September", "other"),
        ("following the planning condition discharge dated 28/01/2026", "28/01/2026", "other"),
        ("awaiting Board action on 2 June 2026", "2 June", "other"),
        ("geotechnical memo dated April 14, 2026", "April 14", "other"),
        ("discussed at the meeting 12 March 2026 with the team", "12 March", "unlabelled"),
    ],
)
def test_the_words_beside_a_date_say_whose_date_it_is(text, value, kind):
    assert date_label(text, text.index(value)) == kind


@pytest.mark.parametrize(
    ("field", "label", "otherwise"),
    [
        ("project_name", "PROJECT:", False),
        ("project_name", "Job:", False),
        ("project_name", "", False),
        ("project_name", "REQUESTED BY:", True),
        ("project_name", "Contractor:", True),
        ("requested_by", "Change requested by:", False),
        ("requested_by", "Project:", True),
        ("requested_by", "Approved by", True),
        ("contract_number", "GC Subcontract No.", False),
        ("contract_number", "Agreement No.", False),
        ("contract_number", "Change Order No.:", True),
        ("contract_number", "Job No.:", True),
        ("change_order_number", "CHANGE ORDER NO.:", False),
        ("change_order_number", "SCO No.:", False),
        ("change_order_number", "Variation Order No:", False),
        ("change_order_number", "No.", False),
        ("change_order_number", "Contract No.:", True),
        ("change_order_number", "Job:", True),
        # The label nearer the value decides.
        ("change_order_number", "AMENDMENT TO CONTRACT (CHANGE ORDER NO.", False),
        ("change_order_number", "Change Order to Contract No.", True),
        ("description", "Contract No.:", False),  # no label rule for this field
    ],
)
def test_a_label_can_say_a_value_belongs_to_another_field(field, label, otherwise):
    assert label_says_otherwise(field, label) is otherwise


@pytest.mark.parametrize(
    ("text", "cues"),
    [
        ("STATUS: APPROVED - fully executed by all parties.", {"approved": "explicit"}),
        ("[X] Approved as noted   [ ] Rejected   [ ] Resubmit", {"approved": "explicit"}),
        ("[ ] Approved   [X] Rejected", {"rejected": "explicit"}),
        ("Accepted:", {"approved": "explicit"}),
        ("Approved by: Helen Tran", {"approved": "explicit"}),
        ("Signed: /s/ R. Calloway   Date: 15-Jan-2026", {"approved": "signature"}),
        ("STATUS: Pending Owner approval", {"pending": "explicit"}),
        ("*** DRAFT - NOT FOR EXECUTION ***", {"draft": "explicit"}),
        ("VOID - superseded by Change Order No. 8", {"void": "explicit"}),
        ("The Owner has not approved this Change Order.", {"rejected": "explicit"}),
        # Words about approval that show none.
        ("This Change Order is not valid until signed by the Owner and the Engineer.", {}),
        ("Upon approval, the Contractor shall proceed with the work.", {}),
        ("Approved by (Owner):   ______________________   Date: ____________", {}),
        ("Accepted by (Contractor):", {}),
        ("Changes must be approved by the Owner in writing.", {}),
        ("Work will not proceed until we receive written approval.", {}),
        ("Authorization: signatures of the Owner and the Contractor appear on page 2.", {}),
        ("This change supersedes field directive FD-09.", {}),
    ],
)
def test_what_a_passage_shows_about_approval(text, cues):
    assert status_cues(text) == cues
