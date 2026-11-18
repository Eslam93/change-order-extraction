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
    duration_label,
    find_dates,
    find_days,
    find_money,
    has_blank_signing_line,
    header_above,
    headings_above,
    is_granted,
    is_hedged,
    is_requested,
    is_schedule_label,
    label_before,
    label_says_otherwise,
    nearest_label,
    no_change_cue,
    reserves_rights,
    schedule_cues,
    status_cues,
    sum_role,
    ticked_statuses,
    title_cues,
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
        # A comma before exactly three digits groups thousands in money, which has two places.
        # (A point there is refused unless the document shows what its points mean: see below.)
        ("1,500", "1500", False),
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
        # Where the document's own figures prove that it writes a decimal point, "1.500" is
        # neither 1500 nor a way anyone writes one and a half: it is refused.
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
        # A heading over a signature block shows nothing by itself: the block may be unsigned.
        ("Accepted:", {}),
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


# ---- Rules tightened after an independent review ------------------------------------------------
#
# Each case below is a reading the first version got wrong, or one that must not be lost by
# refusing its neighbour. `tests/test_false_accepts.py` runs the same cases through the pipeline.


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("$1.2 million", "figure_is_scaled_by_a_word"),
        ("$350 thousand", "figure_is_scaled_by_a_word"),
        ("USD 2.75 MM", "figure_is_scaled_by_a_word"),
        ("$40 K", "figure_is_scaled_by_a_word"),
        ("$7 bn", "figure_is_scaled_by_a_word"),
        # A point before exactly three digits, and no other figure to say what a point means.
        ("1.500", "point_before_three_digits_is_ambiguous"),
        ("KWD 950.250", "point_before_three_digits_is_ambiguous"),
    ],
)
def test_a_figure_that_is_not_the_whole_amount_is_refused(text, problem):
    (money,) = find_money(text)

    assert money.value is None and money.problem == problem


@pytest.mark.parametrize(
    ("text", "mark", "value"),
    [
        ("1.500", ",", "1500"),  # the document writes a decimal comma, so the point groups
        ("$4,250.00\nK. Alvarez", None, "4250.00"),  # an initial on the next line is not a scale
        ("$4,500 M&E works", None, "4500"),  # nor is the M of "M&E"
        ("$4,250.00 m. to follow", None, "4250.00"),
    ],
)
def test_a_figure_beside_something_that_only_looks_like_a_scale_is_read(text, mark, value):
    money = find_money(text, mark)[0]

    assert money.value == Decimal(value) and money.problem is None


@pytest.mark.parametrize(
    ("text", "negative"),
    [
        # Parentheses after an amount in words restate it. They are not an accountant's minus.
        ("Twelve Thousand Four Hundred Fifty Dollars ($12,450.00)", False),
        ("Twelve Thousand Four Hundred Fifty and 00/100 ($12,450.00)", False),
        ("Credit ($12,450.00)", True),
        # A dash set against the figure is a minus sign; a dash between words is punctuation.
        ("Credit \u2013$12,450.00", True),
        ("Net change \u2013 $12,450.00", False),
        ("Items 2,000\u20133,000", False),
    ],
)
def test_whether_a_figure_is_written_as_negative(text, negative):
    assert find_money(text)[-1].negative is negative


@pytest.mark.parametrize(
    ("text", "value", "problem"),
    [
        ("10-14 days", None, "duration_is_a_range"),
        ("10 to 14 days", None, "duration_is_a_range"),
        ("3 or 4 days", None, "duration_is_a_range"),
        ("(14) calendar days", 14, None),
        ("( -5 ) days", 5, None),
        ("one hundred twenty (120) days", 120, None),  # the figure is read; the words are its tail
        ("one hundred twenty days", None, "number_in_words_not_read"),
        ("twenty (120) days", None, "words_and_figures_disagree"),
    ],
)
def test_a_duration_is_one_number_of_days(text, value, problem):
    (days,) = find_days(text)

    assert (days.value, days.problem) == (value, problem)
    assert days.words_agree is False


def test_a_numeric_date_needs_a_four_digit_year():
    # A cost code, a section number and a year-first date all look like "26-05-19".
    assert find_dates("Cost code: 26-05-19   Section 03.30.00   Ref 12/11/25") == []
    assert date_convention(find_dates("Date: 03/04/2026\nCost code: 26-05-19")) is None
    # With a month's name there is nothing to mistake.
    (token,) = find_dates("5 Mar 26")
    assert (token.first, token.second, token.year, token.two_digit_year) == (5, 3, 2026, True)


@pytest.mark.parametrize(
    ("line", "cues", "ticked"),
    [
        # Boxes after their options.
        ("Approved [ ]   Rejected [X]", {"rejected": "explicit"}, {"rejected"}),
        ("Approved: [X]   Rejected: [ ]", {"approved": "explicit"}, {"approved"}),
        ("Approved [ ] Rejected [X]", {"rejected": "explicit"}, {"rejected"}),
        ("Status:  Approved  [ ]  Rejected  [X]", {"rejected": "explicit"}, {"rejected"}),
        # Boxes before their options, with one space or a column gap between.
        ("Status:  [X] Approved   [ ] Rejected", {"approved": "explicit"}, {"approved"}),
        ("[ ]  Approved  [X]  Rejected", {"rejected": "explicit"}, {"rejected"}),
        ("\u2610 Approved \u2612 Rejected", {"rejected": "explicit"}, {"rejected"}),
        # A label and nothing to show for it.
        ("Approved by Owner:", {}, set()),
        ("Approved by Owner: -----", {}, set()),
        ("Approved by Owner: . . . . .", {}, set()),
        ("Approved by ______________", {}, set()),
        ("Accepted by:   Date:", {}, set()),
        # A name after the label is an approval, stated.
        ("Approved by Owner: R. Haddad", {"approved": "explicit"}, set()),
    ],
)
def test_a_tick_box_or_a_blank_decides_what_a_status_word_shows(line, cues, ticked):
    assert status_cues(line) == cues
    assert ticked_statuses(line) == ticked


@pytest.mark.parametrize(
    ("text", "blank"),
    [
        ("Contractor: /s/ J. Smith\nOwner: ______________", True),
        ("Approved by Owner: ______   Date: ____", True),
        ("By:\n\nTitle: Director", True),
        ("Authorized by:   Date:", True),
        # A heading over signed lines, and labels with something after them.
        ("Accepted:\nOwner: /s/ R. Haddad\nContractor: /s/ J. Smith", False),
        ("Approved by: R. Haddad", False),
        ("Requested by:\nBuilding Management", False),
        ("Owner: Cedar Ridge School District", False),
    ],
)
def test_whether_a_signature_block_has_an_unsigned_line(text, blank):
    assert has_blank_signing_line(text) is blank


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Date: 15 January 2026", "own"),
        ("Change Order Date: 15 January 2026", "own"),
        ("Date Prepared: 15 January 2026", "own"),
        ("Date of Issue: 15 January 2026", "own"),
        ("Date of this Change Order: 15 January 2026", "own"),
        ("Original date: 15 January 2026", "own"),
        # Any other word before "date" makes it the date of something else.
        ("Invoice Date: 15 January 2026", "other"),
        ("Start Date: 15 January 2026", "other"),
        ("Meeting Date: 15 January 2026", "other"),
        ("Owner's Date: 15 January 2026", "other"),
        ("Project: Gym Date: 15 January 2026", "other"),  # where the name ends cannot be told
    ],
)
def test_only_the_documents_own_date_label_makes_a_date_its_own(line, kind):
    assert date_label(line, line.index("15")) == kind


def test_a_date_anywhere_under_a_signature_is_the_signatures():
    block = "OWNER\nBy: R. Haddad\nName: R. Haddad\nTitle: Director\nDate: 21 January 2026"
    assert date_label(block, block.index("21")) == "other"

    letter = "/s/ R. Haddad\n\n21 January 2026"
    assert date_label(letter, letter.index("21")) == "other"

    # A header's "Requested by:" is not a signature.
    header = "Change Order No.: 7\nRequested by: Owner\nContract No.: C-1\nDate: 21 January 2026"
    assert date_label(header, header.index("21")) == "own"


@pytest.mark.parametrize(
    ("before", "after", "kind"),
    [
        ("Contract Time: extended by ", ".", "change"),
        ("Add ", " to the schedule.", "change"),
        ("Days", "", "change"),  # a column headed "Days"
        ("Revised Contract Time: ", "", "total"),
        ("The Contract Time prior to this Change Order was ", ".", "total"),
        ("The Contract Time will be (increased) (decreased) by ", ".", "change"),  # a blank form
        ("Net change by previous Change Orders: ", "", "total"),
        ("Contract Time: extended by up to ", ".", "limit"),
        ("Contract Time: extended by ", ", to be confirmed.", "limit"),
        ("Estimated schedule impact: ", "", "limit"),
        ("Payment is due within ", " of substantial completion.", "other"),
        ("This proposal is valid for ", ".", "other"),
        ("Lead time: ", "", "other"),
        ("Crane - ", "", None),
        # Only the duration's own sentence counts.
        ("No change to the Contract Time. Mobilization ", ".", None),
        ("Crane rental: ", ". Schedule impact: none", None),
    ],
)
def test_what_the_words_around_a_duration_make_it(before, after, kind):
    assert duration_label(before, after) == kind


@pytest.mark.parametrize(
    ("label", "role"),
    [
        ("The original Contract Sum was", "original"),
        ("Original Contract Sum:", "original"),
        ("Contract Sum prior to this Change Order:", "before"),
        ("The Contract Sum prior to this Change Order was", "before"),
        ("Previous Contract Sum:", "before"),
        ("New Contract Sum including this Change Order:", "after"),
        ("Revised Contract Sum", "after"),
        ("Net change by previous Change Orders:", "earlier_changes"),
        # A label that does not say which sum, or names no contract sum at all.
        ("Current Contract Sum:", None),
        ("Subtotal:", None),
        ("Total:", None),
        ("Amount of this change order:", None),
    ],
)
def test_a_contract_sum_plays_the_role_its_own_label_gives_it(label, role):
    assert sum_role(label) == role


def test_a_ceiling_said_after_the_figure_is_still_a_ceiling():
    assert amount_label("Amount of this change order: ", " (not to exceed)") == "limit"
    assert amount_label("Amount of this change order: ", " NTE") == "limit"
    assert amount_label("Amount of this change order: ", " estimated") == "limit"
    assert amount_label("Amount of this change order: ", " (add)") == "change"
    # The blank form's "(increased) (decreased)" does not make the sentence a contract total.
    blank = (
        "The Contract Sum will be (increased) (decreased) by this Change Order in the amount of: "
    )
    assert amount_label(blank) == "change"


def test_headings_are_read_for_a_row_whose_own_column_cannot_be_told():
    table = "Item  Description  Add  Deduct\n1  Delete canopy  $12,450.00"
    figure = table.index("$")

    assert header_above(table, figure) is None  # three cells under four headings
    assert amount_cues(headings_above(table, figure)) == {"increase", "decrease"}

    aligned = "Item  Description  Deduct\n1  Delete canopy  $12,450.00"
    assert header_above(aligned, aligned.index("$")) == "Deduct"
    assert headings_above(aligned, aligned.index("$")) == ""  # the column is known


def test_a_title_gives_the_direction_and_a_hedge_takes_a_statement_back():
    assert title_cues("DEDUCTIVE CHANGE ORDER\nProject: X") == {"decrease"}
    assert title_cues("CHANGE ORDER\nThis supersedes the earlier credit.") == set()
    assert is_hedged("No time extension is requested at this time.")
    assert is_hedged("Time: no change, subject to review")
    assert not is_hedged("No change to the Contract Time.")
    assert reserves_rights("Contractor reserves the right to claim additional time.")
    assert not reserves_rights("No change to the Contract Time.")


@pytest.mark.parametrize(
    ("field", "label", "otherwise"),
    [
        ("contract_number", "Contract Sum:", True),
        ("contract_number", "Contract Date:", True),
        ("contract_number", "Contract Time:", True),
        ("change_order_number", "Change Order Date:", True),
        ("change_order_number", "Change Order Amount:", True),
        ("change_order_number", "Date:", True),
        ("contract_number", "Contract No.:", False),
        ("change_order_number", "Change Order No.:", False),
    ],
)
def test_a_label_for_a_date_or_a_sum_does_not_label_an_identifier(field, label, otherwise):
    assert label_says_otherwise(field, label) is otherwise
