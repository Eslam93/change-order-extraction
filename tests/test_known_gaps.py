"""Known gaps: wrong values that can still be published, and correct readings still refused.

After the fixes pinned in `test_false_accepts.py`, a second independent review attacked the new
rules with documents and model replies of its own. One class of its findings needed no mistake
from the model (a figure scaled by a word, "$1.2 mil") and was fixed. The rest are here, not
fixed, each pinned as an expected failure.

Read the two lists differently:

- `WRONG_VALUES`: the model's reply is wrong in the way the case scripts it (it points at a
  line item, reports "no change" for an unticked option, takes a label for an approval), and
  the checks let the value through. With the reply a careful model gives, these documents are
  read correctly or refused.
- `REFUSED`: the reply is correct, and the checks refuse it. The field comes out null and the
  document goes to review. Nothing wrong is published, and a person does work a better lexicon
  would have saved.

Every case is marked `xfail(strict=True)`: a case that starts passing fails the build until it
is moved to `test_false_accepts.py`. That keeps the list in `docs/WRITEUP.md` honest in both
directions.
"""

from __future__ import annotations

from typing import Any

import pytest

from helpers import AMOUNT_LINE, DATE_LINE, STATUS_LINE, TIME_LINE, codes, run_base, stated

A, T, D, S = AMOUNT_LINE, TIME_LINE, DATE_LINE, STATUS_LINE
WRONG_VALUES: list[Any] = []
REFUSED: list[Any] = []


def inc(value: str, *quotes: str) -> dict[str, Any]:
    return stated(value, *quotes, direction="increase")


def dec(value: str, *quotes: str) -> dict[str, Any]:
    return stated(value, *quotes, direction="decrease")


def nc(value: str, *quotes: str) -> dict[str, Any]:
    return stated(value, *quotes, direction="no_change")


def wrong(name: str, field: str, *edits: tuple[str, str], **reply: Any) -> None:
    """A document and a mistaken reply for which `field` should come out null, and does not."""
    WRONG_VALUES.append(pytest.param(field, edits, reply, id=name))


def refused(name: str, field: str, expected: Any, *edits: tuple[str, str], **reply: Any) -> None:
    """A document and a correct reply for which `field` should be `expected`, and is null."""
    REFUSED.append(pytest.param(field, expected, edits, reply, id=name))


# ---- Wrong values --------------------------------------------------------------------------------

# A running total of change orders that happens to equal revised minus original.
G701 = (
    "Original Contract Sum: $100,000.00\n"
    "Net change by previously authorized Change Orders: $20,000.00\n"
    "Contract Sum prior to this Change Order: $120,000.00\n"
    "Amount of this change order (add): $80,000.00\n"
    "New Contract Sum including this Change Order: $200,000.00\n"
    "Total change orders to date: $100,000.00\n"
)
wrong(
    "amount: the total of all change orders to date, vouched for by the contract sums",
    "amount",
    (A, G701),
    amount=inc("$100,000.00", "Total change orders to date: $100,000.00"),
    original_contract_sum=stated("$100,000.00", "Original Contract Sum: $100,000.00"),
    contract_sum_after=stated(
        "$200,000.00", "New Contract Sum including this Change Order: $200,000.00"
    ),
)

# A range or an approximation whose numbers are written in words and figures.
for line, value in (
    ("Contract Time: extended by ten (10) to fourteen (14) days.", "fourteen (14) days"),
    ("Contract Time: extended by about ten (10) days.", "ten (10) days"),
    ("Contract Time: extended by ten (10) days unless float is available.", "ten (10) days"),
    ("Contract Time: extended by seven to ten days.", "ten days"),
    ("Contract Time: extended by about 10 days.", "10 days"),
    ("Expected schedule impact: 10 days", "10 days"),
    ("Time extension: 10 days (not yet agreed)", "10 days"),
):
    wrong(f"schedule, not firm: {line}", "schedule_impact_days", (T, line + "\n"),
          schedule_impact_days=inc(value, line))  # fmt: skip

# "None" and "unchanged" as an option printed on a form, not as the answer.
for line, field, value in (
    ("Cost impact:  [ ] None   [X] Yes, see attached quote", "amount", "None"),
    ("Schedule impact:  [ ] None   [X] Yes - see attached fragnet", "schedule_impact_days", "None"),
    ("Schedule impact:  None [ ]   Yes [X]", "schedule_impact_days", "None"),
    (
        "The Contract Time will be [X] increased  [ ] decreased  [ ] unchanged by 3 days.",
        "schedule_impact_days",
        "unchanged",
    ),
    (
        "The Contract Time will be (increased) (decreased) (unchanged) by ( ) days.",
        "schedule_impact_days",
        "unchanged",
    ),
    ("The Contract Sum will be (increased) (decreased) (unchanged) by $_______.", "amount", "unchanged"),
):  # fmt: skip
    wrong(f"zero from a printed option: {line}", field,
          (A if field == "amount" else T, line + "\n"), **{field: nc(value, line)})  # fmt: skip

# "No change", a zero or "none" that is about something else.
wrong("zero: the deductions column is zero, the additions are not", "amount",
      (A, "Additions: $12,450.00\nDeductions: $0.00\n"),
      amount=nc("$0.00", "Deductions: $0.00"))  # fmt: skip
wrong("zero: no charge for delivery", "amount",
      (A, "Cost: $12,450.00\nNo charge for delivery.\n"),
      amount=nc("No charge", "No charge for delivery."))  # fmt: skip
JOINED = "Cost: see the attached breakdown, Time: none"
wrong("zero: 'none' after a comma belongs to the time", "amount", (A, ""), (T, JOINED + "\n"),
      amount=nc("none", JOINED))  # fmt: skip
CELLS = "See attached  None  Paint columns"
wrong("zero: 'None' in the next column", "amount",
      (A, ""), (T, "Cost  Time  Notes\n" + CELLS + "\n"), amount=nc("None", CELLS))  # fmt: skip
wrong("zero: no days added, five deducted", "schedule_impact_days",
      (T, "Days added: 0   Days deducted: 5\n"),
      schedule_impact_days=nc("0", "Days added: 0   Days deducted: 5"))  # fmt: skip
wrong("zero: the Schedule of Values is not the schedule", "schedule_impact_days",
      (T, "No change to the Schedule of Values.\n"),
      schedule_impact_days=nc("No change", "No change to the Schedule of Values."))  # fmt: skip

# A hedge in other words, or in the next sentence.
for text, value, quote in (
    ("Contract Time: no change; subject to the Engineer's review.", "no change", None),
    (
        "No time extension is requested.\nTime impact to be determined after receipt of steel "
        "delivery dates.",
        "No time extension",
        "No time extension is requested.",
    ),
    (
        "No time extension is requested.\nThis Change Order is issued with full reservation of "
        "rights.",
        "No time extension",
        "No time extension is requested.",
    ),
    ("Contract Time: no change (preliminary)", "no change", None),
):
    wrong("hedged zero: " + text.replace("\n", " / "), "schedule_impact_days", (T, text + "\n"),
          schedule_impact_days=nc(value, quote or text))  # fmt: skip
wrong("hedged zero: final pricing to follow", "amount",
      (A, "Cost: no change; final pricing to follow.\n"),
      amount=nc("no change", "Cost: no change; final pricing to follow."))  # fmt: skip
NOT_PRICED = "Price: N/A - work to be performed on a time and material basis"
wrong("hedged zero: N/A because the work is time and material", "amount",
      (A, NOT_PRICED + "\n"), amount=nc("N/A", NOT_PRICED))  # fmt: skip

# What was requested, where something else was granted, in words the rule does not know.
for block, quote in (
    ("Time extension requested: 10 days\nNo time extension is granted.\n", "Time extension requested: 10 days"),
    ("Time extension requested: 10 days. Time extension granted: none.\n", None),
    ("Time extension requested: 10 days. Denied by Owner.\n", None),
    ("Time extension: 10 days requested, 4 days granted\n", None),
    ("Time extension requested: 10 days\nApproved: 4 days\n", "Time extension requested: 10 days"),
):  # fmt: skip
    wrong("requested, not granted: " + block.strip().replace("\n", " / "), "schedule_impact_days",
          (T, block), schedule_impact_days=inc("10 days", quote or block.strip()))  # fmt: skip
CERTIFIED = "Amount requested: $14,000.00\nAmount certified by Architect: $12,450.00\n"
wrong("requested, not granted: the amount the Architect certified is lower", "amount",
      (A, CERTIFIED), amount=inc("$14,000.00", "Amount requested: $14,000.00"))  # fmt: skip

# Another figure, under a label the lexicon does not rank below the total.
wrong("figure: a line item, with a Grand Total below it", "amount",
      (A, "Demolition: $1,900.00\nNew framing: $10,550.00\nGrand Total: $12,450.00\n"),
      amount=inc("$10,550.00", "New framing: $10,550.00"))  # fmt: skip
wrong("figure: retainage", "amount", (A, "Cost: $12,450.00\nRetainage held: $6,225.00\n"),
      amount=inc("$6,225.00", "Retainage held: $6,225.00"))  # fmt: skip
wrong("figure: an hourly rate", "amount", (A, "Cost: $12,450.00\nRate: $85.00 per hour\n"),
      amount=inc("$85.00", "Rate: $85.00 per hour"))  # fmt: skip
wrong("figure: an alternate that was not accepted", "amount",
      (A, "Cost: $12,450.00\nAlternate No. 2 (not accepted): $8,300.00\n"),
      amount=inc("$8,300.00", "Alternate No. 2 (not accepted): $8,300.00"))  # fmt: skip
LINE_TOTALS = (
    "Item  Description  Qty  Total\n"
    "1  Paint columns  2  $6,000.00\n"
    "2  Primer  1  $6,450.00\n"
    "Lump sum: $12,450.00\n"
)
wrong("figure: one row's total in a table of rows", "amount", (A, LINE_TOTALS),
      amount=inc("$6,000.00", "1  Paint columns  2  $6,000.00"))  # fmt: skip
for line in (
    "Amount of this change order: $22,000.00 (provisional)",
    "Amount of this change order: $22,000.00 (T&M, final cost to be determined)",
    "Amount of this change order: $22,000.00 (subject to final measurement)",
    "ROM cost: $22,000.00",
    "Amount of this change order (add): $22,000.00 per month",
    "Amount of this change order (add): $22,000.00 plus applicable taxes and bond",
):
    wrong(f"figure, not firm: {line}", "amount", (A, line + "\n"), amount=inc("$22,000.00", line))

# An approval read from a label, a negative answer, another party, or another subject.
SQUARES = "Approved " + chr(9633) + "  Rejected " + chr(9632)
for line in (
    "Approved by Owner: not yet",
    "Approved by Owner: pending",
    "Approved by Owner: (not signed)",
    "Approved By   Date",
    "Authorized Signature   Date",
    "Approved: No",
    "Approved: [ ] Yes [X] No",
    SQUARES,
    "Owner: /s/ ______________   Date: ________",
    "Signed copy to follow.",
    "Net change by previously authorized Change Orders: $0.00",
    "This Change Order amends the executed Agreement dated 5 June 2025.",
    "This Change Order is approved when signed by the Owner.",
):
    wrong(f"approval: {line}", "approval_status", (S, line + "\n"),
          approval_status=stated("approved", line))  # fmt: skip
wrong("approval: the Contractor accepted, the Owner's line is empty", "approval_status",
      (S, "Recommended by Architect: J. Lee\nAccepted by Contractor: J. Smith\nOwner: ______________\n"),
      approval_status=stated("approved", "Accepted by Contractor: J. Smith"))  # fmt: skip
for owner in ("Owner:", "Owner: not signed"):
    wrong(f"approval: one signature, and '{owner}'", "approval_status",
          (S, "Contractor: /s/ J. Smith\n" + owner + "\n"),
          approval_status=stated("approved", "Contractor: /s/ J. Smith"))  # fmt: skip

# A date that belongs to something else, in a layout the label rule does not follow.
SIGNED = "Approved by Owner: R. Haddad   Date: 21 January 2026"
wrong("date: the approval's, on the signature line, in a document with no date of its own",
      "date", (D, ""), (S, S + SIGNED + "\n"), date=stated("21 January 2026", SIGNED))  # fmt: skip
wrong("date: the contract's, with its label on the line above", "date",
      (D, "Contract Date:\n5 June 2025\n"), date=stated("5 June 2025", "5 June 2025"))  # fmt: skip
for name, block in (
    ("an empty line inside the signature block", "\nOWNER\nBy: R. Haddad\nTitle: Director\n\nDate: 21 January 2026\n"),
    ("a signature block with no 'By:'", "\nOWNER\nR. Haddad\nDirector\nDate: 21 January 2026\n"),
):  # fmt: skip
    wrong(f"date: a signature's, {name}", "date", (D, ""), (S, S + block),
          date=stated("21 January 2026", "Date: 21 January 2026"))  # fmt: skip
WRAPPED = (
    "The change was directed by the Architect at the site meeting on\n"
    "12 January 2026 and confirmed by email.\n"
)
wrong("date: a meeting's, wrapped to the head of a line", "date", (D, WRAPPED),
      date=stated("12 January 2026", "12 January 2026 and confirmed by email."))  # fmt: skip

# A sign the wording contradicts, in words or marks the direction lexicon does not read.
CREDIT_ITEM = "Credit for unused allowance (included above): $2,000.00"
NET = "Net amount of this change order: $12,450.00"
wrong("sign: a decrease borrowed from another quoted line", "amount",
      (A, CREDIT_ITEM + "\n" + NET + "\n"), amount=dec("$12,450.00", NET, CREDIT_ITEM))  # fmt: skip
IN_WORDS = "Amount: Twelve Thousand Four Hundred Fifty ($12,450.00)"
wrong("sign: parentheses after words that do not end in 'Dollars'", "amount",
      (A, IN_WORDS + "\n"), amount=dec("$12,450.00", IN_WORDS))  # fmt: skip
REDUCED = "Additional cost for reduced-height partitions: $12,450.00"
wrong("sign: 'reduced' describing the work, not the price", "amount",
      (A, REDUCED + "\n"), amount=dec("$12,450.00", REDUCED))  # fmt: skip
PLAIN = "Amount of this change order: $12,450.00"
for line in (
    "Amount of this change order: $12,450.00 - CREDIT",
    "A total of $12,450.00 will be deducted from the Contract Sum.",
    "Amount of this change order: <$12,450.00>",
    "Amount of this change order: 12,450.00-",
):
    wrong(f"sign: a credit read as an increase: {line}", "amount", (A, line + "\n"),
          amount=inc("12,450.00" if line.endswith("-") else "$12,450.00", line))  # fmt: skip
wrong("sign: 'CHANGE ORDER - DEDUCT' as the title", "amount",
      ("CHANGE ORDER\n", "CHANGE ORDER - DEDUCT\n"), (A, PLAIN + "\n"),
      amount=inc("$12,450.00", PLAIN))  # fmt: skip
wrong("sign: a 'Credit' heading over a table of two columns", "amount",
      (A, "Description  Credit\nDelete canopy  $12,450.00\n"),
      amount=inc("$12,450.00", "Delete canopy  $12,450.00"))  # fmt: skip
DEDUCT_DAYS = "Deduct 5 working days from the schedule."
wrong("sign: 'deduct' said of days", "schedule_impact_days", (T, DEDUCT_DAYS + "\n"),
      schedule_impact_days=inc("5 working days", DEDUCT_DAYS))  # fmt: skip

# A duration that is not the change in contract time.
for line, value in (
    ("Lead time of 14 days", "14 days"),
    ("Material lead time is 6 weeks.", "6 weeks"),
    ("Duration of the work: 10 working days", "10 working days"),
    ("Total time extensions to date: 35 days", "35 days"),
    ("Contract Time: 365 calendar days", "365 calendar days"),
    ("Days added: 14   Days deducted: 3", "14"),
    ("Completion date: June 30", "30"),
    ("Contract Time: extended by 3 weeks and 2 days.", "3 weeks"),
    ("Work will take 5 days to complete, with no impact to the completion date.", "5 days"),
):
    wrong(f"duration: {line}", "schedule_impact_days", (T, line + "\n"),
          schedule_impact_days=inc(value, line))  # fmt: skip

# An identifier or a name that is close to the right one.
wrong("identifier: the contract number without its last digit", "contract_number",
      ("Contract No.: MC-2025-014", "Contract No.: MC-2025-0142"),
      contract_number=stated("MC-2025-014", "Contract No.: MC-2025-0142"))  # fmt: skip
wrong("identifier: a proposal number as the contract number", "contract_number",
      ("Contract No.: MC-2025-014\n", "Contract No.: MC-2025-014\nProposal No.: P-118\n"),
      contract_number=stated("P-118", "Proposal No.: P-118"))  # fmt: skip
wrong("identifier: a job number, with its label on the line above", "contract_number",
      ("Contract No.: MC-2025-014\n", "Job No.:\n2214\nContract No.:\nMC-2025-014\n"),
      contract_number=stated("2214", "2214"))  # fmt: skip
wrong("identifier: a PCO number as the change order number", "change_order_number",
      ("CHANGE ORDER\n", "CHANGE ORDER 7\n"), ("Change Order No.: 7\n", "PCO No.: 31\n"),
      change_order_number=stated("31", "PCO No.: 31"))  # fmt: skip
wrong("name: the Architect as the requester", "requested_by",
      (S, S + "Architect: Studio Lee Architects\n"),
      requested_by=stated("Studio Lee Architects", "Architect: Studio Lee Architects"))  # fmt: skip
wrong("name: 'the Owner' out of 'Contractor, not the Owner'", "requested_by",
      ("Requested by: Maple Court Housing LLC", "Requested by: Contractor, not the Owner"),
      requested_by=stated("the Owner", "Requested by: Contractor, not the Owner"))  # fmt: skip


@pytest.mark.xfail(strict=True, reason="known gap: a mistaken reply gets a wrong value published")
@pytest.mark.parametrize(("field", "edits", "reply"), WRONG_VALUES)
def test_a_mistaken_reply_that_still_gets_a_value_published(tmp_path, field, edits, reply):
    result = run_base(tmp_path, *edits, **reply)

    assert result.output.to_dict()[field]["value"] is None


@pytest.mark.xfail(strict=True, reason="known gap: a condition on the approval is not noticed")
@pytest.mark.parametrize(
    "line",
    [
        "Status: Approved in principle; final pricing to follow",
        "Status: Approved for budget purposes only",
    ],
)
def test_an_approval_qualified_in_words_the_lexicon_does_not_know(tmp_path, line):
    result = run_base(tmp_path, (S, line + "\n"), approval_status=stated("approved", line))

    assert codes(result, "approval_status") == {"approval_is_qualified"}


# ---- Correct readings that are refused ---------------------------------------------------------

# A second date that the label rule takes for the document's own.
refused("date: a signature line that carries a date", "date", "2026-01-15",
        (S, S + SIGNED + "\n"))  # fmt: skip
refused("date: another date wrapped to the head of a line", "date", "2026-01-15",
        (D, D + WRAPPED))  # fmt: skip
refused("date: a table of events below the header", "date", "2026-01-15",
        (D, D + "Date  Event  By\n12 January 2026  Site instruction issued  Architect\n"))  # fmt: skip
refused("date: 'This Change Order is dated ...'", "date", "2026-01-15",
        (D, "This Change Order is dated 15 January 2026.\n"),
        date=stated("15 January 2026", "This Change Order is dated 15 January 2026."))  # fmt: skip
refused("number: a mention of another change order in the description", "change_order_number",
        "7", (D, D + "Credit for the work deleted in Change Order No. 5.\n"))  # fmt: skip

# A second figure that the label rule takes for a rival.
refused("amount: beside the total of change orders to date", "amount", 12450,
        (A, A + "Total Change Orders to date: $52,450.00\n"))  # fmt: skip
refused("amount: under a table of line totals", "amount", 12450,
        (A, "Item  Description  Qty  Total\n1  Paint columns  2  $6,000.00\n2  Primer  1  $6,450.00\n" + A))  # fmt: skip
refused("amount: under 'Total direct cost' and a markup", "amount", 12450,
        (A, "Total direct cost: $11,000.00\nMarkup: $1,450.00\n" + A))  # fmt: skip
refused("amount: 'Add' with a zero 'Deduct' beside it", "amount", 12450,
        (A, "Add: $12,450.00\nDeduct: $0.00\n"), amount=inc("$12,450.00", "Add: $12,450.00"))  # fmt: skip
refused("amount: 'Net Add/(Deduct)'", "amount", 12450, (A, "Net Add/(Deduct): $12,450.00\n"),
        amount=inc("$12,450.00", "Net Add/(Deduct): $12,450.00"))  # fmt: skip
for name, lines in (
    ("certified by the Architect", CERTIFIED),
    ("under 'Approved:'", "Amount requested: $14,000.00\nApproved: $12,450.00\n"),
):
    refused(f"amount: the granted figure, {name}", "amount", 12450, (A, lines),
            amount=inc("$12,450.00", lines.splitlines()[1]))  # fmt: skip
TICKED = "The Contract Sum will be [X] increased  [ ] decreased  [ ] unchanged by: $12,450.00"
BALLOT = (
    "The Contract Sum will be " + chr(9746) + " increased " + chr(9744) + " decreased "
    + chr(9744) + " unchanged by $12,450.00"
)  # fmt: skip
for name, line in (("square brackets", TICKED), ("ballot boxes", BALLOT)):
    refused(f"amount: a form with its direction ticked, {name}", "amount", 12450,
            (A, line + "\n"), amount=inc("$12,450.00", line))  # fmt: skip
refused("amount: 'This is a no-cost change order.'", "amount", 0,
        (A, "This is a no-cost change order.\n"),
        amount=nc("no-cost", "This is a no-cost change order."))  # fmt: skip

# A sign the direction lexicon cannot see, so a correct "decrease" is not believed.
refused("sign: parentheses after words that do not end in 'Dollars'", "amount", 12450,
        (A, IN_WORDS + "\n"), amount=inc("$12,450.00", IN_WORDS))  # fmt: skip
refused("sign: 'reduced' describing the work", "amount", 12450,
        (A, REDUCED + "\n"), amount=inc("$12,450.00", REDUCED))  # fmt: skip
for line in (
    "Amount of this change order: $12,450.00 - CREDIT",
    "A total of $12,450.00 will be deducted from the Contract Sum.",
    "Amount of this change order: <$12,450.00>",
    "Amount of this change order: 12,450.00-",
):
    refused(f"sign: a credit, correctly reported: {line}", "amount", -12450, (A, line + "\n"),
            amount=dec("12,450.00" if line.endswith("-") else "$12,450.00", line))  # fmt: skip
DEDUCT_TABLE = (
    "Item  Description  Deduct\n"
    "1  Delete canopy  $2,450.00\n"
    "2  Delete signage  $10,000.00\n"
    "Total  $12,450.00\n"
)
refused("sign: the total row under a 'Deduct' heading", "amount", -12450, (A, DEDUCT_TABLE),
        amount=dec("$12,450.00", "Total  $12,450.00"))  # fmt: skip
refused("sign: a 'Credit' heading over a table of two columns", "amount", -12450,
        (A, "Description  Credit\nDelete canopy  $12,450.00\n"),
        amount=dec("$12,450.00", "Delete canopy  $12,450.00"))  # fmt: skip
refused("sign: 'deduct' said of days", "schedule_impact_days", -5, (T, DEDUCT_DAYS + "\n"),
        schedule_impact_days=dec("5 working days", DEDUCT_DAYS))  # fmt: skip

# A duration or a "no change" in words the lexicon does not know.
for line, value in (
    ("No schedule impact.", "No schedule impact"),
    ("This change will not affect the completion date.", "will not affect the completion date"),
    ("No additional days.", "No additional days"),
    ("This Change Order does not change the Contract Time.", "does not change the Contract Time"),
):
    refused(f"schedule: no change, said as: {line}", "schedule_impact_days", 0,
            (T, line + "\n"), schedule_impact_days=nc(value, line))  # fmt: skip
refused("schedule: 'None' in the Time column", "schedule_impact_days", 0,
        (A, ""), (T, "Cost  Time  Notes\n" + CELLS + "\n"),
        schedule_impact_days=nc("None", CELLS))  # fmt: skip
for line, value, days in (
    ("Contract Time: extended by 3 (three) calendar days.", "3 (three) calendar days", 3),
    (
        "Contract Time: extended by three (3) consecutive calendar days.",
        "three (3) consecutive calendar days",
        3,
    ),
    ("Contract Time: extended by 5 workdays.", "5 workdays", 5),
    ("The Contract Time will be [X] increased  [ ] decreased  [ ] unchanged by 3 days.", "3 days", 3),
):  # fmt: skip
    refused(f"schedule: {line}", "schedule_impact_days", days, (T, line + "\n"),
            schedule_impact_days=inc(value, line))  # fmt: skip
refused("schedule: the days granted, after the days requested on the same line",
        "schedule_impact_days", 4, (T, "Time extension: 10 days requested, 4 days granted\n"),
        schedule_impact_days=inc("4 days", "Time extension: 10 days requested, 4 days granted"))  # fmt: skip
refused("schedule: the days granted, under 'Approved:'", "schedule_impact_days", 4,
        (T, "Time extension requested: 10 days\nApproved: 4 days\n"),
        schedule_impact_days=inc("4 days", "Approved: 4 days"))  # fmt: skip


@pytest.mark.xfail(strict=True, reason="known gap: a correct reply is refused")
@pytest.mark.parametrize(("field", "expected", "edits", "reply"), REFUSED)
def test_a_correct_reply_that_is_still_refused(tmp_path, field, expected, edits, reply):
    result = run_base(tmp_path, *edits, **reply)

    assert result.output.to_dict()[field]["value"] == expected
