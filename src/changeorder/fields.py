"""From the model's candidates to decisions: accepted, absent, or unresolved.

One rule for every field. A value is accepted only when all of these hold:

1. the model says the document states it;
2. at least one of its quotes is found in the document (`evidence.py`);
3. the value is found inside the quoted lines, and is read from the document's own characters
   by a fixed rule (`normalize.py`) - the model's rendering of a number or a date is used only
   to pick the token, never as the value;
4. nothing deterministic contradicts it: the figure's own sign, the wording's direction, the
   label beside it, or the contract sums before and after.

Anything else is `absent` (the model says it is not there) or `unresolved` (something was there
and could not be accepted), with a reason code. Unresolved fields are published as null and
always send the document to a person. A contradiction never produces a value: the pipeline does
not pick a side.

An accepted value may carry caveats (an approximate quote, a date order taken from the
document's other dates, weeks turned into days). They lower its confidence band in
`confidence.py`; they do not change the value.
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from typing import Any

from .document import SourceDocument
from .evidence import collapse, find_quote, locate_passage, locate_text, windows_of
from .models import FIELD_NAMES, Decision
from .normalize import (
    DateToken,
    Days,
    Money,
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
    line_of,
    nearest_label,
    no_change_cue,
    says_price,
    schedule_cues,
    status_cues,
    written_like_money,
)
from .prompt import Candidate, Extraction, SignedCandidate, StatusCandidate

__all__ = ["decide"]

MAX_QUOTES = 5
# One word straight after a figure, bare or alone in parentheses: "credit", "(add)".
_DIRECTLY_AFTER = re.compile(r"\s?(?:\(\w+\)|\w+)")
_BARE_INTEGER = re.compile(r"^\s*([-\u2212+]?)(\d{1,4})\s*$")
_QUALIFIED_APPROVAL = re.compile(
    r"approved\s+(?:as\s+noted|as\s+corrected|with\s+\w+|subject\s+to|in\s+part)"
    r"|(?:partial|conditional)(?:ly)?\s+approv",
    re.IGNORECASE,
)
Window = tuple[int, int]
# What a figure's label can make it instead of the change, and the reason recorded for each.
_NOT_THE_CHANGE = {
    "total": "figure_is_a_contract_total",
    "part": "figure_is_a_part_of_the_change",
    "limit": "figure_is_a_ceiling_or_an_estimate",
    "measure": "figure_is_a_quantity_or_a_rate",
}


class _Context:
    """One document, with its figures, durations and dates found once."""

    def __init__(self, document: SourceDocument, label_checks: bool = True) -> None:
        self.document = document
        self.label_checks = label_checks
        self.text = document.text
        self.decimal_mark = decimal_mark_of(self.text)
        self.money = find_money(self.text, self.decimal_mark)
        self.days = find_days(self.text)
        self.dates = find_dates(self.text)
        self.date_convention = date_convention(self.dates)

    def lines(self, windows: list[Window]) -> str:
        return "\n".join(self.text[start:end] for start, end in windows)


def decide(
    extraction: Extraction, document: SourceDocument, *, label_checks: bool = True
) -> dict[str, Decision]:
    """A decision for each of the nine fields, in contract order.

    `label_checks=False` switches off rule 4's label checks and leaves the rest: a value is
    then accepted wherever its quote is found and it can be read. Nothing in the pipeline does
    that; the stress test does, to measure what those checks are worth.
    """
    context = _Context(document, label_checks)
    decisions = {
        name: _text(context, name, getattr(extraction, name))
        for name in ("project_name", "contract_number", "change_order_number", "requested_by")
    }
    decisions["date"] = _date(context, extraction.date)
    decisions["description"] = _description(context, extraction.description)
    decisions["amount"] = _amount(context, extraction)
    decisions["schedule_impact_days"] = _schedule(context, extraction.schedule_impact_days)
    decisions["approval_status"] = _status(context, extraction.approval_status)
    return {name: decisions[name] for name in FIELD_NAMES}


# ---- Shared steps ------------------------------------------------------------------------------


def _open(
    context: _Context, name: str, candidate: Candidate | StatusCandidate
) -> tuple[Decision, list[Window]]:
    """Steps 1 and 2. Returns the decision so far and the quoted lines; no lines means the
    decision is already final (absent, or unresolved with its reason)."""
    decision = Decision(field=name, state="unresolved", note=candidate.note.strip())
    if candidate.status == "absent":
        decision.state = "absent"
        return decision, []
    quotes = [quote for quote in candidate.quotes if quote.strip()][:MAX_QUOTES]
    decision.evidence = [find_quote(context.document, quote) for quote in quotes]
    if candidate.status == "unclear":
        decision.reasons.append("model_reported_unclear")
        return decision, []
    if not candidate.value.strip():
        decision.reasons.append("stated_without_a_value")
        return decision, []
    windows = windows_of(context.document, decision.evidence)
    if not windows:
        decision.reasons.append("quote_not_found_in_document" if quotes else "no_quote_given")
    return decision, windows


def _unresolved(decision: Decision, reason: str) -> Decision:
    decision.state = "unresolved"
    decision.value = None
    decision.reasons.append(reason)
    return decision


def _accept(context: _Context, decision: Decision, value: Any, span: Window) -> Decision:
    decision.state = "accepted"
    decision.value = value
    decision.span = span
    kinds = [
        item.kind
        for item in decision.evidence
        if item.start is not None
        and any(start <= span[0] < end for start, end in windows_of(context.document, [item]))
    ]
    if kinds and all(kind == "fuzzy" for kind in kinds):
        decision.caveats.append("quote_found_approximately")
    return decision


def _inside(start: int, windows: list[Window]) -> bool:
    """Whether a token starts on a quoted line. A token may run on past the line's end: a
    duration or a date that wraps ("ten (10) calendar" / "days") still belongs to it."""
    return any(low <= start < high for low, high in windows)


def _line(context: _Context, position: int) -> str:
    return context.text[slice(*line_of(context.text, position))]


def _wording(context: _Context, decision: Decision, start: int, end: int) -> str:
    """The words that can say which way the figure at `start`..`end` goes.

    Its label (the text before it on its line); a word that directly follows it ("$5,000
    credit", "4 days earlier", "$7,650.00 (add)"); and the other passages the model quoted, such
    as a title "Deductive Change Order". A remark further along the figure's own line is not
    read: "$13,100.00   (reduced: batts furnished by others)" describes a negotiation, not a sign.
    """
    line_start, line_end = line_of(context.text, start)
    following = _DIRECTLY_AFTER.match(context.text[end:line_end])
    elsewhere = [
        context.text[item.start : item.end]
        for item in decision.evidence
        if item.start is not None
        and item.end is not None
        and not (item.start < line_end and line_start < item.end)
    ]
    return "\n".join(
        [context.text[line_start:start], following.group(0) if following else "", *elsewhere]
    )


def _sign(direction: str, written_negative: bool, cues: set[str]) -> tuple[int | None, str | None]:
    """The sign of a change, or why it cannot be given.

    A figure written as negative, or wording that goes one way only, must agree with the
    direction the model reported. A plain figure with no wording either way is positive as
    written. A decrease that nothing in the document shows is not accepted on the model's word.
    """
    if written_negative:
        return (-1, None) if direction == "decrease" else (None, "direction_contradicts_figure")
    if cues == {"decrease"}:
        return (-1, None) if direction == "decrease" else (None, "direction_contradicts_wording")
    if cues == {"increase"}:
        return (1, None) if direction == "increase" else (None, "direction_contradicts_wording")
    if not cues:
        return (1, None) if direction == "increase" else (None, "decrease_not_shown_by_document")
    return None, "wording_gives_both_directions"


def _no_change(
    context: _Context,
    decision: Decision,
    candidate: SignedCandidate,
    windows: list[Window],
    zero: Decimal | int,
) -> Decision:
    """An explicit zero: the document says the price, or the time, does not change."""
    lines = context.lines(windows)
    stated = f"{candidate.value}\n{lines}"
    figures = [money.value for money in find_money(candidate.value, context.decimal_mark)]
    durations = [days.value for days in find_days(candidate.value)]
    if any(value for value in (*figures, *durations)):
        return _unresolved(decision, "no_change_reported_beside_a_figure")
    cue = no_change_cue(stated)
    zero_written = any(value == 0 for value in (*figures, *durations))
    if not zero_written and cue is None:
        return _unresolved(decision, "no_change_not_shown_by_document")
    located = locate_text(context.document, candidate.value, windows)
    decision.raw = collapse(context.text[slice(*located[0])]) if located else collapse(lines)
    if not zero_written and cue == "not_applicable":
        decision.caveats.append("not_applicable_read_as_no_change")
    return _accept(context, decision, zero, located[0] if located else windows[0])


# ---- Names and identifiers -----------------------------------------------------------------


def _text(context: _Context, name: str, candidate: Candidate) -> Decision:
    decision, windows = _open(context, name, candidate)
    if not windows:
        return decision
    located = locate_text(context.document, candidate.value, windows)
    if located is None:
        return _unresolved(decision, "value_not_in_quoted_text")
    span, in_one_piece = located
    if in_one_piece:
        decision.raw = collapse(context.text[slice(*span)])
    else:
        decision.raw = collapse(candidate.value)
        decision.caveats.append("value_interrupted_by_other_text")
    if context.label_checks and label_says_otherwise(name, nearest_label(context.text, span[0])):
        return _unresolved(decision, "labelled_as_something_else")
    return _accept(context, decision, decision.raw, span)


def _description(context: _Context, candidate: Candidate) -> Decision:
    decision = Decision(field="description", state="unresolved", note=candidate.note.strip())
    if candidate.status == "absent":
        decision.state = "absent"
        return decision
    if candidate.status == "unclear":
        return _unresolved(decision, "model_reported_unclear")
    if not candidate.value.strip():
        return _unresolved(decision, "stated_without_a_value")
    found = locate_passage(context.document, candidate.value)
    if found is None:
        return _unresolved(decision, "wording_not_in_document")
    decision.evidence = found
    decision.raw = " ".join(
        collapse(context.text[item.start : item.end]) for item in found if item.start is not None
    )
    decision.state = "accepted"
    decision.value = decision.raw
    spans = [(item.start, item.end) for item in found if item.start is not None and item.end]
    decision.span = (min(start for start, _ in spans), max(end for _, end in spans))
    if any(item.kind == "fuzzy" for item in found):
        decision.caveats.append("wording_found_approximately")
    return decision


# ---- Date ----------------------------------------------------------------------------------


def _readings(token: DateToken) -> set[datetime.date]:
    """Every calendar date the token could mean."""
    found = {token.resolve(order)[0] for order in ("dmy", "mdy")}
    return {value for value in found if value is not None}


def _date(context: _Context, candidate: Candidate) -> Decision:
    decision, windows = _open(context, "date", candidate)
    if not windows:
        return decision
    wanted = find_dates(candidate.value)
    if len(wanted) != 1:
        return _unresolved(decision, "several_dates_in_value" if wanted else "date_not_readable")
    targets = _readings(wanted[0])
    if not targets:
        return _unresolved(decision, "not_a_calendar_date")
    tokens = [
        token
        for token in context.dates
        if _inside(token.start, windows) and _readings(token) & targets
    ]
    if not tokens:
        return _unresolved(decision, "value_not_in_quoted_text")
    kinds = {token.start: date_label(context.text, token.start) for token in tokens}
    own = [token for token in tokens if kinds[token.start] == "own" or not context.label_checks]
    if not own:
        decision.raw = tokens[0].raw
        if "other" in kinds.values():
            return _unresolved(decision, "date_belongs_to_another_event")
        return _unresolved(decision, "date_not_labelled_as_the_documents_own")
    token = own[0]
    decision.raw = collapse(token.raw)
    value, problem, by_convention = token.resolve(context.date_convention)
    if value is None:
        return _unresolved(decision, problem or "date_not_readable")
    if by_convention:
        decision.caveats.append("day_month_order_from_other_dates")
    if token.two_digit_year:
        decision.caveats.append("two_digit_year")
    return _accept(context, decision, value, (token.start, token.end))


# ---- Amount --------------------------------------------------------------------------------


def _figure(context: _Context, candidate: Candidate) -> Money | None:
    """A contract sum the model pointed at, read from the document; None when it does not hold."""
    if candidate.status != "stated" or not candidate.value.strip():
        return None
    evidence = [find_quote(context.document, quote) for quote in candidate.quotes[:MAX_QUOTES]]
    windows = windows_of(context.document, evidence)
    wanted = {money.value for money in find_money(candidate.value, context.decimal_mark)}
    wanted.discard(None)
    if len(wanted) != 1:
        return None
    value = wanted.pop()
    for money in context.money:
        if money.value == value and _inside(money.start, windows):
            return money
    return None


def _contract_sums(
    context: _Context, extraction: Extraction
) -> tuple[Decimal, Decimal, bool] | None:
    """The contract price before and after this change order, and whether "before" is firm.

    Firm: the document states the price just before this change order, or the original price
    and the total of earlier change orders. Not firm: only the original price is known, so
    earlier change orders the document does not list could explain a difference.
    """
    after = _figure(context, extraction.contract_sum_after)
    if after is None or after.value is None:
        return None
    before = _figure(context, extraction.contract_sum_before)
    if before is not None and before.value is not None:
        return before.value, after.value, True
    original = _figure(context, extraction.original_contract_sum)
    if original is None or original.value is None:
        return None
    previous = _figure(context, extraction.previous_changes)
    if previous is not None and previous.value is not None:
        earlier = -previous.value if previous.negative else previous.value
        return original.value + earlier, after.value, True
    return original.value, after.value, False


def _amount(context: _Context, extraction: Extraction) -> Decision:
    candidate = extraction.amount
    decision, windows = _open(context, "amount", candidate)
    if not windows:
        return decision
    if candidate.direction == "no_change":
        return _no_change(context, decision, candidate, windows, Decimal(0))
    if candidate.direction == "unclear":
        return _unresolved(decision, "direction_not_established")

    wanted = find_money(candidate.value, context.decimal_mark)
    sizes = {money.value for money in wanted if money.value is not None}
    if len(sizes) != 1:
        problems = {money.problem for money in wanted if money.problem}
        reason = "several_figures_in_value" if sizes else "figure_not_readable"
        return _unresolved(
            decision, f"{reason}:{'|'.join(sorted(problems))}" if problems else reason
        )
    size = sizes.pop()
    tokens = [
        money for money in context.money if money.value == size and _inside(money.start, windows)
    ]
    if not tokens:
        return _unresolved(decision, "value_not_in_quoted_text")

    def kind_of(money: Money) -> str | None:
        line_end = line_of(context.text, money.start)[1]
        return amount_label(
            label_before(context.text, money.start), context.text[money.end : line_end]
        )

    # The same figure can stand on two quoted lines; the one labelled as the change is meant.
    order = {"change": 0, None: 1, "part": 2, "limit": 3, "measure": 4, "total": 5}
    token = min(tokens, key=lambda money: order[kind_of(money)])
    decision.raw = token.raw

    wording = _wording(context, decision, token.start, token.end)
    sign, problem = _sign(candidate.direction, token.negative, amount_cues(wording))

    sums = _contract_sums(context, extraction)
    if sums is not None:
        before, after, firm = sums
        difference = after - before
        if abs(difference) == size and size != 0:
            implied = 1 if difference > 0 else -1
            if sign is None and problem in (
                "wording_gives_both_directions",
                "decrease_not_shown_by_document",
            ):
                # The wording could not settle the direction; the contract sums do.
                expected = "increase" if implied > 0 else "decrease"
                if candidate.direction == expected:
                    sign, problem = implied, None
                else:
                    problem = "direction_contradicts_contract_sums"
            elif sign is not None and sign != implied:
                sign, problem = None, "sign_contradicts_contract_sums"
            if sign is not None:
                decision.corroborated = True
        elif firm:
            return _unresolved(decision, "figure_contradicts_contract_sums")
    if sign is None:
        return _unresolved(decision, problem or "direction_not_established")

    if context.label_checks and not decision.corroborated:
        # No arithmetic vouches for this figure, so its label has to.
        kind = kind_of(token)
        if kind in _NOT_THE_CHANGE:
            return _unresolved(decision, _NOT_THE_CHANGE[kind])
        label = label_before(context.text, token.start)
        if kind is None and not says_price(token.raw, label):
            return _unresolved(decision, "figure_is_not_labelled_as_a_price")
        others = [
            money
            for money in context.money
            if money.value not in (None, size) and written_like_money(money.raw)
        ]
        if kind is None and any(kind_of(money) == "change" for money in others):
            return _unresolved(decision, "another_figure_is_labelled_as_the_change")
        if is_requested(nearest_label(context.text, token.start)) and any(
            kind_of(money) == "change" and is_granted(nearest_label(context.text, money.start))
            for money in others
        ):
            return _unresolved(decision, "a_different_figure_was_approved")
    return _accept(context, decision, sign * size, (token.start, token.end))


# ---- Schedule ------------------------------------------------------------------------------


def _schedule(context: _Context, candidate: SignedCandidate) -> Decision:
    decision, windows = _open(context, "schedule_impact_days", candidate)
    if not windows:
        return decision
    if candidate.direction == "no_change":
        return _no_change(context, decision, candidate, windows, 0)
    if candidate.direction == "unclear":
        return _unresolved(decision, "direction_not_established")

    wanted = find_days(candidate.value)
    bare = _BARE_INTEGER.match(candidate.value)
    sizes = {days.value for days in wanted if days.value is not None}
    if not wanted and bare:
        sizes = {int(bare.group(2))}
    if len(sizes) != 1:
        problems = {days.problem for days in wanted if days.problem}
        reason = "several_durations_in_value" if sizes else "duration_not_readable"
        return _unresolved(
            decision, f"{reason}:{'|'.join(sorted(problems))}" if problems else reason
        )
    size = sizes.pop()

    in_windows = [days for days in context.days if _inside(days.start, windows)]
    tokens = [days for days in in_windows if days.value == size]
    if not tokens:
        unreadable = {days.problem for days in in_windows if days.problem}
        if unreadable:
            return _unresolved(decision, f"duration_not_readable:{'|'.join(sorted(unreadable))}")
        token = _bare_duration(context, size, windows) if bare else None
        if token is None:
            return _unresolved(decision, "value_not_in_quoted_text")
        decision.caveats.append("unit_not_stated")
    else:
        token = tokens[0]
    decision.raw = collapse(token.raw)

    if context.label_checks:
        line = _line(context, token.start)
        if not is_schedule_label(f"{label_before(context.text, token.start)} {line}"):
            return _unresolved(decision, "duration_is_not_about_the_schedule")
        if is_requested(nearest_label(context.text, token.start)) and any(
            days.value not in (None, size) and is_granted(nearest_label(context.text, days.start))
            for days in context.days
        ):
            return _unresolved(decision, "a_different_duration_was_granted")

    wording = _wording(context, decision, token.start, token.end)
    sign, problem = _sign(candidate.direction, token.negative, schedule_cues(wording))
    if sign is None:
        return _unresolved(decision, problem or "direction_not_established")
    if token.in_weeks:
        decision.caveats.append("weeks_converted_to_days")
    decision.corroborated = token.words_agree
    return _accept(context, decision, sign * size, (token.start, token.end))


def _bare_duration(context: _Context, size: int, windows: list[Window]) -> Days | None:
    """A number that stands alone on a quoted line whose label is about time: "Days added: 14"."""
    for start, end in windows:
        for match in re.finditer(
            rf"(?<![\w.,/-])([-\u2212]?){size}(?![\w.,/-])", context.text[start:end]
        ):
            return Days(
                value=size,
                raw=match.group(0),
                start=start + match.start(),
                end=start + match.end(),
                negative=bool(match.group(1)),
            )
    return None


# ---- Approval ------------------------------------------------------------------------------


def _status(context: _Context, candidate: StatusCandidate) -> Decision:
    decision, windows = _open(context, "approval_status", candidate)
    if not windows:
        return decision
    lines = context.lines(windows)
    cues = status_cues(lines)
    if candidate.value not in cues:
        return _unresolved(decision, "status_not_shown_by_quoted_text")
    decision.raw = collapse(lines)
    if cues[candidate.value] == "signature":
        decision.caveats.append("approval_from_signature_only")
    if candidate.value == "approved" and _QUALIFIED_APPROVAL.search(lines):
        decision.caveats.append("approval_is_qualified")
    return _accept(context, decision, candidate.value, windows[0])
