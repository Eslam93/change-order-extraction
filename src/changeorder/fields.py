"""From the model's candidates to decisions: accepted, absent, or unresolved.

One rule for every field. A value is accepted only when all of these hold:

1. the model says the document states it;
2. at least one of its quotes is found in the document (`evidence.py`);
3. the value is found inside the quoted lines, in one piece, and is read from the document's
   own characters by a fixed rule (`normalize.py`) - the model's rendering of a number, a date
   or a name is used only to pick the place, never as the value;
4. nothing deterministic contradicts it: the figure's own sign, the wording's direction, the
   label beside it, or the contract sums before and after.

Anything else is `absent` (the model says it is not there) or `unresolved` (something was there
and could not be accepted), with a reason code. Unresolved fields are published as null and
always send the document to a person. A contradiction never produces a value: the pipeline does
not pick a side.

The same holds for a zero. "No change" is accepted when the document says so about this field,
without a hedge, and labels no other figure as the change. The model reporting "no change" is
what starts the check, and decides nothing.

An accepted value may carry caveats (an approximate quote, a date order taken from the
document's other dates, weeks turned into days). They lower its confidence band in
`confidence.py`; they do not change the value.
"""

from __future__ import annotations

import bisect
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
    after_on_line,
    amount_cues,
    amount_label,
    amounts_are_scaled,
    before_on_line,
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
    line_of,
    nearest_label,
    no_change_cue,
    reserves_rights,
    says_price,
    schedule_cues,
    speaks_of_price,
    status_cues,
    sum_role,
    title_cues,
    written_like_money,
)
from .prompt import Candidate, Extraction, SignedCandidate, StatusCandidate

__all__ = ["decide"]

MAX_QUOTES = 5
# How many places a value is looked at when the quoted lines hold it more than once. A change
# order states its amount a handful of times; thousands of copies are not a change order, and
# must not cost a label lookup each.
MAX_PLACES = 100
# One word straight after a figure, bare or alone in parentheses: "credit", "(add)".
_DIRECTLY_AFTER = re.compile(r"\s?(?:\(\w+\)|\w+)")
# A longer remark that opens with the direction: "(credit to Owner)", "(deduct - see item 3)".
_DIRECTION_IN_PARENTHESES = re.compile(
    r"\s?\(\s?(?:credit|deduct(?:ion|ive)?|add(?:ition|itive)?|increase|decrease)\b", re.IGNORECASE
)
_BARE_INTEGER = re.compile(r"^\s*([-\u2212+]?)(\d{1,4})\s*$")
_BARE_ZERO = re.compile(r"(?<![\w.,/$-])0(?![\w.,/%-])")
_QUALIFIED_APPROVAL = re.compile(
    r"\bapproved\b[\s,;:(\-\u2013\u2014]{1,5}(?:as\s+(?:noted|corrected|amended|modified)"
    r"|with\s+\w+|subject\s+to|in\s+part|except\b|provided\s+that|on\s+condition|conditional"
    r"|contingent)"
    r"|(?:partial|conditional)(?:ly)?\s+approv",
    re.IGNORECASE,
)
# Where one statement ends and the next begins: after a full stop or a semicolon, at an empty
# line, and where a new label opens ("Cost: see attached   Time: none" is two statements).
_STATEMENT_BREAK = re.compile(
    r"(?<=[.!?])[ \t]+(?=[A-Z(\[])|(?<=;)[ \t]+|(?<=[.;!?:])[ \t]*\n|\n[ \t]*\n"
    r"|(?:\n|[ \t]{2,})(?=[^\n:]{0,40}:)"
)
MAX_STATEMENTS = 50
Window = tuple[int, int]
# What a figure's label can make it instead of the change, and the reason recorded for each.
_NOT_THE_CHANGE = {
    "total": "figure_is_a_contract_total",
    "part": "figure_is_a_part_of_the_change",
    "limit": "figure_is_a_ceiling_or_an_estimate",
    "measure": "figure_is_a_quantity_or_a_rate",
}
# The same for a length of time.
_NOT_THE_SCHEDULE = {
    "total": "duration_is_a_contract_time_total",
    "limit": "duration_is_a_ceiling_or_an_estimate",
    "other": "duration_is_not_about_the_schedule",
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
        self.title_cues = title_cues(self.text)
        self.amounts_scaled = amounts_are_scaled(self.text)
        self._kinds: dict[tuple[int, int], str | None] = {}
        self._money_starts = [item.start for item in self.money]
        self._days_starts = [item.start for item in self.days]
        self._date_starts = [item.start for item in self.dates]

    def lines(self, windows: list[Window]) -> str:
        return "\n".join(self.text[start:end] for start, end in windows)

    def money_in(self, start: int, end: int) -> list[Money]:
        """The figures that start inside `start`..`end`."""
        low = bisect.bisect_left(self._money_starts, start)
        return self.money[low : bisect.bisect_left(self._money_starts, end)]

    def days_in(self, start: int, end: int) -> list[Days]:
        low = bisect.bisect_left(self._days_starts, start)
        return self.days[low : bisect.bisect_left(self._days_starts, end)]

    def date_at(self, start: int, end: int) -> bool:
        """Whether `start`..`end` touches a date."""
        index = bisect.bisect_left(self._date_starts, end) - 1
        return index >= 0 and self.dates[index].end > start

    def amount_kind(self, money: Money) -> str | None:
        """What the figure's label makes it: see `normalize.amount_label`."""
        key = (money.start, -1)
        if key not in self._kinds:
            self._kinds[key] = amount_label(
                label_before(self.text, money.start), after_on_line(self.text, money.end)
            )
        return self._kinds[key]

    def duration_kind(self, start: int, end: int) -> str | None:
        """What the label of the duration at `start`..`end` makes it."""
        key = (start, end)
        if key not in self._kinds:
            self._kinds[key] = duration_label(
                label_before(self.text, start), after_on_line(self.text, end)
            )
        return self._kinds[key]


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


def _wording(context: _Context, decision: Decision, start: int, end: int) -> str:
    """The words that can say which way the figure at `start`..`end` goes.

    Its label (the text before it on its line); a word that directly follows it ("$5,000
    credit", "4 days earlier", "$7,650.00 (add)", "$4,250.00 (credit to Owner)"); its column's
    heading in a table, or every heading when its column cannot be told; and the other passages
    the model quoted, such as a title "Deductive Change Order". A remark further along the
    figure's own line is not read: "$13,100.00   (reduced: batts furnished by others)" describes
    a negotiation, not a sign.
    """
    line_start, line_end = line_of(context.text, start)
    after = after_on_line(context.text, end)
    following = _DIRECTLY_AFTER.match(after) or _DIRECTION_IN_PARENTHESES.match(after)
    elsewhere = [
        context.text[item.start : item.end]
        for item in decision.evidence
        if item.start is not None
        and item.end is not None
        and not (item.start < line_end and line_start < item.end)
    ]
    return "\n".join(
        [
            before_on_line(context.text, start),
            following.group(0) if following else "",
            header_above(context.text, start) or headings_above(context.text, start),
            *elsewhere,
        ]
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


def _rivalry(label: str, rivals: list[str]) -> str | None:
    """What stands against a value whose label makes it the answer, when other values in the
    document carry such a label too and differ from it.

    What was requested and what was granted are not rivals: the granted one stands, and the
    requested one is refused when a granted one exists. Any other pair is a document that
    gives two answers, and neither is taken.
    """
    if is_requested(label) and any(is_granted(rival) for rival in rivals):
        return "granted_elsewhere"
    unexplained = [rival for rival in rivals if not (is_granted(label) and is_requested(rival))]
    return "second_candidate" if unexplained else None


# ---- An explicit zero ------------------------------------------------------------------------


def _statements(context: _Context, windows: list[Window]) -> list[Window]:
    """The quoted lines cut into statements, each of which says one thing."""
    found: list[Window] = []
    for start, end in windows:
        cursor = start
        for match in _STATEMENT_BREAK.finditer(context.text, start, end):
            found.append((cursor, match.start()))
            cursor = match.end()
        found.append((cursor, end))
    return [(start, end) for start, end in found if context.text[start:end].strip()]


def _states_no_change(
    context: _Context, span: Window, money: bool
) -> tuple[str | None, str | None]:
    """Whether the statement at `span` says the price (or, for `money=False`, the contract time)
    does not change: ("explicit" or "not_applicable", None) when it does, (None, reason) when
    it does not, and (None, None) when it says nothing of the kind.

    It does when it holds a zero under a label that makes the zero the change ("Net change:
    $0.00", a "0" in a column headed "Days"), or says so in words about this field ("No change
    to the Contract Time", "at no cost to the Owner"). It does not when it hedges, or when it
    also labels a figure as the change.
    """
    start, end = span
    text = context.text[start:end]
    how: str | None = None
    reason: str | None = None
    if money:
        for figure in context.money_in(start, end):
            if figure.value != 0:
                continue
            kind = context.amount_kind(figure)
            priced = says_price(figure.raw, label_before(context.text, figure.start))
            if not context.label_checks or kind == "change" or (kind is None and priced):
                how = "explicit"
                break
            reason = reason or _NOT_THE_CHANGE.get(kind or "", "figure_is_not_labelled_as_a_price")
    else:
        zeros = [(days.start, days.end) for days in context.days_in(start, end) if days.value == 0]
        zeros += [
            (start + match.start(), start + match.end())
            for match in _BARE_ZERO.finditer(text)
            if not context.date_at(start + match.start(), start + match.end())
        ]
        for low, high in zeros:
            kind = context.duration_kind(low, high)
            if not context.label_checks or kind == "change":
                how = "explicit"
                break
            reason = reason or _NOT_THE_SCHEDULE.get(
                kind or "", "duration_is_not_about_the_schedule"
            )
    if how is None:
        cue = no_change_cue(text)
        if cue is not None:
            subject = f"{header_above(context.text, start) or ''} {text}"
            about = speaks_of_price(subject) if money else is_schedule_label(subject)
            if about or not context.label_checks:
                how = cue
            else:
                reason = reason or "no_change_is_about_something_else"
    if how is None or not context.label_checks:
        return how, reason

    # The rest of the paragraph may take the statement back: "... reserves the right to claim".
    paragraph = context.text[start : end + 400].split("\n\n")[0]
    if is_hedged(text) or reserves_rights(paragraph):
        return None, "no_change_is_hedged"
    if money:
        for figure in context.money_in(start, end):
            if not figure.value or not written_like_money(figure.raw):
                continue
            kind = context.amount_kind(figure)
            priced = says_price(figure.raw, label_before(context.text, figure.start))
            if kind == "change" or (kind is None and priced):
                return None, "no_change_reported_beside_a_figure"
    elif any(
        days.value and context.duration_kind(days.start, days.end) == "change"
        for days in context.days_in(start, end)
    ):
        return None, "no_change_reported_beside_a_figure"
    return how, None


def _no_change(
    context: _Context,
    decision: Decision,
    candidate: SignedCandidate,
    windows: list[Window],
    money: bool,
) -> Decision:
    """An explicit zero: the document says the price, or the time, does not change."""
    figures = [item.value for item in find_money(candidate.value, context.decimal_mark)]
    durations = [item.value for item in find_days(candidate.value)]
    if any(value for value in (*figures, *durations)):
        return _unresolved(decision, "no_change_reported_beside_a_figure")

    statements = _statements(context, windows)
    located = locate_text(context.document, candidate.value, windows)
    if located is not None:
        # The model pointed at the words; the statement that holds them is the one judged.
        statements = [span for span in statements if span[0] <= located[0] < span[1]] or statements
    reasons: list[str] = []
    for span in statements[:MAX_STATEMENTS]:
        how, reason = _states_no_change(context, span, money)
        if how is None:
            if reason:
                reasons.append(reason)
            continue
        statement = context.text[slice(*span)]
        if context.label_checks:
            # A zero stands only when no other figure is labelled as the change.
            if money:
                rivals = [
                    nearest_label(context.text, figure.start)
                    for figure in context.money
                    if figure.value
                    and written_like_money(figure.raw)
                    and context.amount_kind(figure) == "change"
                ]
                granted, second = (
                    "a_different_figure_was_approved",
                    "another_figure_is_labelled_as_the_change",
                )
            else:
                rivals = [
                    nearest_label(context.text, days.start)
                    for days in context.days
                    if days.value and context.duration_kind(days.start, days.end) == "change"
                ]
                granted, second = (
                    "a_different_duration_was_granted",
                    "another_duration_is_given_for_the_schedule",
                )
            rivalry = _rivalry(statement, rivals)
            if rivalry is not None:
                return _unresolved(decision, granted if rivalry == "granted_elsewhere" else second)
        chosen = located if located is not None else span
        decision.raw = collapse(context.text[slice(*chosen)])
        if how == "not_applicable":
            decision.caveats.append("not_applicable_read_as_no_change")
        return _accept(context, decision, Decimal(0) if money else 0, chosen)
    return _unresolved(decision, reasons[0] if reasons else "no_change_not_shown_by_document")


# ---- Names and identifiers -----------------------------------------------------------------


def _text(context: _Context, name: str, candidate: Candidate) -> Decision:
    decision, windows = _open(context, name, candidate)
    if not windows:
        return decision
    span = locate_text(context.document, candidate.value, windows)
    if span is None:
        return _unresolved(decision, "value_not_in_quoted_text")
    # What is published is the document's own characters, never the model's copy of them.
    decision.raw = collapse(context.text[slice(*span)])
    if context.label_checks:
        if label_says_otherwise(name, nearest_label(context.text, span[0])):
            return _unresolved(decision, "labelled_as_something_else")
        if name in ("contract_number", "change_order_number") and context.date_at(*span):
            return _unresolved(decision, "value_is_part_of_a_date")
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
    ][:MAX_PLACES]
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
    if context.label_checks and any(
        other.resolve(context.date_convention)[0] not in (None, value)
        and date_label(context.text, other.start) == "own"
        for other in context.dates
    ):
        # Two dates, each labelled as the document's own, and they differ.
        return _unresolved(decision, "two_dates_are_labelled_as_the_documents_own")
    if by_convention:
        decision.caveats.append("day_month_order_from_other_dates")
    if token.two_digit_year:
        decision.caveats.append("two_digit_year")
    return _accept(context, decision, value, (token.start, token.end))


# ---- Amount --------------------------------------------------------------------------------


def _figure(
    context: _Context, candidate: Candidate, roles: tuple[str, ...]
) -> tuple[Money, str] | None:
    """A contract sum the model pointed at, read from the document, with the role its own label
    gives it. None when the document does not hold it there, or labels it as something other
    than the sums asked for: the model does not get to say which figure is the revised total."""
    if candidate.status != "stated" or not candidate.value.strip():
        return None
    evidence = [find_quote(context.document, quote) for quote in candidate.quotes[:MAX_QUOTES]]
    windows = windows_of(context.document, evidence)
    wanted = {money.value for money in find_money(candidate.value, context.decimal_mark)}
    wanted.discard(None)
    if len(wanted) != 1:
        return None
    value = wanted.pop()
    places = [
        money for money in context.money if money.value == value and _inside(money.start, windows)
    ]
    for money in places[:MAX_PLACES]:
        role = sum_role(label_before(context.text, money.start))
        if not context.label_checks:
            return money, roles[0]
        if role in roles:
            return money, role
    return None


def _contract_sums(
    context: _Context, extraction: Extraction
) -> tuple[Decimal, Decimal, bool] | None:
    """The contract price before and after this change order, and whether "before" is firm.

    Firm: the document states the price just before this change order, or the original price
    and the total of earlier change orders. Not firm: only the original price is known, so
    earlier change orders the document does not list could explain a difference.
    """
    after = _figure(context, extraction.contract_sum_after, ("after",))
    if after is None or after[0].value is None:
        return None
    before = _figure(context, extraction.contract_sum_before, ("before", "original"))
    if before is not None and before[0].value is not None and before[1] == "before":
        return before[0].value, after[0].value, True
    # A "before" that its own label calls the original sum is the original sum.
    original = _figure(context, extraction.original_contract_sum, ("original",)) or before
    if original is None or original[0].value is None:
        return None
    earlier = _figure(context, extraction.previous_changes, ("earlier_changes",))
    if earlier is not None and earlier[0].value is not None:
        change = -earlier[0].value if earlier[0].negative else earlier[0].value
        return original[0].value + change, after[0].value, True
    return original[0].value, after[0].value, False


def _amount(context: _Context, extraction: Extraction) -> Decision:
    candidate = extraction.amount
    decision, windows = _open(context, "amount", candidate)
    if not windows:
        return decision
    if candidate.direction == "no_change":
        return _no_change(context, decision, candidate, windows, money=True)
    if candidate.direction == "unclear":
        return _unresolved(decision, "direction_not_established")
    if context.amounts_scaled:
        # "(All amounts in thousands of dollars)": no figure here is what it is written as.
        return _unresolved(decision, "amounts_are_stated_in_thousands_or_millions")

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
    ][:MAX_PLACES]
    if not tokens:
        return _unresolved(decision, "value_not_in_quoted_text")

    # The same figure can stand on two quoted lines; the one labelled as the change is meant.
    order = {"change": 0, None: 1, "part": 2, "limit": 3, "measure": 4, "total": 5}
    token = min(tokens, key=lambda money: order[context.amount_kind(money)])
    decision.raw = token.raw
    kind = context.amount_kind(token)
    if context.label_checks and kind == "total":
        # A contract sum is never the change, whatever arithmetic is offered for it.
        return _unresolved(decision, _NOT_THE_CHANGE["total"])

    wording = _wording(context, decision, token.start, token.end)
    cues = amount_cues(wording) | context.title_cues
    sign, problem = _sign(candidate.direction, token.negative, cues)

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
        if kind in _NOT_THE_CHANGE:
            return _unresolved(decision, _NOT_THE_CHANGE[kind])
        label = label_before(context.text, token.start)
        if kind is None and not says_price(token.raw, label):
            return _unresolved(decision, "figure_is_not_labelled_as_a_price")
        others = [
            (money, context.amount_kind(money))
            for money in context.money
            if money.value not in (None, size) and written_like_money(money.raw)
        ]
        rivals = [
            nearest_label(context.text, money.start) for money, other in others if other == "change"
        ]
        if kind is None and rivals:
            return _unresolved(decision, "another_figure_is_labelled_as_the_change")
        line_start, line_end = line_of(context.text, token.start)
        if (
            kind is None
            and header_above(context.text, token.start) is None
            and any(
                other is None and line_start <= money.start < line_end for money, other in others
            )
        ):
            # Two prices on one line and no label to tell them apart.
            return _unresolved(decision, "another_figure_shares_the_line")
        own = nearest_label(context.text, token.start)
        priced = [
            nearest_label(context.text, money.start)
            for money, other in others
            if other is None and says_price(money.raw, label_before(context.text, money.start))
        ]
        rivalry = _rivalry(own, rivals)
        if rivalry == "granted_elsewhere" or (
            is_requested(own) and any(is_granted(other) for other in priced)
        ):
            return _unresolved(decision, "a_different_figure_was_approved")
        if rivalry == "second_candidate":
            return _unresolved(decision, "two_figures_are_labelled_as_the_change")
    return _accept(context, decision, sign * size, (token.start, token.end))


# ---- Schedule ------------------------------------------------------------------------------


def _schedule(context: _Context, candidate: SignedCandidate) -> Decision:
    decision, windows = _open(context, "schedule_impact_days", candidate)
    if not windows:
        return decision
    if candidate.direction == "no_change":
        return _no_change(context, decision, candidate, windows, money=False)
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
    tokens = [days for days in in_windows if days.value == size][:MAX_PLACES]
    if not tokens:
        unreadable = {days.problem for days in in_windows if days.problem}
        if unreadable:
            return _unresolved(decision, f"duration_not_readable:{'|'.join(sorted(unreadable))}")
        found = _bare_duration(context, size, windows) if bare else None
        if found is None:
            return _unresolved(decision, "value_not_in_quoted_text")
        token = found
        decision.caveats.append("unit_not_stated")
    else:
        # The same number of days can stand on two quoted lines; the one about the contract
        # time is meant.
        token = min(
            tokens, key=lambda days: context.duration_kind(days.start, days.end) != "change"
        )
    decision.raw = collapse(token.raw)

    if context.label_checks:
        kind = context.duration_kind(token.start, token.end)
        if kind != "change":
            return _unresolved(
                decision, _NOT_THE_SCHEDULE.get(kind or "", "duration_is_not_about_the_schedule")
            )
        rivals = [
            nearest_label(context.text, days.start)
            for days in context.days
            if days.value not in (None, size)
            and context.duration_kind(days.start, days.end) == "change"
        ]
        rivalry = _rivalry(nearest_label(context.text, token.start), rivals)
        if rivalry == "granted_elsewhere":
            return _unresolved(decision, "a_different_duration_was_granted")
        if rivalry == "second_candidate":
            return _unresolved(decision, "two_durations_are_given_for_the_schedule")

    wording = _wording(context, decision, token.start, token.end)
    sign, problem = _sign(candidate.direction, token.negative, schedule_cues(wording))
    if sign is None:
        return _unresolved(decision, problem or "direction_not_established")
    if token.in_weeks:
        decision.caveats.append("weeks_converted_to_days")
    decision.corroborated = token.words_agree
    return _accept(context, decision, sign * size, (token.start, token.end))


def _bare_duration(context: _Context, size: int, windows: list[Window]) -> Days | None:
    """A number that stands alone on a quoted line whose label is about time: "Days added: 14".
    The day of a date is not one: "Completion date: 30 June 2026" does not say thirty days."""
    pattern = re.compile(rf"(?<![\w.,/$-])([-\u2212]?){size}(?![\w.,/%-])")
    for start, end in windows:
        for match in pattern.finditer(context.text, start, end):
            if context.date_at(match.start(), match.end()):
                continue
            return Days(
                value=size,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
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
        if has_blank_signing_line(context.text):
            # One party signed and another has not: the block is not an approval yet.
            return _unresolved(decision, "signature_block_is_incomplete")
        decision.caveats.append("approval_from_signature_only")
    if candidate.value == "approved" and _QUALIFIED_APPROVAL.search(lines):
        decision.caveats.append("approval_is_qualified")
    return _accept(context, decision, candidate.value, windows[0])
