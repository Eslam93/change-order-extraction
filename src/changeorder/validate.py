"""Checks across the whole document, after each field has been decided.

`fields.py` judges a candidate against its own evidence. These checks look at what the model did
not point to. They come in two strengths:

- A **conflict** means the document gives two answers for one field: two change order numbers,
  an approval beside a rejection. The field becomes unresolved (null), because choosing one
  would be a guess.
- A **suspicion** means the value stands but a person should look: a date before the contract's
  own date, a page that could not be read, text in the document addressed to a model, a field
  the model called absent while the document seems to hold it, a critical field with no value.

Every result is a `Finding` with a code, so review reasons can be tested and counted one by one
and never collapse into a single confidence penalty.
"""

from __future__ import annotations

import re

from .document import SourceDocument
from .models import Decision, Finding
from .normalize import (
    after_on_line,
    amount_label,
    before_on_line,
    date_convention,
    date_label,
    decimal_mark_of,
    find_dates,
    find_days,
    find_money,
    is_schedule_label,
    label_before,
    line_of,
    status_cues,
    ticked_statuses,
)
from .prompt import Extraction
from .vendor.injection_sanitizer import sanitize

__all__ = ["CRITICAL_FIELDS", "validate"]

# A change order cannot be filed, priced or acted on without these, so a missing one always
# goes to a person. The other fields may be legitimately absent: many change orders name no
# requester, state no schedule effect, and some carry no date of their own.
CRITICAL_FIELDS = ("change_order_number", "amount", "approval_status")

_CO_NUMBER_PATTERNS = (
    # "Change Order No. 7", "Change Order Request #17": the words, then a number marker.
    re.compile(
        r"\bchange\s+order\b(?!s)(?:\s+request)?\s*(?:no\.?|number|num\.?|#)\s*[:.#]?\s*"
        r"(?P<id>(?:[A-Za-z]{1,5}[-\s]?)?\d[\w/-]*)",
        re.IGNORECASE,
    ),
    # "Change Order: CO-023": a colon alone counts only where the words are a label of their
    # own (at a line's start or after a column gap) and what follows is an identifier, so that
    # "amount of this change order: 1.234,56" is not read as a number for the change order.
    re.compile(
        r"(?:^|\s{2})change\s+order\s*:\s*(?P<id>(?:[A-Za-z]{1,5}-)?\d[\w/-]*)(?![\w.,]*[.,]\d)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "CO-011", "CO # 9", "C.O. No. 4".
    re.compile(r"\bC\.?O\.?(?:\s*(?:No\.?|NO\.?|Number|#)\s*[:.]?\s*|-)(?P<id>\d[\w/-]*)"),
)
# A number that names another change order: "supersedes Change Order No. 5", "see CO-003".
_REFERENCE = re.compile(
    r"\b(?:previous|prior|earlier|former|original|see|per|refer\w*"
    r"|(?:supersed|replac|revis|amend)\w+(?:\s+by)?)\s*$",
    re.IGNORECASE,
)
# Every pattern here is bounded and none lets a run of punctuation span lines: they read text
# someone else wrote.
_STATUS_LINE = re.compile(
    r"^[^\w\n]{0,20}(?:status|disposition|action\s+taken)\s{0,3}[:\-\u2013]\s{0,3}"
    r"(?P<rest>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
# A line that is about the contract time, said narrowly enough that "at this time" is not one.
_ADDRESSES_SCHEDULE = re.compile(
    r"\bcontract\s+time\b|\bschedule\s+(?:impact|effect)\b|\btime\s+(?:impact|extension|change)\b"
    r"|\bcompletion\s+(?:date|deadline)\b|\bextension\s+of\s+(?:time|the\s+completion)\b"
    r"|\bdays\s+(?:added|deducted)\b|^[^\w\n]{0,20}(?:time|schedule)\s{0,3}:",
    re.IGNORECASE | re.MULTILINE,
)
_CONTRACT_DATE_LABEL = re.compile(
    r"\b(?:contract\s+date|date\s+of\s+(?:the\s+)?(?:original\s+)?(?:contract|agreement)"
    r"|(?:contract|agreement)\s+dated)\W*$",
    re.IGNORECASE,
)


def validate(
    decisions: dict[str, Decision], extraction: Extraction, document: SourceDocument
) -> list[Finding]:
    """Every finding for one document. Conflicts also turn their field unresolved, in place."""
    findings: list[Finding] = []
    findings += _change_order_numbers(decisions["change_order_number"], document)
    findings += _approval(decisions["approval_status"], document)
    findings += _dates(decisions["date"], document)
    findings += _possibly_missed(decisions, document)
    findings += _document(decisions, document, extraction)
    for name, decision in decisions.items():
        if decision.state == "unresolved":
            detail = decision.note or "no value could be accepted"
            findings += [Finding(reason, detail, name) for reason in decision.reasons]
    findings += _critical(decisions)
    return _unique(findings)


def _unique(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str | None]] = set()
    kept: list[Finding] = []
    for finding in findings:
        if (finding.code, finding.field) not in seen:
            seen.add((finding.code, finding.field))
            kept.append(finding)
    return kept


def _reject(decision: Decision, reason: str) -> None:
    decision.state = "unresolved"
    decision.value = None
    if reason not in decision.reasons:
        decision.reasons.append(reason)


# ---- Change order number -----------------------------------------------------------------------


def _key(identifier: str) -> str:
    """An identifier reduced to what makes it this number: "CO-011", "011" and "11" are one."""
    plain = re.sub(r"[^A-Za-z0-9]", "", identifier).upper()
    plain = re.sub(r"^(?:CHANGEORDER|COR|CO)(?=\d)", "", plain)
    return plain.lstrip("0") or "0"


def labelled_change_order_numbers(text: str) -> dict[str, str]:
    """The numbers the document itself labels as this change order's, by key."""
    found: dict[str, str] = {}
    for pattern in _CO_NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            if _REFERENCE.search(text[max(0, match.start() - 20) : match.start()]):
                continue
            found.setdefault(_key(match["id"]), match["id"])
    return found


def _change_order_numbers(decision: Decision, document: SourceDocument) -> list[Finding]:
    labelled = labelled_change_order_numbers(document.text)
    if len(labelled) > 1:
        detail = "the document labels this change order " + " and ".join(sorted(labelled.values()))
        if decision.state == "accepted":
            _reject(decision, "conflicting_change_order_numbers")
            return []
        return [Finding("conflicting_change_order_numbers", detail, decision.field)]
    if decision.state == "accepted" and labelled and _key(str(decision.value)) not in labelled:
        _reject(decision, "number_differs_from_labelled_change_order_number")
    return []


# ---- Approval ----------------------------------------------------------------------------------


def stated_statuses(text: str) -> set[str]:
    """The approval states the document states outright: a status line, a ticked box, or a stamp
    (a short line in capitals). Signature blocks and running prose are not counted here."""
    passages = [match["rest"] for match in _STATUS_LINE.finditer(text)]
    passages += [
        line
        for line in text.splitlines()
        if len(line) <= 60 and any(char.isalpha() for char in line) and line == line.upper()
    ]
    found = ticked_statuses(text)
    for passage in passages:
        found.update(status for status, kind in status_cues(passage).items() if kind == "explicit")
    return found


def _conflicting(states: set[str]) -> bool:
    """Two different states conflict, except a draft that is also pending."""
    return len(states) > 1 and states != {"draft", "pending"}


def _approval(decision: Decision, document: SourceDocument) -> list[Finding]:
    stated = stated_statuses(document.text)
    if decision.state == "accepted":
        if _conflicting(stated | {str(decision.value)}):
            _reject(decision, "conflicting_approval_evidence")
        elif "approval_is_qualified" in decision.caveats:
            return [
                Finding(
                    "approval_is_qualified",
                    "approved with a condition or a note the output cannot carry",
                    decision.field,
                )
            ]
        return []
    if _conflicting(stated):
        return [
            Finding(
                "conflicting_approval_evidence",
                "the document states " + " and ".join(sorted(stated)),
                decision.field,
            )
        ]
    return []


# ---- Dates -------------------------------------------------------------------------------------


def _dates(decision: Decision, document: SourceDocument) -> list[Finding]:
    if decision.state != "accepted":
        return []
    findings: list[Finding] = []
    value = decision.value
    if not 1990 <= value.year <= 2100:
        findings.append(
            Finding("date_outside_plausible_range", f"the year {value.year}", decision.field)
        )
    tokens = find_dates(document.text)
    convention = date_convention(tokens)
    for token in tokens:
        if not _CONTRACT_DATE_LABEL.search(label_before(document.text, token.start)):
            continue
        contract_date = token.resolve(convention)[0]
        if contract_date is not None and value < contract_date:
            findings.append(
                Finding(
                    "date_before_contract_date",
                    f"the change order is dated before the contract ({contract_date.isoformat()})",
                    decision.field,
                )
            )
    return findings


# ---- Fields the model called absent --------------------------------------------------------------


def _possibly_missed(decisions: dict[str, Decision], document: SourceDocument) -> list[Finding]:
    """An absence nothing in the document contradicts is confirmed. One that the document seems
    to contradict is not taken on the model's word: a false "absent" is otherwise invisible."""
    text = document.text
    seen: dict[str, str] = {}

    if decisions["amount"].state == "absent":
        for money in find_money(text, decimal_mark_of(text)):
            kind = amount_label(label_before(text, money.start), after_on_line(text, money.end))
            if money.value and kind == "change":
                seen["amount"] = f"a figure is labelled as the change: {money.raw}"
                break
    if decisions["schedule_impact_days"].state == "absent":
        for days in find_days(text):
            beside = f"{before_on_line(text, days.start)} {after_on_line(text, days.end)}"
            if is_schedule_label(beside):
                seen["schedule_impact_days"] = (
                    f"a duration stands beside schedule wording: {days.raw}"
                )
                break
        else:
            # No duration, but a line that speaks of the contract time: "Time impact: none",
            # "No change to the completion date", "Days added: 8".
            addressed = _ADDRESSES_SCHEDULE.search(text)
            if addressed:
                begin = line_of(text, addressed.start())[0]
                line = text[begin : begin + 80].split(chr(10))[0]
                seen["schedule_impact_days"] = f"the document speaks of the schedule: {line}"
    if decisions["change_order_number"].state == "absent":
        labelled = labelled_change_order_numbers(text)
        if labelled:
            seen["change_order_number"] = (
                "a number is labelled as the change order's: "
                + ", ".join(sorted(labelled.values()))
            )
    if decisions["date"].state == "absent":
        for token in find_dates(text):
            if date_label(text, token.start) == "own":
                seen["date"] = f"a date is labelled as the document's own: {token.raw}"
                break
    if decisions["approval_status"].state == "absent":
        stated = stated_statuses(text)
        if stated:
            seen["approval_status"] = "the document states " + " and ".join(sorted(stated))
    return [Finding("possibly_missed_value", detail, name) for name, detail in seen.items()]


# ---- The document itself -------------------------------------------------------------------------


def _document(
    decisions: dict[str, Decision], document: SourceDocument, extraction: Extraction
) -> list[Finding]:
    findings: list[Finding] = []
    if document.unread_pages:
        pages = ", ".join(str(page) for page in document.unread_pages)
        findings.append(
            Finding(
                "unread_pages", f"page {pages} holds an image and no text layer; it was not read"
            )
        )
    if "pdf_plain_text_fallback" in document.notes:
        findings.append(
            Finding(
                "reading_order_not_rebuilt", "layout extraction failed; rows may be out of order"
            )
        )
    for note in ("decoded_as_windows_1252", "hidden_characters_removed"):
        if note in document.notes:
            findings.append(Finding(note, "recorded while reading the file", review=False))
    suspicious = [
        finding
        for finding in sanitize(document.text, mode="report").findings
        if finding.kind in ("instruction", "role_marker", "delimiter")
    ]
    if suspicious:
        findings.append(
            Finding(
                "instruction_like_text",
                "the document holds text written as an instruction to a model: "
                + suspicious[0].excerpt,
            )
        )
        # Text addressed to the model is not evidence of anything: a value read out of it is
        # dropped, whatever the model made of it.
        for decision in decisions.values():
            if decision.state != "accepted" or decision.span is None:
                continue
            start, end = decision.span
            if any(
                found.start is not None
                and found.end is not None
                and found.start < end
                and start < found.end
                for found in suspicious
            ):
                _reject(decision, "value_read_from_instruction_like_text")
    if extraction.suspicious_instructions:
        findings.append(
            Finding("model_reported_instructions", "the model saw text addressed to it")
        )
    return findings


def _critical(decisions: dict[str, Decision]) -> list[Finding]:
    findings = [
        Finding("critical_field_missing", "the document gives none", name)
        for name in CRITICAL_FIELDS
        if decisions[name].state == "absent"
    ]
    if all(decisions[name].state != "accepted" for name in ("project_name", "contract_number")):
        findings.append(
            Finding(
                "no_contract_reference",
                "neither a project name nor a contract number ties this change order to a contract",
            )
        )
    return findings
