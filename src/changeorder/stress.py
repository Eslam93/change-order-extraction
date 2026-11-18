"""What the checks do when the model is wrong: one mistake at a time, on recorded runs.

On the labelled documents the live models made no mistake that reached the output, so those
runs cannot show what happens when a model does make one. This module makes the mistakes on
purpose. It takes each document's recorded reply, changes one field the way a model plausibly
gets it wrong (the wrong figure, a flipped sign, another date, an approval read from an
approval section, a value reported absent) and runs the deterministic stages on the changed
reply. No model is called.

Each injected mistake ends in one of these ways:

    rejected          the field is published as null, with a reason
    published_wrong   a wrong value went out, at the confidence recorded
    harmless          the changed reply still yields the labelled value (the document states
                      it in two places)
    flagged, silent   for a value falsely reported absent: whether the document was sent to
                      review for that field, or the absence passed unnoticed

This tests the checks, not a model. The mistakes are synthetic and their mix is the author's
guess at what models get wrong, so the rates say how the checks behave under each kind of
mistake. They do not predict how often a real model makes one.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .document import SourceDocument, read_document
from .extract import first_json_object
from .llm import ScriptedModel
from .models import APPROVAL_STATUSES
from .normalize import (
    date_convention,
    decimal_mark_of,
    find_dates,
    find_days,
    find_money,
    line_of,
    written_like_money,
)
from .pipeline import process_document

__all__ = ["ABSENT_KIND", "Injection", "inject", "mistakes", "summarise"]

ABSENT_KIND = "a stated value reported as absent"
_APPROVAL_WORDING = re.compile(r"approv|accept|execut|authori|sign", re.IGNORECASE)


@dataclass(frozen=True)
class Injection:
    document: str
    field: str
    kind: str
    outcome: str
    confidence: float  # of the value that went out; 0.0 when none did
    needs_review: bool
    published: Any
    label: Any


def mistakes(
    reply: dict[str, Any], document: SourceDocument, labels: dict[str, Any]
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Every mistake to play on one document: the field, the kind, and the changed reply."""
    text = document.text

    def line(position: int) -> str:
        return text[slice(*line_of(text, position))]

    def stated(value: str, quote: str, **extra: str) -> dict[str, Any]:
        return {"status": "stated", "value": value, "quotes": [quote], "note": "", **extra}

    def changed(field: str, candidate: dict[str, Any]) -> dict[str, Any]:
        other = copy.deepcopy(reply)
        other[field] = candidate
        return other

    def is_stated(field: str) -> bool:
        return bool(reply[field]["status"] == "stated")

    # The amount: another figure from the same document, a flipped sign, a figure from nowhere.
    amount = labels["amount"]
    size = None if amount is None else abs(Decimal(str(amount)))
    seen: set[Decimal] = set()
    for money in find_money(text, decimal_mark_of(text)):
        if money.value is None or money.value in seen or money.value == size:
            continue
        if not written_like_money(money.raw):  # a bare "8" is a count, not a price
            continue
        seen.add(money.value)
        direction = "decrease" if money.negative else "increase"
        candidate = stated(money.raw, line(money.start), direction=direction)
        yield "amount", "another figure taken for the amount", changed("amount", candidate)
    if is_stated("amount") and reply["amount"]["direction"] in ("increase", "decrease"):
        flipped = "decrease" if reply["amount"]["direction"] == "increase" else "increase"
        candidate = {**reply["amount"], "direction": flipped}
        yield "amount", "the amount's direction reversed", changed("amount", candidate)
        invented = f"${(size or Decimal(0)) + Decimal('1234.56'):,.2f}"
        candidate = {**reply["amount"], "value": invented}
        yield "amount", "a figure that is not in the document", changed("amount", candidate)

    # The date: any other date in the document.
    tokens = find_dates(text)
    convention = date_convention(tokens)
    raw_seen: set[str] = set()
    for token in tokens:
        value = token.resolve(convention)[0]
        if token.raw in raw_seen or (value is not None and value.isoformat() == labels["date"]):
            continue
        raw_seen.add(token.raw)
        candidate = stated(token.raw, line(token.start))
        yield "date", "another date taken for the change order's date", changed("date", candidate)

    # The schedule: another duration, a flipped sign.
    days_label = labels["schedule_impact_days"]
    days_seen: set[int] = set()
    for days in find_days(text):
        if days.value is None or days.value in days_seen:
            continue
        if days_label is not None and days.value == abs(days_label):
            continue
        days_seen.add(days.value)
        candidate = stated(days.raw, line(days.start), direction="increase")
        yield (
            "schedule_impact_days",
            "another duration taken for the schedule impact",
            changed("schedule_impact_days", candidate),
        )
    schedule = reply["schedule_impact_days"]
    if is_stated("schedule_impact_days") and schedule["direction"] in ("increase", "decrease"):
        flipped = "decrease" if schedule["direction"] == "increase" else "increase"
        yield (
            "schedule_impact_days",
            "the schedule's direction reversed",
            changed("schedule_impact_days", {**schedule, "direction": flipped}),
        )

    # Approval: another status on the same evidence, and approval read from approval wording.
    status = labels["approval_status"]
    if is_stated("approval_status"):
        for other in APPROVAL_STATUSES:
            if other != status:
                candidate = {**reply["approval_status"], "value": other}
                yield (
                    "approval_status",
                    "another status claimed on the same quotes",
                    changed("approval_status", candidate),
                )
    if status != "approved":
        wording = dict.fromkeys(
            item.strip() for item in text.splitlines() if _APPROVAL_WORDING.search(item)
        )
        for quote in wording:
            yield (
                "approval_status",
                "approval read from wording about approval",
                changed("approval_status", stated("approved", quote)),
            )

    # Identifiers and names: the right kind of text in the wrong field.
    if is_stated("contract_number") and labels["contract_number"] != labels["change_order_number"]:
        yield (
            "change_order_number",
            "the contract number taken for the change order number",
            changed("change_order_number", copy.deepcopy(reply["contract_number"])),
        )
    if is_stated("requested_by") and labels["requested_by"] != labels["project_name"]:
        yield (
            "project_name",
            "the requester taken for the project name",
            changed("project_name", copy.deepcopy(reply["requested_by"])),
        )

    # A value the document states, reported as absent.
    for field in (
        "amount",
        "schedule_impact_days",
        "change_order_number",
        "date",
        "approval_status",
    ):
        if labels[field] is not None and is_stated(field):
            candidate = {"status": "absent", "value": "", "quotes": [], "note": ""}
            if "direction" in reply[field]:
                candidate["direction"] = "unclear"
            yield field, ABSENT_KIND, changed(field, candidate)


def inject(root: Path, *, label_checks: bool = True) -> list[Injection]:
    """Plays every mistake on every document of a set that has a label and a recorded run.

    With `label_checks=False` the same mistakes meet a pipeline that only checks that a quote
    exists and that the value in it can be read: the ablation that shows what the label
    checks add.
    """
    from .evaluation import outcome  # scoring lives there; imported late to avoid a cycle

    results: list[Injection] = []
    for path in sorted(item for item in (root / "inputs").iterdir() if item.is_file()):
        truth_path = root / "truth" / f"{path.stem}.json"
        recorded = root / "diagnostics" / f"{path.stem}.json"
        if not (truth_path.is_file() and recorded.is_file()):
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        replies = json.loads(recorded.read_text(encoding="utf-8"))["run"]["replies"]
        reply = first_json_object(replies[-1]["text"])
        if reply is None:
            continue
        document = read_document(path)
        labels = truth["fields"]
        for field, kind, changed in mistakes(reply, document, labels):
            result = process_document(
                document, ScriptedModel([json.dumps(changed)]), label_checks=label_checks
            )
            published = result.output.to_dict()[field]["value"]
            if kind == ABSENT_KIND:
                noticed = any(item.review and item.field == field for item in result.findings)
                verdict = "flagged" if noticed else "silent"
            elif published is None:
                verdict = "rejected"
            else:
                scored = outcome(
                    field, published, labels[field], truth.get("also_accept", {}).get(field, ())
                )
                verdict = "harmless" if scored == "correct_value" else "published_wrong"
            results.append(
                Injection(
                    document=path.name,
                    field=field,
                    kind=kind,
                    outcome=verdict,
                    confidence=result.decisions[field].confidence,
                    needs_review=result.output.needs_review,
                    published=published,
                    label=labels[field],
                )
            )
    return results


def summarise(injections: list[Injection]) -> dict[str, Any]:
    """Counts by kind of mistake, and every wrong value that went out."""
    kinds: dict[str, Counter[str]] = {}
    for item in injections:
        kinds.setdefault(item.kind, Counter())[item.outcome] += 1
    wrong = [item for item in injections if item.outcome == "published_wrong"]
    return {
        "injected": len(injections),
        "documents": len({item.document for item in injections}),
        "by_kind": {
            kind: {"injected": sum(counts.values()), **dict(sorted(counts.items()))}
            for kind, counts in kinds.items()
        },
        "published_wrong": len(wrong),
        "published_wrong_by_confidence": {
            f"{confidence:.2f}": count
            for confidence, count in sorted(
                Counter(item.confidence for item in wrong).items(), reverse=True
            )
        },
        "published_wrong_in_documents_flagged_for_review": sum(item.needs_review for item in wrong),
        "silent_absences": [
            {"document": item.document, "field": item.field, "label": item.label}
            for item in injections
            if item.outcome == "silent"
        ],
        "wrong_values": [
            {
                "document": item.document,
                "field": item.field,
                "mistake": item.kind,
                "published": item.published,
                "label": item.label,
                "confidence": item.confidence,
                "needs_review": item.needs_review,
            }
            for item in wrong
        ],
    }
