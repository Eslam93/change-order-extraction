"""Builders the tests share: a model reply in the asked shape, and a run over a small document.

No test here reaches a model. `ScriptedModel` answers with whatever the test wrote, which is how
a test plays a model that is right, wrong, fooled or broken, and then checks what the pipeline
let through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from changeorder.llm import Reply, ScriptedModel
from changeorder.pipeline import Result, process

CANDIDATES = (
    "project_name",
    "contract_number",
    "change_order_number",
    "date",
    "requested_by",
    "description",
    "amount",
    "schedule_impact_days",
    "approval_status",
    "original_contract_sum",
    "previous_changes",
    "contract_sum_before",
    "contract_sum_after",
)


def absent() -> dict[str, Any]:
    return {"status": "absent", "value": "", "quotes": [], "note": ""}


def stated(value: str, *quotes: str, direction: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "status": "stated",
        "value": value,
        "quotes": list(quotes),
        "note": "",
    }
    if direction is not None:
        candidate["direction"] = direction
    return candidate


def unclear(note: str, *quotes: str) -> dict[str, Any]:
    return {"status": "unclear", "value": "", "quotes": list(quotes), "note": note}


def model_reply(**fields: Any) -> str:
    """The JSON a model returns: every field absent unless the test says otherwise."""
    data: dict[str, Any] = {"document_notes": "a test document", "suspicious_instructions": False}
    for name in CANDIDATES:
        data[name] = absent()
    data.update(fields)
    for name in ("amount", "schedule_impact_days"):
        data[name].setdefault("direction", "unclear")
    return json.dumps(data)


# A small, complete change order, and the reply of a model that read it correctly.
BASE = """\
CHANGE ORDER
Project: Maple Court Apartments
Contract No.: MC-2025-014
Change Order No.: 7
Date: 15 January 2026
Requested by: Maple Court Housing LLC

Description: Add a second coat of intumescent paint to the steel columns at Level 1.

Amount of this change order (add): $12,450.00
Contract Time: extended by 3 calendar days.
Status: Approved
"""

AMOUNT_LINE = "Amount of this change order (add): $12,450.00\n"
TIME_LINE = "Contract Time: extended by 3 calendar days.\n"
DATE_LINE = "Date: 15 January 2026\n"
STATUS_LINE = "Status: Approved\n"


def base_fields() -> dict[str, Any]:
    return {
        "project_name": stated("Maple Court Apartments", "Project: Maple Court Apartments"),
        "contract_number": stated("MC-2025-014", "Contract No.: MC-2025-014"),
        "change_order_number": stated("7", "Change Order No.: 7"),
        "date": stated("15 January 2026", "Date: 15 January 2026"),
        "requested_by": stated("Maple Court Housing LLC", "Requested by: Maple Court Housing LLC"),
        "description": stated(
            "Add a second coat of intumescent paint to the steel columns at Level 1."
        ),
        "amount": stated(
            "$12,450.00", "Amount of this change order (add): $12,450.00", direction="increase"
        ),
        "schedule_impact_days": stated(
            "3 calendar days", "Contract Time: extended by 3 calendar days.", direction="increase"
        ),
        "approval_status": stated("approved", "Status: Approved"),
    }


def run(tmp_path: Path, text: str, reply: str | Reply, *, name: str = "change-order.txt") -> Result:
    """The pipeline's result for a text document, with a model that gives exactly `reply`."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return process(path, ScriptedModel([reply]))


def run_base(tmp_path: Path, *edits: tuple[str, str], **fields: Any) -> Result:
    """The base document with some of its text replaced, read by a model whose reply is the
    correct one with some fields replaced."""
    text = BASE
    for old, new in edits:
        assert old in text, f"{old!r} is not in the base document"
        text = text.replace(old, new)
    return run(tmp_path, text, model_reply(**{**base_fields(), **fields}))


def codes(result: Result, field: str | None = None) -> set[str]:
    """The codes of the findings that ask for review, for one field or for all."""
    return {
        finding.code
        for finding in result.findings
        if finding.review and (field is None or finding.field == field)
    }
