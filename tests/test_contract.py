"""The published JSON Schema and the pydantic model: one contract, checked from both sides.

The schema is what a consumer validates against; the pydantic model is what the code builds.
They are written separately, so these tests feed the same documents to both and require the
same verdict, and require the pipeline's real output to pass the published schema.
"""

from __future__ import annotations

import copy
import datetime
import json
from decimal import Decimal
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from changeorder.models import (
    FIELD_NAMES,
    AmountField,
    ChangeOrder,
    DateField,
    DaysField,
    StatusField,
    TextField,
)
from changeorder.pipeline import contract_problems, published_schema
from helpers import run_base

VALID: dict[str, Any] = {
    "project_name": {"value": "Maple Court Apartments", "confidence": 0.9},
    "contract_number": {"value": None, "confidence": 0.0},
    "change_order_number": {"value": "7", "confidence": 0.9},
    "date": {"value": "2026-01-15", "confidence": 0.75},
    "requested_by": {"value": None, "confidence": 0},
    "description": {"value": "Add a second coat of paint.", "confidence": 0.9},
    "amount": {"value": -12450.5, "confidence": 0.97},
    "schedule_impact_days": {"value": 0, "confidence": 0.9},
    "approval_status": {"value": "pending", "confidence": 0.5},
    "needs_review": True,
}


def _pydantic_accepts(data: Any) -> bool:
    try:
        ChangeOrder.model_validate_json(json.dumps(data))
    except ValidationError:
        return False
    return True


def _with(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


def _without(*path: str) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    node = data
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return data


def test_the_published_schema_is_a_valid_draft_2020_12_schema_with_every_field():
    schema = published_schema()

    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["required"] == [*FIELD_NAMES, "needs_review"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {*FIELD_NAMES, "needs_review"}


def test_a_valid_result_passes_both():
    assert contract_problems(VALID) == []
    assert _pydantic_accepts(VALID)


@pytest.mark.parametrize(
    "broken",
    [
        _without("amount"),  # all nine fields are always present
        _without("needs_review"),
        _without("date", "confidence"),
        _without("date", "value"),
        _with(("evidence",), ["page 1"]),  # no extra top-level property
        _with(("amount", "currency"), "USD"),  # each field has exactly value and confidence
        _with(("amount", "value"), "12450.00"),  # a number, not a string
        _with(("amount", "value"), True),
        _with(("schedule_impact_days", "value"), 1.5),  # whole days
        _with(("schedule_impact_days", "value"), "14"),
        _with(("date", "value"), "03/04/2026"),  # ISO 8601 only
        _with(("date", "value"), "2026-02-30"),  # a real date
        _with(("date", "value"), "2026-3-4"),
        _with(("approval_status", "value"), "accepted"),  # one of the documented statuses
        _with(("approval_status", "value"), "Approved"),
        _with(("project_name", "value"), ""),  # missing is null, never an empty string
        _with(("project_name", "value"), "   "),
        _with(("project_name", "value"), 7),
        _with(("amount", "confidence"), 1.2),
        _with(("amount", "confidence"), -0.1),
        _with(("amount", "confidence"), "high"),
        _with(("amount", "confidence"), None),
        _with(("contract_number", "confidence"), 0.5),  # a null value carries confidence 0
        _with(("needs_review",), "true"),
        _with(("needs_review",), 1),
        [VALID],
        "not an object",
    ],
)
def test_what_breaks_the_contract_is_refused_by_the_schema_and_by_the_model_alike(broken):
    assert contract_problems(broken) != []
    assert not _pydantic_accepts(broken)


@pytest.mark.parametrize(
    "fine",
    [
        _with(("amount", "value"), 0),  # an explicit zero is a value
        _with(("amount", "value"), -5000),
        _with(("amount", "value"), None) | {"amount": {"value": None, "confidence": 0.0}},
        _with(("schedule_impact_days", "value"), -10),
        _with(("needs_review",), False),
        {
            **{name: {"value": None, "confidence": 0.0} for name in FIELD_NAMES},
            "needs_review": True,
        },
    ],
)
def test_nulls_zeros_and_negatives_are_all_within_the_contract(fine):
    assert contract_problems(fine) == []
    assert _pydantic_accepts(fine)


def test_the_delivered_json_has_plain_types_in_contract_order():
    order = ChangeOrder(
        project_name=TextField(value="Maple Court Apartments", confidence=0.9),
        contract_number=TextField(value=None, confidence=0.0),
        change_order_number=TextField(value="7", confidence=0.9),
        date=DateField(value=datetime.date(2026, 1, 15), confidence=0.9),
        requested_by=TextField(value=None, confidence=0.0),
        description=TextField(value="Add a second coat of paint.", confidence=0.9),
        amount=AmountField(value=Decimal("-110983.75"), confidence=0.97),
        schedule_impact_days=DaysField(value=-10, confidence=0.9),
        approval_status=StatusField(value="approved", confidence=0.75),
        needs_review=False,
    )

    data = order.to_dict()

    assert list(data) == [*FIELD_NAMES, "needs_review"]
    assert all(set(data[name]) == {"value", "confidence"} for name in FIELD_NAMES)
    assert data["date"]["value"] == "2026-01-15"
    assert data["amount"]["value"] == -110983.75 and isinstance(data["amount"]["value"], float)
    assert json.loads(json.dumps(data)) == data  # nothing in it that JSON cannot carry
    assert contract_problems(data) == []
    # A whole amount goes out as an integer, and cents are never lost to float arithmetic.
    whole = order.model_copy(
        update={"amount": AmountField(value=Decimal("38640.00"), confidence=0.9)}
    )
    assert whole.to_dict()["amount"]["value"] == 38640
    assert isinstance(whole.to_dict()["amount"]["value"], int)
    cents = order.model_copy(
        update={"amount": AmountField(value=Decimal("0.1") + Decimal("0.2"), confidence=0.9)}
    )
    assert cents.to_dict()["amount"]["value"] == 0.3


def test_the_model_refuses_what_the_types_do_not_allow():
    with pytest.raises(ValidationError):
        AmountField(value=Decimal("NaN"), confidence=0.9)
    with pytest.raises(ValidationError):
        AmountField(value=Decimal("Infinity"), confidence=0.9)
    with pytest.raises(ValidationError):
        DaysField(value=True, confidence=0.9)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TextField(value="x", confidence=float("nan"))
    with pytest.raises(ValidationError):
        TextField(value=None, confidence=0.9)


def test_what_the_pipeline_actually_delivers_passes_the_published_schema(tmp_path):
    result = run_base(tmp_path)

    delivered = json.loads(json.dumps(result.output.to_dict()))

    assert contract_problems(delivered) == []
    assert _pydantic_accepts(delivered)
