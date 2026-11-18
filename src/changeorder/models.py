"""The public output contract, and the records the pipeline keeps to itself.

Two shapes live here on purpose. `ChangeOrder` is the published contract: nine fields, each
exactly `{value, confidence}`, plus `needs_review`. `Decision` and `Finding` hold what the pipeline
knows about how it got there (evidence, caveats, review reasons). They never enter the public JSON;
`--diagnostics` writes them to a separate file.
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

FIELD_NAMES: tuple[str, ...] = (
    "project_name",
    "contract_number",
    "change_order_number",
    "date",
    "requested_by",
    "description",
    "amount",
    "schedule_impact_days",
    "approval_status",
)

ApprovalStatus = Literal["approved", "rejected", "pending", "draft", "void"]
APPROVAL_STATUSES: tuple[str, ...] = get_args(ApprovalStatus)

# ---- The published contract --------------------------------------------------------------------

Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
# A string value is never empty or blank: a missing value is null.
NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Field(_Contract):
    confidence: Confidence

    @model_validator(mode="after")
    def _null_has_no_confidence(self) -> Self:
        # The convention for an absent value: null with confidence 0.0, "no accepted extraction".
        if getattr(self, "value", None) is None and self.confidence != 0.0:
            raise ValueError("a null value carries confidence 0.0")
        return self


class TextField(_Field):
    value: NonBlank | None


class DateField(_Field):
    value: datetime.date | None


class AmountField(_Field):
    value: Annotated[Decimal, Field(allow_inf_nan=False)] | None

    @field_validator("value", mode="before")
    @classmethod
    def _a_number_not_text(cls, value: Any) -> Any:
        # Read back from JSON, an amount arrives as an int or a float; text is never one, even
        # text that looks like a number, and neither is true or false.
        if isinstance(value, bool | str):
            raise ValueError("an amount is a number")
        if isinstance(value, int | float):
            return Decimal(str(value))
        return value


class DaysField(_Field):
    value: int | None


class StatusField(_Field):
    value: ApprovalStatus | None


class ChangeOrder(_Contract):
    """One change order, as delivered. `to_dict()` is the JSON that goes to stdout."""

    project_name: TextField
    contract_number: TextField
    change_order_number: TextField
    date: DateField
    requested_by: TextField
    description: TextField
    amount: AmountField
    schedule_impact_days: DaysField
    approval_status: StatusField
    needs_review: bool

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON types: a date as YYYY-MM-DD, an amount as a JSON number."""
        result: dict[str, Any] = {}
        for name in FIELD_NAMES:
            item = getattr(self, name)
            result[name] = {"value": _plain(item.value), "confidence": item.confidence}
        result["needs_review"] = self.needs_review
        return result


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Money is exact in Decimal up to here. JSON has one number type: a whole amount goes out
        # as an integer and anything else as the shortest float that reads back to the same cents.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, datetime.date):
        return value.isoformat()
    return value


# ---- What the pipeline knows and does not publish ----------------------------------------------

State = Literal["accepted", "absent", "unresolved"]


@dataclass(frozen=True)
class Evidence:
    """One quote the model gave, and where code found it in the document (or that it did not)."""

    quote: str
    kind: str  # exact, normalized, fuzzy, not_found
    score: float
    start: int | None = None
    end: int | None = None
    page: int | None = None
    line: int | None = None


@dataclass
class Decision:
    """What became of one field.

    `accepted`: a value that survived every check. `absent`: the document does not state it, as
    far as the model and the deterministic scans can tell. `unresolved`: something was there (a
    candidate, a contradiction, a deferral) but no value could be accepted. The public JSON shows
    the last two alike, as null with confidence 0.0; review logic does not.
    """

    field: str
    state: State
    value: Any = None
    raw: str | None = None  # the value as the document writes it
    span: tuple[int, int] | None = None  # where that is in the document's text
    evidence: list[Evidence] = dataclasses.field(default_factory=list)
    reasons: list[str] = dataclasses.field(default_factory=list)  # why it is unresolved
    caveats: list[str] = dataclasses.field(
        default_factory=list
    )  # what lowered an accepted value's band
    corroborated: bool = False  # an independent deterministic check agrees with the value
    confidence: float = 0.0
    note: str = ""  # the model's own remark, kept for the person who reviews


@dataclass(frozen=True)
class Finding:
    """One result of validation. `review` findings set needs_review; the rest are notes."""

    code: str
    detail: str
    field: str | None = None
    review: bool = True
