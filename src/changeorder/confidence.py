"""How strongly each accepted value is supported, and when a person must look.

Confidence here is a band, not a probability. The model is never asked how sure it is. Each
accepted value starts from what the deterministic checks established and is lowered by what
they could not:

    0.97  corroborated  verbatim evidence, read by rule, and an independent check agrees
                        (the contract sums before and after differ by exactly this amount;
                        a duration written in words and in figures that match)
    0.90  supported     verbatim evidence, read by rule, nothing competing or contradicting,
                        no independent confirmation
    0.75  qualified     one caveat: the quote was found only approximately, the day/month
                        order came from the document's other dates, weeks were converted,
                        approval rests on a signature alone, or the document was not read
                        completely
    0.50  weak          two or more caveats
    0.00  no value      null: absent or unresolved

Each caveat moves a value one band down. The numbers are labels for the bands, chosen before
any measurement and never fitted. They are ordinal: a higher band has passed strictly more
checks. Whether a band deserves its number is an empirical question, answered as far as the
labelled data allows in eval/REPORT.md.

Review does not follow from confidence alone. `needs_review` is true when any review finding
exists: an unresolved field, a conflict, a suspicion, a missing critical field, an unread page
or an emitted value below the threshold. A document can be valid JSON with every value at 0.90
and still need review, and a value at 0.97 can sit in a document that needs it.
"""

from __future__ import annotations

from .document import SourceDocument
from .models import Decision, Finding

__all__ = ["BANDS", "REVIEW_THRESHOLD", "band_of", "score"]

BANDS: tuple[tuple[str, float], ...] = (
    ("weak", 0.50),
    ("qualified", 0.75),
    ("supported", 0.90),
    ("corroborated", 0.97),
)
# An emitted value below this is not accepted without a person: "weak" falls under it.
REVIEW_THRESHOLD = 0.70


def band_of(confidence: float) -> str:
    """The band a published confidence belongs to."""
    for name, value in BANDS:
        if confidence == value:
            return name
    return "no_value"


def document_caveats(document: SourceDocument, findings: list[Finding]) -> list[str]:
    """What weakens every value at once: the reader did not see the whole document as written,
    or the document carries text aimed at the model."""
    codes = {finding.code for finding in findings}
    caveats = []
    if document.unread_pages:
        caveats.append("document_has_unread_pages")
    if "reading_order_not_rebuilt" in codes:
        caveats.append("reading_order_not_rebuilt")
    if codes & {"instruction_like_text", "model_reported_instructions"}:
        caveats.append("document_addresses_the_model")
    return caveats


def score(
    decisions: dict[str, Decision], document: SourceDocument, findings: list[Finding]
) -> list[Finding]:
    """Sets every decision's confidence, and returns a finding for each value under the
    review threshold."""
    shared = document_caveats(document, findings)
    low: list[Finding] = []
    for decision in decisions.values():
        if decision.state != "accepted":
            decision.confidence = 0.0
            continue
        decision.caveats.extend(caveat for caveat in shared if caveat not in decision.caveats)
        start = 3 if decision.corroborated else 2
        name, value = BANDS[max(0, start - len(decision.caveats))]
        decision.confidence = value
        if value < REVIEW_THRESHOLD:
            low.append(
                Finding(
                    "low_confidence",
                    f"{name} ({value:.2f}): " + ", ".join(decision.caveats),
                    decision.field,
                )
            )
    return low
