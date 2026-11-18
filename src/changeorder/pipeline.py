"""One document in, one checked result out.

    read -> ask the model for candidates -> decide each field against its evidence
         -> validate across the document -> score and decide review -> check the contract

Each arrow is one function in one module. Two kinds of failure leave this function as an
exception and never as output: the document could not be read (`DocumentError`) and the model
gave nothing usable (`LLMError`). A document that was read and extracted always returns a
`Result`, however much of it needs review.

The last step checks the exact dictionary that will be printed against the published JSON
Schema. A failure there is a bug in this code, not a property of the document, so it raises
`ContractError` instead of printing something the contract does not allow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from . import __version__
from .confidence import band_of, score
from .config import Limits
from .document import SourceDocument, read_document
from .extract import ExtractionRun, extract_candidates
from .fields import decide
from .llm import ChatModel
from .models import ChangeOrder, Decision, Finding, _plain
from .prompt import PROMPT_LABEL, fingerprint
from .validate import validate

__all__ = [
    "ContractError",
    "Result",
    "contract_problems",
    "process",
    "process_document",
    "published_schema",
]


class ContractError(Exception):
    """The pipeline built something the published schema does not allow: a bug, never data."""


@dataclass(frozen=True)
class Result:
    output: ChangeOrder
    decisions: dict[str, Decision]
    findings: tuple[Finding, ...]
    document: SourceDocument
    run: ExtractionRun
    model: dict[str, object]

    @property
    def review_reasons(self) -> list[str]:
        return [
            f"{finding.field}: {finding.code}" if finding.field else finding.code
            for finding in self.findings
            if finding.review
        ]

    def diagnostics(self) -> dict[str, Any]:
        """Everything the public JSON leaves out: evidence, caveats, findings, and enough about
        the run to reproduce it. The recorded replies let `--replay` rebuild the same output
        with no model and no key."""
        return {
            "document": self.document.describe(),
            "run": {
                "pipeline_version": __version__,
                "prompt": PROMPT_LABEL,
                "prompt_fingerprint": fingerprint(),
                "model": self.model,
                "corrective_retry": self.run.corrected,
                "replies": [reply.as_dict() for reply in self.run.replies],
            },
            "model_notes": self.run.extraction.document_notes,
            "needs_review": self.output.needs_review,
            "findings": [asdict(finding) for finding in self.findings],
            "fields": {
                name: {
                    "state": decision.state,
                    "value": _plain(decision.value),
                    "as_written": decision.raw,
                    "confidence": decision.confidence,
                    "band": band_of(decision.confidence),
                    "corroborated": decision.corroborated,
                    "caveats": decision.caveats,
                    "reasons": decision.reasons,
                    "model_note": decision.note,
                    "evidence": [asdict(item) for item in decision.evidence],
                }
                for name, decision in self.decisions.items()
            },
        }


def process(path: Path, model: ChatModel, *, limits: Limits | None = None) -> Result:
    """The checked result for one file, or `DocumentError`, `LLMError` or `ContractError`."""
    return process_document(read_document(path, limits), model)


def process_document(
    document: SourceDocument, model: ChatModel, *, label_checks: bool = True
) -> Result:
    """The checked result for a document that has been read. `label_checks` is for the stress
    test's ablation only (see `fields.decide`)."""
    run = extract_candidates(document.text, model)
    decisions = decide(run.extraction, document, label_checks=label_checks)
    findings = validate(decisions, run.extraction, document)
    findings += score(decisions, document, findings)

    fields: dict[str, Any] = {
        # Each field's own class checks its value's type: no coercion, no surprises.
        name: ChangeOrder.model_fields[name].annotation(  # type: ignore[misc]
            value=decision.value, confidence=decision.confidence
        )
        for name, decision in decisions.items()
    }
    output = ChangeOrder(**fields, needs_review=any(finding.review for finding in findings))
    problems = contract_problems(output.to_dict())
    if problems:
        raise ContractError("the result breaks the published schema: " + "; ".join(problems))
    return Result(output, decisions, tuple(findings), document, run, model.describe())


@cache
def published_schema() -> dict[str, Any]:
    """The JSON Schema shipped with the package: the contract for what goes to stdout."""
    text = resources.files("changeorder").joinpath("change_order.schema.json").read_text("utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def contract_problems(data: Any) -> list[str]:
    """What the published schema objects to in `data`; empty when it is valid."""
    validator = Draft202012Validator(published_schema(), format_checker=FormatChecker())
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '(root)'}: {error.message}"
        for error in sorted(
            validator.iter_errors(data), key=lambda error: list(error.absolute_path)
        )
    ]
