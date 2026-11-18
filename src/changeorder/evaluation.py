"""Measuring the pipeline against hand labels, and keeping committed outputs honest.

    python -m changeorder.evaluation run samples       call the live model for samples/inputs
    python -m changeorder.evaluation run eval          the same for eval/inputs
    python -m changeorder.evaluation replay samples    no network: rebuild each output from its
    python -m changeorder.evaluation replay eval       recorded replies; fail if one differs
    python -m changeorder.evaluation rebuild eval      no network: rewrite each output from its
                                                       recorded replies with the current code
    python -m changeorder.evaluation report            score both sets; write eval/REPORT.md

A set is a folder with `inputs/` (documents), `truth/` (one hand-written label file per
document), `outputs/` (the JSON the pipeline delivered) and `diagnostics/` (evidence, findings
and the model's raw replies for that run). Labels are written by a person from the document
and never from a model's output. Outputs are written by `run` and never edited: `replay`
proves it, by regenerating every output from the recorded replies with the current code.
`rebuild` is for a change to the deterministic checks: the model's replies are still the ones
a live run recorded, and only what code makes of them is derived again.

Scoring is per field. Each of a document's nine fields ends in one of five outcomes:

    correct_value   the label has a value and the pipeline published the same one
    correct_null    the label is null and the pipeline published null
    wrong_value     both have a value and they differ
    invented        the label is null and the pipeline published a value
    missed          the label has a value and the pipeline published null

`wrong_value` and `invented` are the errors that matter most: a value that looks fine and is
not. `missed` is an abstention; whether it was flagged for review is counted separately.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import llm_settings, load_env_file
from .document import DocumentError
from .llm import LLMError, OpenAICompatibleModel, Reply, ScriptedModel
from .models import FIELD_NAMES
from .pipeline import ContractError, Result, contract_problems, process
from .prompt import PROMPT_LABEL, fingerprint
from .stress import ABSENT_KIND, inject
from .stress import summarise as stress_summary

SETS = ("samples", "eval")
REPORT = Path("eval") / "REPORT.md"
RESULTS = Path("eval") / "results.json"
OUTCOMES = ("correct_value", "correct_null", "wrong_value", "invented", "missed")
# How a field's published value is compared with its label.
KINDS = {
    "project_name": "text",
    "contract_number": "code",
    "change_order_number": "code",
    "date": "exact",
    "requested_by": "text",
    "description": "passage",
    "amount": "money",
    "schedule_impact_days": "exact",
    "approval_status": "exact",
}
# A description counts as right when it shares this much of its wording with the label. The
# pipeline copies wording from the document, so the only freedom left is where the copy starts
# and stops; 0.8 allows a list number or a trailing clause and nothing looser.
PASSAGE_F1 = 0.8


# ---- Comparing a published value with its label ------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold())


def _same(field: str, published: Any, label: Any) -> bool:
    kind = KINDS[field]
    if kind == "text":
        return _words(published) == _words(label)
    if kind == "code":
        return "".join(_words(published)) == "".join(_words(label))
    if kind == "money":
        cents = Decimal("0.01")
        return Decimal(str(published)).quantize(cents) == Decimal(str(label)).quantize(cents)
    if kind == "passage":
        ours, theirs = Counter(_words(published)), Counter(_words(label))
        shared = sum((ours & theirs).values())
        if not shared:
            return False
        precision, recall = shared / sum(ours.values()), shared / sum(theirs.values())
        return 2 * precision * recall / (precision + recall) >= PASSAGE_F1
    return bool(published == label)


def outcome(field: str, published: Any, label: Any, also_accept: Sequence[Any] = ()) -> str:
    if label is None:
        return "correct_null" if published is None else "invented"
    if published is None:
        return "missed"
    accepted = any(_same(field, published, candidate) for candidate in (label, *also_accept))
    return "correct_value" if accepted else "wrong_value"


# ---- One set of documents ----------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """One field of one document, scored."""

    document: str
    field: str
    outcome: str
    published: Any
    label: Any
    confidence: float
    why: str  # the pipeline's own reasons and caveats for this field


@dataclass(frozen=True)
class Document:
    name: str
    needs_review: bool
    review_expected: bool
    fields: tuple[Scored, ...]
    usage: tuple[int, int, float]  # prompt tokens, completion tokens, cost

    @property
    def errors(self) -> int:
        return sum(item.outcome in ("wrong_value", "invented") for item in self.fields)


def _inputs(root: Path) -> list[Path]:
    return sorted(path for path in (root / "inputs").iterdir() if path.is_file())


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def score_set(root: Path) -> list[Document]:
    """Every document of a set that has a label and a committed output."""
    documents: list[Document] = []
    for path in _inputs(root):
        truth_path = root / "truth" / f"{path.stem}.json"
        output_path = root / "outputs" / f"{path.stem}.json"
        if not (truth_path.is_file() and output_path.is_file()):
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        output = json.loads(output_path.read_text(encoding="utf-8"))
        diagnostics_path = root / "diagnostics" / f"{path.stem}.json"
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.is_file()
            else {}
        )
        fields = []
        for name in FIELD_NAMES:
            detail = diagnostics.get("fields", {}).get(name, {})
            why = ", ".join([*detail.get("reasons", []), *detail.get("caveats", [])])
            fields.append(
                Scored(
                    document=path.name,
                    field=name,
                    outcome=outcome(
                        name,
                        output[name]["value"],
                        truth["fields"][name],
                        truth.get("also_accept", {}).get(name, ()),
                    ),
                    published=output[name]["value"],
                    label=truth["fields"][name],
                    confidence=output[name]["confidence"],
                    why=why,
                )
            )
        replies = diagnostics.get("run", {}).get("replies", [])
        usage = (
            sum(reply.get("prompt_tokens", 0) for reply in replies),
            sum(reply.get("completion_tokens", 0) for reply in replies),
            sum(reply.get("cost") or 0.0 for reply in replies),
        )
        documents.append(
            Document(
                path.name, output["needs_review"], truth["review_expected"], tuple(fields), usage
            )
        )
    return documents


# ---- Numbers -----------------------------------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a proportion: honest about small samples, unlike +/- sigma."""
    if total == 0:
        return 0.0, 1.0
    share = successes / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    scale = 1 + z * z / total
    return max(0.0, (centre - spread) / scale), min(1.0, (centre + spread) / scale)


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def field_metrics(fields: Iterable[Scored]) -> dict[str, Any]:
    """The headline rates over a group of scored fields."""
    counts = Counter(item.outcome for item in fields)
    right = counts["correct_value"]
    published = right + counts["wrong_value"] + counts["invented"]
    stated = right + counts["wrong_value"] + counts["missed"]
    not_stated = counts["correct_null"] + counts["invented"]
    total = sum(counts.values())
    return {
        "fields": total,
        **{name: counts[name] for name in OUTCOMES},
        "accuracy": _ratio(right + counts["correct_null"], total),
        "published": published,
        "precision_on_published_values": _ratio(right, published),
        "stated_in_document": stated,
        "recall_of_stated_values": _ratio(right, stated),
        "abstention_rate_on_stated_values": _ratio(counts["missed"], stated),
        "not_stated_in_document": not_stated,
        "null_detection_rate": _ratio(counts["correct_null"], not_stated),
        "false_extraction_rate": _ratio(counts["invented"], not_stated),
    }


def confidence_metrics(fields: Sequence[Scored]) -> dict[str, Any]:
    """Accuracy by confidence band, at thresholds, and the calibration gap, over the values
    that were published. A null has no confidence to assess."""
    published = [item for item in fields if item.published is not None]
    bands: list[dict[str, Any]] = []
    for confidence in sorted({item.confidence for item in published}, reverse=True):
        group = [item for item in published if item.confidence == confidence]
        right = sum(item.outcome == "correct_value" for item in group)
        low, high = wilson(right, len(group))
        bands.append(
            {
                "confidence": confidence,
                "published": len(group),
                "correct": right,
                "accuracy": _ratio(right, len(group)),
                "wilson_95": [round(low, 3), round(high, 3)],
            }
        )
    thresholds: list[dict[str, Any]] = []
    for band in sorted(bands, key=lambda band: band["confidence"]):
        group = [item for item in published if item.confidence >= band["confidence"]]
        wrong = sum(item.outcome != "correct_value" for item in group)
        thresholds.append(
            {
                "at_or_above": band["confidence"],
                "published": len(group),
                "errors": wrong,
                "accuracy": _ratio(len(group) - wrong, len(group)),
                "coverage_of_published": _ratio(len(group), len(published)),
            }
        )
    gap = sum(
        band["published"] / len(published) * abs(band["accuracy"] - band["confidence"])
        for band in bands
    )
    return {
        "bands": bands,
        "thresholds": thresholds,
        "expected_calibration_error": round(gap, 4) if published else None,
    }


def review_metrics(documents: Sequence[Document]) -> dict[str, Any]:
    """How the review decision relates to what a person would say, and to where the errors are."""
    flagged = [document for document in documents if document.needs_review]
    passed = [document for document in documents if not document.needs_review]

    def published(group: Sequence[Document]) -> int:
        return sum(item.published is not None for document in group for item in document.fields)

    def missed(group: Sequence[Document]) -> int:
        return sum(item.outcome == "missed" for document in group for item in document.fields)

    return {
        "documents": len(documents),
        "flagged_for_review": len(flagged),
        "review_rate": _ratio(len(flagged), len(documents)),
        "flagged_and_expected": sum(document.review_expected for document in flagged),
        "flagged_not_expected": sum(not document.review_expected for document in flagged),
        "passed_but_review_expected": sum(document.review_expected for document in passed),
        "passed_without_review": {
            "documents": len(passed),
            "published_values": published(passed),
            "wrong_or_invented_values": sum(document.errors for document in passed),
            "error_rate": _ratio(sum(document.errors for document in passed), published(passed)),
            "values_missed_silently": missed(passed),
        },
        "flagged": {
            "documents": len(flagged),
            "published_values": published(flagged),
            "wrong_or_invented_values": sum(document.errors for document in flagged),
            "values_missed": missed(flagged),
        },
    }


def summarise(documents: Sequence[Document]) -> dict[str, Any]:
    fields = [item for document in documents for item in document.fields]
    return {
        "overall": field_metrics(fields),
        "by_field": {
            name: field_metrics(item for item in fields if item.field == name)
            for name in FIELD_NAMES
        },
        "confidence": confidence_metrics(fields),
        "review": review_metrics(documents),
        "usage": {
            "prompt_tokens": sum(document.usage[0] for document in documents),
            "completion_tokens": sum(document.usage[1] for document in documents),
            "cost": round(sum(document.usage[2] for document in documents), 4),
        },
        "not_correct": [
            {
                "document": item.document,
                "field": item.field,
                "outcome": item.outcome,
                "published": item.published,
                "label": item.label,
                "confidence": item.confidence,
                "why": item.why,
            }
            for item in fields
            if item.outcome not in ("correct_value", "correct_null")
        ],
    }


# ---- The report --------------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _run_details(root: Path) -> dict[str, Any]:
    """What produced a set's outputs, read from its diagnostics."""
    models, prompts, formats = Counter(), Counter(), Counter()  # type: ignore[var-annotated]
    for path in sorted((root / "diagnostics").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8")).get("run", {})
        models[str(run.get("model", {}).get("model"))] += 1
        prompts[f"{run.get('prompt')} ({str(run.get('prompt_fingerprint'))[:12]})"] += 1
        for reply in run.get("replies", [])[-1:]:
            formats[str(reply.get("response_format"))] += 1
    return {"model": dict(models), "prompt": dict(prompts), "response_format": dict(formats)}


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _section(title: str, root: Path, summary: dict[str, Any]) -> list[str]:
    overall, review = summary["overall"], summary["review"]
    confidence = summary["confidence"]
    details = _run_details(root)
    usage = summary["usage"]
    lines = [
        f"## {title}",
        "",
        f"{review['documents']} documents, {overall['fields']} fields. "
        f"Model: {', '.join(details['model']) or 'unknown'}. "
        f"Prompt: {', '.join(details['prompt']) or 'unknown'}. "
        f"Response format: {', '.join(details['response_format']) or 'unknown'}. "
        f"Tokens: {usage['prompt_tokens']:,} prompt, {usage['completion_tokens']:,} completion; "
        f"cost reported by the provider: {usage['cost']:.2f}.",
        "",
        "### Field outcomes",
        "",
        *_table(
            ("Measure", "Value"),
            [
                (
                    "Fields correct (value or null)",
                    f"{overall['correct_value'] + overall['correct_null']} of {overall['fields']} ({_percent(overall['accuracy'])})",
                ),
                (
                    "Published values that are correct",
                    f"{overall['correct_value']} of {overall['published']} ({_percent(overall['precision_on_published_values'])})",
                ),
                ("Wrong values published", overall["wrong_value"]),
                (
                    "Values published where the document has none (false extraction)",
                    f"{overall['invented']} of {overall['not_stated_in_document']} ({_percent(overall['false_extraction_rate'])})",
                ),
                (
                    "Absent values correctly left null",
                    f"{overall['correct_null']} of {overall['not_stated_in_document']} ({_percent(overall['null_detection_rate'])})",
                ),
                (
                    "Stated values recovered",
                    f"{overall['correct_value']} of {overall['stated_in_document']} ({_percent(overall['recall_of_stated_values'])})",
                ),
                (
                    "Stated values left null (abstentions)",
                    f"{overall['missed']} of {overall['stated_in_document']} ({_percent(overall['abstention_rate_on_stated_values'])})",
                ),
            ],
        ),
        "",
        "### By field",
        "",
        *_table(
            (
                "Field",
                "Correct",
                "Published",
                "Right when published",
                "Wrong",
                "Invented",
                "Missed",
            ),
            [
                (
                    f"`{name}`",
                    f"{row['correct_value'] + row['correct_null']}/{row['fields']}",
                    row["published"],
                    _percent(row["precision_on_published_values"]),
                    row["wrong_value"],
                    row["invented"],
                    row["missed"],
                )
                for name, row in summary["by_field"].items()
            ],
        ),
        "",
        "### Confidence",
        "",
        *_table(
            ("Confidence", "Published", "Correct", "Accuracy", "95% Wilson interval"),
            [
                (
                    f"{band['confidence']:.2f}",
                    band["published"],
                    band["correct"],
                    _percent(band["accuracy"]),
                    f"{band['wilson_95'][0]:.0%} to {band['wilson_95'][1]:.0%}",
                )
                for band in confidence["bands"]
            ],
        ),
        "",
        *_table(
            (
                "Accept at or above",
                "Published values kept",
                "Errors among them",
                "Accuracy",
                "Share of published values",
            ),
            [
                (
                    f"{row['at_or_above']:.2f}",
                    row["published"],
                    row["errors"],
                    _percent(row["accuracy"]),
                    _percent(row["coverage_of_published"]),
                )
                for row in confidence["thresholds"]
            ],
        ),
        "",
        f"Expected calibration error over published values: {confidence['expected_calibration_error']}. "
        "With this few values in each band it is a description of this run, not an estimate.",
        "",
        "### Review",
        "",
        *_table(
            ("Measure", "Value"),
            [
                (
                    "Documents flagged for review",
                    f"{review['flagged_for_review']} of {review['documents']} ({_percent(review['review_rate'])})",
                ),
                ("Flagged, and the label says review is warranted", review["flagged_and_expected"]),
                (
                    "Flagged, and the label says it is not (unnecessary review)",
                    review["flagged_not_expected"],
                ),
                (
                    "Passed, and the label says review is warranted (missed review)",
                    review["passed_but_review_expected"],
                ),
                (
                    "Wrong or invented values in documents passed without review",
                    f"{review['passed_without_review']['wrong_or_invented_values']} of {review['passed_without_review']['published_values']} published",
                ),
                (
                    "Stated values silently left null in passed documents",
                    review["passed_without_review"]["values_missed_silently"],
                ),
                (
                    "Wrong or invented values in flagged documents",
                    f"{review['flagged']['wrong_or_invented_values']} of {review['flagged']['published_values']} published",
                ),
                (
                    "Stated values left null in flagged documents",
                    review["flagged"]["values_missed"],
                ),
            ],
        ),
        "",
        "### Every field that is not correct",
        "",
    ]
    if summary["not_correct"]:
        lines += _table(
            (
                "Document",
                "Field",
                "Outcome",
                "Published",
                "Label",
                "Confidence",
                "Pipeline's reason",
            ),
            [
                (
                    f"`{row['document']}`",
                    f"`{row['field']}`",
                    row["outcome"],
                    f"`{json.dumps(row['published'], ensure_ascii=False)[:60]}`",
                    f"`{json.dumps(row['label'], ensure_ascii=False)[:60]}`",
                    row["confidence"],
                    row["why"] or "none recorded",
                )
                for row in summary["not_correct"]
            ],
        )
    else:
        lines.append("None.")
    return [*lines, ""]


PREAMBLE = """\
# Evaluation report

Generated by `python -m changeorder.evaluation report` from the committed outputs, the recorded
model replies and the hand-written labels. Nothing here is typed in by hand: `make eval-replay`
rebuilds every output from the recorded replies and fails if an output or this report differs.

Read the numbers with these limits in mind. docs/WRITEUP.md discusses each.

- **Every document is synthetic and was written by the author of the pipeline.** The six samples
  and the edge cases share one person's idea of what a change order looks like. Real documents
  will be harder in ways these are not.
- **These are development results, not a held-out test.** The labels were written before any
  model read the documents, but the checks were then improved against these same documents.
  The numbers show that the pipeline behaves as intended on the cases it was built against.
  They overstate what to expect on documents nobody has seen.
- **The sets are small.** The intervals below are wide, and a band with a handful of values
  says almost nothing about that band.
- **The edge cases over-represent trouble.** They were chosen because they are hard, so the
  share of documents sent to review here is far above what ordinary documents would give.
- **Confidence values are band labels fixed in advance** (0.97, 0.90, 0.75, 0.50), not fitted
  to these results and not calibrated probabilities.
- **One run per document.** Each section names the model that produced its replies. A live
  model is not deterministic, and another run can differ. When a deterministic check changes,
  outputs are rebuilt from the recorded replies, so no number here rests on a second call.
"""

STRESS_INTRO = """\
## What the checks do when the model is wrong

No model made a mistake that reached an output above, so those runs cannot show what a mistake
would do. Here each document's recorded reply is changed in one field, the way a model
plausibly gets it wrong, and the deterministic stages run on the changed reply
(`src/changeorder/stress.py`; no model is called). A mistake is **refused** when the field is
published as null, **harmless** when the changed reply still yields the labelled value, and
**published** when a wrong value goes out.
"""

STRESS_NOTE = """\
The mistakes are synthetic and their mix is the author's guess at what models get wrong, so
these shares are not error rates. They show which kinds of mistake the checks refuse, and which
kinds still pass. The checks that refuse them were tightened against this same list, so it is
a record of what they now cover, not an independent test of them.
"""


def _stress_section(stress: dict[str, Any]) -> list[str]:
    by_kind = stress["by_kind"]
    absent = by_kind.get(ABSENT_KIND, {})
    without = stress["without_label_checks"]
    lines = [
        STRESS_INTRO,
        f"{stress['injected']} mistakes were played on {stress['documents']} documents.",
        "",
        *_table(
            (
                "Mistake played",
                "Played",
                "Refused",
                "Harmless",
                "Wrong value published",
                "Published if labels were not checked",
            ),
            [
                (
                    kind.capitalize(),
                    row["injected"],
                    row.get("rejected", 0),
                    row.get("harmless", 0),
                    row.get("published_wrong", 0),
                    without["by_kind"].get(kind, 0),
                )
                for kind, row in by_kind.items()
                if kind != ABSENT_KIND
            ],
        ),
        "",
        f"The last column is an ablation. With the label checks switched off, so that a value is "
        f"accepted wherever its quote is found and it can be read, {without['published_wrong']} "
        f"of these mistakes are published instead of {stress['published_wrong']}. A quote that "
        "exists is not a quote that supports the field.",
        "",
        f"A value the document states, reported as absent, was played {absent.get('injected', 0)} "
        f"times. {absent.get('flagged', 0)} sent the document to review for that field; "
        f"{absent.get('silent', 0)} passed unnoticed.",
        "",
    ]
    wrong = stress["wrong_values"]
    if wrong:
        bands = ", ".join(
            f"{count} at {confidence}"
            for confidence, count in stress["published_wrong_by_confidence"].items()
        )
        lines += [
            f"Wrong values published: {stress['published_wrong']} ({bands}), "
            f"{stress['published_wrong_in_documents_flagged_for_review']} of them in documents "
            "flagged for review.",
            "",
            *_table(
                ("Document", "Field", "Mistake", "Published", "Label", "Confidence", "Flagged"),
                [
                    (
                        f"`{row['document']}`",
                        f"`{row['field']}`",
                        row["mistake"],
                        f"`{json.dumps(row['published'], ensure_ascii=False)[:50]}`",
                        f"`{json.dumps(row['label'], ensure_ascii=False)[:50]}`",
                        row["confidence"],
                        "yes" if row["needs_review"] else "no",
                    )
                    for row in wrong
                ],
            ),
            "",
        ]
    else:
        lines += ["No wrong value was published.", ""]
    if stress["silent_absences"]:
        lines += [
            "False absences that passed unnoticed:",
            "",
            *_table(
                ("Document", "Field", "Label"),
                [
                    (
                        f"`{row['document']}`",
                        f"`{row['field']}`",
                        f"`{json.dumps(row['label'], ensure_ascii=False)[:50]}`",
                    )
                    for row in stress["silent_absences"]
                ],
            ),
            "",
        ]
    return [*lines, STRESS_NOTE]


def render(
    summaries: dict[str, tuple[Path, dict[str, Any]]], stress: dict[str, Any] | None = None
) -> str:
    lines = [PREAMBLE]
    titles = {
        "samples": "The six sample documents",
        "eval": "The labelled edge cases",
    }
    for name, (root, summary) in summaries.items():
        lines += _section(titles.get(name, name), root, summary)
    if stress is not None:
        lines += _stress_section(stress)
    return "\n".join(lines).rstrip() + "\n"


# ---- Running -----------------------------------------------------------------------------------


def _write(result: Result, root: Path, stem: str) -> None:
    for folder, data in (
        ("outputs", result.output.to_dict()),
        ("diagnostics", result.diagnostics()),
    ):
        target = root / folder / f"{stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_dump(data), encoding="utf-8", newline="\n")


def run_live(root: Path, only: Sequence[str] = ()) -> int:
    """Calls the configured model for every input of a set and writes what came back."""
    load_env_file()
    settings = llm_settings()
    paths = [path for path in _inputs(root) if not only or any(part in path.name for part in only)]

    def one(path: Path) -> tuple[Path, Result | Exception]:
        model = OpenAICompatibleModel(settings)
        try:
            return path, process(path, model)
        except (DocumentError, LLMError, ContractError) as exc:
            return path, exc
        finally:
            model.close()

    failed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, result in pool.map(one, paths):
            if isinstance(result, Exception):
                failed += 1
                print(f"FAILED  {path.name}: {type(result).__name__}: {result}", file=sys.stderr)
                continue
            _write(result, root, path.stem)
            print(
                f"ok      {path.name}: needs_review={str(result.output.needs_review).lower()}"
                + (f" ({'; '.join(result.review_reasons)})" if result.review_reasons else ""),
                file=sys.stderr,
            )
    print(
        f"{len(paths) - failed} of {len(paths)} documents processed with {settings.model}",
        file=sys.stderr,
    )
    return 1 if failed else 0


def _recorded(root: Path, path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The run recorded for one input, or why it cannot be replayed."""
    recorded = root / "diagnostics" / f"{path.stem}.json"
    if not recorded.is_file():
        return None, "no recorded run"
    run: dict[str, Any] = json.loads(recorded.read_text(encoding="utf-8"))["run"]
    if run.get("prompt_fingerprint") != fingerprint():
        return None, f"recorded with another prompt than {PROMPT_LABEL}; run it live again"
    return run, None


def _replayed(path: Path, run: dict[str, Any]) -> Result:
    """The result the current code gives for the replies a live run recorded."""
    model = ScriptedModel(
        (Reply.from_dict(reply) for reply in run["replies"]),
        description={**run.get("model", {}), "rebuilt_from_recorded_replies": True},
    )
    return process(path, model)


def replay(root: Path) -> int:
    """Rebuilds every committed output from its recorded replies with the current code. Any
    difference means an output was edited by hand, or the code changed after it was recorded."""
    problems: list[str] = []
    checked = 0
    for path in _inputs(root):
        committed = root / "outputs" / f"{path.stem}.json"
        run, problem = _recorded(root, path)
        if run is None or not committed.is_file():
            problems.append(f"{path.name}: {problem or 'no committed output'}")
            continue
        try:
            rebuilt = _replayed(path, run).output.to_dict()
        except (DocumentError, LLMError, ContractError) as exc:
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        checked += 1
        if _dump(rebuilt) != committed.read_text(encoding="utf-8"):
            problems.append(f"{path.name}: the committed output differs from the rebuilt one")
        problems += [f"{path.name}: {problem}" for problem in contract_problems(rebuilt)]
    for problem in problems:
        print(f"MISMATCH  {problem}", file=sys.stderr)
    print(
        f"{checked} outputs under {root} rebuilt from recorded replies; {len(problems)} problems",
        file=sys.stderr,
    )
    return 1 if problems else 0


def rebuild(root: Path) -> int:
    """Rewrites every output and its diagnostics from the recorded replies with the current
    code: what to run after changing a check, when no new model call is wanted or needed."""
    failed = 0
    for path in _inputs(root):
        run, problem = _recorded(root, path)
        if run is None:
            failed += 1
            print(f"SKIPPED  {path.name}: {problem}", file=sys.stderr)
            continue
        try:
            _write(_replayed(path, run), root, path.stem)
        except (DocumentError, LLMError, ContractError) as exc:
            failed += 1
            print(f"FAILED  {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(
        f"outputs under {root} rebuilt from recorded replies; {failed} not rebuilt", file=sys.stderr
    )
    return 1 if failed else 0


def build_report() -> tuple[str, dict[str, Any]]:
    """The report and the same numbers as data, from what is committed under each set."""
    summaries = {
        name: (Path(name), summarise(score_set(Path(name))))
        for name in SETS
        if (Path(name) / "outputs").is_dir()
    }
    injections = [item for name in summaries for item in inject(Path(name))]
    stress = stress_summary(injections) if injections else None
    if stress is not None:
        # The same mistakes against quote matching alone: what the label checks are worth.
        bare = stress_summary(
            [item for name in summaries for item in inject(Path(name), label_checks=False)]
        )
        stress["without_label_checks"] = {
            "published_wrong": bare["published_wrong"],
            "by_kind": {
                kind: row.get("published_wrong", 0) for kind, row in bare["by_kind"].items()
            },
        }
    results: dict[str, Any] = {name: summary for name, (_, summary) in summaries.items()}
    if stress is not None:
        results["stress"] = stress
    return render(summaries, stress), results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m changeorder.evaluation", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "replay", "rebuild"):
        command = commands.add_parser(name)
        command.add_argument("set", choices=SETS)
        if name == "run":
            command.add_argument(
                "only", nargs="*", help="run only the inputs whose name contains one of these"
            )
    commands.add_parser("report")
    args = parser.parse_args(argv)

    if args.command == "run":
        status = run_live(Path(args.set), args.only)
    elif args.command == "replay":
        status = replay(Path(args.set))
    elif args.command == "rebuild":
        status = rebuild(Path(args.set))
    else:
        status = 0
    report, results = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "replay":
        if not REPORT.is_file() or REPORT.read_text(encoding="utf-8") != report:
            print(f"MISMATCH  {REPORT} is not what the committed outputs produce", file=sys.stderr)
            status = 1
    else:
        REPORT.write_text(report, encoding="utf-8", newline="\n")
        RESULTS.write_text(_dump(results), encoding="utf-8", newline="\n")
        print(f"report written to {REPORT}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
