"""Finding what the model quoted in the document, and where.

A quote is evidence only if code can find it. `find_quote` looks for it in the canonical text in
three tiers and reports the tier: `exact` (character for character), `normalized` (after folding
case, typographic quotes and dashes, ligatures and white space, so a quote that joins two lines
with a space still matches) and `fuzzy` (nearly all of its words, in order, close together).
The tiers and the map back to original positions come from the Toolkit's quote-matching core.

The fuzzy tier is deliberately strict here (85% of the quote's words, filling 75% of the
stretch): it exists to forgive a dropped word or a line-wrap, not to accept a paraphrase. A
value that rests on a fuzzy quote is marked, and scores lower.

Finding a quote proves the text exists. It does not prove the text supports the value: that is
the job of `fields.py`, which locates the value inside the quoted lines and reads it from there.
"""

from __future__ import annotations

import re

from .document import SourceDocument
from .models import Evidence
from .vendor.quote_matching import LATIN, MARKUP, FuzzyPolicy, PatternRule, Rule, align

__all__ = ["collapse", "find_quote", "locate_passage", "locate_text", "windows_of"]

# A word broken at a line end ("tenant-\nimprovement") reads as one hyphenated word.
RULES: tuple[Rule, ...] = (
    *LATIN,
    PatternRule("line_end_hyphen", re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)"), "-", stage=MARKUP),
)
QUOTE_FUZZY = FuzzyPolicy(coverage=0.85, density=0.75)
# A value whose words sit on one line with another column's text between them.
SPLIT_VALUE = FuzzyPolicy(coverage=1.0, density=0.5)
PASSAGE_FUZZY = FuzzyPolicy(coverage=0.9, density=0.8)
# The fuzzy tier's cost grows with the quote's length, and the quote comes from a model.
MAX_FUZZY_CHARS = 600
# Fewer characters than this match almost anywhere, so they show nothing.
MIN_QUOTE_CHARS = 4

_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+|\n+")
_SPACE = re.compile(r"\s+")


def collapse(text: str) -> str:
    """The text on one line, with single spaces."""
    return _SPACE.sub(" ", text).strip()


def find_quote(document: SourceDocument, quote: str) -> Evidence:
    """Where the quote is in the document, and how it was found."""
    wanted = quote.strip()
    if sum(not char.isspace() for char in wanted) < MIN_QUOTE_CHARS:
        return Evidence(quote, "too_short", 0.0)
    fuzzy = QUOTE_FUZZY if len(wanted) <= MAX_FUZZY_CHARS else None
    found = align(document.text, wanted, fuzzy=fuzzy, rules=RULES)
    if found.span is None:
        return Evidence(quote, "not_found", 0.0)
    start, end = found.span
    page, line = document.locate(start)
    return Evidence(quote, found.kind, round(found.score, 3), start, end, page, line)


def windows_of(document: SourceDocument, evidence: list[Evidence]) -> list[tuple[int, int]]:
    """The whole lines that the found quotes touch. A value is looked for inside these: the
    model pointed at the line, and code reads the line."""
    text = document.text
    windows: list[tuple[int, int]] = []
    for item in evidence:
        if item.start is None or item.end is None:
            continue
        start = text.rfind("\n", 0, item.start) + 1
        end = text.find("\n", max(item.end - 1, item.start))
        windows.append((start, len(text) if end == -1 else end))
    return sorted(set(windows))


def locate_text(
    document: SourceDocument, value: str, windows: list[tuple[int, int]]
) -> tuple[tuple[int, int], bool] | None:
    """Where `value` is inside the quoted lines, and whether it stands there in one piece.

    One piece: the value is the document's own characters, and the caller slices them out. Not
    one piece: every word of it is there in order with other text between (a name that a second
    column interrupts), which is weaker and is marked. None: the quoted lines do not hold it.
    """
    wanted = value.strip()
    if not wanted or not windows:
        return None
    found = align(document.text, wanted, windows=windows, rules=RULES)
    if found.kind in ("exact", "normalized") and found.span is not None:
        return found.span, True
    if len(wanted) <= MAX_FUZZY_CHARS:
        found = align(document.text, wanted, windows=windows, fuzzy=SPLIT_VALUE, rules=RULES)
        if found.kind == "fuzzy" and found.span is not None:
            return found.span, False
    return None


def locate_passage(document: SourceDocument, passage: str) -> list[Evidence] | None:
    """A passage the model copied, found in the document piece by piece.

    The passage is found whole when it can be; otherwise sentence by sentence, in order, each
    one exactly, after folding, or nearly word for word. The caller rebuilds the passage from
    the spans, so what is published is the document's wording, never the model's. None when any
    sentence is not in the document: a description is supported entirely or not at all.
    """
    text = document.text
    wanted = passage.strip()
    if not wanted:
        return None

    def evidence(piece: str, kind: str, score: float, span: tuple[int, int]) -> Evidence:
        page, line = document.locate(span[0])
        return Evidence(piece, kind, round(score, 3), span[0], span[1], page, line)

    whole = align(text, wanted, rules=RULES)
    if whole.span is not None:
        return [evidence(wanted, whole.kind, whole.score, whole.span)]

    pieces: list[str] = []
    for piece in (part.strip() for part in _SENTENCE_END.split(wanted)):
        if not piece:
            continue
        # A fragment of a few characters matches anywhere; it rides with its neighbour.
        if pieces and len(pieces[-1]) < 20:
            pieces[-1] = f"{pieces[-1]} {piece}"
        else:
            pieces.append(piece)
    found: list[Evidence] = []
    cursor = 0
    for piece in pieces:
        fuzzy = PASSAGE_FUZZY if len(piece) <= MAX_FUZZY_CHARS else None
        match = align(text, piece, after=cursor, fuzzy=fuzzy, rules=RULES)
        if match.span is None:
            return None
        found.append(evidence(piece, match.kind, match.score, match.span))
        cursor = match.span[1]
    return found
