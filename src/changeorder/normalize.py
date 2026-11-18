"""Reading values out of the document's own characters.

Money, day counts and dates are found as tokens in the text with their positions, and read by
fixed rules that refuse what they cannot settle (a date such as 03/04/2026, a number such as
1.500 in a document that writes 4,120.50). The lexicons at the end say which way a change goes,
whether a figure's label makes it the change or a contract total, whether a date belongs to some
other event, and what a passage shows about approval.

Pure functions on strings. Nothing here calls a model, and nothing here guesses: every refusal
carries a reason that ends up in the diagnostics.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from .vendor.number_reading import infer_decimal_mark, read_number

__all__ = [
    "DateToken",
    "Days",
    "Money",
    "amount_cues",
    "amount_label",
    "date_convention",
    "date_label",
    "decimal_mark_of",
    "find_dates",
    "find_days",
    "find_money",
    "header_above",
    "is_granted",
    "is_requested",
    "is_schedule_label",
    "label_before",
    "label_says_otherwise",
    "line_of",
    "nearest_label",
    "no_change_cue",
    "says_price",
    "schedule_cues",
    "status_cues",
    "written_like_money",
]

# ---- Lines and labels -----------------------------------------------------------------------


def line_of(text: str, position: int) -> tuple[int, int]:
    """The start and end of the line that holds `position`."""
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return start, len(text) if end == -1 else end


# A figure standing on its own: not the digits of a code such as "COR-031" or "RFI-112".
_FIGURE = re.compile(r"(?<![\w.,/-])\d[\d,.:/]*(?![\w-])")


def label_before(text: str, start: int) -> str:
    """The words that label the value starting at `start`.

    Nearest first: the text between the previous figure on the line and the value, so that on a
    line of several pairs ("Contract Date: 5 June 2025   Change Order Date: 18 March 2026") each
    value gets its own label. If that holds no words, everything before the value on its line
    ("Labor - 28 hrs @ 92.00   2,576.00"). If the line holds none either, the line above: a form
    that puts the label over the value.
    """

    def has_words(piece: str) -> bool:
        return sum(char.isalpha() for char in piece) >= 3

    heading = header_above(text, start)
    if heading is not None:  # a table row: the label is the column's heading, not a neighbour
        return heading
    line_start, _ = line_of(text, start)
    before = text[line_start:start]
    figures = list(_FIGURE.finditer(before))
    nearest = before[figures[-1].end() :] if figures else before
    if has_words(nearest):
        return nearest
    if has_words(before):
        return before
    above = text[:line_start].rstrip("\n")
    return f"{above[above.rfind(chr(10)) + 1 :]} {before}"


# ---- Money ----------------------------------------------------------------------------------

_NUMBER = r"""(?:
      \d{1,3}(?:,\d{3})+(?:\.\d+)?                      # 1,234,567.89
    | \d{1,3}(?:\.\d{3})+(?:,\d+)?                      # 1.234.567,89
    | \d{1,3}(?:[\u00a0\u202f]\d{3})+(?:[.,]\d+)?       # 1 234 567,89 with no-break spaces
    | \d{1,3}(?:\ \d{3})+,\d{2}(?!\d)                   # 1 234 567,89 with plain spaces
    | \d+(?:[.,]\d+)?                                   # 1234.56 and 12,50
)"""
_CURRENCY = r"(?:US\$|C\$|A\$|\$|\u20ac|\u00a3|USD|CAD|AUD|EUR|GBP|AED|SAR)"
_MONEY = re.compile(
    rf"""
    (?<![\w.,/-])                                # not the tail of a code, a date or a number
    (?P<open>\(\s*)?
    (?P<minus>(?<![\w)])[-\u2212])?              # a minus sign touching the figure, not a dash
    (?P<currency>{_CURRENCY}\s*)?
    (?P<minus2>[-\u2212])?
    (?P<number>{_NUMBER})
    (?![\w%/]|[.,]\d)                            # not a percentage, a code, a date or a rate's head
    (?P<currency2>\s?(?:USD|CAD|AUD|EUR|GBP|AED|SAR)\b)?
    (?P<close>\s*\))?
    (?P<credit>\s?CR\b)?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Money:
    """One figure as written. `value` is its size without a sign; `negative` says whether the
    document wrote it as negative (parentheses, a minus sign, or CR)."""

    value: Decimal | None
    raw: str
    start: int
    end: int
    negative: bool
    problem: str | None = None  # why the figure could not be read


def find_money(text: str, decimal_mark: str | None = None) -> list[Money]:
    """Every figure in the text that is written like an amount of money."""
    found: list[Money] = []
    for match in _MONEY.finditer(text):
        read = read_number(match["number"], max_places=2, decimal_mark=decimal_mark)
        value: Decimal | None = read.value
        problem: str | None = read.reason
        if value is not None and value.as_tuple().exponent < -2:  # type: ignore[operator]
            # "1.500" where the document writes "4,120.50": by the rule it is one and a half,
            # which no one writes that way. Refused, not read as 1.50 and not as 1500.
            value, problem = None, "more_than_two_decimal_places"
        negative = bool(
            (match["open"] and match["close"])
            or match["minus"]
            or match["minus2"]
            or match["credit"]
        )
        # A lone parenthesis belongs to the sentence, not to the figure.
        start = match.start() if match["open"] and match["close"] else match.start("number")
        end = match.end() if match["open"] and match["close"] else match.end("number")
        if not (match["open"] and match["close"]):
            for group in ("minus", "currency"):
                if match[group]:
                    start = min(start, match.start(group))
            for group in ("currency2", "credit"):
                if match[group]:
                    end = max(end, match.end(group))
        found.append(Money(value, text[start:end], start, end, negative, problem))
    return found


def decimal_mark_of(text: str) -> str | None:
    """The document's decimal mark ("." or ","), from the figures that prove one, or None."""
    return infer_decimal_mark(match["number"] for match in _MONEY.finditer(text))


# ---- Days -----------------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_WORD_NUMBER = (
    rf"(?:(?:{'|'.join(_TENS)})(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
    rf"|{'|'.join(sorted(_UNITS, key=len, reverse=True))})"
)
_DAYS = re.compile(
    rf"""
    (?<![\w.,/])
    (?:(?P<words>{_WORD_NUMBER})\s*)?
    (?:\(\s*)?
    (?P<minus>[-\u2212])?(?P<plus>\+)?
    (?P<number>\d+(?:[.,]\d+)?)?
    (?:\s*\))?
    [\s-]*(?:(?:calendar|working|business|work|consecutive)[\s-]+)?
    (?P<unit>days?|weeks?|months?)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _words_value(words: str) -> int:
    parts = re.split(r"[-\s]+", words.lower())
    return sum(_TENS.get(part, _UNITS.get(part, 0)) for part in parts)


@dataclass(frozen=True)
class Days:
    """A length of time as written. `value` is in days, without a sign."""

    value: int | None
    raw: str
    start: int
    end: int
    negative: bool
    in_weeks: bool = False  # converted from weeks: seven days each
    words_agree: bool = False  # written in words and in figures, and they agree
    problem: str | None = None


def find_days(text: str) -> list[Days]:
    """Every length of time in the text: "fourteen (14) calendar days", "4 working days"."""
    found: list[Days] = []
    for match in _DAYS.finditer(text):
        words, number = match["words"], match["number"]
        if not words and not number:
            continue
        unit = match["unit"].lower()
        from_words = _words_value(words) if words else None
        value: int | None = None
        problem: str | None = None
        if number is not None:
            if not number.isdigit():
                problem = "not_whole_days"
            elif from_words is not None and from_words != int(number):
                problem = "words_and_figures_disagree"
            else:
                value = int(number)
        else:
            value = from_words
        if unit.startswith("month"):
            value, problem = None, "months_have_no_fixed_length"
        in_weeks = unit.startswith("week")
        if value is not None and in_weeks:
            value *= 7
        found.append(
            Days(
                value=value,
                raw=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
                negative=bool(match["minus"]),
                in_weeks=in_weeks,
                words_agree=bool(words and number and problem is None),
                problem=problem,
            )
        )
    return found


# ---- Dates ----------------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = (
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "iso",
        re.compile(
            r"(?<![\d/.-])(?P<y>\d{4})(?P<s>[-/.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})(?![\d/-])"
        ),
    ),
    (
        "day_month_name",
        re.compile(
            rf"(?<![\d/.-])(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[\s./-]*{_MONTH}\.?[\s.,/-]*"
            r"(?P<y>\d{4}|\d{2})(?![\d/-])",
            re.IGNORECASE,
        ),
    ),
    (
        "month_name_day",
        re.compile(
            rf"\b{_MONTH}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})(?![\d/-])",
            re.IGNORECASE,
        ),
    ),
    (
        "numeric",
        re.compile(
            r"(?<![\d/.-])(?P<a>\d{1,2})(?P<s>[/.-])(?P<b>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})(?![\d/-])"
        ),
    ),
)


def _year(text: str) -> int:
    year = int(text)
    if len(text) == 4:
        return year
    return 2000 + year if year < 80 else 1900 + year


@dataclass(frozen=True)
class DateToken:
    """A date as written. A numeric date keeps its two leading numbers, because which is the day
    may not be decidable from the token alone."""

    raw: str
    start: int
    end: int
    kind: str  # iso, month_name, numeric
    year: int
    first: int  # the day for iso and month_name; the first number for numeric
    second: int  # the month for iso and month_name; the second number for numeric
    two_digit_year: bool = False

    @property
    def proves(self) -> str | None:
        """The order a numeric date proves: "dmy" when its first number cannot be a month, "mdy"
        when its second cannot."""
        if self.kind != "numeric":
            return None
        if self.first > 12 >= self.second:
            return "dmy"
        if self.second > 12 >= self.first:
            return "mdy"
        return None

    def resolve(self, convention: str | None) -> tuple[datetime.date | None, str | None, bool]:
        """The date, or why not, and whether the document's convention had to decide it."""
        used_convention = False
        if self.kind != "numeric":
            day, month = self.first, self.second
        elif self.first == self.second:
            day = month = self.first
        else:
            order = self.proves
            if order is None:
                if convention is None:
                    return None, "day_and_month_ambiguous", False
                order, used_convention = convention, True
            day, month = (self.first, self.second) if order == "dmy" else (self.second, self.first)
        try:
            return datetime.date(self.year, month, day), None, used_convention
        except ValueError:
            return None, "not_a_calendar_date", False


def find_dates(text: str) -> list[DateToken]:
    """Every date in the text that names a day, a month and a year."""
    found: list[DateToken] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append(match.span())
            year_text = match["y"]
            if kind == "numeric":
                first, second, name = int(match["a"]), int(match["b"]), "numeric"
            else:
                month = int(match["m"]) if kind == "iso" else _MONTHS[match["month"][:3].lower()]
                first, second = int(match["d"]), month
                name = "iso" if kind == "iso" else "month_name"
            found.append(
                DateToken(
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    kind=name,
                    year=_year(year_text),
                    first=first,
                    second=second,
                    two_digit_year=len(year_text) == 2,
                )
            )
    return sorted(found, key=lambda token: token.start)


def date_convention(tokens: Iterable[DateToken]) -> str | None:
    """The day/month order the document's own numeric dates prove: "dmy", "mdy", or None when
    none proves one or they disagree."""
    proofs = {token.proves for token in tokens} - {None}
    return proofs.pop() if len(proofs) == 1 else None


# ---- Which way a change goes -----------------------------------------------------------------

_AMOUNT_UP = re.compile(r"\b(?:add(?:s|ed|ing|ition|itive)?|increas\w*|extra)\b", re.IGNORECASE)
_AMOUNT_DOWN = re.compile(
    r"\b(?:deduct\w*|credit\w*|decreas\w*|reduc\w*|savings?|rebate|refund)\b", re.IGNORECASE
)
_TIME_UP = re.compile(
    r"\b(?:extend\w*|extension|increas\w*|add(?:s|ed|ing|ition|itional)?|delay\w*|later)\b",
    re.IGNORECASE,
)
_TIME_DOWN = re.compile(
    r"\b(?:reduc\w*|decreas\w*|shorten\w*|accelerat\w*|advanc\w*|earlier|sav(?:e|es|ed|ings?))\b",
    re.IGNORECASE,
)
_NO_CHANGE = re.compile(
    r"\b(?:no\s+(?:net\s+)?(?:change|changes|adjustment|increase|decrease|extension|impact"
    r"|effect|cost|charge|time|additional\s+\w+)"
    r"|unchanged|not\s+(?:be\s+)?(?:changed|affected|adjusted|extended|impacted|increased)"
    r"|remains?\s+(?:the\s+same|unchanged)|zero|nil|none)\b",
    re.IGNORECASE,
)
_NOT_APPLICABLE = re.compile(r"\bn\s?/\s?a\b|\bnot\s+applicable\b", re.IGNORECASE)


def _cues(text: str, up: re.Pattern[str], down: re.Pattern[str]) -> set[str]:
    found = set()
    if up.search(text):
        found.add("increase")
    if down.search(text):
        found.add("decrease")
    return found


def amount_cues(text: str) -> set[str]:
    """The directions the wording gives a price change: "increase", "decrease", both or neither.
    Both is what a blank form reads like ("will be (increased) (decreased) by")."""
    return _cues(text, _AMOUNT_UP, _AMOUNT_DOWN)


def schedule_cues(text: str) -> set[str]:
    """The directions the wording gives a change in contract time."""
    return _cues(text, _TIME_UP, _TIME_DOWN)


def no_change_cue(text: str) -> str | None:
    """ "explicit" when the text says nothing changes, "not_applicable" for N/A, else None."""
    if _NO_CHANGE.search(text):
        return "explicit"
    return "not_applicable" if _NOT_APPLICABLE.search(text) else None


# ---- What the words beside a value make it -----------------------------------------------------
#
# A quote that holds a value proves the value is in the document. It does not prove the value
# is what the field asks for: the revised contract total is in the document too, and so are the
# signature dates and the requester's name. The words beside a value say what it is. These
# lexicons read them, so that a value sitting under the wrong label is refused.

_COLUMN_GAP = re.compile(r"\s{2,}")


def header_above(text: str, start: int) -> str | None:
    """The column heading over the value at `start`, when its line is a row under a header row.

    A header row is the line above when it has the same number of columns as this one (three
    or more) and holds no figures and no colons, as a row of headings does and a row of
    "Label: value" pairs does not. None for any other line.
    """
    line_start, line_end = line_of(text, start)
    if line_start == 0:
        return None
    header = text[slice(*line_of(text, line_start - 1))].strip()
    if not header or ":" in header or any(char.isdigit() for char in header):
        return None
    headings = _COLUMN_GAP.split(header)
    row = text[line_start:line_end]
    if len(headings) < 3 or len(headings) != len(_COLUMN_GAP.split(row.strip())):
        return None
    column = len(_COLUMN_GAP.findall(row[: start - line_start].lstrip()))
    return headings[column] if column < len(headings) else None


def nearest_label(text: str, start: int) -> str:
    """The label right beside the value at `start`: its column heading when the line is a table
    row; otherwise the last stretch of words before it on its line, after the previous figure
    and after the last column gap. Empty when the value opens its line or follows a figure."""
    heading = header_above(text, start)
    if heading is not None:
        return heading
    line_start, _ = line_of(text, start)
    before = text[line_start:start]
    figures = list(_FIGURE.finditer(before))
    if figures:
        before = before[figures[-1].end() :]
    pieces = [piece for piece in _COLUMN_GAP.split(before) if piece.strip()]
    return pieces[-1].strip() if pieces else ""


_CONTRACT = r"(?:sub)?(?:contract|agreement)\s+(?:sum|amount|price|value|total)"
_TOTAL_LABEL = re.compile(
    rf"\b(?:original|revised|new|current|adjusted|previous|prior|updated|final|total)\s+{_CONTRACT}"
    rf"|\b{_CONTRACT}\s+(?:prior\s+to|before|after|including|to\s+date|as\s+of|was\b|remains\b"
    r"|will\s+be(?!\s+(?:increased|decreased|changed|reduced|adjusted|unchanged)))"
    r"|\bnet\s+change\s+by\s+(?:previous|previously|prior)"
    r"|\bprevious(?:ly)?\s+(?:authori[sz]ed\s+|approved\s+)?change\s+orders",
    re.IGNORECASE,
)
# A ceiling or a guess, not a price: "not to exceed $22,000.00", "budget estimate".
_LIMIT_LABEL = re.compile(
    r"\bnot\s+to\s+exceed\b|\bn\.?t\.?e\.?\b|\bceiling\b|\ballowance\b|\bbudget\w*|\bestimat\w*"
    r"|\bapprox\w*|\bup\s+to\b|\bmaximum\b",
    re.IGNORECASE,
)
_CHANGE_LABEL = re.compile(
    r"\b(?:increased|decreased|changed|reduced|adjusted)\s+by\b"
    rf"|\b(?:increas|decreas|reduc)\w*\s+(?:the\s+)?{_CONTRACT}\s+by\b"
    rf"|\b(?:change|increase|decrease|addition|add|adjustment|credit|deduct\w*)\s+(?:in|to|of|from)"
    rf"\s+(?:the\s+)?{_CONTRACT}"
    rf"|\b{_CONTRACT}\s+(?:increase|decrease|change|adjustment|addition|credit)\b"
    r"|\bamount\s+of\s+(?:this\s+|the\s+)?change"
    r"|\bchange\s+order\s+(?:amount|total|sum|value|price)"
    r"|\b(?:net|total)\s+(?:this\s+)?(?:change|add|deduct|credit|increase|decrease|cost|price)\b"
    r"|\btotal\s+(?:proposed|requested|approved|this)\b"
    r"|\b(?:amount|total)\s+(?:approved|requested|proposed)\b"
    r"|\b(?:approved|requested|proposed)\s+(?:amount|total)\b"
    # "ADD:", "Net change - DEDUCT": the word right beside the figure, not a verb in a sentence.
    r"|\b(?:add|deduct|credit)\b[\s:)\-]*$",
    re.IGNORECASE,
)
_PART_LABEL = re.compile(
    r"\b(?:sub-?total|sales\s+tax|tax|vat|overhead|profit|oh\s*&\s*p|mark-?up|fee|bond|insurance"
    r"|unit\s+price|labou?r|materials?|equipment|freight|contingency)\b",
    re.IGNORECASE,
)
# A plain "TOTAL" at the head of its label, when nothing more specific was said.
_PLAIN_TOTAL = re.compile(r"^\W*total\b", re.IGNORECASE)
# A figure followed by a unit is a quantity; one after "@" or before "/unit" is a rate.
_MEASURE_AFTER = re.compile(
    r"^\s?(?:sq\.?\s?ft|sf|lf|cy|sy|tons?|feet|foot|ft|hrs?|hours?|ea|each|m2|m3|lbs?|kg"
    r"|gal(?:lons?)?|inch(?:es)?|%)\b|^\s?/\s?[A-Za-z]",
    re.IGNORECASE,
)
_RATE_BEFORE = re.compile(r"(?:@|\bat|\bx)\s*$", re.IGNORECASE)
_REQUESTED = re.compile(r"\b(?:request|propos|claim|submit)\w*", re.IGNORECASE)
_GRANTED = re.compile(r"\b(?:approv|grant|authori[sz]|accept|agree)\w*", re.IGNORECASE)


def amount_label(before: str, after: str = "") -> str | None:
    """What the words around a figure make it.

    "total": a contract sum, or the sum of earlier change orders. "limit": a ceiling or an
    estimate. "change": the change itself. "part": a line, a subtotal, a markup. "measure": a
    quantity or a unit rate. None when the words do not say.
    """
    if _MEASURE_AFTER.match(after) or _RATE_BEFORE.search(before):
        return "measure"
    if _TOTAL_LABEL.search(before):
        return "total"
    if _LIMIT_LABEL.search(before):
        return "limit"
    if _CHANGE_LABEL.search(before) or _CHANGE_LABEL.search(after):
        return "change"
    if _PART_LABEL.search(before):
        return "part"
    return "change" if _PLAIN_TOTAL.search(before) else None


_HAS_CURRENCY = re.compile(_CURRENCY)
_MONEY_SHAPED = re.compile(rf"{_CURRENCY}|[.,]\d")
_PRICE_WORDS = re.compile(
    r"\b(?:amount|price|cost|sum|total|value|charge|credit|deduct\w*|add|increase|decrease)\b",
    re.IGNORECASE,
)


def written_like_money(raw: str) -> bool:
    """Whether a figure is written as an amount: with a currency sign or code, or with a decimal
    or thousands mark. A bare "6" is a count, a number or an identifier."""
    return bool(_MONEY_SHAPED.search(raw))


def says_price(raw: str, label: str) -> bool:
    """Whether anything marks a figure as a price at all: a currency on the figure, or a word
    for money in its label. "Below elevation 412.0" has neither."""
    return bool(_HAS_CURRENCY.search(raw) or _PRICE_WORDS.search(label))


def is_requested(label: str) -> bool:
    """Whether a label marks what was asked for, as opposed to what was granted."""
    return bool(_REQUESTED.search(label)) and not _GRANTED.search(label)


def is_granted(label: str) -> bool:
    return bool(_GRANTED.search(label))


_SCHEDULE_LABEL = re.compile(
    r"\b(?:time|schedule|completion|duration|extension|extend\w*|delay\w*|accelerat\w*"
    r"|days?\s+(?:added|deducted|granted))\b",
    re.IGNORECASE,
)


def is_schedule_label(label: str) -> bool:
    """Whether the words before a duration are about the contract time. "Valid for 15 days" and
    "Crane - 1 day" are durations too, and neither is the schedule impact."""
    return bool(_SCHEDULE_LABEL.search(label))


_OTHER_EVENT = re.compile(
    r"\b(?:contract\s+date|date\s+of\s+(?:the\s+)?(?:original\s+)?(?:contract|agreement)"
    r"|(?:contract|agreement)\s+dated|notice\s+to\s+proceed"
    r"|substantial\s+(?:completion|performance)|completion|complete[ds]?\s+by"
    r"|signed|signature|received|due|valid\s+(?:until|through)|expir\w+|effective"
    r"|approved|accepted|executed|authori[sz]ed|rejected|void(?:ed)?|answered|action|revised"
    r"|(?:memo|letter|email|drawing|sketch|report|proposal|quote|quotation|survey|discharge"
    r"|invoice|meeting|walk)\s+(?:dated|of|on))\b"
    r"|\bby\s*:",
    re.IGNORECASE,
)
_OWN_DATE = re.compile(
    r"\b(?:date|dated|issued|issue|prepared|sent)\W*"
    r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day\W*)?$",
    re.IGNORECASE,
)
_SIGNATURE_CONTEXT = re.compile(r"\bby\s*:|/s/|\bsigned\b|\bsignature\b", re.IGNORECASE)


def date_label(text: str, start: int) -> str:
    """What the words beside the date at `start` make it.

    "own": labelled as the document's date ("Date:", "Issued:", "Date Prepared:"), or standing
    alone at the head of its line as a letter's date does. "other": the date of something else
    (the contract, a signature, a completion, an approval, a referenced letter), or a date in
    a signature block. "unlabelled": in running text with no label that says either.
    """
    line_start, _ = line_of(text, start)
    before = text[line_start:start]
    above = text[:line_start].rstrip("\n")
    previous = above[above.rfind("\n") + 1 :]
    label = nearest_label(text, start)
    if _OTHER_EVENT.search(label) or _SIGNATURE_CONTEXT.search(before):
        return "other"
    if _OWN_DATE.search(label):
        # "Date:" under a line of "By:" entries is the date of a signature.
        return "other" if _SIGNATURE_CONTEXT.search(previous) else "own"
    if not any(char.isalpha() for char in label):
        return "own" if not before.strip() else "unlabelled"
    return "unlabelled"


# The right kind of text under the wrong label: for each field, the labels that say the value
# is something else, and the labels that say it is this field after all. The one nearer the
# value wins: "AMENDMENT TO CONTRACT (CHANGE ORDER NO. 2" labels a change order number.
_LABELS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "project_name": (
        re.compile(
            r"\b(?:requested|initiated|instructed|directed|prepared|submitted|approved|accepted"
            r"|signed|recommended)\s+by\b"
            r"|\b(?:owner|contractor|subcontractor|architect|engineer|client|employer|from|to"
            r"|attn|cc)\s*:"
            r"|\b(?:contract|agreement|change\s+order|job|project)\s*(?:no\.?|number|#|ref)",
            re.IGNORECASE,
        ),
        re.compile(r"\bproject(?:\s+name)?\s*:|\bjob\s*(?:name)?\s*:", re.IGNORECASE),
    ),
    "requested_by": (
        re.compile(
            r"\b(?:project|job|contract|agreement|subject)\s*(?:name|no\.?|number|#)?\s*:"
            r"|\b(?:approved|accepted|signed|prepared|recommended)\s+by\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:requested|initiated|instructed|directed)\s+by\b", re.IGNORECASE),
    ),
    "contract_number": (
        re.compile(
            r"\bchange\s+order\b|\bC\.?O\.?R?\s*(?:no\.?|number|#)"
            r"|\b(?:job|project|rfi|asi|invoice|bulletin|p\.?o\.?|purchase\s+order|permit)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:sub)?contract\b|\bagreement\b", re.IGNORECASE),
    ),
    "change_order_number": (
        re.compile(
            r"\b(?:sub)?contract\b|\bagreement\b"
            r"|\b(?:job|project|rfi|asi|invoice|bulletin|p\.?o\.?|purchase\s+order|permit"
            r"|drawing|sketch)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bchange\s+order\b|\bvariation\b|\bmodification\b|\bamendment\s+no"
            r"|\b(?:C\.?O\.?R?|[FSP]CO)\b",
            re.IGNORECASE,
        ),
    ),
}


def label_says_otherwise(field: str, label: str) -> bool:
    """Whether the label beside a value says it belongs to a different field."""
    if field not in _LABELS:
        return False
    wrong, right = _LABELS[field]
    against = [match.end() for match in wrong.finditer(label)]
    in_favour = [match.end() for match in right.finditer(label)]
    return bool(against) and (not in_favour or max(against) > max(in_favour))


# ---- What a passage shows about approval ------------------------------------------------------

_STATUS_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rejected",
        re.compile(
            r"\b(?:rejected|denied|declined|disapproved|refused)\b|\bnot\s+approved\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pending",
        re.compile(
            r"\b(?:pending|awaiting|under\s+review|unsigned|submitted\s+for"
            r"|not\s+yet\s+(?:been\s+)?(?:approved|signed|executed|accepted)"
            r"|for\s+(?:your\s+|owner'?s?\s+)?(?:review|approval|signature))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "draft",
        re.compile(
            r"\bdraft\b|\bpreliminary\b|\bnot\s+for\s+(?:execution|construction)\b", re.IGNORECASE
        ),
    ),
    (
        "void",
        re.compile(r"\b(?:void(?:ed)?|cancell?ed|withdrawn|rescinded|superseded)\b", re.IGNORECASE),
    ),
    (
        "approved",
        re.compile(r"\b(?:approved|accepted|executed|authori[sz]ed|ratified)\b", re.IGNORECASE),
    ),
)
# A signature mark, or the word "signed" used as a fact: "K. Alvarez (signed)", "Signed for the
# Employer: R. Haddad". "Unsigned" is another word, and "sign here" is not "signed".
_SIGNATURE = re.compile(r"(?P<mark>/s/\s*\S)|\bsigned\b", re.IGNORECASE)
_EMPTY_BOX = re.compile(r"(?:\[\s*\]|\(\s*\)|\u2610)\s*$")
# A negation earlier in the same clause: "has not been approved", "No change to the Contract is
# authorized", "Not issued for signature".
_NEGATION = re.compile(r"\b(?:not|never|no|without|neither|nor)\b[^.;:\n]{0,45}$", re.IGNORECASE)
# Words that make what follows a condition or an instruction, not a statement of fact.
_CONDITION = re.compile(
    r"\b(?:until|unless|upon|once|when|if|after|before|subject\s+to|requires?|must|shall|should"
    r"|will|to\s+be|prior\s+to|valid)\b[^.;:\n]{0,60}$",
    re.IGNORECASE,
)
# "Approved by (Owner): ______" or "Approved by:" with nothing after: a label, not a signature.
_BLANK_LABEL = re.compile(
    r"^\s*by\b[^\w\n]*(?:\([^)\n]*\))?[\s:]*(?:[_.\-]{3,}.*)?$", re.IGNORECASE
)
# The same with a party named: "Approved by Utility Director: ______", "Accepted for the
# Conservancy: ______". What follows the label's colon is a blank, not a name.
_BLANK_AFTER = re.compile(r"^[^:\n]{0,50}:?\s*[_.]{3,}")


def status_cues(text: str) -> dict[str, str]:
    """The approval states a passage gives evidence of, each "explicit" (a statement, a stamp, a
    ticked box) or "signature" (a completed signature and nothing more).

    A word does not count when it is an unticked option, is negated ("not approved" counts as
    rejected instead), sits in a condition ("not valid until signed", "must be approved by"), or
    labels a blank ("Approved by: ______").
    """
    found: dict[str, str] = {}
    for line in text.splitlines():
        for status, pattern in _STATUS_WORDS:
            for match in pattern.finditer(line):
                before, after = line[: match.start()], line[match.end() :]
                if _EMPTY_BOX.search(before) or _BLANK_AFTER.match(after):
                    continue
                if status == "approved" and (
                    _NEGATION.search(before)
                    or _CONDITION.search(before)
                    or _BLANK_LABEL.match(after)
                ):
                    continue
                if status in ("pending", "draft", "void") and _NEGATION.search(before):
                    continue
                found[status] = "explicit"
        for match in _SIGNATURE.finditer(line):
            before, after = line[: match.start()], line[match.end() :]
            if match["mark"] or not (
                _NEGATION.search(before)
                or _CONDITION.search(before)
                or _BLANK_LABEL.match(after)
                or _BLANK_AFTER.match(after)
            ):
                found.setdefault("approved", "signature")
    return found
