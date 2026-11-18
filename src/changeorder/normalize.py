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

import bisect
import datetime
import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from .vendor.number_reading import infer_decimal_mark, read_number

__all__ = [
    "DateToken",
    "Days",
    "Money",
    "after_on_line",
    "amount_cues",
    "amount_label",
    "amounts_are_scaled",
    "before_on_line",
    "date_convention",
    "date_label",
    "decimal_mark_of",
    "duration_label",
    "find_dates",
    "find_days",
    "find_money",
    "has_blank_signing_line",
    "header_above",
    "headings_above",
    "is_granted",
    "is_hedged",
    "is_requested",
    "is_schedule_label",
    "label_before",
    "label_says_otherwise",
    "line_above",
    "line_of",
    "lines_above",
    "nearest_label",
    "no_change_cue",
    "reserves_rights",
    "says_price",
    "schedule_cues",
    "speaks_of_price",
    "status_cues",
    "sum_role",
    "ticked_statuses",
    "title_cues",
    "written_like_money",
]

# ---- Lines and labels -----------------------------------------------------------------------

# How far along its line a label is looked for, and the longest line read as a table row. A
# label sits beside its value. A line of many thousands of characters is a paragraph or an
# attack, and must not cost its whole length for every value on it.
_REACH = 240
_LONGEST_ROW = 1000


@functools.lru_cache(maxsize=8)
def _line_breaks(text: str) -> list[int]:
    """Where the text's line breaks are. Found once per text: a line can be as long as the
    whole document, and looking for its ends again for every value on it adds up."""
    breaks: list[int] = []
    position = text.find("\n")
    while position != -1:
        breaks.append(position)
        position = text.find("\n", position + 1)
    return breaks


def line_of(text: str, position: int) -> tuple[int, int]:
    """The start and end of the line that holds `position`."""
    breaks = _line_breaks(text)
    index = bisect.bisect_left(breaks, position)
    start = breaks[index - 1] + 1 if index else 0
    return start, breaks[index] if index < len(breaks) else len(text)


def before_on_line(text: str, start: int) -> str:
    """The text before `start` on its line, or the last `_REACH` characters of it."""
    return text[max(line_of(text, start)[0], start - _REACH) : start]


def after_on_line(text: str, end: int) -> str:
    """The text after `end` on its line, or the first `_REACH` characters of it."""
    return text[end : min(line_of(text, end)[1], end + _REACH)]


def line_above(text: str, start: int) -> str:
    """The nearest line above the one that holds `start` that is not empty."""
    end = line_of(text, start)[0]
    while end > 0 and text[end - 1] == "\n":
        end -= 1
    return text[max(text.rfind("\n", 0, end) + 1, end - _REACH) : end]


def lines_above(text: str, start: int, most: int) -> str:
    """The lines directly above the one that holds `start`: at most `most` of them, and none
    beyond an empty line. They are the rest of the block the line belongs to."""
    end = line_of(text, start)[0] - 1  # the line break that ends the line above
    found: list[str] = []
    while end > 0 and len(found) < most:
        begin = text.rfind("\n", 0, end) + 1
        if begin >= end:
            break
        found.append(text[max(begin, end - _REACH) : end])
        end = begin - 1
    return "\n".join(reversed(found))


_THREE_LETTERS = re.compile(r"[^\W\d_][^\n]*?[^\W\d_][^\n]*?[^\W\d_]")
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
        return _THREE_LETTERS.search(piece) is not None

    heading = header_above(text, start)
    if heading is not None:  # a table row: the label is the column's heading, not a neighbour
        return heading
    before = before_on_line(text, start)
    figures = list(_FIGURE.finditer(before))
    nearest = before[figures[-1].end() :] if figures else before
    if has_words(nearest):
        return nearest
    if has_words(before):
        return before
    return f"{line_above(text, start)} {before}"


# ---- Money ----------------------------------------------------------------------------------
#
# Every quantifier below is bounded. These patterns run over text someone else wrote, and an
# unbounded one invites a line of twenty thousand hyphens to cost a minute.

_NUMBER = r"""(?:
      \d{1,3}(?:,\d{3}){1,6}(?:\.\d{1,6})?                  # 1,234,567.89
    | \d{1,3}(?:\.\d{3}){1,6}(?:,\d{1,6})?                  # 1.234.567,89
    | \d{1,3}(?:[\u00a0\u202f]\d{3}){1,6}(?:[.,]\d{1,6})?   # 1 234 567,89 with no-break spaces
    | \d{1,3}(?:\ \d{3}){1,6},\d{2}(?!\d)                   # 1 234 567,89 with plain spaces
    | \d{1,15}(?:[.,]\d{1,6})?                              # 1234.56 and 12,50
)"""
_CURRENCY = r"(?:US\$|C\$|A\$|\$|\u20ac|\u00a3|USD|CAD|AUD|EUR|GBP|AED|SAR)"
_MONEY = re.compile(
    rf"""
    (?<![\w.,/-])                                # not the tail of a code, a date or a number
    (?P<open>\(\s?)?
    (?P<minus>(?<![\w)])[-\u2212\u2013])?        # a minus sign touching the figure, not a dash
    (?P<currency>{_CURRENCY}\s{{0,3}})?
    (?P<minus2>[-\u2212])?
    (?P<number>{_NUMBER})
    (?![\w%/]|[.,]\d)                            # not a percentage, a code, a date or a rate's head
    (?P<currency2>\s?(?:USD|CAD|AUD|EUR|GBP|AED|SAR)\b)?
    (?P<close>\s?\))?
    (?P<credit>\s?CR\b)?
    """,
    re.VERBOSE,
)
# "$1.2 million", "$1.2 mil", "350 thou.", "2.75 MM", "40 k": the figure is not the amount. The
# word may follow a gap, or open the next line when the sentence wraps. A single letter counts
# only right beside the figure and as a word of its own: not the M of "M&E", not an initial
# across a column gap or on the next line. "$350 K. Alvarez" is refused too: the price of not
# reading "$350 K." as three hundred and fifty.
_SCALE_WORD = (
    r"(?:millions?|billions?|trillions?|thousands?|hundreds?|mil|mill|mio|thou|lakhs?|crores?"
    r"|mm|bn|mn)"
)
_SCALE_AFTER = re.compile(
    rf"[ \t]{{0,3}}(?:\n[ \t]{{0,3}})?{_SCALE_WORD}\b|[ \t]?[km]\b(?![&/'-])", re.IGNORECASE
)
# A note that scales every figure in the document: "(All amounts in thousands of dollars)",
# "Figures in $000", "USD '000".
_SCALED_DOCUMENT = re.compile(
    r"\b(?:amounts?|figures?|values?|prices?|costs?|dollars|sums?|totals?)\s+(?:are\s+|is\s+)?"
    r"(?:shown\s+|stated\s+|expressed\s+|given\s+|presented\s+|reported\s+)?in\s+"
    r"(?:(?:thousands|millions|hundreds)\b|\$?\s?'?0{3}s?\b(?![.,]\d)|\$?k\b)"
    r"|\(\s?in\s+(?:thousands|millions)\b[^)\n]{0,30}\)"
    r"|\(\s?(?:in\s+)?(?:\$|USD)?\s?'?0{3}s?\s?\)"
    r"|(?:\$|USD)\s?'?0{3}s?\b(?![.,]\d)",
    re.IGNORECASE,
)
# "Twelve Thousand Four Hundred Fifty Dollars ($12,450.00)": the parentheses restate the words.
_IN_WORDS_BEFORE = re.compile(
    r"\b(?:dollars?|euros?|pounds?|cents?|hundred|thousand|million|only|\d{1,2}/100)\W{0,3}$",
    re.IGNORECASE,
)
_POINT_AND_THREE = re.compile(r"\d{1,3}\.\d{3}")


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
        number = match["number"]
        read = read_number(number, max_places=2, decimal_mark=decimal_mark)
        value: Decimal | None = read.value
        problem: str | None = read.reason
        if value is not None and value.as_tuple().exponent < -2:  # type: ignore[operator]
            # "1.500" where the document writes "4,120.50": by the rule it is one and a half,
            # which no one writes that way. Refused, not read as 1.50 and not as 1500.
            value, problem = None, "more_than_two_decimal_places"
        elif value is not None and decimal_mark is None and _POINT_AND_THREE.fullmatch(number):
            # "950.250" with nothing else to go by: 950,250, or 950.25 in a currency of three
            # decimals. A comma before three digits is the usual thousands mark; a point is not.
            value, problem = None, "point_before_three_digits_is_ambiguous"
        elif value is not None and _SCALE_AFTER.match(text, match.end()):
            value, problem = None, "figure_is_scaled_by_a_word"
        enclosed = bool(match["open"] and match["close"])
        restated = enclosed and bool(
            _IN_WORDS_BEFORE.search(text[max(0, match.start() - 40) : match.start()])
        )
        negative = bool(
            (enclosed and not restated) or match["minus"] or match["minus2"] or match["credit"]
        )
        # A lone parenthesis belongs to the sentence, not to the figure.
        start = match.start() if enclosed else match.start("number")
        end = match.end() if enclosed else match.end("number")
        if not enclosed:
            for group in ("minus", "currency"):
                if match[group]:
                    start = min(start, match.start(group))
            for group in ("currency2", "credit"):
                if match[group]:
                    end = max(end, match.end(group))
        found.append(Money(value, text[start:end], start, end, negative, problem))
    return found


def amounts_are_scaled(text: str) -> bool:
    """Whether the document says its figures stand for thousands or millions. A figure read
    from such a document as it is written would be wrong by that factor."""
    return bool(_SCALED_DOCUMENT.search(text))


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
_FIGURE_OF_DAYS = r"\d{1,5}(?:[.,]\d{1,3})?"
_DAYS = re.compile(
    rf"""
    (?<![\w.,/])
    (?:
        (?P<words>{_WORD_NUMBER})
        (?:\s?\(\s?(?P<restated>{_FIGURE_OF_DAYS})\s?\))?    # "fourteen (14)"
      |
        (?P<open>\(\s?)?                                     # "(14) days", "( -5 ) days"
        (?P<minus>[-\u2212])?\+?
        (?P<number>{_FIGURE_OF_DAYS})
        (?(open)\s?\))
    )
    [\s-]{{0,3}}(?:(?:calendar|working|business|work|consecutive)[\s-]{{1,3}})?
    (?P<unit>days?|weeks?|months?)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
# "10-14 days", "10 to 14 days": a range is not a number of days.
_RANGE_BEFORE = re.compile(r"\d\s?(?:-|\u2013|to|or)\s?$", re.IGNORECASE)
# "one hundred twenty (120) days": the words this reader knows are only the number's tail.
_LONGER_NUMBER_BEFORE = re.compile(r"\b(?:hundred|thousand)(?:\s+and)?[\s-]{1,3}$", re.IGNORECASE)


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
        words = match["words"]
        number = match["restated"] if words else match["number"]
        unit = match["unit"].lower()
        from_words = _words_value(words) if words else None
        tail_only = bool(words) and bool(
            _LONGER_NUMBER_BEFORE.search(text[max(0, match.start() - 16) : match.start()])
        )
        if tail_only:
            from_words = None
        value: int | None = None
        problem: str | None = None
        if number is not None:
            if not number.isdigit():
                problem = "not_whole_days"
            elif from_words is not None and from_words != int(number):
                problem = "words_and_figures_disagree"
            else:
                value = int(number)
        elif tail_only:
            problem = "number_in_words_not_read"
        else:
            value = from_words
        if unit.startswith("month"):
            value, problem = None, "months_have_no_fixed_length"
        elif _RANGE_BEFORE.search(text[max(0, match.start() - 6) : match.start()]):
            value, problem = None, "duration_is_a_range"
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
                words_agree=bool(words and number and problem is None and not tail_only),
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
            rf"(?<![\d/.-])(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[\s./-]{{0,3}}{_MONTH}\.?[\s.,/-]{{0,3}}"
            r"(?P<y>\d{4}|\d{2})(?![\d/-])",
            re.IGNORECASE,
        ),
    ),
    (
        "month_name_day",
        re.compile(
            rf"\b{_MONTH}\.?\s{{1,3}}(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s{{1,3}}(?P<y>\d{{4}})"
            r"(?![\d/-])",
            re.IGNORECASE,
        ),
    ),
    (
        # A numeric date needs a four-digit year. "26-05-19" is as likely a cost code, a section
        # number or a year-first date, and must not be read as a date or prove anything.
        "numeric",
        re.compile(
            r"(?<![\d/.-])(?P<a>\d{1,2})(?P<s>[/.-])(?P<b>\d{1,2})(?P=s)(?P<y>\d{4})(?![\d/-])"
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
    taken: list[tuple[int, int]] = []  # the spans already read as dates, in order

    def is_taken(start: int, end: int) -> bool:
        index = bisect.bisect_left(taken, (start, start))
        return (index < len(taken) and taken[index][0] < end) or (
            index > 0 and taken[index - 1][1] > start
        )

    for kind, pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if is_taken(match.start(), match.end()):
                continue
            year_text = match["y"]
            if kind == "numeric":
                first, second, name = int(match["a"]), int(match["b"]), "numeric"
            elif kind == "iso":
                first, second, name = int(match["d"]), int(match["m"]), "iso"
            else:
                # casefold, not lower: the pattern is case-insensitive, and a long s matches "s".
                month = _MONTHS.get(match["month"].casefold()[:3])
                if month is None:
                    continue
                first, second, name = int(match["d"]), month, "month_name"
            bisect.insort(taken, match.span())
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
# A title that gives the direction of the whole document: "DEDUCTIVE CHANGE ORDER".
_TITLE_DIRECTION = re.compile(
    r"^[^\w\n]{0,3}(?:(?P<down>deductive|deduct|credit)|(?P<up>additive|add))\s+"
    r"(?:change\s+order|c\.?o\.?\b|modification|variation)",
    re.IGNORECASE | re.MULTILINE,
)
# Words that leave a statement open: "No time extension is requested at this time" does not
# say the time is unchanged, and neither does one made while reserving the right to claim.
_HEDGED = re.compile(
    r"\bat\s+(?:this|the\s+present)\s+time\b|\bat\s+present\b|\bfor\s+now\b"
    r"|\bto\s+be\s+(?:determined|confirmed|advised|agreed|assessed|evaluated|negotiated|submitted)\b"
    r"|\bt\.?b\.?[dca]\b|\bpending\b|\bsubject\s+to\b|\bnot\s+yet\b|\bunknown\b|\bunder\s+review\b"
    r"|\bcannot\s+(?:yet\s+)?be\s+determined\b|\bunless\b|\bprovided\s+that\b|\bexcept\b",
    re.IGNORECASE,
)
_RESERVATION = re.compile(
    r"\breserv\w+\s+(?:the\s+|its\s+|their\s+|all\s+)?rights?\b|\bwithout\s+prejudice\b",
    re.IGNORECASE,
)


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


def title_cues(text: str) -> set[str]:
    """The directions the document's own title lines give the change."""
    found = set()
    for match in _TITLE_DIRECTION.finditer(text):
        found.add("decrease" if match["down"] else "increase")
    return found


def no_change_cue(text: str) -> str | None:
    """ "explicit" when the text says nothing changes, "not_applicable" for N/A, else None."""
    if _NO_CHANGE.search(text):
        return "explicit"
    return "not_applicable" if _NOT_APPLICABLE.search(text) else None


def is_hedged(text: str) -> bool:
    """Whether a statement leaves its subject open ("at this time", "to be confirmed")."""
    return bool(_HEDGED.search(text))


def reserves_rights(text: str) -> bool:
    """Whether the text reserves a right to claim later."""
    return bool(_RESERVATION.search(text))


# ---- What the words beside a value make it -----------------------------------------------------
#
# A quote that holds a value proves the value is in the document. It does not prove the value
# is what the field asks for: the revised contract total is in the document too, and so are the
# signature dates and the requester's name. The words beside a value say what it is. These
# lexicons read them, so that a value sitting under the wrong label is refused.

_COLUMN_GAP = re.compile(r"\s{2,}")


def _row_above(text: str, start: int) -> tuple[list[str], str, int] | None:
    """The headings of the line above when it reads as a header row, with this line and the
    position of `start` on it. A header row has three or more columns and holds no figures and
    no colons, as a row of headings does and a row of "Label: value" pairs does not."""
    line_start, line_end = line_of(text, start)
    if line_start == 0 or line_end - line_start > _LONGEST_ROW:
        return None
    header_start, header_end = line_of(text, line_start - 1)
    if header_end - header_start > _LONGEST_ROW:
        return None
    header = text[header_start:header_end].strip()
    if not header or ":" in header or any(char.isdigit() for char in header):
        return None
    headings = _COLUMN_GAP.split(header)
    if len(headings) < 3:
        return None
    return headings, text[line_start:line_end], start - line_start


def header_above(text: str, start: int) -> str | None:
    """The column heading over the value at `start`, when its line is a row under a header row
    with the same number of columns. None for any other line."""
    found = _row_above(text, start)
    if found is None:
        return None
    headings, row, position = found
    if len(headings) != len(_COLUMN_GAP.split(row.strip())):
        return None
    column = len(_COLUMN_GAP.findall(row[:position].lstrip()))
    return headings[column] if column < len(headings) else None


def headings_above(text: str, start: int) -> str:
    """Every heading of the header row over the value at `start`, when its own column cannot be
    told: a row with fewer cells than the header has headings, as under "Add" and "Deduct"
    columns of which one is empty. Empty when the line above is no header row, or when the
    value's column is known."""
    found = _row_above(text, start)
    if found is None:
        return ""
    headings, row, _ = found
    cells = len(_COLUMN_GAP.split(row.strip()))
    return "  ".join(headings) if 2 <= cells < len(headings) else ""


def nearest_label(text: str, start: int) -> str:
    """The label right beside the value at `start`: its column heading when the line is a table
    row; otherwise the last stretch of words before it on its line, after the previous figure
    and after the last column gap. Empty when the value opens its line or follows a figure."""
    heading = header_above(text, start)
    if heading is not None:
        return heading
    before = before_on_line(text, start)
    figures = list(_FIGURE.finditer(before))
    if figures:
        before = before[figures[-1].end() :]
    pieces = [piece for piece in _COLUMN_GAP.split(before) if piece.strip()]
    return pieces[-1].strip() if pieces else ""


_CONTRACT = r"(?:sub)?(?:contract|agreement)\s+(?:sum|amount|price|value|total)"
_TOTAL_LABEL = re.compile(
    rf"\b(?:original|revised|new|current|adjusted|previous|prior|updated|final|total)\s+{_CONTRACT}"
    rf"|\b{_CONTRACT}\s+(?:prior\s+to|before|after|including|to\s+date|as\s+of|was\b|remains\b"
    r"|will\s+be(?!\W{1,3}(?:increased|decreased|changed|reduced|adjusted|unchanged)))"
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
# The same said after the figure: "$22,000.00 (not to exceed)", "$5,000 estimated".
_LIMIT_AFTER = re.compile(
    r"^[\s,(\-\u2013]{0,3}(?:not\s+to\s+exceed|n\.?t\.?e\b|max(?:imum)?\b|est(?:imated?)?\b|approx\w*"
    r"|budget\w*|allowance\b|ceiling\b|or\s+less\b|t\.?b\.?[dc]\b"
    r"|to\s+be\s+(?:confirmed|determined|agreed))",
    re.IGNORECASE,
)
_CHANGE_LABEL = re.compile(
    r"\b(?:increased|decreased|changed|reduced|adjusted)\)?\s+by\b"
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
    r"|\b(?:add|deduct|credit)\b[\s:)\-]{0,6}$",
    re.IGNORECASE,
)
_PART_LABEL = re.compile(
    r"\b(?:sub-?total|sales\s+tax|tax|vat|overhead|profit|oh\s*&\s*p|mark-?up|fee|bond|insurance"
    r"|unit\s+price|labou?r|materials?|equipment|freight|contingency)\b",
    re.IGNORECASE,
)
# A plain "TOTAL" at the head of its label, when nothing more specific was said.
_PLAIN_TOTAL = re.compile(r"^\W{0,6}total\b", re.IGNORECASE)
# A figure followed by a unit is a quantity; one after "@" or before "/unit" is a rate.
_MEASURE_AFTER = re.compile(
    r"^\s?(?:sq\.?\s?ft|sf|lf|cy|sy|tons?|feet|foot|ft|hrs?|hours?|ea|each|m2|m3|lbs?|kg"
    r"|gal(?:lons?)?|inch(?:es)?|%)\b|^\s?/\s?[A-Za-z]",
    re.IGNORECASE,
)
_RATE_BEFORE = re.compile(r"(?:@|\bat|\bx)\s{0,3}$", re.IGNORECASE)
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
    if _LIMIT_LABEL.search(before) or _LIMIT_AFTER.match(after):
        return "limit"
    if _CHANGE_LABEL.search(before) or _CHANGE_LABEL.search(after):
        return "change"
    if _PART_LABEL.search(before):
        return "part"
    return "change" if _PLAIN_TOTAL.search(before) else None


_SUM_AFTER = re.compile(
    r"\b(?:revised|new|adjusted|updated)\b|\bafter\b|\bincluding\b|\bwill\s+be\b|\bis\s+now\b",
    re.IGNORECASE,
)
_SUM_BEFORE = re.compile(
    r"\bprior\s+to\b|\bbefore\b|\bwas\b|\b(?:previous|present|existing)\s", re.IGNORECASE
)
_SUM_ORIGINAL = re.compile(r"\b(?:original|initial|base)\b", re.IGNORECASE)
_SUM_EARLIER_CHANGES = re.compile(
    r"\bnet\s+change\s+by\s+(?:previous|previously|prior)"
    r"|\b(?:previous|prior)(?:ly)?\s+(?:authori[sz]ed\s+|approved\s+)?change\s+orders",
    re.IGNORECASE,
)


def sum_role(label: str) -> str | None:
    """Which contract sum a label names: "original", "before" (just before this change order),
    "after", or "earlier_changes" (the total of earlier change orders). None when the label
    names no contract sum, or does not say which one ("Current Contract Sum")."""
    if not _TOTAL_LABEL.search(label):
        return None
    if _SUM_EARLIER_CHANGES.search(label):
        return "earlier_changes"
    after = bool(_SUM_AFTER.search(label))
    if _SUM_ORIGINAL.search(label):
        return None if after else "original"
    if _SUM_BEFORE.search(label):
        return None if after else "before"
    return "after" if after else None


_HAS_CURRENCY = re.compile(_CURRENCY)
_MONEY_SHAPED = re.compile(rf"{_CURRENCY}|[.,]\d")
_PRICE_WORDS = re.compile(
    r"\b(?:amount|price|cost|sum|total|value|charge|credit|deduct\w*|add|increase|decrease)\b",
    re.IGNORECASE,
)
_PRICE_SUBJECT = re.compile(
    r"\b(?:price|costs?|amount|sum|charges?|compensation|fee|value)\b", re.IGNORECASE
)


def written_like_money(raw: str) -> bool:
    """Whether a figure is written as an amount: with a currency sign or code, or with a decimal
    or thousands mark. A bare "6" is a count, a number or an identifier."""
    return bool(_MONEY_SHAPED.search(raw))


def says_price(raw: str, label: str) -> bool:
    """Whether anything marks a figure as a price at all: a currency on the figure, or a word
    for money in its label. "Below elevation 412.0" has neither."""
    return bool(_HAS_CURRENCY.search(raw) or _PRICE_WORDS.search(label))


def speaks_of_price(text: str) -> bool:
    """Whether a statement is about the price: "no change to the Contract Sum", "no cost"."""
    return bool(_PRICE_SUBJECT.search(text))


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
# A column headed "Days" and nothing else.
_DAYS_HEADING = re.compile(
    r"^\W{0,3}(?:(?:calendar|working|business)\s+)?days\W{0,3}$", re.IGNORECASE
)
# The contract time itself, or the days of earlier change orders: a total, not this change.
_TIME_TOTAL = re.compile(
    r"\b(?:original|revised|new|current|adjusted|previous|prior|updated)\s+(?:contract\s+)?"
    r"(?:time|duration|period)\b(?!\s+(?:extension|impact|change|adjustment))"
    r"|\btotal\s+contract\s+(?:time|duration|period)\b"
    r"|\bcontract\s+(?:time|duration|period)\s+(?:prior\s+to|before|after|including|to\s+date"
    r"|as\s+of|was\b|is\s+now\b|remains\b"
    r"|will\s+be(?!\W{1,3}(?:increased|decreased|changed|reduced|adjusted|unchanged|extended"
    r"|shortened)))"
    r"|\bnet\s+change\s+by\s+(?:previous|previously|prior)"
    r"|\b(?:previous|prior)(?:ly)?\s+(?:authori[sz]ed\s+|approved\s+)?change\s+orders",
    re.IGNORECASE,
)
# A guess, a ceiling or a floor, said before the duration or after it.
_UNSURE_BEFORE = re.compile(
    r"\b(?:up\s+to|approx\w*|estimat\w*|not\s+to\s+exceed|maximum|max|at\s+least|minimum"
    r"|no\s+more\s+than|no\s+less\s+than|possibl[ey]|potential\w*|anticipat\w*|may\s+be|could\s+be"
    r"|might\s+be|between)\b[^.;\n]{0,30}$",
    re.IGNORECASE,
)
_UNSURE_AFTER = re.compile(
    r"^[\s,;:(\-\u2013]{0,4}(?:to\s+be\s+(?:confirmed|determined|agreed|advised|verified)"
    r"|t\.?b\.?[dc]\b|approx\w*|est(?:imated?)?\b|or\s+more\b|or\s+less\b|more\s+or\s+less\b"
    r"|max(?:imum)?\b|min(?:imum)?\b|pending\b|subject\s+to\b|tentative\w*|if\b)",
    re.IGNORECASE,
)
# A term for something else: a payment, a period of validity, a lead time, a deadline to act.
_OTHER_TERM = re.compile(
    r"\b(?:within|valid\s+for|good\s+for|open\s+for|net|every|per|each|after|first|last|next"
    r"|past|lead[\s-]?time)\W{0,3}$"
    r"|\b(?:payment|pay|paid|invoic\w+|warranty|guarantee|retainage|validity)\b[^.;\n]{0,40}$",
    re.IGNORECASE,
)
_SENTENCE_START = re.compile(r"[.;!?]\s+(?=[A-Z(\[])")
_SENTENCE_END = re.compile(r"[.;!?](?:\s|$)|\s{2,}")


def is_schedule_label(label: str) -> bool:
    """Whether words are about the contract time. "Valid for 15 days" and "Crane - 1 day" are
    durations too, and neither is the schedule impact."""
    return bool(_SCHEDULE_LABEL.search(label) or _DAYS_HEADING.match(label))


def duration_label(before: str, after: str = "") -> str | None:
    """What the words around a length of time make it.

    "total": the contract time itself, or the days of earlier change orders. "limit": a
    ceiling, a floor, an estimate, or a number still to be confirmed. "other": a term for
    something else (a payment, a period of validity, a lead time). "change": the words are
    about the contract time. None when they do not say.

    Only the duration's own sentence is read: what a neighbouring sentence or the next column
    says about time is not said about this number.
    """
    starts = [match.end() for match in _SENTENCE_START.finditer(before)]
    if starts:
        before = before[starts[-1] :]
    end = _SENTENCE_END.search(after)
    if end:
        after = after[: end.start()]
    if _TIME_TOTAL.search(before):
        return "total"
    if _UNSURE_BEFORE.search(before) or _UNSURE_AFTER.match(after):
        return "limit"
    if _OTHER_TERM.search(before):
        return "other"
    if is_schedule_label(before) or _SCHEDULE_LABEL.search(after):
        return "change"
    return None


_OTHER_EVENT = re.compile(
    r"\b(?:contract\s+date|date\s+of\s+(?:the\s+)?(?:original\s+)?(?:contract|agreement)"
    r"|(?:contract|agreement)\s+dated|notice\s+to\s+proceed"
    r"|substantial\s+(?:completion|performance)|completion|complete[ds]?\s+by"
    r"|signed|signature|received|due|valid\s+(?:until|through)|expir\w+|effective"
    r"|approved|accepted|executed|authori[sz]ed|rejected|void(?:ed)?|answered|action|revised"
    r"|(?:memo|letter|email|drawing|sketch|report|proposal|quote|quotation|survey|discharge"
    r"|invoice|meeting|walk)\s+(?:dated|of|on))\b"
    r"|\bby\s{0,3}:",
    re.IGNORECASE,
)
_OWN_DATE = re.compile(
    r"\b(?:date|dated|issued|issue|issuance|prepared|submitted|written|initiated|created|sent"
    r"|date\s+of\s+(?:this\s+|the\s+)?(?:change\s+order|request|document|proposal))"
    r"\W{0,12}(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day\W{0,6})?$",
    re.IGNORECASE,
)
# The words that may stand before "date" in the label of the document's own date. Any other
# word makes it the date of something else: "Invoice Date", "Start Date", "Meeting Date".
_OWN_DATE_WORDS = frozenset(
    [
        "change",
        "order",
        "co",
        "cor",
        "pco",
        "cco",
        "ccd",
        "document",
        "doc",
        "this",
        "request",
        "proposal",
        "variation",
        "modification",
        "amendment",
        "directive",
        "form",
        "no",
        "number",
        "of",
        "the",
        "on",
        "date",
        "dated",
        "issue",
        "issued",
        "issuance",
        "prepared",
        "submitted",
        "submission",
        "written",
        "initiated",
        "created",
        "creation",
        "sent",
        "original",
        "today",
        "s",
    ]
)
_WORD = re.compile(r"[a-z]+")
_SIGNATURE_CONTEXT = re.compile(r"\bby\s{0,3}:|/s/|\bsigned\b|\bsignature\b", re.IGNORECASE)
# A line of a signature block, wherever it stands in the block above a date.
_SIGNING = re.compile(
    r"^[^\w\n]{0,3}by\s{0,3}:|\s{2}by\s{0,3}:"
    r"|\b(?:approved|accepted|authori[sz]ed|agreed|recommended|reviewed|signed|executed"
    r"|witnessed)\s+by\b|/s/|\bsignatures?\b|\bsigned\b",
    re.IGNORECASE | re.MULTILINE,
)


def _labels_the_documents_date(label: str) -> bool:
    match = _OWN_DATE.search(label)
    if match is None:
        return False
    lead = label[: match.start()]
    lead = lead[lead.rfind(":") + 1 :]  # what follows an earlier label's colon
    return all(word in _OWN_DATE_WORDS for word in _WORD.findall(lead.lower()))


def date_label(text: str, start: int) -> str:
    """What the words beside the date at `start` make it.

    "own": labelled as the document's date ("Date:", "Issued:", "Change Order Date:", "Date
    Prepared:"), or standing alone at the head of its line as a letter's date does. "other": the
    date of something else (the contract, an invoice, a completion, an approval, a referenced
    letter), or a date in a signature block. "unlabelled": in running text with no label that
    says either.
    """
    before = before_on_line(text, start)
    label = nearest_label(text, start)
    if _OTHER_EVENT.search(label) or _SIGNATURE_CONTEXT.search(before):
        return "other"
    alone = not any(char.isalpha() for char in label)
    if _OWN_DATE.search(label) or (alone and not before.strip()):
        # "Date:" under a line of "By:" entries, or further down a signature block, is the
        # date of a signature.
        if _SIGNATURE_CONTEXT.search(line_above(text, start)) or _SIGNING.search(
            lines_above(text, start, 6)
        ):
            return "other"
        return "own" if alone or _labels_the_documents_date(label) else "other"
    return "unlabelled"


# The right kind of text under the wrong label: for each field, the labels that say the value
# is something else, and the labels that say it is this field after all. The one nearer the
# value wins: "AMENDMENT TO CONTRACT (CHANGE ORDER NO. 2" labels a change order number.
_NOT_AN_IDENTIFIER = (
    r"|\b(?:date|dated|amount|total|sum|price|value|time|days|page|sheet|phone|tel|fax)\b"
)
_LABELS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "project_name": (
        re.compile(
            r"\b(?:requested|initiated|instructed|directed|prepared|submitted|approved|accepted"
            r"|signed|recommended)\s+by\b"
            r"|\b(?:owner|contractor|subcontractor|architect|engineer|client|employer|from|to"
            r"|attn|cc)\s{0,3}:"
            r"|\b(?:contract|agreement|change\s+order|job|project)\s{0,3}(?:no\.?|number|#|ref)",
            re.IGNORECASE,
        ),
        re.compile(r"\bproject(?:\s+name)?\s{0,3}:|\bjob\s{0,3}(?:name)?\s{0,3}:", re.IGNORECASE),
    ),
    "requested_by": (
        re.compile(
            r"\b(?:project|job|contract|agreement|subject)\s{0,3}(?:name|no\.?|number|#)?\s{0,3}:"
            r"|\b(?:approved|accepted|signed|prepared|recommended)\s+by\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:requested|initiated|instructed|directed)\s+by\b", re.IGNORECASE),
    ),
    "contract_number": (
        re.compile(
            r"\bchange\s+order\b|\bC\.?O\.?R?\s{0,3}(?:no\.?|number|#)"
            r"|\b(?:job|project|rfi|asi|invoice|bulletin|p\.?o\.?|purchase\s+order|permit)\b"
            + _NOT_AN_IDENTIFIER,
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:sub)?contract\b|\bagreement\b", re.IGNORECASE),
    ),
    "change_order_number": (
        re.compile(
            r"\b(?:sub)?contract\b|\bagreement\b"
            r"|\b(?:job|project|rfi|asi|invoice|bulletin|p\.?o\.?|purchase\s+order|permit"
            r"|drawing|sketch)\b" + _NOT_AN_IDENTIFIER,
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
_SIGNATURE = re.compile(r"(?P<mark>/s/\s{0,3}\S)|\bsigned\b", re.IGNORECASE)
_BOX = re.compile(
    r"(?P<ticked>\[\s?[xX\u2713\u2714]\s?\]|\(\s?[xX\u2713\u2714]\s?\)|[\u2611\u2612])"
    r"|(?P<empty>\[[\s_]{0,3}\]|\(\s{0,3}\)|\u2610)"
)
# How much of a line is read on either side of a word. The patterns below reach no further.
_NEAR = 96
# A negation earlier in the same clause: "has not been approved", "No change to the Contract is
# authorized", "Not issued for signature".
_NEGATION = re.compile(r"\b(?:not|never|no|without|neither|nor)\b[^.;:\n]{0,45}$", re.IGNORECASE)
# Words that make what follows a condition or an instruction, not a statement of fact.
_CONDITION = re.compile(
    r"\b(?:until|unless|upon|once|when|if|after|before|subject\s+to|requires?|must|shall|should"
    r"|will|to\s+be|prior\s+to|valid)\b[^.;:\n]{0,60}$",
    re.IGNORECASE,
)
_FILL = r"[_.\-\u2013\u2014]"
# What stands where a name or a mark would: nothing, a run of fill characters ("____", "- - -",
# ". . . ."), or the next label of a signature block.
_NOTHING = rf"(?:$|{_FILL}{{2,}}|(?:{_FILL}\s){{2,}}|\s{{0,6}}(?:date|name|title)\b)"
# "Approved by (Owner): ______", "Approved by ______", or "Approved by:" with nothing after: a
# label, not a signature.
_BLANK_LABEL = re.compile(
    r"^\s{0,3}by\b(?:[^\w\n]|_){0,40}(?:\([^)\n]{0,60}\))?(?:[^\w\n]|_){0,200}$", re.IGNORECASE
)
# The same with a party named: "Approved by Utility Director: ______", "Accepted for the
# Conservancy:". What follows the label's colon is a blank, not a name.
_BLANK_AFTER = re.compile(rf"^[^:\n]{{0,50}}?(?::\s{{0,3}}{_NOTHING}|\s?[_.]{{3,}})", re.IGNORECASE)
# A line of a signature block that nobody has signed. A label with nothing after it counts
# only when nothing stands under it either: "Accepted:" over two signed lines is a heading.
_UNSIGNED = (
    rf"(?:$(?=\n[ \t]*\n|\n?\Z)|{_FILL}{{2,}}|(?:{_FILL}\s){{2,}}|\s{{0,6}}(?:date|name|title)\b)"
)
_BLANK_SIGNING_LINE = re.compile(
    r"(?:\b(?:approved|accepted|authori[sz]ed|agreed)\b[^:\n]{0,40}|\bsignature"
    r"|(?<!requested )(?<!prepared )(?<!initiated )(?<!submitted )\bby)"
    rf"\s?:\s{{0,3}}{_UNSIGNED}"
    r"|\b(?:owner|architect|engineer|contractor|client|employer)\b[^:\n]{0,30}"
    rf":\s{{0,3}}(?:{_FILL}{{2,}}|(?:{_FILL}\s){{2,}})"
    rf"|\b(?:approved|accepted|authori[sz]ed)\s+by\s?{_FILL}{{3,}}",
    re.IGNORECASE | re.MULTILINE,
)


def _boxes_lead(line: str, boxes: list[re.Match[str]]) -> bool:
    """Whether the tick boxes on a line stand before their options ("[X] Approved  [ ] Rejected")
    or after them ("Approved [ ]  Rejected [X]")."""
    first = len(line) - len(line.lstrip())
    last = len(line.rstrip())
    lead = follow = 0
    for box in boxes[:50]:
        before = line[max(0, box.start() - 4) : box.start()]
        after = line[box.end() : box.end() + 4]
        opens = (
            box.start() == first
            or before.endswith("  ")
            or before.rstrip().endswith((":", ";", ",", "|", "/"))
        )
        closes = (
            box.end() == last
            or after.startswith("  ")
            or after.lstrip().startswith((";", ",", "|", "/"))
        )
        if opens and re.match(r"\s{0,3}[A-Za-z]", after):
            lead += 1
        elif closes and re.search(r"[A-Za-z][:.]?\s{0,3}$", before):
            follow += 1
    if lead != follow:
        return lead > follow
    word = re.search(r"[A-Za-z]", line)
    return word is None or boxes[0].start() < word.start()


def _marks(line: str) -> list[tuple[str, str]]:
    """Each approval state a line gives evidence of, and how: "ticked" (its own box is ticked),
    "explicit" (stated), or "signature" (a completed signature and nothing more)."""
    found: list[tuple[str, str]] = []
    boxes = list(_BOX.finditer(line))
    starts = [box.start() for box in boxes]
    leading = _boxes_lead(line, boxes) if boxes else True

    def own_box(start: int, end: int) -> str | None:
        # The nearest box on the side where this line puts its boxes, if it is close by.
        if not boxes:
            return None
        if leading:
            index = bisect.bisect_right(starts, start) - 1
            if index < 0 or start - boxes[index].end() > 60:
                return None
        else:
            index = bisect.bisect_left(starts, end)
            if index >= len(boxes) or boxes[index].start() - end > 60:
                return None
        return "ticked" if boxes[index]["ticked"] else "empty"

    for status, pattern in _STATUS_WORDS:
        for match in pattern.finditer(line):
            before = line[max(0, match.start() - _NEAR) : match.start()]
            after = line[match.end() : match.end() + _NEAR]
            box = own_box(match.start(), match.end())
            if box == "empty" or _BLANK_AFTER.match(after):
                continue
            if status == "approved" and (
                _NEGATION.search(before) or _CONDITION.search(before) or _BLANK_LABEL.match(after)
            ):
                continue
            if status in ("pending", "draft", "void") and _NEGATION.search(before):
                continue
            found.append((status, "ticked" if box == "ticked" else "explicit"))
    for match in _SIGNATURE.finditer(line):
        before = line[max(0, match.start() - _NEAR) : match.start()]
        after = line[match.end() : match.end() + _NEAR]
        if match["mark"] or not (
            _NEGATION.search(before)
            or _CONDITION.search(before)
            or _BLANK_LABEL.match(after)
            or _BLANK_AFTER.match(after)
        ):
            found.append(("approved", "signature"))
    return found


def status_cues(text: str) -> dict[str, str]:
    """The approval states a passage gives evidence of, each "explicit" (a statement, a stamp, a
    ticked box) or "signature" (a completed signature and nothing more).

    A word does not count when its own tick box is empty, when it is negated ("not approved"
    counts as rejected instead), sits in a condition ("not valid until signed", "must be
    approved by"), or labels a blank ("Approved by: ______", "Approved by Owner:").
    """
    found: dict[str, str] = {}
    for line in text.splitlines():
        for status, how in _marks(line):
            if how == "signature":
                found.setdefault(status, "signature")
            else:
                found[status] = "explicit"
    return found


def ticked_statuses(text: str) -> set[str]:
    """The approval states whose own tick box is ticked."""
    return {status for line in text.splitlines() for status, how in _marks(line) if how == "ticked"}


def has_blank_signing_line(text: str) -> bool:
    """Whether a signature block still has a line nobody signed: "Approved by Owner: ______",
    "By:" with nothing after it."""
    return bool(_BLANK_SIGNING_LINE.search(text))
