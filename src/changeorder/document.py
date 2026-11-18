"""Reading one file into text the rest of the pipeline can check quotes against.

A PDF's text layer is read page by page with pypdf in layout mode, which rebuilds rows from the
positions of the text. Without it, a table written to the page a column at a time comes out as a
list of descriptions followed by a list of figures, and a label lands on a different line from
its value. A plain-text file is decoded as UTF-8 (UTF-16 with a byte-order mark; Windows-1252 as
a recorded last resort).

Either way the result is one canonical text: invisible characters removed, NFC, one line ending,
runs of spaces reduced to two (enough to keep a column gap visible), blank lines collapsed. Every
position the pipeline keeps (a quote's span, a value's page and line) is a position in that text,
and it is exactly the text the model is shown.

Nothing here guesses. A file that is not a PDF or text, an encrypted or damaged PDF, a document
with no usable text, garbled text and a document over a limit all raise `DocumentError`. A page
that holds an image and no text (a scanned page) is not an error by itself: the pages that can be
read are read, and the page is listed in `unread_pages` so the result is sent to a person.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Limits
from .vendor.injection_sanitizer import remove_hidden

__all__ = ["DocumentError", "Page", "SourceDocument", "read_document"]

# First bytes of files people send by mistake, with what to tell them.
_NOT_SUPPORTED: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG", "This is an image. Only a PDF with a text layer or a plain-text file is read."),
    (
        b"\xff\xd8\xff",
        "This is an image. Only a PDF with a text layer or a plain-text file is read.",
    ),
    (b"GIF8", "This is an image. Only a PDF with a text layer or a plain-text file is read."),
    (b"II*\x00", "This is an image. Only a PDF with a text layer or a plain-text file is read."),
    (b"MM\x00*", "This is an image. Only a PDF with a text layer or a plain-text file is read."),
    (
        b"PK\x03\x04",
        "This is a ZIP or Office file. Export it to PDF, or paste its text into a file.",
    ),
    (b"\xd0\xcf\x11\xe0", "This is an old Office file. Export it to PDF, or paste its text."),
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_GAPS = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_BREAK = "\n\n"


class DocumentError(Exception):
    """The document could not be read. `code` is stable; `message` is safe to show a person."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Page:
    number: int  # from 1, as in the file
    start: int  # where its text starts in SourceDocument.text
    end: int


@dataclass(frozen=True)
class SourceDocument:
    name: str
    kind: str  # "pdf" or "text"
    text: str
    pages: tuple[Page, ...]  # the pages that have text
    page_count: int
    unread_pages: tuple[int, ...]  # pages with an image and no text layer
    notes: tuple[str, ...]  # what reading had to do that a person may want to know
    sha256: str

    def locate(self, offset: int) -> tuple[int, int]:
        """The page and the line on that page (both from 1) of a position in `text`."""
        page = self.pages[0]
        for candidate in self.pages:
            if candidate.start <= offset:
                page = candidate
        return page.number, self.text.count("\n", page.start, offset) + 1

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "pages": self.page_count,
            "characters": len(self.text),
            "unread_pages": list(self.unread_pages),
            "notes": list(self.notes),
        }


def read_document(path: Path, limits: Limits | None = None) -> SourceDocument:
    """The file as canonical text with its pages, or `DocumentError`."""
    limits = limits or Limits()
    if not path.is_file():
        raise DocumentError("not_found", f"No file at {path}.")
    size = path.stat().st_size
    if size > limits.max_bytes:
        raise DocumentError(
            "too_large",
            f"The file is {size:,} bytes; the limit is {limits.max_bytes:,}. Nothing was read.",
        )
    data = path.read_bytes()
    if not data.strip():
        raise DocumentError("no_text", "The file is empty.")

    notes: list[str] = []
    unread: list[int] = []
    if b"%PDF-" in data[:1024]:
        kind = "pdf"
        page_texts, page_count, unread, notes = _read_pdf(data, limits)
    else:
        for signature, message in _NOT_SUPPORTED:
            if data.startswith(signature):
                raise DocumentError("unsupported_type", message)
        kind = "text"
        page_texts, page_count = [(1, _decode(data, notes))], 1

    pieces: list[str] = []
    pages: list[Page] = []
    offset = 0
    hidden = False
    for number, raw in page_texts:
        cleaned = remove_hidden(raw)
        hidden = hidden or bool(cleaned.findings)
        text = _canonical(cleaned.text)
        if not text:
            continue
        pieces.append(text)
        pages.append(Page(number, offset, offset + len(text)))
        offset += len(text) + len(_PAGE_BREAK)
    if hidden:
        notes.append("hidden_characters_removed")
    text = _PAGE_BREAK.join(pieces)

    _check_quality(text, kind, unread, limits)
    if len(text) > limits.max_chars:
        raise DocumentError(
            "too_large",
            f"The document has {len(text):,} characters of text; the limit is "
            f"{limits.max_chars:,}. It was refused, not cut.",
        )
    return SourceDocument(
        name=path.name,
        kind=kind,
        text=text,
        pages=tuple(pages),
        page_count=page_count,
        unread_pages=tuple(unread),
        notes=tuple(notes),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _decode(data: bytes, notes: list[str]) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            raise DocumentError("unsupported_type", "The file is not readable text.") from None
    if b"\x00" in data:
        raise DocumentError("unsupported_type", "The file is binary, not a PDF or plain text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Text pasted from an older Windows program. The guess is recorded, and any byte it
        # cannot place becomes U+FFFD, which the quality check below counts.
        notes.append("decoded_as_windows_1252")
        return data.decode("cp1252", errors="replace")


def _canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = _CONTROL.sub("", text)
    lines = (_GAPS.sub("  ", line.replace("\t", "  ")).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _check_quality(text: str, kind: str, unread: list[int], limits: Limits) -> None:
    visible = [char for char in text if not char.isspace()]
    if len(visible) < limits.min_chars:
        if kind == "pdf":
            detail = (
                f" Pages {', '.join(map(str, unread))} hold an image and no text." if unread else ""
            )
            raise DocumentError(
                "no_text",
                "The PDF has no text layer to read: it looks scanned, and this pipeline does "
                "no OCR." + detail,
            )
        raise DocumentError("no_text", "The file holds too little text to be a change order.")
    garbage = sum(
        1 for char in visible if char == "\ufffd" or unicodedata.category(char) in ("Co", "Cn")
    )
    readable = sum(1 for char in visible if char.isalnum())
    if garbage / len(visible) > 0.05 or readable / len(visible) < 0.35:
        raise DocumentError(
            "unreadable_text",
            "The text that came out of the file is garbled (undecodable characters or almost no "
            "letters), so nothing reliable can be extracted from it.",
        )


def _read_pdf(
    data: bytes, limits: Limits
) -> tuple[list[tuple[int, str]], int, list[int], list[str]]:
    """Each page's text, the page count, the pages with an image and no text, and notes."""
    from pypdf import PdfReader

    texts: list[tuple[int, str]] = []
    unread: list[int] = []
    notes: list[str] = []
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("encrypted", "The PDF is protected by a password.")
        count = len(reader.pages)
        if count > limits.max_pages:
            raise DocumentError(
                "too_large",
                f"The PDF has {count} pages; the limit is {limits.max_pages}. Nothing was read.",
            )
        for number, page in enumerate(reader.pages, start=1):
            text = _page_text(page, notes)
            if text.strip():
                texts.append((number, text))
            elif _has_image(page):
                unread.append(number)
    except (DocumentError, MemoryError):
        raise
    # pypdf raises far more than its own error classes on a damaged file (KeyError, ValueError,
    # NotImplementedError for a filter it does not know, DependencyError for AES without the
    # cryptography package). Every one means the same thing here: this PDF cannot be read.
    except Exception as exc:
        if "crypt" in str(exc).lower():
            raise DocumentError(
                "encrypted", "The PDF is encrypted and could not be opened."
            ) from exc
        raise DocumentError("malformed", "The PDF could not be read.") from exc
    return texts, count, unread, notes


def _page_text(page: Any, notes: list[str]) -> str:
    try:
        text: str = page.extract_text(extraction_mode="layout") or ""
    except Exception:  # layout mode is newer and stricter than the plain reader
        text = ""
    if text.strip():
        return text
    plain: str = page.extract_text() or ""
    if plain.strip() and "pdf_plain_text_fallback" not in notes:
        notes.append("pdf_plain_text_fallback")
    return plain


def _has_image(page: Any) -> bool:
    """Whether a page with no text draws an image or a form: content that was not read."""
    try:
        resources = page.get("/Resources")
        xobjects = resources.get_object().get("/XObject") if resources is not None else None
        return bool(xobjects and len(xobjects.get_object()) > 0)
    except Exception:  # content we cannot even list is content we did not read
        return True
