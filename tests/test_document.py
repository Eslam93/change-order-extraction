"""Reading a file: what is read, what is refused, and what is read with a warning.

The PDFs are built in the test with reportlab (a development dependency), so every case is a
real PDF that pypdf has to parse, not a mocked reader.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from changeorder.config import Limits
from changeorder.document import DocumentError, read_document
from changeorder.llm import ScriptedModel
from changeorder.pipeline import process
from helpers import base_fields, codes, model_reply

TEXT_LINES = (
    "CHANGE ORDER",
    "Project: Maple Court Apartments",
    "Contract No.: MC-2025-014",
    "Change Order No.: 7",
    "Date: 15 January 2026",
    "Requested by: Maple Court Housing LLC",
    "Description: Add a second coat of intumescent paint to the steel columns at Level 1.",
    "Amount of this change order (add): $12,450.00",
    "Contract Time: extended by 3 calendar days.",
    "Status: Approved",
)


def _scan() -> ImageReader:
    """A page-sized bitmap: what a scanner produces, with no text layer."""
    image = Image.new("L", (300, 400), 235)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def make_pdf(
    path: Path, pages: list[tuple[str, ...] | None], *, password: str | None = None
) -> Path:
    """A PDF with one page per item: lines of text, or None for a scanned page."""
    canvas = Canvas(str(path), pagesize=letter, invariant=1, encrypt=password)
    for lines in pages:
        if lines is None:
            canvas.drawImage(_scan(), 0, 0, width=letter[0], height=letter[1])
        else:
            y = 720
            for line in lines:
                canvas.drawString(72, y, line)
                y -= 18
        canvas.showPage()
    canvas.save()
    return path


# ---- What is read -------------------------------------------------------------------------------


def test_a_text_file_is_read_as_it_is_with_one_line_ending_and_no_byte_order_mark(tmp_path):
    path = tmp_path / "co.txt"
    path.write_bytes(
        "﻿CHANGE ORDER\r\nProject:\tMaple Court Apartments   \r\n\r\n\r\n\r\nAmount: $12,450.00\r\n".encode()
    )

    document = read_document(path)

    assert document.kind == "text"
    assert document.text == "CHANGE ORDER\nProject:  Maple Court Apartments\n\nAmount: $12,450.00"
    assert document.unread_pages == () and document.notes == ()
    assert len(document.sha256) == 64


def test_a_pdf_text_layer_is_read_page_by_page_with_positions_that_point_back(tmp_path):
    path = make_pdf(tmp_path / "co.pdf", [TEXT_LINES[:5], TEXT_LINES[5:]])

    document = read_document(path)

    assert document.kind == "pdf" and document.page_count == 2
    assert [page.number for page in document.pages] == [1, 2]
    for page in document.pages:
        assert document.text[page.start : page.end].count("\n") + 1 == 5
    assert document.locate(document.text.index("Change Order No.")) == (1, 4)
    assert document.locate(document.text.index("Amount of this")) == (2, 3)


def test_a_table_written_to_the_page_column_by_column_is_read_row_by_row(tmp_path):
    # Every description is drawn first, then every amount: the order a PDF often stores a table in.
    path = tmp_path / "table.pdf"
    canvas = Canvas(str(path), pagesize=letter, invariant=1)
    rows = (("Labor", "6,272.00"), ("Material", "9,415.50"), ("TOTAL THIS CHANGE", "$21,318.13"))
    for index, (label, _) in enumerate(rows):
        canvas.drawString(72, 700 - 18 * index, label)
    for index, (_, amount) in enumerate(rows):
        canvas.drawRightString(500, 700 - 18 * index, amount)
    canvas.save()

    lines = read_document(path, Limits(min_chars=10)).text.splitlines()

    assert lines == ["Labor  6,272.00", "Material  9,415.50", "TOTAL THIS CHANGE  $21,318.13"]


def test_text_from_an_older_windows_program_is_read_and_the_guess_is_recorded(tmp_path):
    path = tmp_path / "co.txt"
    path.write_bytes(
        "Project: Caf\xe9 Riviera \x96 Phase 2 fit-out for the Owner, amount $12,450.00".encode(
            "latin-1"
        )
    )

    document = read_document(path)

    assert "Café Riviera – Phase 2" in document.text
    assert document.notes == ("decoded_as_windows_1252",)


def test_invisible_characters_are_removed_before_anything_is_matched(tmp_path):
    path = tmp_path / "co.txt"
    path.write_text(
        "Project: Maple​ Court­ Apartments, amount of this change order $12,450.00", encoding="utf-8"
    )

    document = read_document(path)

    assert "Maple Court Apartments" in document.text
    assert document.notes == ("hidden_characters_removed",)


# ---- What is refused ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"", "no_text"),
        (b"   \n\n  ", "no_text"),
        (b"CO 7", "no_text"),  # too little to be a document
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "unsupported_type"),
        (b"PK\x03\x04" + b"\x00" * 64, "unsupported_type"),  # a .docx or .zip
        (b"\x00\x01\x02\x03 binary \x00" * 20, "unsupported_type"),
        (b"%PDF-1.7\nthis is not really a pdf at all\n%%EOF", "malformed"),
        ("���� text ������������ lost ��������������������".encode(), "unreadable_text"),
        (("_" * 300 + " . " * 100).encode(), "unreadable_text"),  # almost no letters
    ],
)  # fmt: skip
def test_a_file_that_cannot_be_read_reliably_is_an_error_not_a_result(tmp_path, content, code):
    path = tmp_path / "upload.bin"
    path.write_bytes(content)

    with pytest.raises(DocumentError) as raised:
        read_document(path)

    assert raised.value.code == code
    assert raised.value.message  # something to tell the person


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(DocumentError) as raised:
        read_document(tmp_path / "nowhere.pdf")
    assert raised.value.code == "not_found"


def test_a_scanned_pdf_is_refused_because_nothing_was_read(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", [None, None])

    with pytest.raises(DocumentError) as raised:
        read_document(path)

    assert raised.value.code == "no_text"
    assert "no OCR" in raised.value.message and "1, 2" in raised.value.message


def test_a_password_protected_pdf_is_refused(tmp_path):
    path = make_pdf(tmp_path / "locked.pdf", [TEXT_LINES], password="s3cret")

    with pytest.raises(DocumentError) as raised:
        read_document(path)

    assert raised.value.code == "encrypted"


@pytest.mark.parametrize(
    ("limits", "detail"),
    [
        (Limits(max_bytes=200), "bytes"),
        (Limits(max_pages=1), "pages"),
        (Limits(max_chars=120), "characters"),
    ],
)
def test_a_document_over_a_limit_is_refused_never_cut(tmp_path, limits, detail):
    path = make_pdf(tmp_path / "long.pdf", [TEXT_LINES, TEXT_LINES])

    with pytest.raises(DocumentError) as raised:
        read_document(path, limits)

    assert raised.value.code == "too_large"
    assert detail in raised.value.message


# ---- What is read with a warning ------------------------------------------------------------------


def test_a_scanned_page_among_text_pages_is_reported_not_ignored(tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf", [TEXT_LINES, None, ()])

    document = read_document(path)

    assert document.page_count == 3
    assert document.unread_pages == (2,)  # page 3 is blank: nothing there to miss
    assert [page.number for page in document.pages] == [1]


def test_a_document_with_an_unread_page_is_processed_but_never_passed_without_review(tmp_path):
    # Everything on page 1 checks out. Page 2 is a scan the reader could not see.
    path = make_pdf(tmp_path / "mixed.pdf", [TEXT_LINES, None])

    result = process(path, ScriptedModel([model_reply(**base_fields())]))

    out = result.output.to_dict()
    assert out["needs_review"] is True
    assert codes(result) == {"unread_pages"}
    assert out["amount"] == {"value": 12450, "confidence": 0.75}
    assert all(out[name]["confidence"] == 0.75 for name in out if name != "needs_review")
    assert "document_has_unread_pages" in result.decisions["amount"].caveats
