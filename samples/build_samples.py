"""Builds the four PDF samples in samples/inputs/. The two .txt samples are plain files.

    uv run python samples/build_samples.py

The documents are synthetic: invented projects, companies, people and figures, written to carry
the problems real change orders have (see samples/README.md). They are drawn with absolute
positions, the way a form or an accounting export is, and two of them are deliberately written
to the page out of reading order (a column at a time), because that is what a PDF's text layer
often looks like to a parser. Page 2 of sample 6 is a bitmap with no text layer, standing in for
a scanned signature page.

reportlab and Pillow are development dependencies; the pipeline itself never imports them.
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

OUT = Path(__file__).parent / "inputs"
WIDTH, HEIGHT = letter
LEFT, RIGHT = 54.0, WIDTH - 54.0


class Page:
    """A cursor over one canvas: text goes where it is told, top down."""

    def __init__(self, path: Path) -> None:
        # invariant=1: no creation date and a fixed document id, so a rebuild changes no byte
        # unless the content changed.
        self.canvas = Canvas(str(path), pagesize=letter, invariant=1)
        self.y = HEIGHT - 54.0

    def text(self, x: float, text: str, *, font: str = "Helvetica", size: float = 10) -> None:
        self.canvas.setFont(font, size)
        self.canvas.drawString(x, self.y, text)

    def right(self, x: float, text: str, *, font: str = "Helvetica", size: float = 10) -> None:
        self.canvas.setFont(font, size)
        self.canvas.drawRightString(x, self.y, text)

    def centre(self, text: str, *, font: str = "Helvetica-Bold", size: float = 16) -> None:
        self.canvas.setFont(font, size)
        self.canvas.drawCentredString(WIDTH / 2, self.y, text)

    def down(self, points: float = 14.0) -> None:
        self.y -= points

    def rule(self, x0: float = LEFT, x1: float = RIGHT, *, width: float = 0.6) -> None:
        self.canvas.setLineWidth(width)
        self.canvas.line(x0, self.y, x1, self.y)

    def paragraph(
        self,
        x: float,
        text: str,
        *,
        width: float,
        font: str = "Helvetica",
        size: float = 10,
        leading: float = 13.0,
    ) -> None:
        """Left-aligned text wrapped at `width`, one drawn line per wrapped line."""
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if stringWidth(trial, font, size) > width and line:
                self.text(x, line, font=font, size=size)
                self.down(leading)
                line = word
            else:
                line = trial
        if line:
            self.text(x, line, font=font, size=size)
            self.down(leading)

    def leader(self, label: str, amount: str, *, font: str = "Helvetica", size: float = 10) -> None:
        """A label, a dotted leader and a right-aligned figure, as on a contract-sum summary."""
        self.text(LEFT, label, font=font, size=size)
        start = LEFT + stringWidth(label, font, size) + 6
        end = RIGHT - stringWidth(amount, font, size) - 6
        self.canvas.setDash(1, 3)
        self.canvas.setLineWidth(0.5)
        self.canvas.line(start, self.y + 2, end, self.y + 2)
        self.canvas.setDash()
        self.right(RIGHT, amount, font=font, size=size)
        self.down(15)

    def new_page(self) -> None:
        self.canvas.showPage()
        self.y = HEIGHT - 54.0

    def save(self) -> None:
        self.canvas.save()


def standard_form() -> None:
    """Sample 1: a clean owner/architect form with a contract-sum summary. The easy case."""
    page = Page(OUT / "co-01-standard-form.pdf")
    page.centre("CHANGE ORDER")
    page.down(16)
    page.canvas.setFont("Helvetica-Oblique", 8)
    page.canvas.drawCentredString(
        WIDTH / 2,
        page.y,
        "This Change Order is not valid until signed by the Architect, the Contractor and the Owner.",
    )
    page.down(20)
    page.rule()
    page.down(16)

    rows = [
        ("PROJECT:", "Harborview Elementary School – Classroom Wing Addition",
         "CHANGE ORDER NO.:", "004"),
        ("", "1420 Marina Way, Port Alder", "DATE:", "March 18, 2026"),
        ("OWNER:", "Port Alder Unified School District", "CONTRACT NO.:", "PAUSD-2025-0117"),
        ("CONTRACTOR:", "Castellan Builders, Inc.", "CONTRACT DATE:", "June 2, 2025"),
        ("ARCHITECT:", "Brightline Design Group", "", ""),
    ]
    for left_label, left_value, right_label, right_value in rows:
        page.text(LEFT, left_label, font="Helvetica-Bold", size=9)
        page.text(LEFT + 78, left_value, size=9)
        page.text(372, right_label, font="Helvetica-Bold", size=9)
        page.text(372 + 96, right_value, size=9)
        page.down(14)
    page.text(LEFT, "REQUESTED BY:", font="Helvetica-Bold", size=9)
    page.text(LEFT + 78, "Port Alder Unified School District", size=9)
    page.down(12)
    page.rule()
    page.down(18)

    page.text(LEFT, "The Contract is changed as follows:", font="Helvetica-Bold")
    page.down(16)
    for item in (
        "1. Furnish and install twelve (12) additional data drops in Classrooms 201–206 per "
        "Owner Bulletin No. 3.",
        "2. Upgrade corridor flooring from VCT to luxury vinyl tile (approx. 2,400 sq. ft.) per "
        "ASI-07.",
    ):
        page.paragraph(LEFT + 10, item, width=RIGHT - LEFT - 20)
        page.down(3)
    page.down(10)

    page.leader("The original Contract Sum was", "$ 4,860,000.00")
    page.leader("Net change by previously authorized Change Orders", "$ 73,215.00")
    page.leader("The Contract Sum prior to this Change Order was", "$ 4,933,215.00")
    page.leader(
        "The Contract Sum will be increased by this Change Order in the amount of",
        "$ 38,640.00",
        font="Helvetica-Bold",
    )
    page.leader("The new Contract Sum including this Change Order will be", "$ 4,971,855.00")
    page.down(6)
    page.paragraph(
        LEFT,
        "The Contract Time will be increased by fourteen (14) calendar days. The date of "
        "Substantial Completion as of the date of this Change Order therefore is August 28, 2026.",
        width=RIGHT - LEFT,
    )
    page.down(14)
    page.rule()
    page.down(16)

    columns = [
        (LEFT, "ARCHITECT", "Brightline Design Group", "Dana Whitlock", "03/19/2026"),
        (LEFT + 172, "CONTRACTOR", "Castellan Builders, Inc.", "Marcus Oyelaran", "03/20/2026"),
        (LEFT + 344, "OWNER", "Port Alder Unified School District", "Helen Tran", "03/24/2026"),
    ]
    top = page.y
    for x, role, firm, name, signed in columns:
        page.y = top
        page.text(x, role, font="Helvetica-Bold", size=9)
        page.down(13)
        page.text(x, firm, size=8)
        page.down(13)
        page.text(x, f"By: {name}", size=9)
        page.down(13)
        page.text(x, f"Date: {signed}", size=9)
    page.down(24)
    page.text(LEFT, "STATUS: APPROVED – fully executed by all parties.", font="Helvetica-Bold")
    page.save()


def deductive_credit() -> None:
    """Sample 3: a credit. Figures in parentheses, a shortened schedule, and a header whose two
    blocks are written to the page one after the other, not row by row."""
    page = Page(OUT / "co-03-deductive-credit.pdf")
    serif, bold = "Times-Roman", "Times-Bold"

    # The left block first, then the right block: out of reading order on purpose.
    top = page.y
    for line, font, size in (
        ("NORTHGATE CONSTRUCTORS LLC", bold, 13),
        ("2200 Foundry Road, Suite 400", serif, 9),
        ("General Contractor", serif, 9),
    ):
        page.text(LEFT, line, font=font, size=size)
        page.down(14)
    page.y = top
    for line, font, size in (
        ("CHANGE ORDER", bold, 15),
        ("No. CO-011 (Deductive)", serif, 11),
        ("Date: 12-Jan-2026", serif, 11),
    ):
        page.right(RIGHT, line, font=font, size=size)
        page.down(16)
    page.down(6)
    page.rule()
    page.down(18)

    for label, value in (
        ("Project:", "Riverbend Logistics Center – Building C"),
        ("Owner:", "Calloway Industrial Properties"),
        ("Contract / Agreement No.:", "CIP-RLC-0932"),
        ("Initiated by:", "Calloway Industrial Properties (Owner)"),
    ):
        page.text(LEFT, label, font=bold, size=10.5)
        page.text(LEFT + 132, value, font=serif, size=10.5)
        page.down(15)
    page.down(8)

    page.text(LEFT, "Description of Change:", font=bold, size=10.5)
    page.down(15)
    page.paragraph(
        LEFT,
        "Delete the mezzanine office build-out (Rooms M-101 through M-108) from the Contract, "
        "including metal stud framing, gypsum board, acoustical ceilings, doors and hardware, and "
        "associated electrical rough-in. Owner will complete this work under a separate "
        "tenant-improvement contract.",
        width=RIGHT - LEFT,
        font=serif,
        size=10.5,
        leading=14,
    )
    page.down(10)

    page.text(LEFT, "Cost Breakdown", font=bold, size=10.5)
    page.down(15)
    page.text(LEFT + 8, "Item", font=bold, size=10)
    page.right(RIGHT - 8, "Amount", font=bold, size=10)
    page.down(5)
    page.rule()
    page.down(14)
    for label, amount in (
        ("Framing and gypsum board (credit)", "(61,200.00)"),
        ("Acoustical ceilings (credit)", "(18,450.00)"),
        ("Doors, frames and hardware (credit)", "(22,300.00)"),
        ("Electrical rough-in (credit)", "(14,875.00)"),
        ("Subtotal", "(116,825.00)"),
        ("Contractor's retained OH&P (5%)", "5,841.25"),
    ):
        page.text(LEFT + 8, label, font=serif, size=10)
        page.right(RIGHT - 8, amount, font=serif, size=10)
        page.down(14)
    page.rule()
    page.down(15)
    page.text(LEFT + 8, "NET CHANGE – DEDUCT", font=bold, size=11)
    page.right(RIGHT - 8, "($110,983.75)", font=bold, size=11)
    page.down(24)

    for label, amount in (
        ("Original Contract Sum", "$12,480,000.00"),
        ("Net change by prior Change Orders", "$264,310.00"),
        ("Contract Sum prior to this Change Order", "$12,744,310.00"),
        ("Revised Contract Sum", "$12,633,326.25"),
    ):
        page.text(LEFT + 8, label, font=serif, size=10)
        page.right(RIGHT - 8, amount, font=serif, size=10)
        page.down(14)
    page.down(10)

    page.paragraph(
        LEFT,
        "Contract Time: The date of Substantial Completion is advanced; the Contract Time is "
        "decreased by ten (10) calendar days.",
        width=RIGHT - LEFT,
        font=serif,
        size=10.5,
        leading=14,
    )
    page.down(14)
    page.text(LEFT, "Accepted:", font=bold, size=10.5)
    page.down(16)
    for party, signer, signed in (
        ("Owner: Calloway Industrial Properties", "/s/ R. Calloway", "15-Jan-2026"),
        ("Contractor: Northgate Constructors LLC", "/s/ L. Ferreira", "14-Jan-2026"),
    ):
        page.text(LEFT, party, font=serif, size=10)
        page.text(LEFT + 236, f"Signed: {signer}", font=serif, size=10)
        page.text(LEFT + 392, f"Date: {signed}", font=serif, size=10)
        page.down(15)
    page.save()


def subcontractor_multi_amount() -> None:
    """Sample 4: a subcontractor's priced proposal that the general contractor approved for less.
    A line-item table written to the page column by column, a proposed total, an approved amount,
    a revised subcontract value, and a date that reads two ways."""
    page = Page(OUT / "co-04-subcontractor-multi-amount.pdf")
    mono, bold = "Courier", "Courier-Bold"

    page.text(LEFT, "DELMAR ELECTRIC CO.", font=bold, size=12)
    page.right(RIGHT, "CHANGE ORDER PROPOSAL / AUTHORIZATION", font=bold, size=10)
    page.down(16)
    page.text(LEFT, "Job: 2214  Eastgate Transit Plaza – Parking Structure P2", font=mono, size=9)
    page.right(RIGHT, "CO # 9", font=bold, size=10)
    page.down(13)
    page.text(LEFT, "GC Subcontract No. ETP-SC-0457", font=mono, size=9)
    page.right(RIGHT, "Rev. 2", font=mono, size=9)
    page.down(13)
    page.text(LEFT, "Date 03/04/2026", font=mono, size=9)
    page.down(16)
    page.rule()
    page.down(15)

    page.paragraph(
        LEFT,
        "Requested by: Hallmark General Contractors – J. Ortega (Project Engineer), "
        "RFI-112 response",
        width=RIGHT - LEFT,
        font=mono,
        size=9,
        leading=12,
    )
    page.down(6)
    page.paragraph(
        LEFT,
        "Work description: Provide and install EV charging infrastructure at Level 1: (8) "
        "dual-port Level 2 chargers, 400A distribution panel EV1, feeders from switchboard MSB, "
        "underground conduit and trenching (approx. 310 LF), and bollards per sketch SK-E-14.",
        width=RIGHT - LEFT,
        font=mono,
        size=9,
        leading=12,
    )
    page.down(10)

    # The table, one column at a time: every description, then every quantity, and so on.
    headers = ("Item", "Description", "Qty", "Unit", "Unit price", "Extended")
    xs = (LEFT, LEFT + 34, 318.0, 356.0, 470.0, RIGHT)
    right_aligned = (False, False, True, False, True, True)
    items = (
        ("1", "Chargers, dual-port L2", "8", "EA", "4,150.00", "33,200.00"),
        ("2", "Panel EV1 400A", "1", "EA", "7,900.00", "7,900.00"),
        ("3", "Feeders & conduit", "310", "LF", "46.00", "14,260.00"),
        ("4", "Trenching / backfill", "310", "LF", "38.50", "11,935.00"),
        ("5", "Bollards", "16", "EA", "285.00", "4,560.00"),
    )
    top = page.y
    for column, (header, x, align_right) in enumerate(zip(headers, xs, right_aligned, strict=True)):
        page.y = top
        draw = page.right if align_right else page.text
        draw(x, header, font=bold, size=9)
        page.down(14)
        for row in items:
            draw(x, row[column], font=mono, size=9)
            page.down(13)
    page.y = top - 6
    page.rule()
    page.y = top - 14 - 13 * len(items) + 4
    page.rule()
    page.down(14)

    for label, amount, font in (
        ("Subtotal", "71,855.00", mono),
        ("Overhead & profit 12%", "8,622.60", mono),
        ("Sales tax on material (8.25% of 45,660.00)", "3,766.95", mono),
        ("Bond 1%", "842.45", mono),
        ("TOTAL PROPOSED", "85,087.00", bold),
    ):
        page.text(LEFT + 150, label, font=font, size=9)
        page.right(RIGHT, amount, font=font, size=9)
        page.down(13)
    page.down(8)

    page.paragraph(
        LEFT,
        "GC REVIEW: Unit price for trenching reduced to $31.00/LF per subcontract unit rates; "
        "bond not reimbursable.",
        width=RIGHT - LEFT,
        font=mono,
        size=9,
        leading=12,
    )
    page.down(4)
    for label, amount, font in (
        ("AMOUNT APPROVED:", "$81,640.55", bold),
        ("Previous subcontract value:", "$1,946,300.00", mono),
        ("Revised subcontract value:", "$2,027,940.55", mono),
    ):
        page.text(LEFT, label, font=font, size=9.5)
        page.right(LEFT + 300, amount, font=font, size=9.5)
        page.down(13)
    page.down(8)
    page.text(
        LEFT,
        "Time extension requested: 6 working days. Time extension granted: 4 working days.",
        font=mono,
        size=9,
    )
    page.down(22)

    page.text(LEFT, "AUTHORIZATION", font=bold, size=10)
    page.down(15)
    page.text(LEFT, "[X] Approved as noted   [ ] Rejected   [ ] Resubmit", font=mono, size=9.5)
    page.down(15)
    page.text(
        LEFT,
        "Hallmark General Contractors   By: Karen Iwasaki, Project Manager   Date 03/09/2026",
        font=mono,
        size=9,
    )
    page.save()


def _handwriting(draw: ImageDraw.ImageDraw, rng: random.Random, x: int, y: int, length: int) -> None:
    """A scribble that reads as a signature: a wandering stroke with loops."""
    points = []
    for step in range(length):
        t = step / 6.0
        points.append(
            (
                x + step * 3.2 + 9 * math.sin(t * 1.7) + rng.uniform(-1.5, 1.5),
                y + 16 * math.sin(t * 2.3 + 0.6) * math.cos(t * 0.45) + rng.uniform(-1.5, 1.5),
            )
        )
    draw.line(points, fill=25, width=3, joint="curve")


def _signature_scan() -> Image.Image:
    """Page 2 of sample 6 as a scanner would deliver it: a greyscale bitmap, slightly skewed."""
    rng = random.Random(23)
    image = Image.new("L", (1275, 1650), 246)
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=38)
    body = ImageFont.load_default(size=26)
    small = ImageFont.load_default(size=22)

    draw.text((110, 130), "CHANGE ORDER CO-023", font=title, fill=30)
    draw.text((110, 185), "Stonebridge Residences - Tower B", font=body, fill=40)
    draw.text((930, 140), "Page 2 of 2", font=small, fill=60)
    draw.line([(110, 240), (1165, 240)], fill=60, width=2)
    draw.text((110, 290), "ACCEPTED AND APPROVED", font=title, fill=30)
    draw.text(
        (110, 350),
        "The parties agree to the change in Contract price and time on page 1.",
        font=small,
        fill=50,
    )

    for top, party, name, signed in (
        (470, "OWNER: Stonebridge Urban Living LP", "A. Mercier, Development Director", "11 May 2026"),
        (790, "CONTRACTOR: Kestrel & Vance Construction Ltd.", "P. Vance, Principal", "8 May 2026"),
    ):
        draw.text((110, top), party, font=body, fill=35)
        draw.line([(110, top + 150), (700, top + 150)], fill=70, width=2)
        _handwriting(draw, rng, 150, top + 105, 150)
        draw.text((110, top + 162), name, font=small, fill=55)
        draw.text((780, top + 110), "Date:", font=body, fill=35)
        draw.line([(860, top + 150), (1165, top + 150)], fill=70, width=2)
        # The date, "handwritten": drawn larger, then nudged and blurred below.
        draw.text((880 + rng.randint(-6, 6), top + 102), signed, font=body, fill=20)

    stamp = Image.new("L", (430, 170), 0)
    ink = ImageDraw.Draw(stamp)
    ink.rectangle([(4, 4), (425, 165)], outline=255, width=7)
    ink.text((62, 28), "APPROVED", font=ImageFont.load_default(size=58), fill=255)
    ink.text((112, 108), "11 MAY 2026", font=ImageFont.load_default(size=30), fill=255)
    stamp = stamp.rotate(-11, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(95, (720, 1180), stamp)

    for _ in range(900):  # the dust a flatbed leaves
        px, py = rng.randrange(1275), rng.randrange(1650)
        image.putpixel((px, py), rng.randrange(120, 215))
    image = image.rotate(0.9, resample=Image.Resampling.BICUBIC, fillcolor=243)
    return image.filter(ImageFilter.GaussianBlur(0.7))


def scanned_signature_page() -> None:
    """Sample 6: page 1 has a text layer; page 2, where the approval is, is a scan."""
    page = Page(OUT / "co-06-scanned-signature-page.pdf")
    page.text(LEFT, "CHANGE ORDER", font="Helvetica-Bold", size=15)
    page.right(RIGHT, "Page 1 of 2", size=9)
    page.down(22)
    rows = [
        ("Stonebridge Residences – Tower B", "Change Order: CO-023"),
        ("Owner: Stonebridge Urban Living LP", "Issued: 7 May 2026"),
        ("Contractor: Kestrel & Vance Construction Ltd.", "Contract: SUL-TB-GC-001"),
    ]
    for left_text, right_text in rows:
        page.text(LEFT, left_text, size=10)
        page.right(RIGHT, right_text, size=10)
        page.down(14)
    page.down(4)
    page.rule()
    page.down(18)

    page.text(LEFT, "Change requested by: Stonebridge Urban Living LP – Design Bulletin DB-19")
    page.down(22)
    page.text(LEFT, "Scope:", font="Helvetica-Bold")
    page.down(15)
    page.paragraph(
        LEFT,
        "Revise Level 14 amenity terrace: replace specified concrete pavers with porcelain pavers "
        "on adjustable pedestals (approx. 3,600 sq ft), add two (2) gas fire tables with dedicated "
        "gas line, and add in-slab conduit for future terrace lighting.",
        width=RIGHT - LEFT,
    )
    page.down(12)

    page.text(LEFT, "Summary of cost:", font="Helvetica-Bold")
    page.down(16)
    for label, amount in (
        ("Pavers and pedestals – material and installation", "41,870.00"),
        ("Gas fire tables and piping", "17,340.00"),
        ("In-slab conduit", "4,125.00"),
        ("Credit for specified concrete pavers", "(12,600.00)"),
        ("General conditions (5 days)", "3,750.00"),
        ("Fee (6%)", "3,269.10"),
    ):
        page.text(LEFT + 12, label)
        page.right(RIGHT - 60, amount)
        page.down(14)
    page.rule(LEFT + 12, RIGHT - 60)
    page.down(15)
    page.text(LEFT + 12, "Change Order total (add)", font="Helvetica-Bold")
    page.right(RIGHT - 60, "$57,754.10", font="Helvetica-Bold")
    page.down(24)

    for label, amount in (
        ("Contract price before this Change Order", "$31,912,480.00"),
        ("Contract price after this Change Order", "$31,970,234.10"),
    ):
        page.text(LEFT + 12, label)
        page.right(RIGHT - 60, amount)
        page.down(14)
    page.down(12)

    page.paragraph(
        LEFT,
        "Schedule: The Substantial Performance date is extended by 5 working days, from "
        "30 October 2026 to 6 November 2026.",
        width=RIGHT - LEFT,
    )
    page.down(10)
    page.paragraph(
        LEFT,
        "Authorization: signatures of the Owner and the Contractor appear on page 2.",
        width=RIGHT - LEFT,
    )

    page.new_page()
    scan = io.BytesIO()
    _signature_scan().save(scan, format="PNG", optimize=False)
    scan.seek(0)
    page.canvas.drawImage(ImageReader(scan), 0, 0, width=WIDTH, height=HEIGHT)
    page.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (standard_form, deductive_credit, subcontractor_multi_amount, scanned_signature_page):
        build()
    for path in sorted(OUT.glob("*.pdf")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
