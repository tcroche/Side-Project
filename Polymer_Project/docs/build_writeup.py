"""
build_writeup.py -- generates the one-page Polymer Tech Expo write-up (PDF).

Constraints from the brief: max 1 page, PDF, minimum font size 11 pt (applied
to EVERY text element, diagram and table included).

    python build_writeup.py            -> Theo_Crochemar_Universite_Paris_1_Pantheon-Sorbonne.pdf

Every figure quoted in the text is produced by a script in the repository:
run_real_case.py (M2 deflation, concentration, PBO), run_bench.py --llm
(benchmark table, self-audit block), pytest (test count).
"""

from __future__ import annotations

import base64
import os
import re
import zlib

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = os.environ.get(
    "WRITEUP_OUT", "Theo_Crochemar_Universite_Paris_1_Pantheon-Sorbonne.pdf"
)
# Principle 7 (disclose what is incomplete), enforced here rather than promised:
# the "one command, one report" wording was gated until the report layer existed.
# run_report.py shipped on 2026-08-15, so the claim is now true and the default
# flipped. BIA_ONE_COMMAND=0 restores the pre-B5 wording.
ONE_COMMAND = os.environ.get("BIA_ONE_COMMAND", "1") == "1"

# --- fonts ------------------------------------------------------------------
# Carlito is metric-compatible with Calibri, so either produces the SAME line
# breaks and therefore the same one-page fit. The path used to be hard-coded to
# a Linux directory, which meant this script could not run on the machine that
# owns the project. It now searches, in order: an explicit override, Carlito
# where Linux and LibreOffice put it, then Calibri on Windows and macOS.
#
# A font that is not metric-compatible would silently reflow the page. It is not
# silent here: verify_output() re-reads the PDF and fails if the result is not
# exactly one page with nothing below 11 pt, so a wrong font is caught by the
# build rather than by a reader.

FONT_CANDIDATES = (
    # (regular, bold, italic, bold-italic) tried in order
    ("/usr/share/fonts/truetype/crosextra/Carlito-{}.ttf",
     ("Regular", "Bold", "Italic", "BoldItalic")),
    (os.path.expanduser("~/.local/share/fonts/Carlito-{}.ttf"),
     ("Regular", "Bold", "Italic", "BoldItalic")),
    ("C:/Program Files/LibreOffice/share/fonts/truetype/Carlito-{}.ttf",
     ("Regular", "Bold", "Italic", "BoldItalic")),
    ("C:/Windows/Fonts/calibri{}.ttf", ("", "b", "i", "z")),
    ("/System/Library/Fonts/Supplemental/Calibri{}.ttf", ("", " Bold", " Italic",
                                                          " Bold Italic")),
)
FONT_ENV = "BIA_WRITEUP_FONT_DIR"


def _resolve_font_files() -> tuple[str, str, str, str]:
    """Return the four faces, or explain exactly what to install."""
    override = os.environ.get(FONT_ENV)
    candidates = list(FONT_CANDIDATES)
    if override:
        candidates.insert(0, (os.path.join(override, "Carlito-{}.ttf"),
                              ("Regular", "Bold", "Italic", "BoldItalic")))
    tried = []
    for pattern, suffixes in candidates:
        paths = tuple(pattern.format(s) for s in suffixes)
        tried.extend(paths)
        if all(os.path.isfile(p) for p in paths):
            return paths
    raise SystemExit(
        "No Calibri-metric font found, so the write-up cannot be built with the "
        "layout it was designed for.\n"
        "Install Carlito (Linux: fonts-crosextra-carlito; anywhere: it ships with "
        "LibreOffice), or set "
        f"{FONT_ENV} to a directory holding Carlito-Regular.ttf and its Bold, "
        "Italic and BoldItalic faces.\nTried:\n  " + "\n  ".join(tried)
    )


_regular, _bold, _italic, _bolditalic = _resolve_font_files()
pdfmetrics.registerFont(TTFont("Carlito", _regular))
pdfmetrics.registerFont(TTFont("Carlito-Bold", _bold))
pdfmetrics.registerFont(TTFont("Carlito-Italic", _italic))
pdfmetrics.registerFont(TTFont("Carlito-BoldItalic", _bolditalic))
pdfmetrics.registerFontFamily(
    "Carlito", normal="Carlito", bold="Carlito-Bold",
    italic="Carlito-Italic", boldItalic="Carlito-BoldItalic",
)

BODY_PT = 11.0          # the brief's minimum; nothing on the page is smaller
# The graphics renderer's initial (unused) font state defaults to 10 pt; raise it
# so that no Tf operator below 11 pt appears anywhere in the PDF, drawn or not.
import reportlab.graphics.shapes as _shapes
_shapes.STATE_DEFAULTS["fontName"] = "Carlito"
_shapes.STATE_DEFAULTS["fontSize"] = BODY_PT
NAVY = colors.HexColor("#1F3A5F")
RULE = colors.HexColor("#9DB0C7")
INK = colors.HexColor("#1A1A1A")
LANE_STAT = colors.HexColor("#E8EEF7")
LANE_CODE = colors.HexColor("#EAF3EA")
LANE_OUT = colors.HexColor("#F6EFE3")

styles = {
    "title": ParagraphStyle(
        "title", fontName="Carlito-Bold", fontSize=14, leading=16.5,
        textColor=NAVY, spaceAfter=1,
    ),
    "sub": ParagraphStyle(
        "sub", fontName="Carlito", fontSize=BODY_PT, leading=13,
        textColor=INK, spaceAfter=3,
    ),
    "h": ParagraphStyle(
        "h", fontName="Carlito-Bold", fontSize=BODY_PT + 0.5, leading=13.5,
        textColor=NAVY, spaceBefore=3.8, spaceAfter=1.0,
    ),
    "body": ParagraphStyle(
        "body", fontName="Carlito", fontSize=BODY_PT, leading=12.4,
        textColor=INK, alignment=TA_JUSTIFY,
    ),
    "cell": ParagraphStyle(
        "cell", fontName="Carlito", fontSize=BODY_PT, leading=12.4, textColor=INK,
    ),
    "cellb": ParagraphStyle(
        "cellb", fontName="Carlito-Bold", fontSize=BODY_PT, leading=12.4, textColor=INK,
    ),
}


class HRule(Flowable):
    def __init__(self, width, thickness=0.6, color=RULE, space=2):
        super().__init__()
        self.width, self.thickness, self.color, self.space = width, thickness, color, space

    def wrap(self, aw, ah):
        return self.width, self.thickness + self.space

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space / 2, self.width, self.space / 2)


# ---------------------------------------------------------------------------
# Architecture diagram (all strings at 11 pt)
# ---------------------------------------------------------------------------


def lane_box(d, x, y, w, h, fill, title, lines):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=RULE, strokeWidth=0.6, rx=3, ry=3))
    ty = y + h - 13
    d.add(String(x + 6, ty, title, fontName="Carlito-Bold", fontSize=BODY_PT, fillColor=NAVY))
    for line in lines:
        ty -= 12.6
        d.add(String(x + 6, ty, line, fontName="Carlito", fontSize=BODY_PT, fillColor=INK))


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=NAVY, strokeWidth=1.1))
    # arrow head pointing along +x
    d.add(Polygon([x2, y2, x2 - 6, y2 + 3.2, x2 - 6, y2 - 3.2],
                  fillColor=NAVY, strokeColor=NAVY, strokeWidth=0.5))


def architecture(width):
    h = 68
    d = Drawing(width, h)
    gap = 26
    w_in = 118
    w_mid = 196
    w_out = width - w_in - w_mid - 2 * gap
    x_in, x_mid, x_out = 0, w_in + gap, w_in + gap + w_mid + gap

    lane_box(d, x_in, 4, w_in, h - 8, colors.white, "Backtest under audit",
             ["trial returns (T x N)", "strategy source code", "the M2 case, or yours"])

    top_h = 33
    lane_box(d, x_mid, h - top_h, w_mid, top_h, LANE_STAT, "Statistical deflation",
             ["PSR, DSR, E[max SR], MinTRL, PBO"])
    lane_box(d, x_mid, 0, w_mid, top_h, LANE_CODE, "Code leakage, two detectors",
             ["8 AST rules  |  LLM pass, grounded"])

    lane_box(d, x_out, 4, w_out, h - 8, LANE_OUT, "Audit report",
             ["probabilities in [0, 1], labelled",
              "findings with verified lines",
              "LLM section kept separate"])

    ymid, y_top, y_bot = h / 2, h - top_h / 2, top_h / 2
    # left bus: input -> both lanes
    bx = x_in + w_in + gap / 2
    d.add(Line(x_in + w_in, ymid, bx, ymid, strokeColor=NAVY, strokeWidth=1.1))
    d.add(Line(bx, y_bot, bx, y_top, strokeColor=NAVY, strokeWidth=1.1))
    arrow(d, bx, y_top, x_mid - 1, y_top)
    arrow(d, bx, y_bot, x_mid - 1, y_bot)
    # right bus: both lanes -> report
    rx = x_mid + w_mid + gap / 2
    d.add(Line(x_mid + w_mid, y_top, rx, y_top, strokeColor=NAVY, strokeWidth=1.1))
    d.add(Line(x_mid + w_mid, y_bot, rx, y_bot, strokeColor=NAVY, strokeWidth=1.1))
    d.add(Line(rx, y_bot, rx, y_top, strokeColor=NAVY, strokeWidth=1.1))
    arrow(d, rx, ymid, x_out - 1, ymid)
    return d


class DrawingFlowable(Flowable):
    def __init__(self, drawing):
        super().__init__()
        self.d = drawing

    def wrap(self, aw, ah):
        return self.d.width, self.d.height

    def draw(self):
        renderPDF.draw(self.d, self.canv, 0, 0)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

P = lambda text, style="body": Paragraph(text, styles[style])  # noqa: E731

TITLE = "Backtest Integrity Auditor (BIA): deflate the numbers, read the code that produced them"
#: The test count is read from the environment so the PDF cannot quietly
#: disagree with the suite: BIA_TEST_COUNT=$(pytest -q | ...) before building,
#: or accept the default recorded at the last verified run.
TEST_COUNT = os.environ.get("BIA_TEST_COUNT", "288")
SUBTITLE = ("Théo Crochemar, M.Sc. Applied Mathematics &amp; Quantitative Finance, "
            "Université Paris 1 Panthéon-Sorbonne. Polymer Tech Expo 2026. "
            f"Python, {TEST_COUNT} tests, GitHub: tcroche.")

PROBLEM = (
    "A backtest can be wrong in two independent ways. Statistically: when many "
    "configurations are tried and the best is reported, the in-sample Sharpe ratio is "
    "inflated by selection, a bias Bailey and López de Prado quantified (Deflated "
    "Sharpe Ratio, Probability of Backtest Overfitting) but that few pipelines compute. "
    "In code: look-ahead leaks such as a negative shift, a centred window or a scaler "
    "fitted before the split feed the future into the signal, and finding them is "
    "manual review. My own M.Sc. systematic-trading project showed both: an in-sample "
    "Sharpe of 1.9 built on 18 tuned configurations, and a write-up in which I had "
    "written that it \u201ccollapsed to a deflated 0.92\u201d, treating a DSR, a probability, "
    "as a Sharpe ratio. The tool exists to catch exactly that, in the numbers and in "
    "the prose."
)

SOLUTION = (
    ("BIA is a Python tool that audits a backtest on two axes and writes one report. "
     if ONE_COMMAND else
     "BIA is a Python tool that audits a backtest on two axes and reports on each. ")
    + "The statistical axis takes the matrix of trial returns and computes PSR, "
    "E[max SR], DSR, MinTRL and CSCV/PBO with its simulated null, plus concentration "
    "diagnostics: share of "
    "P&amp;L in the best day, Sharpe without it, correlation between trials. The code "
    "axis scans the strategy source with eight deterministic AST rules (exact lines), "
    "then a language-model pass for what syntax cannot express, such as "
    "a same-day aggregate broadcast to every intraday bar. The two detectors are never "
    "merged. On my M2 case the report "
    "reproduces the original DSR (0.929 with the Russell 2000, 0.378 without), finds 70% of the in-sample profit in a single session (2025-04-09) with "
    "18 of 18 grid cells drawing more than half their P&amp;L from that day, and no leak "
    "in the code (0 rule detections; the semantic pass raised only questions, each "
    "resolved by the engine's own convention): the code was clean, the leak was in the "
    "selection."
)

USE_OF_AI = (
    "AI plays two roles, kept apart. <b>As a component</b>: Claude (claude-sonnet-4-6, "
    "Anthropic API, temperature 0, JSON only) performs the semantic code pass. The "
    "prompt lives in a versioned YAML registry with a changelog (v1.0.0 to v1.2.0). "
    "Its system prompt says, in short: <i>\u201cReport only leakage that no syntactic rule "
    "can express. Every finding must cite line numbers that exist. Every finding "
    "carries an external_dependency field: null only if the leak is established within "
    "this file alone; otherwise name the external fact to check, set severity to "
    "review, and make the fix conditional on it.\u201d</i> The harness trusts none of it: "
    "line numbers are checked against the file, snippets come from the real source, "
    "and severity is capped at \u201creview\u201d in code unless the model declares the "
    "leak self-contained. Responses are cached by SHA-256 of the exact prompt pair. "
    "<b>As a coding assistant</b>: "
    "Claude in the chat interface produced scaffolds, first implementations and tests; "
    "I checked every formula against the papers and kept a dated DEVLOG of prompts, "
    "iterations and mistakes."
)

IMPACT_INTRO = (
    "For a multi-manager platform, BIA is a pre-mortem on any strategy submission: "
    + ("one command answers " if ONE_COMMAND else "its two scripts answer ")
    + "how many trials were run, how concentrated the P&amp;L is and "
    "whether the code sees the future. Measured "
    "on a seeded benchmark of 16 files (11 leaks, 6 controls), plus the tool's own "
    "source and my real strategy code audited live in the same run:"
)

IMPACT_OUTRO = (
    "0 invented line numbers over 16 files; 0 findings on the tool's 28 files and the "
    "16 M2 files. Each half catches what the other cannot: that complementarity, not "
    "round numbers on a small benchmark we wrote ourselves, is the result."
)

REFLECTIONS = (
    "Three lessons, each with a scar. <b>A prompt rule is a request, code is a "
    "guarantee</b>: prompt v1.1.0 asked the model to cap cross-file-conditional "
    "findings at \u201creview\u201d; on real code it returned \u201cmedium\u201d and a fix "
    "that would have delayed a causal signal, so the cap moved into the harness. "
    "<b>Negative instructions are weakly obeyed</b>: the model re-reported 6 "
    "of 8 patterns it was told to leave to the rules; the deterministic layer is the "
    "floor. <b>Benchmarks audit their author</b>: seeding trapped files "
    "exposed a precision gap in one rule and naming artefacts in my own cases. "
    "Challenges: unit discipline (DSR is a probability, and a guard raises on the "
    "mix-up), reproducing published numbers to the digit, and a PBO whose own null has "
    "standard deviation 0.21 at T = 103. Next: "
    + ("a UI layer, " if ONE_COMMAND else "a one-command report and UI layer, ")
    + "validation on third-party code with independently known leaks, and run-to-run "
    "variance of the semantic pass."
)


def bench_table(width):
    rows = [
        [P("Detector", "cellb"), P("Syntactic leaks (9)", "cellb"),
         P("Semantic leaks (2)", "cellb"), P("FP on 6 control files", "cellb")],
        [P("AST rules", "cell"), P("9 / 9", "cell"), P("0 / 2", "cell"), P("0", "cell")],
        [P("LLM pass (prompt v1.2.0)", "cell"), P("7 / 9", "cell"), P("2 / 2", "cell"), P("0", "cell")],
        [P("Combined", "cellb"), P("9 / 9", "cellb"), P("2 / 2", "cellb"), P("0", "cellb")],
    ]
    col = [width * 0.32, width * 0.22, width * 0.22, width * 0.24]
    t = Table(rows, colWidths=col, rowHeights=[13.6] * 4)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), LANE_STAT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Carlito"),
        ("FONTSIZE", (0, 0), (-1, -1), BODY_PT),
    ]))
    return t


def build() -> str:
    left = right = 12 * mm
    top, bottom = 9 * mm, 8 * mm
    doc = SimpleDocTemplate(
        OUT, pagesize=A4, leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom,
        title="Backtest Integrity Auditor, Polymer Tech Expo 2026 write-up",
        author="Théo Crochemar",
        initialFontName="Carlito", initialFontSize=BODY_PT, initialLeading=12.4,
    )
    width = A4[0] - left - right

    story = [
        P(TITLE, "title"),
        P(SUBTITLE, "sub"),
        HRule(width, thickness=1.0, color=NAVY, space=3),

        P("Problem statement", "h"), P(PROBLEM),
        P("Solution overview", "h"), P(SOLUTION),
        Spacer(1, 3), DrawingFlowable(architecture(width)), Spacer(1, 1),
        P("Use of AI", "h"), P(USE_OF_AI),
        P("Impact &amp; value", "h"), P(IMPACT_INTRO),
        Spacer(1, 2), bench_table(width), Spacer(1, 2), P(IMPACT_OUTRO),
        P("Reflections", "h"), P(REFLECTIONS),
    ]
    doc.build(story)
    return OUT


# ---------------------------------------------------------------------------
# The brief's two hard constraints, checked against the artefact
# ---------------------------------------------------------------------------

MIN_PT = 11.0
# Font resources are named like /F1+0, so the name class cannot be \w.
_TF = re.compile(rb"/[^\s/]+\s+([\d.]+)\s+Tf")


def _page_streams(raw: bytes) -> list[bytes]:
    """Content streams of a PDF, decoded.

    reportlab writes page content as ASCII85 over Flate, so both layers have to
    come off before any operator is visible. Deliberately small: the only PDF
    this function ever reads is the one the code above just wrote.
    """
    out = []
    for match in re.finditer(rb"stream(.*?)endstream", raw, re.S):
        data = match.group(1).lstrip(b"\r\n")
        stripped = data.strip()
        if stripped.endswith(b"~>"):
            try:
                data = base64.a85decode(stripped, adobe=True)
            except ValueError:
                pass
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass
        out.append(data)
    return out


def verify_output(path: str = None) -> dict:
    """Re-read the generated PDF and enforce the brief: one page, nothing under
    11 pt. Raises SystemExit with the offending sizes rather than printing a
    reassuring line, because a write-up that silently ran to two pages would be
    rejected without appeal."""
    path = path or OUT
    with open(path, "rb") as handle:
        raw = handle.read()

    n_pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    sizes = sorted({round(float(m.group(1)), 2)
                    for stream in _page_streams(raw)
                    for m in _TF.finditer(stream)})
    too_small = [s for s in sizes if s < MIN_PT]

    problems = []
    if n_pages != 1:
        problems.append(f"the write-up is {n_pages} page(s); the brief allows 1")
    if too_small:
        problems.append(f"font size(s) below {MIN_PT:g} pt: {too_small}")
    if problems:
        raise SystemExit("Write-up rejected by its own check:\n  "
                         + "\n  ".join(problems))
    return {"pages": n_pages, "font_sizes_pt": sizes, "path": path,
            "bytes": len(raw)}


if __name__ == "__main__":
    path = build()
    result = verify_output(path)
    print(f"{path}")
    print(f"  pages          : {result['pages']}  (the brief allows 1)")
    print(f"  font sizes, pt : {result['font_sizes_pt']}  (minimum allowed 11)")
    print(f"  size           : {result['bytes'] / 1024:.0f} KB")
