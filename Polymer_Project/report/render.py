"""
Self-contained HTML rendering of an audit report.

Design constraints, all of them deliberate:

* ONE file, no external asset, no CDN, no web font, no script. The report opens
  offline, can be mailed, and prints to PDF from any browser (Ctrl+P, A4 rules
  below). `tests/test_report.py` asserts the absence of every external hook
  rather than trusting this docstring.
* Every string that comes from source code or from a model is HTML-escaped.
  Findings quote real code, and real code contains `<`, `>` and `&`; an
  unescaped snippet would silently corrupt the page. There is a test for it.
* The three sections are rendered by three separate functions and never share
  a data structure. Corroboration is an annotation on the semantic side only.
* Probabilities are labelled as probabilities everywhere they appear. PSR, DSR
  and PBO are in [0, 1] and are never described as Sharpe ratios.
* Anything not run is stated as not run, with the reason, rather than being
  silently absent from the page.

The visual system, and why none of it is decoration
---------------------------------------------------
Four kinds of evidence appear on this page and must never be read as one:

    measured   solid rule, verification tick   arithmetic in this repository
    parsed     solid rule, verification tick   Python AST, exact line numbers
    model      DASHED rule, no tick            a question, not a measurement
    unseen     no rule, no tick                outside any static analysis

The tiers are separated by LINE QUALITY before colour, because line quality
survives grayscale printing, photocopying and colour blindness. The tick is the
auditor's "vouched to source" mark, drawn with CSS borders so it needs no glyph
and no font file, and it is deliberately absent from the model tier.

Colour is rationed to one job each: red marks a failing verdict and nothing
else, slate marks the model tier and nothing else. Everything else is ink.

Punctuation: nothing this module writes uses an em dash or an en dash, so the
page reads the same way to a French and an English reader. Text quoted from a
model is NOT touched: a finding's explanation is evidence and is reproduced
character for character, dashes included. `tests/test_report.py` pins both
halves of that rule.

EVERY MARK PLOTS A NUMBER THE TOOL COMPUTED. The two charts are drawn here, in
Python, from the fields of the DeflationSection they sit next to: the dots on
the sensitivity chart are the rows of the table below it, and the marker on the
PBO chart is the observed PBO inside its own simulated null. Nothing on this
page is illustrative, and every figure a chart shows also appears as text, so
no information is carried by colour or by a chart alone.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable

from auditor.schema import MANUAL_CHECKLIST, RULES, Finding
from report.corroboration import CorroborationSummary
from report.deflation import DeflationSection

E = html.escape

#: tier -> (display name, one-line gloss)
TIERS = {
    "measured": ("Measured", "Arithmetic, recomputable from the command below."),
    "parsed": ("Parsed", "Python AST. Exact lines, same answer every run."),
    "model": ("Model", "Unverified reasoning. Read as questions."),
    "unseen": ("Unseen", "No static analysis can answer these."),
    "record": ("Record", "How this page was produced."),
}

CSS = """
:root{
  --page:#f1f1ee;    /* the desk the sheet sits on */
  --sheet:#ffffff;
  --ink:#131313;
  --ink-2:#6a6a66;
  --rule:#e3e3df;
  --rule-2:#c9c9c3;
  --flag:#c8371b;    /* a failing verdict. Nothing else is ever this colour. */
  --model:#35638f;   /* the model tier. Nothing else is ever this colour.    */
  --sans:"Segoe UI Variable Display","Segoe UI Variable Text","Segoe UI",
         Inter,"Helvetica Neue",Helvetica,Arial,sans-serif;
  --mono:"Cascadia Mono",Consolas,"SF Mono","DejaVu Sans Mono",ui-monospace,monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
  font:15px/1.55 var(--sans);font-weight:400}
.sheet{max-width:1180px;margin:26px auto;background:var(--sheet);
  border:1px solid var(--rule);padding:30px 44px 40px}
a{color:inherit;text-decoration:none}
a:focus-visible{outline:2px solid var(--flag);outline-offset:3px}

/* ---- machine-written things ---- */
.mono,code,pre,td.num,th.num,.where,.kvblock,.runline,.serial,.chart text{
  font-family:var(--mono);font-variant-numeric:tabular-nums}
.label{font-size:10px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--ink-2);margin:0;font-weight:600}

/* ---- top bar ---- */
.topbar{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px solid var(--ink);padding-bottom:10px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:.06em;font-size:15px}
.brand span{font-weight:400;letter-spacing:.15em;font-size:10px;color:var(--ink-2);
  margin-left:14px;text-transform:uppercase}

/* ---- masthead ---- */
.masthead{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,.72fr)
  minmax(0,1.05fr);column-gap:38px;row-gap:22px;padding:34px 0 30px}
h1{font-size:54px;line-height:1;letter-spacing:-.032em;font-weight:600;
  margin:0;max-width:16ch}
.standfirst{margin:18px 0 0;color:var(--ink-2);font-size:16px;max-width:34ch}
.runline{margin:20px 0 0;font-size:10.5px;line-height:1.7;color:var(--ink-2);
  word-break:break-word}
.hero-val{font-size:60px;line-height:1;letter-spacing:-.035em;font-weight:600;
  margin:12px 0 0;font-family:var(--mono);font-variant-numeric:tabular-nums}
.hero.fail .hero-val,.hero.fail .hero-verdict{color:var(--flag)}
.hero-note{margin:14px 0 0;font-size:11px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-2)}
.hero-verdict{margin:6px 0 0;font-size:17px;font-weight:600;letter-spacing:.07em}
.chart-cap{margin:8px 0 0;font-size:11px;color:var(--ink-2);line-height:1.45}
svg.chart{width:100%;height:auto;display:block}
.chart text{font-size:8.5px;fill:var(--ink-2)}

/* ---- the four-tier strip ---- */
.strip{display:grid;grid-template-columns:repeat(4,1fr);
  border-top:1px solid var(--ink);border-bottom:1px solid var(--ink)}
.cell{padding:16px 20px 18px;display:block;border-left:1px solid var(--rule)}
.cell:first-child{border-left:none}
.cell .label{display:flex;align-items:center}
.cell-val{font-size:34px;line-height:1.05;letter-spacing:-.03em;font-weight:600;
  margin:10px 0 2px;font-family:var(--mono);font-variant-numeric:tabular-nums}
.cell-state{font-size:13.5px}
.cell[data-tier="measured"] .cell-state.fail{color:var(--flag)}
.cell[data-tier="model"] .cell-val,.cell[data-tier="model"] .cell-state{color:var(--model)}
.cell[data-tier="model"]{border-left:1px dashed var(--model)}
.cell[data-tier="unseen"] .cell-val,.cell[data-tier="unseen"] .cell-state{color:var(--ink-2)}
a.cell:hover{background:#fafaf9}
.tick{display:inline-block;width:8px;height:4px;border-left:2px solid var(--flag);
  border-bottom:2px solid var(--flag);transform:rotate(-45deg);flex:none;
  margin:0 8px 2px 0}

/* ---- sections ---- */
.sec{padding:34px 0 6px;border-top:1px solid var(--rule)}
.sec:first-of-type{border-top:none}
.sec-head{display:flex;align-items:baseline;margin-bottom:22px}
.serial{font-size:26px;font-weight:600;letter-spacing:-.02em;margin-right:20px;
  color:var(--rule-2);flex:none}
.sec[data-tier="model"] .serial{color:var(--model)}
h2{font-size:19px;letter-spacing:.02em;text-transform:uppercase;font-weight:600;
  margin:0}
h3{font-size:15px;font-weight:600;margin:26px 0 6px;letter-spacing:.01em}
p{margin:9px 0}
.cols{display:grid;grid-template-columns:minmax(0,290px) minmax(0,1fr);
  column-gap:44px;row-gap:20px;align-items:start}
.duo{display:grid;grid-template-columns:1fr 1fr;column-gap:44px;row-gap:26px}
.lead{color:var(--ink-2);max-width:44ch;margin-top:0}
.kv{max-width:78ch}

/* ---- key/value block, the run's own facts ---- */
.kvblock{font-size:12.5px;line-height:1.85;margin:16px 0 0}
.kvblock div{display:flex;justify-content:space-between;border-bottom:1px dotted var(--rule)}
.kvblock span:last-child{color:var(--ink);padding-left:16px;text-align:right}
.kvblock span:first-child{color:var(--ink-2)}
.kvblock .fail{color:var(--flag);font-weight:600}

/* ---- data ---- */
table{border-collapse:collapse;width:100%;margin:10px 0 4px;font-size:13px}
th{font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;
  color:var(--ink-2);text-align:left;padding:0 12px 7px 0;
  border-bottom:1px solid var(--ink)}
td{padding:7px 12px 7px 0;border-bottom:1px solid var(--rule);vertical-align:top}
td:last-child,th:last-child{padding-right:0}
td.num,th.num{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;white-space:nowrap}
tr.cont td{color:var(--ink-2);border-bottom-style:dashed}
.prov td:first-child{white-space:nowrap;width:1%;padding-right:22px}
td.fail{color:var(--flag)}
.prob{display:block;font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink-2);margin-top:2px}

/* ---- concentration bar ---- */
.bar{height:26px;display:flex;margin:14px 0 6px;border:1px solid var(--rule)}
.bar i{display:block;height:100%}
.bar .top{background:var(--flag)}
.bar .rest{background:var(--rule)}
.bar-key{font-size:11.5px;color:var(--ink-2);display:flex;flex-wrap:wrap}
.bar-key span{margin-right:22px}
.swatch{display:inline-block;width:9px;height:9px;margin-right:6px}
.swatch.top{background:var(--flag)}
.swatch.rest{background:var(--rule)}

/* ---- counts ---- */
.count{border:1px solid var(--rule);padding:16px 18px;margin:0 0 4px;
  display:flex;align-items:baseline}
.sec[data-tier="model"] .count{border:1px dashed var(--model)}
/* Direct child only: a <b> inside the count's prose must stay prose. */
.count>b{font-size:40px;line-height:.9;letter-spacing:-.03em;font-weight:600;
  margin-right:14px;font-family:var(--mono)}
.sec[data-tier="model"] .count>b{color:var(--model)}
.count .label{margin-bottom:4px}
.count p{margin:4px 0 0;font-size:12.5px;color:var(--ink-2);max-width:38ch}

/* ---- findings ---- */
.panel{border-top:1px solid var(--ink);margin-top:26px}
.panel-head{font-size:10px;letter-spacing:.15em;text-transform:uppercase;
  font-weight:600;padding:9px 0;border-bottom:1px solid var(--rule)}
.find{display:grid;grid-template-columns:minmax(0,330px) minmax(0,1fr);
  column-gap:40px;padding:20px 0;border-bottom:1px solid var(--rule)}
.find-id{display:flex;align-items:baseline;flex-wrap:wrap}
.find-id>*{margin:0 10px 6px 0}
.find-no{font-family:var(--mono);font-size:12px;color:var(--ink-2)}
.where{font-size:11.5px;color:var(--ink-2);word-break:break-all;flex-basis:100%}
pre.snippet{font-size:11.5px;line-height:1.5;background:#fafaf8;
  border-left:2px solid var(--rule-2);padding:8px 11px;margin:10px 0 0;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.panel.sem pre.snippet{border-left:2px dashed var(--model)}
.f{margin:0 0 12px;max-width:70ch}
.f:last-child{margin-bottom:0}
.f b{display:block;font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-2);margin-bottom:2px}
.tag{font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;
  border:1px solid var(--rule-2);padding:3px 8px;white-space:nowrap}
.tag-review{border-color:var(--model);color:var(--model)}
.tag-medium,.tag-high{border-color:var(--flag);color:var(--flag)}
.tag-novel{border-color:var(--ink);color:var(--ink)}

/* ---- notes, not-run, checklist ---- */
.note{border-left:2px solid var(--rule-2);padding:2px 0 2px 14px;margin:12px 0;
  font-size:12.5px;color:var(--ink-2);max-width:74ch}
.notrun{border:1px dashed var(--ink-2);padding:16px 18px;margin:4px 0;
  max-width:76ch}
.notrun b{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.15em;
  text-transform:uppercase;margin-bottom:5px}
.epistemic{border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);
  padding:12px 0;margin:0 0 18px;font-size:13.5px;max-width:80ch}
.cards{display:grid;grid-template-columns:repeat(5,1fr);column-gap:16px;row-gap:16px;
  margin-top:6px}
.card{border-top:1px solid var(--rule-2);padding:12px 0 0}
.card .n{font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.card p{font-size:13px;margin:7px 0 0}
code{font-size:12px}
footer{display:flex;justify-content:space-between;border-top:1px solid var(--ink);
  margin-top:34px;padding-top:11px;font-size:11px;color:var(--ink-2);flex-wrap:wrap}
footer p{margin:0;max-width:74ch}

@media (max-width:1000px){
  .sheet{padding:24px 22px 32px;margin:0}
  .masthead{grid-template-columns:1fr;padding:26px 0 24px}
  h1{font-size:42px;max-width:none}
  .strip{grid-template-columns:repeat(2,1fr)}
  .cell:nth-child(3){border-left:1px dashed var(--model)}
  .cell:nth-child(odd){border-left:none}
  .cell:nth-child(n+3){border-top:1px solid var(--rule)}
  .cols,.duo{grid-template-columns:1fr}
  .find{grid-template-columns:1fr;row-gap:14px}
  .cards{grid-template-columns:repeat(2,1fr)}
}
@media (max-width:520px){
  h1{font-size:32px}
  .hero-val{font-size:44px}
  .strip{grid-template-columns:1fr}
  .cell{border-left:none!important;border-top:1px solid var(--rule)}
  .cell:first-child{border-top:none}
  .cards{grid-template-columns:1fr}
}
@media print{
  @page{size:A4;margin:12mm}
  body{background:#fff;font-size:9.5pt}
  .sheet{margin:0;padding:0;border:none;max-width:none}
  h1{font-size:34pt}
  .hero-val{font-size:30pt}
  .cell-val{font-size:20pt}
  .masthead{grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);row-gap:14px}
  .masthead>div:last-child{grid-column:1/-1}
  .cols{grid-template-columns:minmax(0,190px) minmax(0,1fr);column-gap:26px}
  .duo{column-gap:26px}
  .cards{grid-template-columns:repeat(3,1fr)}
  .find,table,.note,.notrun,.count,.card,.strip{break-inside:avoid;
    page-break-inside:avoid}
  h2,h3,.sec-head{break-after:avoid;page-break-after:avoid}
  .sec{padding-top:16px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _tag(text: str, kind: str) -> str:
    return f'<span class="tag tag-{E(kind)}">{E(text)}</span>'


def _pct(x: float) -> str:
    return f"{x:.3f}"


def _periods(value: float) -> str:
    """MinTRL is infinite when the observed Sharpe ratio does not exceed the
    benchmark: no amount of track record would clear the threshold. Printing
    'inf periods' is technically true and practically unreadable."""
    if value != value:                      # NaN
        return "not computable"
    if value == float("inf"):
        return "unattainable (the observed Sharpe never clears the benchmark)"
    return f"{value:.0f} periods"


def _where(f: Finding) -> str:
    span = f"{f.line_start}" if f.line_end == f.line_start else f"{f.line_start}-{f.line_end}"
    return f"{E(f.filename)}:{span}"


def _kvblock(pairs: Iterable[tuple[str, str, bool]]) -> str:
    rows = "".join(
        f'<div><span>{E(k)}</span>'
        f'<span class="{"fail" if fail else ""}">{E(v)}</span></div>'
        for k, v, fail in pairs
    )
    return f'<div class="kvblock">{rows}</div>'


def _section(serial: str, tier: str, title: str, body: str) -> str:
    return (
        f'<section class="sec" data-tier="{E(tier)}" id="s{E(serial)}">'
        f'<div class="sec-head"><span class="serial">{E(serial)}</span>'
        f"<h2>{E(title)}</h2></div>{body}</section>"
    )


# ---------------------------------------------------------------------------
# Charts. Drawn from the section's own fields; never illustrative.
# ---------------------------------------------------------------------------


def _svg_dsr_sensitivity(s: DeflationSection) -> tuple[str, str]:
    """Every row of the sensitivity table as one dot, against the threshold.

    The point of the picture is that the DSR moves a lot with the assumption
    about V[SR] and not at all across the threshold: the rejection is not an
    artefact of one modelling choice.
    """
    values: list[float] = []
    for r in s.rows:
        values.append(r.dsr)
        if r.dsr_correlation_adjusted is not None:
            values.append(r.dsr_correlation_adjusted)
    if not values:
        return "", ""

    w, h = 320.0, 118.0
    left, right, top, bottom = 34.0, 10.0, 12.0, 10.0
    plot_w, plot_h = w - left - right, h - top - bottom
    thr = s.headline.confidence

    def x_of(i: int) -> float:
        n = max(len(values) - 1, 1)
        return left + (plot_w * i / n if len(values) > 1 else plot_w / 2)

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(1.0, v)))

    spoken = ", ".join(f"{v:.3f}" for v in values)
    parts = [f'<svg class="chart" viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
             f'aria-label="Deflated Sharpe ratio under {len(values)} assumptions: '
             f'{spoken}. All below the {thr:.2f} threshold." '
             f'xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
                 f'stroke="var(--rule-2)" stroke-width="1"/>')
    for grid in (0.0, 0.5, 1.0):
        y = y_of(grid)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" '
                     f'stroke="var(--rule)" stroke-width="1"/>')
        # 1.00 goes unlabelled: it sits a few pixels from the threshold tick and
        # the two labels would collide at print size.
        if abs(grid - thr) > 0.15:
            parts.append(f'<text x="{left - 5}" y="{y + 3:.1f}" text-anchor="end">'
                         f"{grid:.2f}</text>")
    # The threshold is a tick on the axis, not a floating label: nothing to collide
    # with the dots, and it survives being printed small.
    y_thr = y_of(thr)
    parts.append(f'<line x1="{left}" y1="{y_thr:.1f}" x2="{w - right}" y2="{y_thr:.1f}" '
                 f'stroke="var(--ink)" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{left - 5}" y="{y_thr + 3:.1f}" text-anchor="end" '
                 f'fill="var(--ink)">{thr:.2f}</text>')
    for i, value in enumerate(values):
        x, y = x_of(i), y_of(value)
        if y + 8 < top + plot_h:
            parts.append(f'<line x1="{x:.1f}" y1="{y + 6:.1f}" x2="{x:.1f}" '
                         f'y2="{top + plot_h}" stroke="var(--rule-2)" stroke-width="1"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="var(--flag)"/>')
    parts.append("</svg>")
    caption = (
        f"One dot per row of the sensitivity table in section 01: the DSR under each "
        f"assumption about the number of trials and about V[SR]. The dashed line is "
        f"the {thr:.2f} threshold; every assumption lands below it."
    )
    return "".join(parts), caption


def _svg_pbo_null(s: DeflationSection) -> str:
    """The observed PBO placed inside its own simulated null distribution."""
    w, h = 320.0, 82.0
    left, right = 12.0, 12.0
    axis_y = 46.0
    span = w - left - right

    def x_of(v: float) -> float:
        return left + span * max(0.0, min(1.0, v))

    lo, hi = s.pbo_null_lo, s.pbo_null_hi
    obs, mean = s.pbo.pbo, s.pbo_null_mean
    parts = [f'<svg class="chart" viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
             f'aria-label="Observed PBO {obs:.3f} against a simulated null centred on '
             f'{mean:.3f}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<rect x="{x_of(lo):.1f}" y="{axis_y - 11:.0f}" '
                 f'width="{x_of(hi) - x_of(lo):.1f}" height="22" fill="var(--rule)"/>')
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{w - right}" y2="{axis_y}" '
                 f'stroke="var(--ink)" stroke-width="1"/>')
    parts.append(f'<line x1="{x_of(mean):.1f}" y1="{axis_y - 13:.0f}" '
                 f'x2="{x_of(mean):.1f}" y2="{axis_y + 13:.0f}" '
                 f'stroke="var(--ink-2)" stroke-width="1"/>')
    parts.append(f'<text x="{x_of(mean):.1f}" y="{axis_y + 25:.0f}" '
                 f'text-anchor="middle">null mean {mean:.2f}</text>')
    x_obs = x_of(obs)
    parts.append(f'<polygon points="{x_obs:.1f},{axis_y - 3:.0f} '
                 f'{x_obs - 5:.1f},{axis_y - 15:.0f} {x_obs + 5:.1f},{axis_y - 15:.0f}" '
                 f'fill="var(--flag)"/>')
    parts.append(f'<text x="{x_obs:.1f}" y="{axis_y - 19:.0f}" text-anchor="end" '
                 f'fill="var(--flag)">observed {obs:.3f}</text>')
    for v in (0.0, 1.0):
        parts.append(f'<text x="{x_of(v):.1f}" y="{axis_y + 25:.0f}" '
                     f'text-anchor="{"start" if v == 0 else "end"}">{v:.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _concentration_bar(s) -> str:
    """One horizontal bar: the share of the total carried by a single observation."""
    top = max(0.0, min(1.0, s.top_1_share))
    return (
        f'<div class="bar" role="img" aria-label="The single best observation carries '
        f'{top:.1%} of the total return">'
        f'<i class="top" style="width:{top * 100:.1f}%"></i>'
        f'<i class="rest" style="width:{(1 - top) * 100:.1f}%"></i></div>'
        f'<p class="bar-key"><span><i class="swatch top"></i>'
        f"{E(s.top_day_label)}, {top:.1%} of the total</span>"
        f'<span><i class="swatch rest"></i>every other observation, '
        f"{1 - top:.1%}</span></p>"
    )


# ---------------------------------------------------------------------------
# Section 1 -- statistical deflation
# ---------------------------------------------------------------------------


def render_deflation(sections: Iterable[DeflationSection]) -> str:
    sections = list(sections)
    if not sections:
        body = (
            '<div class="notrun"><b>Not run</b> No trial-returns matrix was '
            "supplied, so no deflation was computed. This section being empty is "
            "not evidence that the backtest survives deflation: it means the "
            "question was not asked. Pass <code>--trials</code> and "
            "<code>--meta</code> to answer it.</div>"
        )
        return _section("01", "measured", "Statistical deflation", body)

    out = [
        '<p class="epistemic">PSR, DSR and PBO below are <b>probabilities in '
        "[0, 1]</b>, never Sharpe ratios. A DSR of 0.92 against a 0.95 threshold "
        "is a rejection, not a degraded Sharpe ratio.</p>"
    ]

    for s in sections:
        v = s.headline
        out.append(f"<h3>{E(s.label)}</h3>")
        out.append('<div class="cols">')

        # --- left rail: the run's own facts, as a machine would print them ---
        out.append(
            '<div><p class="lead">The selected configuration is deflated against '
            "the best of N trials under the null that no trial has skill, then "
            "cross-validated for overfitting and inspected for concentration.</p>"
            + _kvblock([
                ("selected configuration", s.selected_trial, False),
                ("observations", f"{s.n_obs}", False),
                ("grid trials", f"{s.n_grid_trials}", False),
                ("trials in total", f"{s.n_all_trials}", False),
                ("moment estimator", s.moment_estimator, False),
                ("Sharpe, per period", f"{s.sr_per_period:.6f}", False),
                ("Sharpe, annualized", f"{s.sr_annual:.4f}", False),
                ("DSR", f"{v.dsr:.3f}", not v.passes),
                ("threshold", f"{v.confidence:.3f}", False),
                ("verdict", v.verdict, not v.passes),
            ])
            + "</div>"
        )

        # --- right: the quantities table ---
        out.append("<div>")
        out.append("<table><tr><th>Quantity</th><th class='num'>Value</th>"
                   "<th>Reading</th></tr>")
        rows = [
            ("PSR vs zero <span class='prob'>probability in [0,1]</span>",
             _pct(v.psr_vs_zero), False,
             "Probability the true Sharpe ratio exceeds zero."),
            ("DSR <span class='prob'>probability in [0,1]</span>", _pct(v.dsr),
             not v.passes,
             f"Probability the true Sharpe ratio exceeds E[max SR] over "
             f"{v.n_trials} trials. Threshold {v.confidence:.2f}."),
            ("E[max SR] under H0", f"{v.expected_max_sr_annual:.4f}", False,
             "Annualized Sharpe ratio the best of N lucky trials would show."),
            ("MinTRL, observed moments", _periods(v.min_track_record_periods), False,
             "Observations needed to clear the threshold."),
            ("MinTRL, if returns were normal",
             _periods(v.min_track_record_periods_if_normal), False,
             "The gap between these two lines is the track record that "
             "non-normality is buying, or costing."),
            ("Skewness / kurtosis", f"{v.skew:+.2f} / {v.kurtosis:.2f}", False,
             f"Variance term {v.variance_term:.3f}: "
             + ("below 1, so non-normality is INFLATING the probability."
                if v.variance_term < 1 else "at or above 1, deflating the probability.")),
            ("Mean pairwise trial correlation", f"{s.mean_trial_correlation:+.3f}",
             False, "How much the trials repeat one another."),
        ]
        for name, value, fail, reading in rows:
            klass = "num fail" if fail else "num"
            out.append(f"<tr><td>{name}</td><td class='{klass}'>{E(value)}</td>"
                       f"<td>{reading}</td></tr>")
        out.append("</table></div></div>")

        # --- sensitivity ---
        out.append("<h3>Sensitivity of the DSR to N and to V[SR]</h3>")
        out.append('<div class="cols"><div>')
        out.append(
            f'<p class="lead">V[SR] in per-period units: empirical across the grid '
            f"{s.v_empirical_grid:.3e}, across all trials {s.v_empirical_all:.3e}, "
            f"theoretical under H0 (1/T, T={s.n_obs}) {s.v_null:.3e}; ratio "
            f"empirical/theoretical {s.v_ratio:.2f}. {E(s.v_ratio_reading)}</p>"
        )
        out.append("</div><div>")
        out.append("<table><tr><th class='num'>N</th><th>V[SR] source</th>"
                   "<th class='num'>E[max SR] ann.</th>"
                   "<th class='num'>DSR</th><th>Verdict</th></tr>")
        has_adjusted = False
        for r in s.rows:
            out.append(
                f"<tr><td class='num'>{r.n_trials}</td>"
                f"<td>{E(r.v_sr_source_label)}</td>"
                f"<td class='num'>{r.expected_max_sr_annual:.4f}</td>"
                f"<td class='num'>{_pct(r.dsr)}</td><td>{E(r.verdict)}</td></tr>"
            )
            if r.dsr_correlation_adjusted is not None:
                has_adjusted = True
                adj = r.dsr_correlation_adjusted
                out.append(
                    f"<tr class='cont'><td class='num'>{r.n_trials}</td>"
                    f"<td>same row, shrunk by (1-rho)</td>"
                    f"<td class='num'>n/a</td><td class='num'>{_pct(adj)}</td>"
                    f"<td>{'PASS' if adj >= v.confidence else 'REJECT'}</td></tr>"
                )
        out.append("</table>")
        if has_adjusted:
            out.append(
                '<p class="note">The shrunk line re-reads the row above it after '
                "discounting N by the correlation between trials: highly correlated "
                "configurations are not independent bets. It carries no E[max SR] of "
                "its own, which is why that cell reads n/a rather than blank.</p>"
            )
        out.append("</div></div>")

        # --- concentration ---
        c = s.concentration
        out.append("<h3>Where the profit actually came from</h3>")
        out.append('<div class="cols"><div>')
        out.append(
            f'<p class="lead">Verdict <b>{E(c.verdict)}</b>. Sharpe {c.sharpe_full:.2f} '
            f"on the full sample, {c.sharpe_ex_top_day:.2f} with the single best "
            f"observation removed."
            + (f" {c.days_to_zero} observation(s) removed take the total to zero."
               if c.days_to_zero is not None else "")
            + "</p>"
        )
        out.append("</div><div>")
        out.append(_concentration_bar(c))
        out.append(
            f'<p class="kv">Top 3 observations {c.top_3_share:.1%} of the total, so '
            f"everything outside them is net negative; {c.positive_day_share:.1%} of "
            f"observations are positive; median {c.median_return:+.5f}.</p>"
        )
        if len(s.shared):
            out.append("<table><tr><th>Observation</th>"
                       "<th class='num'>Configurations depending on it</th></tr>")
            for label, row in s.shared.head(3).iterrows():
                stamp = getattr(label, "date", lambda: label)()
                out.append(
                    f"<tr><td>{E(str(stamp))}</td><td class='num'>"
                    f"{int(row['n_trials_dependent'])} / {s.n_grid_trials}</td></tr>"
                )
            out.append("</table>")
            out.append(
                '<p class="note">When every cell of a grid leans on the same '
                "observation, the grid was never exploring different strategies: it "
                "was re-expressing one event.</p>"
            )
        out.append("</div></div>")

        # --- PBO ---
        p = s.pbo
        out.append("<h3>Backtest overfitting (CSCV)</h3>")
        out.append('<div class="cols"><div>')
        out.append(
            f'<p class="lead">PBO <b>{_pct(p.pbo)}</b>, the probability in [0,1] that '
            "the in-sample winner is below median out-of-sample, from "
            f"{p.n_combinations} combinations of {p.n_blocks} blocks over "
            f"{p.n_obs_used} observations and {p.n_configs} configurations. Median "
            f"out-of-sample rank of the winner {p.median_oos_rank:.3f}; degradation "
            f"slope {p.degradation_slope:+.2f}.</p>"
        )
        out.append("</div><div>")
        out.append(_svg_pbo_null(s))
        out.append(
            f'<p class="chart-cap">Simulated null at the same dimensions '
            f"({N_SIMS_LABEL}): mean {s.pbo_null_mean:.3f}, standard deviation "
            f"{s.pbo_null_sd:.3f}, 90% interval [{s.pbo_null_lo:.3f}, "
            f"{s.pbo_null_hi:.3f}]. The observed PBO sits at percentile "
            f"<b>{s.pbo_percentile:.2f}</b> of that null. {E(s.pbo_reading)}</p>"
        )
        for note in p.notes:
            out.append(f'<p class="note">{E(note)}</p>')
        out.append("</div></div>")

    return _section("01", "measured", "Statistical deflation", "\n".join(out))


N_SIMS_LABEL = "pure noise, same T, N and S"


# ---------------------------------------------------------------------------
# Section 2 -- deterministic findings
# ---------------------------------------------------------------------------


def render_ast(findings: list[Finding], n_files: int) -> str:
    n = len(findings)
    out = ['<div class="duo"><div>']
    out.append(
        f'<p class="lead">Exact syntactic matches from the Python parser: '
        f"{len(RULES)} rules, line numbers taken from the AST. They run "
        f"offline, cost nothing and give the same answer on every run.</p>"
    )
    out.append("</div><div>")
    out.append(
        f'<div class="count"><b>{n}</b><div><p class="label">'
        + ("finding" if n == 1 else "findings")
        + f'</p><p>{n_files} file(s) scanned. '
        + ("No deterministic leakage pattern matched. This is a statement about the "
           "CODE only: it says nothing about how many configurations were tried, "
           "nor about the vintage of the data." if not n else
           "Each one below is an exact match, not an opinion.")
        + "</p></div></div>"
    )
    out.append("</div></div>")

    if findings:
        out.append('<div class="panel"><p class="panel-head">Deterministic findings</p>')
        order = {"high": 0, "medium": 1, "review": 2}
        for i, f in enumerate(sorted(findings,
                                     key=lambda f: (order[f.severity], f.filename,
                                                    f.line_start)), start=1):
            out.append(
                f'<div class="find"><div><div class="find-id">'
                f'<span class="find-no">{i:02d}</span>{_tag(f.severity.upper(), f.severity)}'
                f"<b>{E(f.rule_id)}</b>"
                f'<span class="where">{_where(f)}</span></div>'
                f"<pre class='snippet'>{E(f.snippet)}</pre></div>"
                f"<div><p class='f'><b>{E(f.title)}</b>{E(f.explanation)}</p>"
                f"<p class='f'><b>Fix</b>{E(f.suggested_fix)}</p></div></div>"
            )
        out.append("</div>")
    return _section("02", "parsed", "Code findings (deterministic)", "\n".join(out))


# ---------------------------------------------------------------------------
# Section 3 -- semantic findings, kept apart
# ---------------------------------------------------------------------------


def render_semantic(
    summary: CorroborationSummary | None,
    *,
    ran: bool,
    prompt_version: str,
    n_rejected: int,
    n_capped: int,
    reason_not_run: str = "",
) -> str:
    title = "Semantic findings (model), to verify"
    if not ran:
        body = (
            '<div class="notrun"><b>Not run</b> '
            + E(reason_not_run or "The semantic pass was not enabled.")
            + " An empty semantic section is not evidence that the code is free of "
            "semantic leakage: it means the question was not asked.</div>"
        )
        return _section("03", "model", title, body)

    assert summary is not None
    out = ['<div class="duo"><div>']
    out.append(
        '<p class="lead">A language model reads what no syntactic rule can express. '
        f"Line numbers were verified against the file and snippets extracted from the "
        f"real source by this tool, but the REASONING is the model&rsquo;s: <b>each one "
        f"is a question, not a verdict</b>, and they are never merged into the "
        f"deterministic section above. Prompt <code>{E(prompt_version)}</code>.</p>"
    )
    out.append("</div><div>")
    out.append(
        f'<div class="count"><b>{summary.n_semantic}</b><div><p class="label">'
        + ("question to verify" if summary.n_semantic == 1 else "questions to verify")
        + f"</p><p>{n_rejected} rejected by grounding &middot; {n_capped} severity "
        f"capped at review by the harness &middot; {summary.n_corroborated} "
        f"corroborating a deterministic rule &middot; <b>{summary.n_novel} outside "
        f"the rules&rsquo; reach</b>. The last number is the semantic pass&rsquo;s "
        f"actual contribution: the rest restates what section 02 already found "
        f"deterministically.</p></div></div>"
    )
    out.append("</div></div>")

    if not summary.n_semantic:
        out.append('<p class="note">The model returned no semantic finding.</p>')
    else:
        out.append('<div class="panel sem"><p class="panel-head">'
                   "Semantic review findings, unverified</p>")
        for i, a in enumerate(summary.annotated, start=1):
            f = a.finding
            right = [f"<p class='f'><b>Question</b>{E(f.explanation)}</p>"]
            if a.is_corroborated:
                right.append(f"<p class='f'><b>Cross-check</b>{E(a.label)} in section "
                             "02. Reported here for context only; the two detectors "
                             "are counted separately.</p>")
            if f.external_dependency:
                right.append(f"<p class='f'><b>Fact to verify</b>"
                             f"{E(f.external_dependency)}</p>")
            if f.capped_from:
                right.append(f"<p class='f'><b>Severity capped</b>by the harness from "
                             f"{E(f.capped_from.upper())} to REVIEW: the finding "
                             "depends on code outside this file, so it cannot be "
                             "established here.</p>")
            right.append(f"<p class='f'><b>Suggested fix</b>{E(f.suggested_fix)}</p>")
            out.append(
                f'<div class="find"><div><div class="find-id">'
                f'<span class="find-no">{i:02d}</span>'
                f"{_tag(f.severity.upper(), f.severity)}"
                + (_tag("novel", "novel") if not a.is_corroborated else "")
                + f'<span class="where">{_where(f)}</span></div>'
                f"<pre class='snippet'>{E(f.snippet)}</pre></div>"
                f"<div>{''.join(right)}</div></div>"
            )
        out.append("</div>")
    return _section("03", "model", title, "\n".join(out))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_checklist() -> str:
    cards = "".join(
        f'<div class="card"><span class="n">{i:02d}</span><p>{E(item)}</p></div>'
        for i, item in enumerate(MANUAL_CHECKLIST, start=1)
    )
    body = (
        '<div class="duo"><div><p class="lead">These are properties of the DATA, not '
        "of the code. Presenting them as questions rather than as detections is the "
        "point: a static analyser cannot decide them, and claiming otherwise would be "
        "the same overstatement this tool exists to catch.</p></div><div></div></div>"
        f'<div class="cards">{cards}</div>'
    )
    return _section("04", "unseen", "What this audit cannot see", body)


@dataclass(frozen=True)
class LedgerEntry:
    """One cell of the strip under the masthead: what a tier actually returned.

    Every value here is a real measured figure with its unit and threshold
    named, or the words "Not run". A tier that did not run keeps its cell: a
    missing cell would read as nothing to report, which is the failure mode
    this whole report is built against.
    """

    tier: str
    label: str
    value: str
    reading: str
    anchor: str = ""
    failing: bool = False


def render_ledger(entries: Iterable[LedgerEntry]) -> str:
    entries = list(entries)
    if not entries:
        return ""
    cells = []
    for e in entries:
        name, _ = TIERS[e.tier]
        tick = '<span class="tick"></span>' if e.tier in ("measured", "parsed") else ""
        inner = (
            f'<p class="label">{tick}{E(name)}</p>'
            f'<p class="cell-val">{E(e.value)}</p>'
            f'<p class="cell-state{" fail" if e.failing else ""}">{E(e.reading)}</p>'
            f'<p class="label" style="margin-top:8px">{E(e.label)}</p>'
        )
        tag = "a" if e.anchor else "div"
        href = f' href="#{E(e.anchor)}"' if e.anchor else ""
        cells.append(f'<{tag} class="cell" data-tier="{E(e.tier)}"{href}>{inner}</{tag}>')
    return f'<div class="strip">{"".join(cells)}</div>'


@dataclass(frozen=True)
class Hero:
    """The headline figure: one number, its threshold, its verdict, its chart."""

    label: str
    value: str
    note: str
    verdict: str
    failing: bool = False
    chart_svg: str = ""
    caption: str = ""


def hero_from_section(s: DeflationSection | None) -> Hero:
    if s is None:
        return Hero(label="Statistical deflation", value="Not run",
                    note="no trial matrix supplied", verdict="", failing=False)
    chart, caption = _svg_dsr_sensitivity(s)
    return Hero(
        label="DSR (measured)",
        value=f"{s.headline.dsr:.3f}",
        note=f"threshold {s.headline.confidence:.3f}",
        verdict=s.headline.verdict,
        failing=not s.headline.passes,
        chart_svg=chart,
        caption=caption,
    )


STANDFIRST = ("Statistical validity and code integrity, examined separately and "
              "reported separately.")


def render_page(
    *,
    title: str,
    subtitle: str,
    deflation_html: str,
    ast_html: str,
    semantic_html: str,
    provenance: dict[str, str],
    ledger: Iterable[LedgerEntry] | None = None,
    hero: Hero | None = None,
) -> str:
    prov = "".join(
        f"<tr><td>{E(k)}</td><td><code>{E(v)}</code></td></tr>"
        for k, v in provenance.items()
    )
    record = _section(
        "05", "record", "Provenance",
        '<p class="lead">Every figure above is produced by a script in this '
        "repository and can be recomputed from the command below. The two charts "
        "are drawn from the same fields as the tables beside them.</p>"
        f"<table class='prov'><tr><th>Item</th><th>Value</th></tr>{prov}</table>"
    )
    generated = provenance.get("generated", "")
    hero = hero or Hero(label="Statistical deflation", value="Not run",
                        note="no trial matrix supplied", verdict="")
    hero_html = (
        f'<div class="hero{" fail" if hero.failing else ""}">'
        f'<p class="label">{E(hero.label)}</p>'
        f'<p class="hero-val">{E(hero.value)}</p>'
        f'<p class="hero-note">{E(hero.note)}</p>'
        + (f'<p class="hero-verdict">{E(hero.verdict)}</p>' if hero.verdict else "")
        + "</div>"
    )
    chart_html = (f'<div>{hero.chart_svg}<p class="chart-cap">{E(hero.caption)}</p></div>'
                  if hero.chart_svg else "<div></div>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)}</title><style>{CSS}</style></head><body>
<div class="sheet">
<div class="topbar"><p class="brand">BIA<span>Backtest Integrity Auditor</span></p>
<p class="label">Run {E(generated)}</p></div>
<header class="masthead">
<div><h1>{E(title)}</h1><p class="standfirst">{STANDFIRST}</p>
<p class="runline">{E(subtitle)}</p></div>
{hero_html}
{chart_html}
</header>
{render_ledger(ledger or [])}
<main>
{deflation_html}
{ast_html}
{semantic_html}
{render_checklist()}
{record}
</main>
<footer><p>Backtest Integrity Auditor. PSR, DSR and PBO are probabilities in
[0, 1], never Sharpe ratios. Deterministic and semantic findings are reported in
separate sections and are never merged. Solid rules mark what this tool verified
itself; dashed rules mark what a language model proposed and nobody has checked.</p>
<p>{E(generated)}</p>
</footer>
</div>
</body></html>"""


__all__ = [
    "Hero",
    "LedgerEntry",
    "TIERS",
    "hero_from_section",
    "render_ast",
    "render_checklist",
    "render_deflation",
    "render_ledger",
    "render_page",
    "render_semantic",
]
