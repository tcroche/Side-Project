"""Tests for the report layer.

Three guarantees, all executable:

1. Corroboration ANNOTATES the semantic side and never touches the
   deterministic one. The two detectors stay separate objects, separate
   sections, separate counters.
2. The HTML is safe and honest: code snippets are escaped (real findings quote
   real code, which contains `<`, `>` and `&`), and any section that was not
   run says so instead of rendering as empty-therefore-clean.
3. The report's deflation reproduces exactly what `run_real_case.py` computes,
   so the two entry points can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auditor.schema import Finding
from core.stats import deflate, mean_offdiagonal_correlation, sharpe_moments
from core.units import TRADING_DAYS_PER_YEAR, to_per_period
from report.corroboration import annotate
from report.deflation import MOMENT_ESTIMATOR, run_deflation
from report.render import (
    Hero,
    LedgerEntry,
    hero_from_section,
    render_ast,
    render_deflation,
    render_ledger,
    render_page,
    render_semantic,
)


def make_finding(line_start, line_end=None, *, detector="ast", rule_id="R1",
                 severity="high", filename="strategy.py", snippet="x = 1",
                 explanation="why", fix="fix", external_dependency=None,
                 capped_from=None):
    return Finding(
        rule_id=rule_id, title="t", severity=severity,
        line_start=line_start, line_end=line_end if line_end is not None else line_start,
        snippet=snippet, explanation=explanation, suggested_fix=fix,
        detector=detector, filename=filename,
        external_dependency=external_dependency, capped_from=capped_from,
    )


# ---------------------------------------------------------------------------
# 1) Corroboration annotates; it never merges
# ---------------------------------------------------------------------------


def test_overlapping_semantic_finding_is_annotated_not_merged():
    ast = [make_finding(11, rule_id="R1")]
    sem = [make_finding(10, 14, detector="llm", rule_id="SEM", severity="medium")]
    summary = annotate(ast, sem)

    assert summary.n_semantic == 1 and summary.n_corroborated == 1 and summary.n_novel == 0
    assert summary.annotated[0].corroborates == (("R1", 11),)
    assert "corroborates R1 at line 11" == summary.annotated[0].label
    # The AST finding is untouched and still its own object.
    assert ast == [make_finding(11, rule_id="R1")]
    # The annotation wraps the semantic finding; it does not rewrite it.
    assert summary.annotated[0].finding is sem[0]


def test_non_overlapping_semantic_finding_is_novel():
    summary = annotate([make_finding(11)], [make_finding(40, 44, detector="llm")])
    assert summary.n_novel == 1 and summary.n_corroborated == 0
    assert summary.annotated[0].label == ""
    assert summary.novel_findings[0].finding.line_start == 40


def test_overlap_requires_the_same_file():
    ast = [make_finding(11, filename="a.py")]
    sem = [make_finding(11, filename="b.py", detector="llm")]
    assert annotate(ast, sem).n_novel == 1


def test_path_separators_do_not_defeat_the_match():
    """Windows and POSIX spellings of the same file must still match."""
    ast = [make_finding(11, filename="m2_backtester\\strategy.py")]
    sem = [make_finding(11, filename="m2_backtester/strategy.py", detector="llm")]
    assert annotate(ast, sem).n_corroborated == 1


def test_several_rules_on_one_semantic_finding_are_all_listed_once():
    ast = [make_finding(10, rule_id="R1"), make_finding(12, rule_id="R8"),
           make_finding(12, rule_id="R8")]
    sem = [make_finding(9, 13, detector="llm")]
    summary = annotate(ast, sem)
    assert summary.annotated[0].corroborates == (("R1", 10), ("R8", 12))


def test_no_ast_findings_leaves_every_semantic_finding_novel():
    summary = annotate([], [make_finding(1, detector="llm"), make_finding(9, detector="llm")])
    assert (summary.n_semantic, summary.n_corroborated, summary.n_novel) == (2, 0, 2)


# ---------------------------------------------------------------------------
# 2) The HTML is safe and honest
# ---------------------------------------------------------------------------


def test_code_snippets_are_html_escaped():
    """Findings quote real code. Unescaped, `df[df['a'] < 0] & mask` would
    silently corrupt the page."""
    nasty = "df[df['a'] < 0] & mask  # <script>alert(1)</script>"
    html = render_ast([make_finding(3, snippet=nasty)], n_files=1)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp; mask" in html


def test_semantic_explanations_are_escaped_too():
    sem = [make_finding(3, detector="llm", explanation="uses x < y & z",
                        external_dependency="check <caller>")]
    html = render_semantic(annotate([], sem), ran=True, prompt_version="1.2.0",
                           n_rejected=0, n_capped=0)
    assert "x &lt; y &amp; z" in html
    assert "<caller>" not in html


def test_missing_deflation_says_not_run_rather_than_looking_clean():
    html = render_deflation([])
    assert "Not run" in html
    assert "not evidence that the backtest survives deflation" in html


def test_missing_semantic_pass_says_not_run_with_its_reason():
    html = render_semantic(None, ran=False, prompt_version="", n_rejected=0,
                           n_capped=0, reason_not_run="No API key was available.")
    assert "Not run" in html
    assert "No API key was available." in html
    assert "not evidence that the code is free of semantic leakage" in html


def test_semantic_section_states_its_epistemic_status_and_the_novel_count():
    sem = [make_finding(5, detector="llm"), make_finding(50, detector="llm")]
    html = render_semantic(annotate([make_finding(5)], sem), ran=True,
                           prompt_version="1.2.0", n_rejected=1, n_capped=2)
    assert "question, not a verdict" in html
    assert "never merged" in html
    assert "1 outside the rules" in html
    assert "1 rejected by grounding" in html
    assert "2 severity capped" in html


def test_capped_finding_shows_the_original_severity_and_the_external_fact():
    sem = [make_finding(21, detector="llm", severity="review", capped_from="medium",
                        external_dependency="The engine's position/return alignment.")]
    html = render_semantic(annotate([], sem), ran=True, prompt_version="1.2.0",
                           n_rejected=0, n_capped=1)
    assert "capped" in html.lower() and "MEDIUM" in html
    assert "position/return alignment" in html


def test_page_is_self_contained_and_labels_probabilities():
    page = render_page(
        title="T", subtitle="S",
        deflation_html=render_deflation([]),
        ast_html=render_ast([], n_files=0),
        semantic_html=render_semantic(None, ran=False, prompt_version="",
                                      n_rejected=0, n_capped=0),
        provenance={"generated": "now"},
    )
    assert "http://" not in page and "https://" not in page   # no external asset
    assert "<style>" in page                                   # CSS inlined
    assert "probabilities in\n[0, 1], never Sharpe ratios" in page or \
           "probabilities in [0, 1], never Sharpe ratios" in page.replace("\n", " ")
    assert "What this audit cannot see" in page


def test_page_keeps_the_two_detectors_in_separate_sections():
    ast_html = render_ast([make_finding(11, rule_id="R1")], n_files=1)
    sem_html = render_semantic(
        annotate([make_finding(11, rule_id="R1")],
                 [make_finding(11, detector="llm", rule_id="SEM")]),
        ran=True, prompt_version="1.2.0", n_rejected=0, n_capped=0,
    )
    # The corroboration annotation exists only on the semantic side.
    assert "corroborates" in sem_html
    assert "corroborates" not in ast_html
    assert "SEM" not in ast_html


# ---------------------------------------------------------------------------
# 3) The report's deflation matches run_real_case.py
# ---------------------------------------------------------------------------


def _fixture_frame():
    """A small stand-in with the exact shape of the M2 export."""
    rng = np.random.default_rng(0)
    n_days, n_grid, n_sweep = 103, 18, 3
    ids = [f"t{i:02d}" for i in range(n_grid + n_sweep)]
    dates = pd.bdate_range("2025-01-06", periods=n_days)
    common = rng.normal(0.0, 0.0015, size=(n_days, 1))
    idio = rng.normal(0.0, 0.0006, size=(n_days, len(ids)))
    frame = pd.DataFrame(common + idio, index=dates, columns=ids)

    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    meta = pd.DataFrame({
        "trial_id": ids,
        "trial_kind": ["grid"] * n_grid + ["gamma_sweep"] * n_sweep,
        "is_frozen_cell": [i == 4 for i in range(len(ids))],
        "sharpe_annual_with_rut": [
            float(frame[t].mean() / frame[t].std(ddof=1) * ann) for t in ids
        ],
    })
    return frame, meta


@pytest.fixture
def fixture_trials():
    return _fixture_frame()


def test_report_deflation_reproduces_the_run_real_case_headline(fixture_trials):
    frame, meta = fixture_trials
    section = run_deflation(frame, meta, label="fixture",
                            sharpe_col="sharpe_annual_with_rut")

    # Recompute the way run_real_case.analyse does, independently.
    grid_ids = [t for t in meta.loc[meta["trial_kind"] == "grid", "trial_id"]]
    selected = meta.loc[meta["is_frozen_cell"], "trial_id"].iloc[0]
    moments = sharpe_moments(
        frame[selected].dropna().to_numpy(dtype=float),
        periods_per_year=TRADING_DAYS_PER_YEAR,
        moment_estimator=MOMENT_ESTIMATOR,
    )
    v_grid = float(np.var(
        [to_per_period(float(meta.loc[meta["trial_id"] == t,
                                      "sharpe_annual_with_rut"].iloc[0]),
                       TRADING_DAYS_PER_YEAR) for t in grid_ids], ddof=1))
    rho = mean_offdiagonal_correlation(frame[grid_ids].dropna().to_numpy(dtype=float))
    expected = deflate(moments, n_trials=len(grid_ids),
                       sr_variance_across_trials=v_grid,
                       mean_trial_correlation=float(rho),
                       v_sr_source="empirical_cross_section")

    assert section.headline.dsr == pytest.approx(expected.dsr)
    assert section.headline.expected_max_sr_annual == pytest.approx(
        expected.expected_max_sr_annual)
    assert section.selected_trial == selected
    assert section.n_grid_trials == len(grid_ids)


def test_deflation_outputs_are_probabilities_not_sharpe_ratios(fixture_trials):
    """The invariant this whole project exists to protect."""
    frame, meta = fixture_trials
    s = run_deflation(frame, meta, label="fixture", sharpe_col="sharpe_annual_with_rut")
    for value in (s.headline.dsr, s.headline.psr_vs_zero, s.pbo.pbo, s.pbo_percentile):
        assert 0.0 <= value <= 1.0


def test_deflation_section_renders_the_probability_disclaimer(fixture_trials):
    frame, meta = fixture_trials
    s = run_deflation(frame, meta, label="fixture", sharpe_col="sharpe_annual_with_rut")
    html = render_deflation([s])
    assert "probabilities in [0, 1]" in html
    assert "never Sharpe ratios" in html
    assert "rejection, not a degraded Sharpe ratio" in html


def test_infinite_mintrl_is_readable_not_inf(fixture_trials):
    """MinTRL is infinite when the observed Sharpe never clears the benchmark.
    'inf periods' is true and unreadable; the report must say what it means."""
    frame, meta = fixture_trials
    s = run_deflation(frame, meta, label="fixture", sharpe_col="sharpe_annual_with_rut")
    html = render_deflation([s])
    assert "inf periods" not in html
    if s.headline.min_track_record_periods == float("inf"):
        assert "unattainable" in html


def test_subtitle_entities_are_not_double_escaped():
    """render_page escapes the subtitle, as it must; callers therefore pass
    plain text. An HTML entity written by a caller would render literally."""
    page = render_page(
        title="T", subtitle="a  |  b",
        deflation_html="", ast_html="", semantic_html="",
        provenance={"generated": "now"},
    )
    assert "a  |  b" in page
    assert "&amp;middot;" not in page


# ---------------------------------------------------------------------------
# 4) The visual system is a guarantee, not a claim in a docstring
# ---------------------------------------------------------------------------


def _full_page(**kwargs):
    """A complete page with one finding of each kind, for whole-document checks."""
    ast = [make_finding(11, rule_id="R1", severity="high")]
    sem = [make_finding(40, 44, detector="llm", rule_id="SEM", severity="review",
                        external_dependency="the caller's P&L convention")]
    defaults = dict(
        title="Backtest Integrity Audit", subtitle="a  |  b",
        deflation_html=render_deflation([]),
        ast_html=render_ast(ast, n_files=3),
        semantic_html=render_semantic(annotate(ast, sem), ran=True,
                                      prompt_version="1.2.0", n_rejected=0, n_capped=0),
        provenance={"generated": "now"},
        ledger=[
            LedgerEntry("measured", "Statistical deflation", "Not run", "not asked", "s1"),
            LedgerEntry("parsed", "Deterministic rules", "1 finding", "3 files", "s2"),
            LedgerEntry("model", "Semantic pass", "1 question", "1 novel", "s3"),
            LedgerEntry("unseen", "Data questions", "5 open", "invisible to code", "s4"),
        ],
    )
    defaults.update(kwargs)
    return render_page(**defaults)


def test_page_hooks_nothing_external():
    """One file, offline, mailable. Asserted rather than asserted-in-a-comment:
    no network origin, no linked asset, no web font, no script."""
    page = _full_page()
    for hook in ("http://", "https://", "//fonts.", "<script", "<link", "@import",
                 "url(", "srcset", "<img"):
        assert hook not in page, f"the report reaches outside itself via {hook!r}"


def test_severity_and_verdict_are_never_colour_alone():
    """The page is printed to PDF and read in grayscale. Every state that
    matters must survive the loss of colour as words."""
    page = _full_page()
    assert "HIGH" in page and "REVIEW" in page          # severities, as text
    assert "novel" in page                              # the contribution counter
    assert "Not run" in page                            # a tier that did not run


def test_the_model_tier_is_typeset_as_provisional():
    """Solid rules for what the tool verified, dashed for what it did not.

    The distinction is the report's whole thesis, and it must hold in grayscale,
    so it is carried by line quality and pinned here rather than left to colour.
    """
    page = _full_page()
    assert 'data-tier="model"' in page
    assert page.count("dashed var(--model)") >= 3, \
        "the model tier lost its dashed rules; it now looks as solid as a measurement"
    assert "question, not a verdict" in page
    assert "dashed rules mark what a language model proposed" in page


def test_colour_is_rationed_to_two_jobs():
    """--flag marks a failing verdict, --model marks the model tier. If either
    hue leaks into ordinary chrome, the page stops encoding anything."""
    css = render_page.__globals__["CSS"]
    for token, allowed in (("var(--flag)", 12), ("var(--model)", 14)):
        assert css.count(token) <= allowed, f"{token} is spreading into decoration"


def test_every_charted_number_is_also_written_out():
    """No figure may exist only as a mark: charts are drawn from the same fields
    as the tables beside them, and both must be readable without the other."""
    frame, meta = _fixture_frame()
    section = run_deflation(frame, meta, label="fixture",
                            sharpe_col="sharpe_annual_with_rut")
    html = render_deflation([section])
    page = _full_page(deflation_html=html,
                      hero=hero_from_section(section))
    assert f"{section.pbo.pbo:.3f}" in page                    # PBO chart marker
    assert f"{section.pbo_null_mean:.3f}" in page              # its null
    assert f"{section.headline.dsr:.3f}" in page               # hero + sensitivity dot
    assert f"{section.concentration.top_1_share:.1%}" in page  # concentration bar
    assert "<svg" in page and 'role="img"' in page             # charts are labelled
    assert page.count('aria-label') >= 3


def test_only_verified_tiers_carry_the_tick():
    """The tick is the auditor's 'vouched to source' mark. It appears on the
    measured and parsed gutters and nowhere else."""
    for tier in ("measured", "parsed"):
        assert '<span class="tick"></span>' in render_ledger(
            [LedgerEntry(tier, "l", "v", "r")])
    for tier in ("model", "unseen"):
        assert '<span class="tick"></span>' not in render_ledger(
            [LedgerEntry(tier, "l", "v", "r")])


def test_a_tier_that_did_not_run_still_occupies_its_cell():
    """A missing cell would read as nothing to report."""
    html = render_ledger([LedgerEntry("model", "Semantic pass", "Not run",
                                      "absence is not evidence of clean code", "s3")])
    assert "Not run" in html and "absence is not evidence" in html


def test_the_page_prints_to_a4():
    page = _full_page()
    assert "@page{size:A4" in page.replace("\n", "").replace(" ", "")
    assert "page-break-inside:avoid" in page or "break-inside:avoid" in page


def test_ledger_entries_are_escaped_like_everything_else():
    html = render_ledger([LedgerEntry("model", "<b>l</b>", "1 & 2", "a < b", "s3")])
    assert "<b>l</b>" not in html and "&lt;b&gt;l&lt;/b&gt;" in html
    assert "1 &amp; 2" in html and "a &lt; b" in html


def test_the_tool_writes_no_em_dashes_but_quotes_the_model_verbatim():
    """Two rules that pull in opposite directions, both pinned.

    Nothing BIA writes itself uses an em or en dash. Text that came from the
    model is evidence, so it is reproduced character for character: normalising
    a quotation would mean the page no longer shows what the model actually
    said.
    """
    page = _full_page()
    for dash in ("\u2014", "\u2013", "&mdash;", "&ndash;"):
        assert dash not in page, f"BIA wrote a {dash!r} of its own"

    quoted = "charged at P[t] \u2014 the same price used to generate the signal"
    sem = [make_finding(3, detector="llm", explanation=quoted)]
    html = render_semantic(annotate([], sem), ran=True, prompt_version="1.2.0",
                           n_rejected=0, n_capped=0)
    assert quoted in html
