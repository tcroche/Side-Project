"""Truth-known tests for the seeded-bug benchmark.

Two guarantees, both executable:

1. The ground-truth registry can never drift from the detectors: for every
   trapped case, the recorded (rule, line) pairs must EQUAL what the real
   scanner produces, and every non-trapped case must be AST-silent -- the
   "invisible to syntax" claim about the semantic cases, made a test.

2. The scoring itself is validated on miniatures where every number is known
   by hand: threshold behaviour, overlap matching, precision/recall/F1,
   control-file false positives, and the no-NaN edge.
"""

from __future__ import annotations

import glob
import os

import pytest

from auditor.ast_scan import audit_file
from auditor.schema import Finding
from bench.score import (
    DETECTION_SEVERITIES,
    is_detection,
    merge_findings,
    overlaps,
    score_detector,
)
from bench.truth import CASES, CASES_DIR, Case, Leak, case_path

TRAPPED = [case for case in CASES if case.kind == "trapped"]
AST_SILENT = [case for case in CASES if case.kind != "trapped"]


# ---------------------------------------------------------------------------
# 1) The registry and the detectors agree, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", TRAPPED, ids=[c.filename for c in TRAPPED])
def test_trapped_case_fires_exactly_its_seeded_rules(case):
    actual = {(f.rule_id, f.line_start) for f in audit_file(case_path(case.filename))}
    expected = {(leak.rule_hint, leak.line_start) for leak in case.leaks}
    assert actual == expected, (
        f"{case.filename}: scanner produced {sorted(actual)}, "
        f"truth records {sorted(expected)} -- one of them is wrong."
    )


@pytest.mark.parametrize("case", AST_SILENT, ids=[c.filename for c in AST_SILENT])
def test_non_trapped_case_is_ast_silent(case):
    findings = audit_file(case_path(case.filename))
    assert findings == [], (
        f"{case.filename} ({case.kind}) fired {[str(f) for f in findings]} -- "
        f"either the case leaks syntactically or a rule lost precision."
    )


def test_registry_and_disk_agree_both_ways():
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(CASES_DIR, "*.py"))}
    in_truth = {case.filename for case in CASES}
    assert on_disk == in_truth


def test_truth_lines_exist_in_their_files():
    for case in CASES:
        n_lines = len(open(case_path(case.filename), encoding="utf-8").read().splitlines())
        for leak in case.leaks:
            assert 1 <= leak.line_start <= leak.line_end <= n_lines


# ---------------------------------------------------------------------------
# 2) The scoring, validated on hand-checkable miniatures
# ---------------------------------------------------------------------------


def make_finding(line_start, line_end=None, severity="high", detector="ast"):
    return Finding(
        rule_id="RX",
        title="miniature",
        severity=severity,
        line_start=line_start,
        line_end=line_end if line_end is not None else line_start,
        snippet="",
        explanation="miniature",
        suggested_fix="miniature",
        detector=detector,
        filename="mini.py",
    )


MINI_CASES = (
    Case("mini.py", "trapped", (
        Leak("m1", "catalogue", "RX", 10, 12, "seeded"),
    )),
    Case("mini_clean.py", "clean"),
)


def test_review_findings_are_questions_not_detections():
    assert DETECTION_SEVERITIES == {"high", "medium"}
    assert not is_detection(make_finding(10, severity="review"))
    assert is_detection(make_finding(10, severity="medium"))


def test_overlap_is_inclusive_on_both_edges():
    leak = MINI_CASES[0].leaks[0]                      # lines 10..12
    assert overlaps(make_finding(12, 20), leak)        # touches the end
    assert overlaps(make_finding(1, 10), leak)         # touches the start
    assert not overlaps(make_finding(13, 15), leak)
    assert not overlaps(make_finding(1, 9), leak)


def test_precision_recall_f1_on_a_hand_checked_scenario():
    findings = {
        "mini.py": [
            make_finding(11),                          # TP
            make_finding(40, severity="medium"),       # FP (no leak there)
            make_finding(10, severity="review"),       # question, not scored
        ],
    }
    score = score_detector("mini", findings, MINI_CASES)
    assert (score.tp, score.fp, score.n_detections) == (1, 1, 2)
    assert score.precision == pytest.approx(0.5)
    assert score.recall_overall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(2 / 3)
    assert score.review_questions == 1


def test_two_detections_on_one_leak_are_two_true_positives():
    findings = {"mini.py": [make_finding(10), make_finding(11, detector="llm")]}
    score = score_detector("mini", findings, MINI_CASES)
    assert (score.tp, score.fp) == (2, 0)
    assert score.leaks_found_catalogue == 1            # recall counts the LEAK once


def test_detection_on_a_control_file_is_a_clean_false_positive():
    findings = {"mini_clean.py": [make_finding(5)]}
    score = score_detector("mini", findings, MINI_CASES)
    assert score.fp == 1
    assert score.clean_false_positives == 1
    assert score.recall_overall == 0.0                 # the seeded leak was missed


def test_zero_detections_yield_zeros_not_nans():
    score = score_detector("empty", {}, MINI_CASES)
    assert (score.precision, score.recall_overall, score.f1) == (0.0, 0.0, 0.0)


def test_hybrid_union_recovers_what_one_detector_missed():
    ast_only = {"mini.py": [make_finding(40, severity="medium")]}   # FP only
    llm_only = {"mini.py": [make_finding(11, detector="llm")]}      # the TP
    hybrid = merge_findings(ast_only, llm_only)
    score = score_detector("hybrid", hybrid, MINI_CASES)
    assert score.recall_overall == pytest.approx(1.0)
    assert (score.tp, score.fp) == (1, 1)


def test_semantic_family_recall_is_tracked_separately():
    cases = (
        Case("sem_mini.py", "semantic", (
            Leak("s", "semantic", "SEM", 3, 8, "seeded semantic"),
        )),
    )
    hit = {"sem_mini.py": [make_finding(5, severity="medium", detector="llm")]}
    score = score_detector("llm", hit, cases)
    assert score.recall_semantic == pytest.approx(1.0)
    assert score.recall_catalogue == 0.0               # no catalogue leaks exist
    assert score.tp_semantic == 1 and score.tp_catalogue == 0
