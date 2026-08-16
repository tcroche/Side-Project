"""
Ground truth for the seeded-bug benchmark.

Every entry below was VERIFIED against the real scanner before being written
down: for trapped cases the recorded rule and line are exactly what
`auditor.ast_scan.audit_file` produces, and `tests/test_bench.py` re-asserts
that equality in CI, so this registry can never drift silently away from the
detectors it scores.

Case kinds
----------
  trapped    contains seeded catalogue leaks (AST territory)
  semantic   contains a seeded leak NO syntactic rule can express (LLM territory)
  clean      contains nothing; false-positive control
  dependent  a signal-only file whose causality depends on an unseen caller;
             the correct semantic answer is a QUESTION at severity "review"
             naming the external convention -- not a verdict, and not silence

Line ranges for semantic leaks are deliberately generous (the whole leaking
construct): a language model may anchor its finding on any line of it, and
overlap-based matching is the honest way to score localisation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")


@dataclass(frozen=True)
class Leak:
    leak_id: str
    family: str          # "catalogue" (AST territory) | "semantic" (LLM territory)
    rule_hint: str       # expected rule for catalogue leaks; "SEM" otherwise
    line_start: int
    line_end: int
    note: str


@dataclass(frozen=True)
class Case:
    filename: str
    kind: str            # "trapped" | "clean" | "semantic" | "dependent"
    leaks: tuple[Leak, ...] = ()


CASES: tuple[Case, ...] = (
    # --- trapped: catalogue leaks, one surgical seed per entry --------------
    Case("trap01_future_feature.py", "trapped", (
        Leak("t01", "catalogue", "R1", 11, 11, "feature is tomorrow's close"),
    )),
    Case("trap02_smoothed_signal.py", "trapped", (
        Leak("t02", "catalogue", "R2", 11, 11, "centred rolling window"),
    )),
    Case("trap03_full_sample_scaler.py", "trapped", (
        Leak("t03", "catalogue", "R3", 21, 21, "scaler fitted before the split"),
    )),
    Case("trap04_backfilled_prices.py", "trapped", (
        Leak("t04", "catalogue", "R8", 11, 11, "backward fill of price gaps"),
    )),
    Case("trap05_fit_on_test.py", "trapped", (
        Leak("t05", "catalogue", "R9", 22, 22, "model fitted on the test slice"),
    )),
    Case("trap06_global_zscore.py", "trapped", (
        Leak("t06", "catalogue", "R10", 12, 12, "whole-sample z-score signal"),
    )),
    Case("trap07_target_as_feature.py", "trapped", (
        Leak("t07", "catalogue", "R5", 12, 12, "feature built from label values"),
    )),
    Case("trap08_two_leaks.py", "trapped", (
        Leak("t08a", "catalogue", "R8", 10, 10, "fillna(method='bfill') loader"),
        Leak("t08b", "catalogue", "R1", 15, 15, "signal is P[t+2]"),
    )),
    # --- clean: the false-positive control ----------------------------------
    Case("clean01_causal_engine.py", "clean"),
    Case("clean02_lagged_features.py", "clean"),
    Case("clean03_train_only_scaler.py", "clean"),
    Case("clean04_alignment_and_sharpe.py", "clean"),
    Case("clean05_expanding_norm.py", "clean"),
    # --- semantic: invisible to syntax by design ----------------------------
    Case("sem01_same_day_aggregate.py", "semantic", (
        Leak("s01", "semantic", "SEM", 12, 17,
             "groupby(day).transform('mean'): every bar sees the whole session"),
    )),
    Case("sem02_indexed_lookahead.py", "semantic", (
        Leak("s02", "semantic", "SEM", 10, 19,
             "loop indexes prices[t + horizon] while building an input for t"),
    )),
    # --- dependent: the expected answer is a question ------------------------
    Case("dep01_signal_only.py", "dependent"),
)


def case_path(filename: str) -> str:
    return os.path.join(CASES_DIR, filename)


def cases_by_file() -> dict[str, Case]:
    return {case.filename: case for case in CASES}


def total_leaks(family: str | None = None) -> int:
    return sum(
        1
        for case in CASES
        for leak in case.leaks
        if family is None or leak.family == family
    )


__all__ = ["CASES", "CASES_DIR", "Case", "Leak", "case_path", "cases_by_file", "total_leaks"]
