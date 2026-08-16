"""
Finding schema and the deterministic rule catalogue.

Scope discipline
----------------
Only patterns that are genuinely visible in the SYNTAX of a Python file belong
here. Two categories that are commonly listed as "leakage rules" are excluded
on purpose, because static analysis cannot decide them:

  * survivorship bias from an index's CURRENT constituents -- the code that
    reads a ticker list looks identical whether the list is point-in-time or
    not. This is a property of the DATA, not of the syntax.
  * non-point-in-time fundamentals (restated values used at the original date)
    -- likewise a data-vintage question.

Both are surfaced as questions in the report's manual checklist rather than
claimed as detections. Claiming to detect them would be the same kind of
overstatement this tool exists to catch.

Severity
--------
  high    the pattern uses future information in essentially every context
  medium  the pattern usually leaks, but a legitimate use exists
  review  a heuristic; a human has to look. Never merged with the two above.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

Severity = Literal["high", "medium", "review"]
Detector = Literal["ast", "llm"]


@dataclass(frozen=True)
class Finding:
    """One leakage finding, always anchored to real line numbers.

    The last two fields exist for the semantic (LLM) detector only and stay
    None for AST findings:

      * external_dependency -- the exact out-of-file fact the finding hinges
        on, as declared by the model (None = the model declared the leak
        established within this file alone);
      * capped_from -- when the harness deterministically capped the severity
        at "review" (dependency declared, or field missing), the severity the
        model originally claimed. None = no cap was applied.
    """

    rule_id: str
    title: str
    severity: Severity
    line_start: int
    line_end: int
    snippet: str
    explanation: str
    suggested_fix: str
    detector: Detector = "ast"
    filename: str = ""
    external_dependency: str | None = None
    capped_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        where = f"{self.filename}:{self.line_start}" if self.filename else f"line {self.line_start}"
        return f"[{self.severity.upper():6}] {self.rule_id} {where} -- {self.title}"


def findings_to_json(findings: list[Finding], *, indent: int = 2) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=indent)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    severity: Severity
    explanation: str
    suggested_fix: str
    detectable: str  # what the AST can and cannot see, stated plainly


RULES: dict[str, Rule] = {
    "R1": Rule(
        rule_id="R1",
        title="Negative shift on a feature",
        severity="high",
        explanation=(
            "A negative shift moves future values backwards onto the current "
            "row, so the feature at time t contains information from t+k. This "
            "is the most direct form of look-ahead there is."
        ),
        suggested_fix=(
            "Use a positive shift to lag a feature. A negative shift is only "
            "legitimate when building the TARGET of a supervised problem, and "
            "the target must never be fed back in as an input."
        ),
        detectable=(
            "Fully detectable: the sign of the shift argument is syntactic. "
            "One deliberate exemption: a negative shift assigned to a bare "
            "target-named variable (label construction) does not fire, as "
            "the fix text prescribes; re-use of that label as a feature is "
            "R5's territory."
        ),
    ),
    "R2": Rule(
        rule_id="R2",
        title="Centred rolling window",
        severity="high",
        explanation=(
            "A centred window spans observations on both sides of the current "
            "point, so half of every value is computed from the future."
        ),
        suggested_fix=(
            "Drop center=True. Rolling windows default to trailing, which is "
            "causal, and that default is what a backtest needs."
        ),
        detectable="Fully detectable: center=True is an explicit keyword.",
    ),
    "R3": Rule(
        rule_id="R3",
        title="Preprocessor fitted before the train/test split",
        severity="medium",
        explanation=(
            "A scaler, decomposition or feature selector fitted on the whole "
            "dataset learns statistics that include the test period, so the "
            "test set is no longer unseen."
        ),
        suggested_fix=(
            "Fit on the training slice only and transform the test slice with "
            "that fitted object, or put the preprocessor inside a Pipeline "
            "evaluated by a time-aware splitter."
        ),
        detectable=(
            "Partially detectable: the call is visible, but whether its "
            "argument is a training slice often is not. Reported as medium."
        ),
    ),
    "R4": Rule(
        rule_id="R4",
        title="Possible same-bar execution",
        severity="review",
        explanation=(
            "A position appears to be multiplied by a return over the same "
            "index, with no lag anywhere in the expression. If the signal is "
            "computed from a bar's close and the trade is booked at that same "
            "close, the trade uses a price that was not knowable when the "
            "decision was made."
        ),
        suggested_fix=(
            "Hold the position decided at t over the interval t to t+1: either "
            "lag the position by one bar, or write the P&L explicitly as "
            "position[t] * (price[t+1] - price[t])."
        ),
        detectable=(
            "Heuristic only. Naming conventions drive it, and an engine that "
            "indexes the lag inside a loop will not match. Always 'review'."
        ),
    ),
    "R5": Rule(
        rule_id="R5",
        title="Feature built from the target",
        severity="high",
        explanation=(
            "A feature is assigned from an expression that references the "
            "target variable. The model then sees the answer among its inputs."
        ),
        suggested_fix=(
            "Build features only from information available at prediction "
            "time. If a transformation of the target is genuinely needed, it "
            "belongs to the label pipeline, not the feature matrix."
        ),
        detectable=(
            "Partially detectable through naming (y, target, label, "
            "future_/fwd_ prefixes). Renaming defeats it."
        ),
    ),
    "R8": Rule(
        rule_id="R8",
        title="Backward fill or interpolation over time",
        severity="high",
        explanation=(
            "Backward filling propagates a later observation onto an earlier "
            "timestamp. Interpolation does the same thing more smoothly: the "
            "interpolated value depends on the next known point."
        ),
        suggested_fix=(
            "Use a forward fill for time series. If a gap cannot be filled "
            "causally, leave it missing and handle it downstream."
        ),
        detectable="Fully detectable: bfill, backfill and interpolate are named calls.",
    ),
    "R9": Rule(
        rule_id="R9",
        title="Model or search fitted on test data",
        severity="high",
        explanation=(
            "Fitting on a variable whose name marks it as the test or "
            "validation set turns the held-out data into training data, and "
            "the reported score into an in-sample score."
        ),
        suggested_fix=(
            "Fit on the training set, tune on a validation split carved out of "
            "it, and touch the test set once, at the very end."
        ),
        detectable="Detectable through argument naming, which is reliable in practice.",
    ),
    "R10": Rule(
        rule_id="R10",
        title="Normalisation by whole-sample statistics",
        severity="medium",
        explanation=(
            "Subtracting a mean or dividing by a standard deviation computed "
            "over the entire history injects information from the end of the "
            "sample into rows at the beginning."
        ),
        suggested_fix=(
            "Use an expanding or rolling statistic, or compute the statistic "
            "on the training period and freeze it for everything after."
        ),
        detectable=(
            "Partially detectable: a z-score shape is visible, but whether the "
            "statistic came from the full sample or a training slice is not "
            "always. Reported as medium."
        ),
    ),
}

#: Checks a syntax tree cannot make. These are printed as questions, never as findings.
MANUAL_CHECKLIST: list[str] = [
    "Is the trading universe defined from constituents as of each date, or from "
    "today's index membership? Static analysis cannot tell these apart.",
    "Are fundamental or accounting inputs point-in-time, or restated values "
    "attributed to their original date?",
    "Do corporate actions, delistings and halted names appear in the history, or "
    "were they dropped when the data was pulled?",
    "Does the transaction-cost model reflect the liquidity actually available at "
    "the size being simulated?",
    "How many configurations were tried in total, including the ones that were "
    "abandoned and never written down?",
]


__all__ = [
    "Finding",
    "Rule",
    "RULES",
    "MANUAL_CHECKLIST",
    "Severity",
    "Detector",
    "findings_to_json",
]
