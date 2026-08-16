"""Tests for the deterministic AST scanner.

Every rule gets at least one TRUE POSITIVE (the pattern must fire) and one TRUE
NEGATIVE (a legitimate near-miss must NOT fire). False positives on clean code
are the failure mode that destroys trust in a linter, so the negatives matter
as much as the positives.

Run from the project root: pytest -q
"""

from __future__ import annotations

import json
import textwrap

import pytest

from auditor.ast_scan import audit_source
from auditor.schema import MANUAL_CHECKLIST, RULES, findings_to_json


def rules_fired(code: str) -> set[str]:
    return {f.rule_id for f in audit_source(textwrap.dedent(code))}


# ---------------------------------------------------------------------------
# R1 -- negative shift
# ---------------------------------------------------------------------------


def test_r1_fires_on_negative_shift():
    assert "R1" in rules_fired("df['feature'] = df['close'].shift(-1)")


def test_r1_fires_on_negative_periods_keyword():
    assert "R1" in rules_fired("df['f'] = df['close'].shift(periods=-3)")


def test_r1_does_not_fire_on_a_positive_lag():
    assert "R1" not in rules_fired("df['feature'] = df['close'].shift(1)")


def test_r1_ignores_the_pattern_inside_a_string():
    """The reason this is an AST scanner and not a regular expression."""
    assert "R1" not in rules_fired("note = 'never call shift(-1) on a feature'")


def test_r1_ignores_the_pattern_inside_a_comment():
    assert "R1" not in rules_fired("x = 1  # do not use shift(-1) here")


def test_r1_exempts_label_construction_by_name():
    """R1's own fix text: 'A negative shift is only legitimate when building
    the TARGET.' The code now implements the exemption it was prescribing --
    found by seeding the benchmark, where trap07 builds its label with
    shift(-1) and R1 flagged the one construction its documentation calls
    legitimate."""
    assert "R1" not in rules_fired("target = close.pct_change().shift(-1)")


def test_r1_exempts_target_prefixed_names():
    assert "R1" not in rules_fired("y_next = close.shift(-1)")
    assert "R1" not in rules_fired("fwd_ret = close.pct_change().shift(-5)")


def test_r1_exemption_hands_reuse_of_the_label_to_r5():
    fired = rules_fired(
        """
        target = close.pct_change().shift(-1)
        edge = target.rolling(5).mean()
        """
    )
    assert "R1" not in fired
    assert "R5" in fired


def test_r1_exemption_is_narrow_a_subscript_label_still_fires():
    """When unsure, a leak detector keeps firing: only a BARE target-named
    variable is exempt, not a column assignment."""
    assert "R1" in rules_fired("df['target'] = df['close'].shift(-1)")


# ---------------------------------------------------------------------------
# R2 -- centred rolling window
# ---------------------------------------------------------------------------


def test_r2_fires_on_centred_window():
    assert "R2" in rules_fired("df['ma'] = df['close'].rolling(20, center=True).mean()")


def test_r2_does_not_fire_on_a_trailing_window():
    assert "R2" not in rules_fired("df['ma'] = df['close'].rolling(20).mean()")


def test_r2_does_not_fire_when_center_is_explicitly_false():
    assert "R2" not in rules_fired("df['ma'] = df['c'].rolling(20, center=False).mean()")


# ---------------------------------------------------------------------------
# R3 -- preprocessor fitted before the split
# ---------------------------------------------------------------------------


def test_r3_fires_when_a_scaler_is_fitted_on_everything():
    assert "R3" in rules_fired(
        """
        scaler = StandardScaler()
        scaler.fit(features)
        """
    )


def test_r3_does_not_fire_when_fitted_on_a_training_slice():
    assert "R3" not in rules_fired(
        """
        scaler = StandardScaler()
        scaler.fit(X_train)
        """
    )


def test_r3_does_not_fire_on_a_plain_model_fit():
    """A model fitted on unlabelled-by-name data is not a preprocessor leak."""
    assert "R3" not in rules_fired("model.fit(features, labels)")


# ---------------------------------------------------------------------------
# R4 -- possible same-bar execution
# ---------------------------------------------------------------------------


def test_r4_fires_on_unlagged_position_times_return():
    fired = rules_fired("pnl = position * returns")
    assert "R4" in fired


def test_r4_does_not_fire_when_the_position_is_lagged():
    assert "R4" not in rules_fired("pnl = position.shift(1) * returns")


def test_r4_is_always_review_severity():
    findings = audit_source("pnl = signal * ret")
    r4 = [f for f in findings if f.rule_id == "R4"]
    assert r4 and all(f.severity == "review" for f in r4)


# ---------------------------------------------------------------------------
# R5 -- feature built from the target
# ---------------------------------------------------------------------------


def test_r5_fires_when_a_feature_references_the_target():
    assert "R5" in rules_fired("df['feature'] = y * 2")


def test_r5_fires_on_a_forward_prefixed_source():
    assert "R5" in rules_fired("df['momentum'] = future_return.rolling(5).mean()")


def test_r5_does_not_fire_when_the_target_itself_is_built():
    assert "R5" not in rules_fired("y = df['close'].pct_change()")


def test_r5_does_not_fire_on_index_alignment():
    """Aligning on y_test.index uses metadata, not values.

    This was a genuine false positive found on a trapped benchmark script, and
    it is the kind that destroys trust in a linter faster than a missed finding.
    """
    assert "R5" not in rules_fired(
        "position = pd.Series(model.predict(X_test), index=y_test.index)"
    )
    assert "R5" not in rules_fired("returns = prices['close'].loc[y_test.index]")
    assert "R5" not in rules_fired("n = y_train.shape[0]")


def test_r5_still_fires_when_target_values_are_used():
    assert "R5" in rules_fired("feature = y_test.rolling(3).mean()")


def test_r5_fires_on_a_plain_alias_of_the_target():
    assert "R5" in rules_fired("df['feature'] = y")


class TestSelfAuditFalsePositives:
    """Regression corpus harvested by running the scanner on ITS OWN source.

    Auditing the auditor produced six findings, all false positives, in two
    classes: target-vocabulary homonyms used non-numerically (``targets`` as
    AST assignment targets, ``labels``/``label`` as axis labels), and R10
    matching a cross-sectional Sharpe ratio because any mean/std division
    looked like a normalisation. Each snippet below is lifted from the real
    file that fired.
    """

    def test_iterating_over_a_variable_named_targets_is_not_leakage(self):
        # auditor/ast_scan.py -- AST assignment targets, nothing to do with ML.
        assert "R5" not in rules_fired('assigned = " ".join(t for t in targets if t)')

    def test_listcomp_over_labels_is_not_leakage(self):
        # core/concentration.py -- observation labels for display.
        assert "R5" not in rules_fired("index = [str(x) for x in labels][: len(values)]")

    def test_getattr_on_label_is_not_leakage(self):
        # run_real_case.py -- formatting a date label.
        assert "R5" not in rules_fired('stamp = getattr(label, "date", lambda: label)()')

    def test_a_cross_sectional_sharpe_ratio_is_not_a_normalisation(self):
        # run_deflation_demo.py and tests -- mean/std over trials IS the point.
        assert "R10" not in rules_fired(
            "srs = matrix.mean(axis=0) / matrix.std(axis=0, ddof=1)"
        )

    def test_a_plain_sharpe_ratio_is_not_a_normalisation(self):
        assert "R10" not in rules_fired("sr = returns.mean() / returns.std()")

    def test_the_z_score_shape_still_fires_after_the_tightening(self):
        assert "R10" in rules_fired(
            "z = (df['close'] - df['close'].mean()) / df['close'].std()"
        )

    def test_min_max_scaling_still_fires(self):
        assert "R10" in rules_fired(
            "scaled = (x - x.min()) / (x.max() - x.min())"
        )

    def test_str_of_target_is_not_leakage(self):
        assert "R5" not in rules_fired("caption = str(y_test)")

    def test_len_of_target_is_not_leakage(self):
        assert "R5" not in rules_fired("n = len(y_train)")

    def test_but_arithmetic_inside_a_call_still_fires(self):
        # str(y * 2): the values feed a BinOp before reaching the builtin.
        assert "R5" in rules_fired("caption = str(y * 2)")


# ---------------------------------------------------------------------------
# R8 -- backward fill and interpolation
# ---------------------------------------------------------------------------


def test_r8_fires_on_bfill():
    assert "R8" in rules_fired("df = df.bfill()")


def test_r8_fires_on_fillna_with_a_backfill_method():
    assert "R8" in rules_fired("df = df.fillna(method='bfill')")


def test_r8_fires_on_interpolate():
    assert "R8" in rules_fired("df = df.interpolate()")


def test_r8_fires_on_reindex_with_backfill():
    assert "R8" in rules_fired("df = df.reindex(idx, method='bfill')")


def test_r8_does_not_fire_on_forward_fill():
    assert "R8" not in rules_fired("df = df.ffill()")


def test_r8_does_not_fire_on_a_constant_fill():
    assert "R8" not in rules_fired("df = df.fillna(0.0)")


# ---------------------------------------------------------------------------
# R9 -- fitting on test data
# ---------------------------------------------------------------------------


def test_r9_fires_when_fitting_on_the_test_set():
    assert "R9" in rules_fired("model.fit(X_test, y_test)")


def test_r9_fires_on_a_validation_named_variable():
    assert "R9" in rules_fired("search.fit(X_valid, y_valid)")


def test_r9_does_not_fire_on_a_training_fit():
    assert "R9" not in rules_fired("model.fit(X_train, y_train)")


def test_r9_does_not_fire_on_predicting_from_test_data():
    assert "R9" not in rules_fired("preds = model.predict(X_test)")


# ---------------------------------------------------------------------------
# R10 -- whole-sample normalisation
# ---------------------------------------------------------------------------


def test_r10_fires_on_a_full_sample_z_score():
    assert "R10" in rules_fired("z = (df['close'] - df['close'].mean()) / df['close'].std()")


def test_r10_does_not_fire_on_a_rolling_z_score():
    code = """
    roll = df['close'].rolling(30)
    z = (df['close'] - roll.mean()) / roll.std()
    """
    assert "R10" not in rules_fired(code)


def test_r10_does_not_fire_on_an_expanding_z_score():
    code = "z = (s - s.expanding().mean()) / s.expanding().std()"
    assert "R10" not in rules_fired(code)


# ---------------------------------------------------------------------------
# Engine-level guarantees
# ---------------------------------------------------------------------------


def test_every_finding_points_at_a_real_line():
    code = textwrap.dedent(
        """
        import pandas as pd

        df = pd.read_csv('prices.csv')
        df['f1'] = df['close'].shift(-1)
        df['f2'] = df['close'].rolling(10, center=True).mean()
        df = df.bfill()
        model.fit(X_test, y_test)
        """
    )
    n_lines = len(code.splitlines())
    findings = audit_source(code, filename="demo.py")
    assert findings
    for finding in findings:
        assert 1 <= finding.line_start <= n_lines
        assert finding.line_end >= finding.line_start
        assert finding.snippet


def test_findings_are_sorted_by_line():
    code = textwrap.dedent(
        """
        df = df.bfill()
        df['a'] = df['c'].shift(-1)
        df['b'] = df['c'].rolling(5, center=True).mean()
        """
    )
    lines = [f.line_start for f in audit_source(code)]
    assert lines == sorted(lines)


def test_clean_causal_code_produces_no_findings():
    """The false-positive control. This is a correct, causal backtest fragment."""
    code = textwrap.dedent(
        """
        import numpy as np
        import pandas as pd

        def compute_positions(close, window=120, k_entry=3.0):
            mean = close.rolling(window).mean()
            std = close.rolling(window).std()
            z = (close - mean) / std
            raw = np.where(z > k_entry, 1.0, np.where(z < -k_entry, -1.0, 0.0))
            return pd.Series(raw, index=close.index)

        def backtest(close):
            pos = compute_positions(close)
            dP = np.diff(close.values)
            held = pos.values[:-1]
            gross = float((held * dP).sum())
            return gross
        """
    )
    assert audit_source(code) == []


def test_a_syntax_error_is_reported_rather_than_raised():
    findings = audit_source("def broken(:\n    pass\n", filename="bad.py")
    assert len(findings) == 1
    assert findings[0].rule_id == "PARSE"


def test_findings_serialise_to_valid_json():
    findings = audit_source("df['f'] = df['c'].shift(-1)", filename="x.py")
    payload = json.loads(findings_to_json(findings))
    assert payload[0]["rule_id"] == "R1"
    assert payload[0]["detector"] == "ast"
    assert set(payload[0]) >= {
        "rule_id", "severity", "line_start", "line_end", "snippet",
        "explanation", "suggested_fix",
    }


def test_severity_values_are_within_the_declared_vocabulary():
    assert {rule.severity for rule in RULES.values()} <= {"high", "medium", "review"}


def test_every_rule_documents_what_it_cannot_see():
    for rule in RULES.values():
        assert rule.detectable, f"{rule.rule_id} has no scope statement"


def test_undetectable_checks_are_questions_not_rules():
    """Survivorship and point-in-time issues must not be claimed as detections."""
    assert not any(
        "survivor" in rule.title.lower() or "point-in-time" in rule.title.lower()
        for rule in RULES.values()
    )
    joined = " ".join(MANUAL_CHECKLIST).lower()
    assert "constituents" in joined and "point-in-time" in joined


@pytest.mark.parametrize("rule_id", ["R1", "R2", "R3", "R4", "R5", "R8", "R9", "R10"])
def test_rule_catalogue_is_complete(rule_id):
    rule = RULES[rule_id]
    assert rule.title and rule.explanation and rule.suggested_fix
