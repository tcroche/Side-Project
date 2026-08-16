"""Tests for core.stats and core.units.

Two kinds of checks:
  1. ANALYTIC cases where the answer is known in closed form and recomputed
     here by an independent expression (not by calling the function under test).
  2. INVARIANTS that must hold for any valid implementation (monotonicity,
     scaling, boundary behaviour).

Run from the project root:
    pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from core import units
from core.stats import (
    EULER_MASCHERONI,
    correlation_adjusted_variance,
    deflate,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    mean_offdiagonal_correlation,
    min_track_record_length,
    psr,
    psr_radicand,
    sharpe_moments,
    sr_variance_under_null,
)
from core.units import UnitError

# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def test_annualization_round_trip():
    sr_annual = 1.93
    sr_daily = units.to_per_period(sr_annual, units.TRADING_DAYS_PER_YEAR)
    assert sr_daily == pytest.approx(1.93 / math.sqrt(252))
    assert units.to_annual(sr_daily, units.TRADING_DAYS_PER_YEAR) == pytest.approx(1.93)


def test_guard_rejects_annualized_sharpe():
    """The whole point: an annualized Sharpe must not silently enter PSR."""
    with pytest.raises(UnitError):
        units.check_per_period_sharpe(1.93, name="sr_hat_per_period")


def test_guard_accepts_plausible_per_period_sharpe():
    value = units.check_per_period_sharpe(0.1216, name="sr_hat_per_period")
    assert value == pytest.approx(0.1216)


def test_lo_factor_reduces_to_sqrt_k_when_uncorrelated():
    assert units.lo_annualization_factor(0.0, 252) == pytest.approx(math.sqrt(252))


def test_positive_autocorrelation_lowers_annualization_factor():
    """Positive autocorrelation makes naive sqrt(k) scaling optimistic."""
    assert units.lo_annualization_factor(0.2, 252) < units.lo_annualization_factor(0.0, 252)


# ---------------------------------------------------------------------------
# PSR -- analytic cases
# ---------------------------------------------------------------------------


def test_psr_equals_one_half_when_benchmark_equals_estimate():
    """SR_hat == SR* gives z = 0, hence Phi(0) = 0.5 exactly."""
    assert psr(0.1, 0.1, n_obs=100, skew=0.0, kurtosis=3.0) == pytest.approx(0.5)


def test_psr_matches_independent_closed_form_under_normality():
    """Recompute the formula by hand rather than trusting the implementation."""
    sr_hat, sr_star, n_obs = 0.10, 0.0, 100
    skew, kurt = 0.0, 3.0

    radicand = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    expected_z = (sr_hat - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(radicand)
    expected = float(sps.norm.cdf(expected_z))

    assert psr(sr_hat, sr_star, n_obs, skew, kurt) == pytest.approx(expected)
    # Sanity on the magnitude: z is just under 1, so PSR sits in the low 0.8s.
    assert 0.80 < expected < 0.85


def test_psr_negative_skew_reduces_confidence():
    """Negative skew inflates the variance term, so PSR must fall."""
    base = psr(0.10, 0.0, 100, skew=0.0, kurtosis=3.0)
    negative_skew = psr(0.10, 0.0, 100, skew=-1.0, kurtosis=3.0)
    assert negative_skew < base


def test_psr_fat_tails_reduce_confidence():
    base = psr(0.10, 0.0, 100, skew=0.0, kurtosis=3.0)
    fat = psr(0.10, 0.0, 100, skew=0.0, kurtosis=12.0)
    assert fat < base


def test_psr_increases_with_sample_size():
    short = psr(0.10, 0.0, 50, 0.0, 3.0)
    long = psr(0.10, 0.0, 500, 0.0, 3.0)
    assert long > short


def test_psr_is_bounded_probability():
    for n in (10, 100, 5000):
        value = psr(0.05, 0.0, n, 0.0, 3.0)
        assert 0.0 <= value <= 1.0


def test_psr_rejects_annualized_input_by_default():
    with pytest.raises(UnitError):
        psr(1.93, 0.0, n_obs=250, skew=0.0, kurtosis=3.0)


def test_psr_allows_annualized_input_when_units_relaxed():
    value = psr(1.93, 0.0, 250, 0.0, 3.0, strict_units=False)
    assert value == pytest.approx(1.0, abs=1e-6)  # absurd, which is the point


def test_min_track_record_length_is_infinite_below_benchmark():
    assert min_track_record_length(0.05, 0.10, 0.0, 3.0) == math.inf


def test_min_track_record_length_matches_closed_form():
    sr_hat, sr_star, skew, kurt = 0.10, 0.02, 0.0, 3.0
    radicand = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    z = float(sps.norm.ppf(0.95))
    expected = 1.0 + radicand * (z / (sr_hat - sr_star)) ** 2
    assert min_track_record_length(sr_hat, sr_star, skew, kurt) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# E[max SR]
# ---------------------------------------------------------------------------


def test_single_trial_has_nothing_to_deflate():
    assert expected_max_sharpe(1, sr_variance_across_trials=0.01) == 0.0


def test_expected_max_sharpe_matches_closed_form():
    n, var = 30, 0.0025
    z1 = float(sps.norm.ppf(1.0 - 1.0 / n))
    z2 = float(sps.norm.ppf(1.0 - 1.0 / (n * math.e)))
    expected = math.sqrt(var) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    assert expected_max_sharpe(n, var) == pytest.approx(expected)


def test_expected_max_sharpe_increases_with_trials():
    var = 0.0025
    values = [expected_max_sharpe(n, var) for n in (2, 5, 10, 30, 100, 1000)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_expected_max_sharpe_scales_linearly_with_dispersion():
    """Doubling the standard deviation of trial Sharpes doubles E[max SR]."""
    base = expected_max_sharpe(30, 0.0025)          # sd = 0.05
    doubled = expected_max_sharpe(30, 0.0025 * 4)   # sd = 0.10
    assert doubled == pytest.approx(2.0 * base)


def test_expected_max_sharpe_approaches_sqrt_two_log_n_from_below():
    """The maximum of N normal draws grows like sqrt(2 ln N).

    Convergence is slow and strictly from below: the second-order term
    -(ln ln N + ln 4*pi) / (2*sqrt(2 ln N)) is negative. So the correct
    statement is that the ratio increases towards 1, not that it equals 1.
    """
    ratios = [
        expected_max_sharpe(n, 1.0) / math.sqrt(2.0 * math.log(n))
        for n in (10, 100, 1_000, 10_000, 100_000)
    ]
    assert all(r < 1.0 for r in ratios)
    assert all(b > a for a, b in zip(ratios, ratios[1:]))


def test_expected_max_sharpe_matches_exact_expectation_of_the_maximum():
    """Validate the Bailey & Lopez de Prado approximation numerically.

    The exact expectation of the maximum of N iid standard normals is

        E[max] = integral of x * N * phi(x) * Phi(x)**(N-1) dx

    The closed-form approximation should track it within a few percent over
    the range of trial counts a real grid search produces.
    """
    from scipy import integrate

    def exact_expected_max(n: int) -> float:
        integrand = lambda x: x * n * sps.norm.pdf(x) * sps.norm.cdf(x) ** (n - 1)
        value, _ = integrate.quad(integrand, -12.0, 12.0, limit=400)
        return value

    for n in (10, 30, 100, 1_000):
        approx = expected_max_sharpe(n, 1.0)
        exact = exact_expected_max(n)
        assert abs(approx - exact) / exact < 0.03, f"N={n}: {approx=} vs {exact=}"


def test_zero_dispersion_gives_zero_threshold():
    assert expected_max_sharpe(50, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Effective number of trials
# ---------------------------------------------------------------------------


def test_effective_trials_equals_n_when_independent():
    assert effective_number_of_trials(30, 0.0) == pytest.approx(30.0)


def test_effective_trials_collapses_when_perfectly_correlated():
    assert effective_number_of_trials(30, 0.999) == pytest.approx(1.0, abs=0.05)


def test_effective_trials_is_monotone_in_correlation():
    values = [effective_number_of_trials(30, rho) for rho in (0.0, 0.2, 0.5, 0.9)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_mean_offdiagonal_correlation_of_independent_columns_is_near_zero():
    rng = np.random.default_rng(7)
    mat = rng.normal(size=(4000, 12))
    assert abs(mean_offdiagonal_correlation(mat)) < 0.05


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------


def test_dsr_is_below_psr_against_zero():
    """Deflating against a positive threshold can only lower the probability."""
    kwargs = dict(n_obs=250, skew=0.0, kurtosis=3.0)
    naive = psr(0.10, 0.0, **kwargs)
    dsr = deflated_sharpe_ratio(
        0.10, n_trials=30, sr_variance_across_trials=0.0025, **kwargs
    )
    assert dsr < naive


def test_dsr_decreases_as_trials_increase():
    kwargs = dict(n_obs=250, skew=0.0, kurtosis=3.0, sr_variance_across_trials=0.0025)
    values = [deflated_sharpe_ratio(0.10, n_trials=n, **kwargs) for n in (2, 10, 50, 500)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_dsr_with_one_trial_equals_psr_against_zero():
    kwargs = dict(n_obs=250, skew=0.0, kurtosis=3.0)
    assert deflated_sharpe_ratio(
        0.10, n_trials=1, sr_variance_across_trials=0.0025, **kwargs
    ) == pytest.approx(psr(0.10, 0.0, **kwargs))


def test_dsr_is_a_probability_not_a_sharpe_ratio():
    """Guards the exact conflation this project exists to prevent."""
    value = deflated_sharpe_ratio(0.12, 250, 0.0, 3.0, 30, 0.0025)
    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Moments and end-to-end
# ---------------------------------------------------------------------------


def test_sharpe_moments_recovers_known_parameters():
    rng = np.random.default_rng(42)
    mu, sigma, n = 0.0008, 0.01, 20_000
    returns = rng.normal(mu, sigma, n)
    m = sharpe_moments(returns, periods_per_year=252)

    assert m.n_obs == n
    assert m.sr_per_period == pytest.approx(mu / sigma, abs=0.01)
    assert m.skew == pytest.approx(0.0, abs=0.05)
    assert m.kurtosis == pytest.approx(3.0, abs=0.10)
    assert m.sr_annual == pytest.approx(m.sr_per_period * math.sqrt(252))


def test_sharpe_moments_rejects_constant_series():
    with pytest.raises(ValueError):
        sharpe_moments(np.ones(100))


def test_deflate_report_is_consistent_and_readable():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0006, 0.01, 1000)
    m = sharpe_moments(returns, periods_per_year=252)

    report = deflate(m, n_trials=30, sr_variance_across_trials=0.0025)

    assert 0.0 <= report.dsr <= 1.0
    assert report.dsr <= report.psr_vs_zero
    assert report.verdict in {"PASS", "REJECT"}
    assert report.passes == (report.dsr >= report.confidence)
    assert report.expected_max_sr_annual == pytest.approx(
        report.expected_max_sr_per_period * math.sqrt(252)
    )
    text = report.to_text()
    assert "NOT a Sharpe ratio" in text
    assert "per period" in text


def test_empirical_v_sr_does_not_get_a_second_correlation_adjustment():
    """The cross-sectional V[SR] already absorbs trial correlation.

    Applying N_eff on top would double-count and can remove the deflation
    entirely. With an empirical V[SR], rho is reported but not applied.
    """
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0006, 0.01, 1000)
    m = sharpe_moments(returns, periods_per_year=252)

    plain = deflate(m, n_trials=30, sr_variance_across_trials=0.0025)
    with_rho = deflate(
        m, n_trials=30, sr_variance_across_trials=0.0025, mean_trial_correlation=0.9
    )
    assert with_rho.dsr == pytest.approx(plain.dsr)
    assert with_rho.dsr_correlation_adjusted is None
    assert with_rho.n_trials_effective is not None and with_rho.n_trials_effective < 30
    assert any("already absorbed" in note for note in with_rho.notes)


def test_marginal_v_sr_is_shrunk_by_one_minus_rho():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0006, 0.01, 1000)
    m = sharpe_moments(returns, periods_per_year=252)

    report = deflate(
        m,
        n_trials=30,
        sr_variance_across_trials=sr_variance_under_null(m.n_obs),
        mean_trial_correlation=0.9,
        v_sr_source="marginal",
    )
    assert report.dsr_correlation_adjusted is not None
    assert report.dsr_correlation_adjusted > report.dsr
    assert any("shrunk by" in note for note in report.notes)


def test_deflate_rejects_an_unknown_variance_source():
    m = sharpe_moments(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    with pytest.raises(ValueError, match="v_sr_source"):
        deflate(m, 10, 0.0025, v_sr_source="whatever")


def test_correlation_adjusted_variance_matches_one_minus_rho():
    assert correlation_adjusted_variance(0.01, 0.75) == pytest.approx(0.0025)
    assert correlation_adjusted_variance(0.01, 0.0) == pytest.approx(0.01)
    assert correlation_adjusted_variance(0.01, -0.3) == pytest.approx(0.01)


def test_cross_sectional_variance_absorbs_correlation_by_construction():
    """The identity the correction rests on: E[cross-sectional V] = sigma^2 (1 - rho)."""
    rng = np.random.default_rng(0)
    n_obs, n_trials, rho = 103, 18, 0.9
    empirical = []
    for _ in range(1500):
        common = rng.normal(0, 1, size=(n_obs, 1))
        idio = rng.normal(0, 1, size=(n_obs, n_trials))
        panel = math.sqrt(rho) * common + math.sqrt(1 - rho) * idio
        srs = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
        empirical.append(np.var(srs, ddof=1))
    marginal = sr_variance_under_null(n_obs)
    assert float(np.mean(empirical)) / marginal == pytest.approx(1 - rho, abs=0.03)


def test_psr_radicand_below_one_means_skew_is_helping():
    positive_skew = psr_radicand(0.12, skew=6.0, kurtosis=50.0)
    normal = psr_radicand(0.12, skew=0.0, kurtosis=3.0)
    assert positive_skew < 1.0 < normal or positive_skew < normal
    assert psr(0.12, 0.0, 103, 6.0, 50.0) > psr(0.12, 0.0, 103, 0.0, 3.0)


def test_min_trl_under_normality_is_reported_separately():
    """Fat right tails shorten the apparent track record requirement."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0006, 0.01, 300)
    returns[10] += 0.15  # one enormous positive day
    m = sharpe_moments(returns, periods_per_year=252, moment_estimator="simple")
    report = deflate(m, n_trials=18, sr_variance_across_trials=0.0002)
    assert report.min_track_record_periods < report.min_track_record_periods_if_normal
    assert report.variance_term < 1.0


def test_pure_noise_strategies_are_rejected():
    """Known-truth validation: 30 strategies with zero true edge.

    The best of 30 noise strategies looks good against SR* = 0, but the DSR
    must not accept it. This is the case where the right answer is known by
    construction.
    """
    rng = np.random.default_rng(2024)
    n_trials, n_obs = 30, 500
    matrix = rng.normal(0.0, 0.01, size=(n_obs, n_trials))  # true Sharpe = 0

    srs = matrix.mean(axis=0) / matrix.std(axis=0, ddof=1)
    best = int(np.argmax(srs))

    m = sharpe_moments(matrix[:, best], periods_per_year=252)
    report = deflate(m, n_trials=n_trials, sr_variance_across_trials=float(np.var(srs, ddof=1)))

    assert report.psr_vs_zero > 0.90, "the winner should look good naively"
    assert not report.passes, "the DSR must reject a strategy with no true edge"
