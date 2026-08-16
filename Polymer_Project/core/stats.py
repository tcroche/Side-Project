"""
Statistical core of the Backtest Integrity Auditor.

Implements, with explicit units at every step:

  * PSR  -- Probabilistic Sharpe Ratio      (Bailey & Lopez de Prado, 2012)
  * E[max SR] under the null of no skill    (Bailey & Lopez de Prado, 2014)
  * DSR  -- Deflated Sharpe Ratio           (Bailey & Lopez de Prado, 2014)
  * N_eff -- heuristic effective trial count under correlated trials

IMPORTANT -- what PSR and DSR actually are
------------------------------------------
PSR and DSR are PROBABILITIES in [0, 1], not Sharpe ratios. DSR = 0.92 means
"there is a 92% probability that the true Sharpe ratio exceeds the threshold
that pure luck would have produced given N trials". It does NOT mean "a
deflated Sharpe ratio of 0.92". The conventional acceptance threshold is 0.95,
so DSR = 0.92 is a REJECTION, not a mild degradation.

References
----------
Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
    Frontier. Journal of Risk, 15(2), 3-44.
Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
    Journal of Portfolio Management, 40(5), 94-107.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats as sps

from core.units import (
    TRADING_DAYS_PER_YEAR,
    UnitError,
    check_per_period_sharpe,
    to_annual,
)

#: Euler-Mascheroni constant, used in the E[max SR] approximation.
EULER_MASCHERONI = 0.577215664901532860606512090082

#: Conventional acceptance threshold for PSR / DSR (one-sided, 95%).
DEFAULT_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Descriptive moments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeMoments:
    """Per-period Sharpe ratio and the higher moments PSR needs."""

    sr_per_period: float
    skew: float
    kurtosis: float  # non-excess: a normal distribution gives 3.0
    n_obs: int
    periods_per_year: int

    @property
    def sr_annual(self) -> float:
        return to_annual(self.sr_per_period, self.periods_per_year)

    @property
    def excess_kurtosis(self) -> float:
        return self.kurtosis - 3.0


def sharpe_moments(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_per_period: float = 0.0,
    ddof: int = 1,
    moment_estimator: str = "unbiased",
) -> SharpeMoments:
    """Compute the per-period Sharpe ratio and the moments PSR requires.

    `returns` must be a series of PERIODIC returns (e.g. daily), not prices and
    not cumulative returns. The resulting Sharpe ratio is per-period, matching
    the periodicity of `returns`, which is exactly what `psr` expects.

    Kurtosis is returned in NON-EXCESS form (normal = 3.0), the convention used
    in the PSR formula as published.

    `moment_estimator` selects the skewness / kurtosis estimator:

    - ``"unbiased"`` (default): the bias-corrected sample estimators, as in
      ``scipy.stats.skew(bias=False)``.
    - ``"simple"``: the plain moment ratios ``mean(z**3)`` and ``mean(z**4)``
      where ``z`` is standardized with ``ddof=1``. This is what the M2
      backtester's ``dsr.py`` uses, so it is the setting that reproduces the
      original figures exactly. The two differ by a factor of about
      ``sqrt(n(n-1))/(n-2)`` on the skewness, which is 1.5% at n = 103.

    Neither is wrong; what matters is saying which one produced a published
    number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_obs = arr.size
    if n_obs < 3:
        raise ValueError(f"Need at least 3 finite returns, got {n_obs}.")

    excess = arr - risk_free_per_period
    sd = float(np.std(excess, ddof=ddof))
    if sd <= 0.0:
        raise ValueError("Return series has zero volatility; Sharpe ratio undefined.")

    sr = float(np.mean(excess)) / sd

    if moment_estimator == "unbiased":
        skew = float(sps.skew(excess, bias=False))
        kurt = float(sps.kurtosis(excess, fisher=False, bias=False))
    elif moment_estimator == "simple":
        z = (excess - np.mean(excess)) / sd
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4))
    else:
        raise ValueError(
            f"moment_estimator must be 'unbiased' or 'simple', got {moment_estimator!r}."
        )

    return SharpeMoments(
        sr_per_period=sr,
        skew=skew,
        kurtosis=kurt,
        n_obs=n_obs,
        periods_per_year=periods_per_year,
    )


def sr_variance_under_null(n_obs: int, sr_per_period: float = 0.0) -> float:
    """Theoretical sampling variance of an estimated Sharpe ratio (Lo, 2002, IID).

        Var(SR_hat) = (1 + SR**2 / 2) / T

    Under the null of no skill (SR = 0) this is simply 1/T.

    Why this matters for the DSR
    ----------------------------
    E[max SR] needs V[SR], the variance of the trial Sharpe ratios. The usual
    estimate is the empirical variance across the N trials, but that quantity
    conflates two things: the sampling noise of each estimate, and genuine
    differences in true Sharpe ratio between configurations. Bailey & Lopez de
    Prado's derivation assumes all trials have a true Sharpe of zero, so V[SR]
    should be pure sampling noise.

    Comparing the empirical V[SR] with this theoretical value is a cheap
    diagnostic:

    - empirical ~ theoretical -> the trials really do look like noise, and the
      deflation is well specified;
    - empirical >> theoretical -> the configurations genuinely differ, so V[SR]
      is inflated and the DSR over-deflates (conservative);
    - empirical << theoretical -> the trials are so correlated that they barely
      explore anything, so N overstates the real number of independent bets.

    Report both. It is a one-line check that anticipates the question "how did
    you estimate V[SR]?".
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}.")
    return (1.0 + 0.5 * sr_per_period**2) / n_obs


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------


def psr(
    sr_hat_per_period: float,
    sr_benchmark_per_period: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    strict_units: bool = True,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark).

        PSR(SR*) = Phi[ (SR_hat - SR*) * sqrt(T - 1)
                        / sqrt(1 - g3*SR_hat + ((g4 - 1)/4) * SR_hat**2) ]

    All Sharpe ratios must be PER-PERIOD and on the same time scale as `n_obs`,
    `skew` and `kurtosis`. Passing an annualized Sharpe here is the classic
    error; `strict_units=True` makes it raise instead of returning a
    meaningless number.

    Parameters
    ----------
    sr_hat_per_period : float
        Observed (estimated) Sharpe ratio, per period.
    sr_benchmark_per_period : float
        Threshold SR*, per period. Use 0.0 for "better than nothing"; use
        `expected_max_sharpe(...)` to obtain the DSR.
    n_obs : int
        Number of return observations T.
    skew : float
        Sample skewness of the returns (gamma_3).
    kurtosis : float
        Sample kurtosis of the returns, NON-EXCESS (gamma_4; normal = 3.0).

    Returns
    -------
    float
        A probability in [0, 1].
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}.")

    check_per_period_sharpe(
        sr_hat_per_period, name="sr_hat_per_period", strict=strict_units
    )
    check_per_period_sharpe(
        sr_benchmark_per_period, name="sr_benchmark_per_period", strict=strict_units
    )

    radicand = (
        1.0
        - skew * sr_hat_per_period
        + ((kurtosis - 1.0) / 4.0) * sr_hat_per_period**2
    )
    if radicand <= 0.0:
        raise ValueError(
            f"PSR variance term is non-positive (radicand={radicand:.6g}). This "
            f"happens with extreme skew/kurtosis combined with a large Sharpe "
            f"ratio; the asymptotic approximation does not hold here."
        )

    z = (sr_hat_per_period - sr_benchmark_per_period) * math.sqrt(n_obs - 1) / math.sqrt(
        radicand
    )
    return float(sps.norm.cdf(z))


def min_track_record_length(
    sr_hat_per_period: float,
    sr_benchmark_per_period: float,
    skew: float,
    kurtosis: float,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """MinTRL: number of observations needed for PSR to reach `confidence`.

    Bailey & Lopez de Prado (2012). Answers "how long must the track record be
    before this Sharpe ratio is statistically distinguishable from SR*?".
    Returned in the same periodicity as the returns.
    """
    diff = sr_hat_per_period - sr_benchmark_per_period
    if diff <= 0.0:
        return math.inf
    radicand = (
        1.0
        - skew * sr_hat_per_period
        + ((kurtosis - 1.0) / 4.0) * sr_hat_per_period**2
    )
    if radicand <= 0.0:
        raise ValueError("PSR variance term is non-positive; MinTRL undefined.")
    z = float(sps.norm.ppf(confidence))
    return 1.0 + radicand * (z / diff) ** 2


# ---------------------------------------------------------------------------
# Expected maximum Sharpe ratio under the null
# ---------------------------------------------------------------------------


def expected_max_sharpe(
    n_trials: int,
    sr_variance_across_trials: float,
) -> float:
    """E[max SR] over N independent trials whose TRUE Sharpe ratio is zero.

        E[max_N SR] ~ sqrt(V[SR]) * [ (1 - g) * Phi^-1(1 - 1/N)
                                      + g * Phi^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant. This is the Sharpe ratio that pure
    luck would produce given N attempts: the benchmark the DSR deflates against.

    `sr_variance_across_trials` is the variance of the ESTIMATED Sharpe ratios
    across the N trials, in per-period units. It measures how much the trials
    scatter, i.e. how much room luck had.

    The maximum of N standard normal draws grows like sqrt(2 * ln N): this is
    why "trying more configurations" mechanically inflates the best observed
    Sharpe ratio, with no skill involved.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if sr_variance_across_trials < 0.0:
        raise ValueError("sr_variance_across_trials must be >= 0.")
    if n_trials == 1:
        # A single trial involves no selection, so there is nothing to deflate.
        return 0.0

    sd = math.sqrt(sr_variance_across_trials)
    z1 = float(sps.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def correlation_adjusted_variance(
    sr_variance_marginal: float, mean_correlation: float
) -> float:
    """Shrink a MARGINAL variance of trial Sharpe ratios for trial correlation.

        V_effective = V_marginal * (1 - rho_bar)

    Why this, and not an effective trial count
    ------------------------------------------
    Write each trial's estimated Sharpe ratio under the null as
    ``e_i = sigma * (sqrt(rho) * Z + sqrt(1 - rho) * Y_i)``, with ``Z`` a factor
    common to every trial and ``Y_i`` independent. Then

        max_i e_i = sigma * (sqrt(rho) * Z + sqrt(1 - rho) * max_i Y_i)

    so ``E[max] = sigma * sqrt(1 - rho) * E[max of N independent draws]``.
    Correlation shrinks the luck threshold through the variance, not through N.

    The practical consequence is important: the CROSS-SECTIONAL sample variance
    of the N estimated Sharpe ratios already equals ``sigma**2 * (1 - rho)`` in
    expectation, because the common component cancels out of a cross-sectional
    variance. So when V[SR] is estimated empirically across trials, the
    correlation is ALREADY absorbed and no further adjustment is legitimate.
    Apply this function only when V[SR] comes from a marginal source such as
    `sr_variance_under_null`.
    """
    if not -1.0 < mean_correlation < 1.0:
        raise ValueError("mean_correlation must lie strictly between -1 and 1.")
    if sr_variance_marginal < 0.0:
        raise ValueError("sr_variance_marginal must be >= 0.")
    return sr_variance_marginal * (1.0 - max(mean_correlation, 0.0))


def psr_radicand(sr_hat_per_period: float, skew: float, kurtosis: float) -> float:
    """The variance term inside the PSR square root.

        1 - g3 * SR + ((g4 - 1) / 4) * SR**2

    Worth inspecting directly. Below 1 it INFLATES the z-score, so positive
    skewness makes a strategy look better, not worse. As it approaches zero the
    asymptotic approximation stops being usable, and below zero the formula has
    no meaning at all.
    """
    return (
        1.0
        - skew * sr_hat_per_period
        + ((kurtosis - 1.0) / 4.0) * sr_hat_per_period**2
    )


def effective_number_of_trials(
    n_trials: int,
    mean_correlation: float,
) -> float:
    """Heuristic effective trial count under equicorrelated trials.

        N_eff = N / (1 + (N - 1) * rho_bar)

    WARNING -- do not combine this with an empirical V[SR].
    ------------------------------------------------------
    This adjustment and the cross-sectional estimate of V[SR] correct for the
    SAME thing. Using both double-counts the correlation and can drive N_eff to
    1, which removes the deflation entirely. See `correlation_adjusted_variance`
    for the correct treatment. It is kept here because it is the answer people
    expect to "what if the trials are correlated?", and because it is a
    reasonable descriptive statistic of how much independent exploration a grid
    really contains.

    Report it ALONGSIDE the naive N, labelled as a heuristic, never as the
    number that produced a published DSR.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if not -1.0 < mean_correlation < 1.0:
        raise ValueError("mean_correlation must lie strictly between -1 and 1.")
    rho = max(mean_correlation, 0.0)
    n_eff = n_trials / (1.0 + (n_trials - 1) * rho)
    return float(min(max(n_eff, 1.0), n_trials))


def mean_offdiagonal_correlation(returns_matrix: np.ndarray) -> float:
    """Mean pairwise correlation between trial return series.

    `returns_matrix` is T x N: one column per trial, one row per period.
    Feeds `effective_number_of_trials`.
    """
    mat = np.asarray(returns_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        raise ValueError("returns_matrix must be 2-D with at least 2 columns.")
    corr = np.corrcoef(mat, rowvar=False)
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off_diag))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio and the full report
# ---------------------------------------------------------------------------


@dataclass
class DeflationReport:
    """Every intermediate quantity of a deflation, with its unit.

    Built so that the whole computation can be read aloud step by step, which
    is exactly what a technical panel asks for.
    """

    sr_hat_per_period: float
    sr_hat_annual: float
    periods_per_year: int
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    sr_variance_across_trials: float
    expected_max_sr_per_period: float
    expected_max_sr_annual: float
    psr_vs_zero: float
    dsr: float
    min_track_record_periods: float
    min_track_record_periods_if_normal: float
    variance_term: float
    v_sr_source: str
    confidence: float = DEFAULT_CONFIDENCE
    mean_trial_correlation: float | None = None
    n_trials_effective: float | None = None
    dsr_correlation_adjusted: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """True if the DSR clears the confidence threshold."""
        return self.dsr >= self.confidence

    @property
    def verdict(self) -> str:
        return "PASS" if self.passes else "REJECT"

    def to_text(self) -> str:
        """Step-by-step narration, units stated at every line."""
        lines = [
            "DEFLATION REPORT",
            "=" * 68,
            f"Observed Sharpe ratio    : {self.sr_hat_per_period:.6f} per period "
            f"({self.sr_hat_annual:.4f} annualized, {self.periods_per_year} periods/year)",
            f"Sample size              : {self.n_obs} observations (periods)",
            f"Skewness (gamma_3)       : {self.skew:+.4f} (dimensionless)",
            f"Kurtosis (gamma_4)       : {self.kurtosis:.4f} (dimensionless, normal = 3)",
            "",
            f"Number of trials N       : {self.n_trials} (configurations actually tried)",
            f"Variance of trial SRs    : {self.sr_variance_across_trials:.6e} "
            f"(per-period Sharpe, squared)",
            f"E[max SR] under H0       : {self.expected_max_sr_per_period:.6f} per period "
            f"({self.expected_max_sr_annual:.4f} annualized)",
            "    -> this is the Sharpe ratio pure luck would produce with N tries.",
            "",
            f"PSR vs SR* = 0           : {self.psr_vs_zero:.4f} "
            f"(probability, in [0,1] -- NOT a Sharpe ratio)",
            f"DSR = PSR vs E[max SR]   : {self.dsr:.4f} "
            f"(probability, in [0,1] -- NOT a Sharpe ratio)",
            f"Threshold                : {self.confidence:.2f}",
            f"VERDICT                  : {self.verdict}",
            "",
            f"MinTRL vs E[max SR]      : {self._format_min_trl()}",
            "    -> observations needed before this Sharpe ratio could clear the",
            "       threshold. It separates 'no edge' from 'not enough evidence yet'.",
            f"MinTRL if returns were normal: {self._format_min_trl_normal()}",
            "",
            f"PSR variance term        : {self.variance_term:.4f}",
            f"    -> below 1 it INFLATES the z-score. {self._variance_term_comment()}",
            f"V[SR] source             : {self.v_sr_source}",
        ]
        if self.mean_trial_correlation is not None:
            lines += [
                "",
                "Trial correlation:",
                f"  mean pairwise rho_bar  : {self.mean_trial_correlation:+.3f}",
                f"  N_eff (descriptive)    : {self.n_trials_effective:.1f} "
                f"(vs naive N = {self.n_trials})",
            ]
            if self.dsr_correlation_adjusted is not None:
                lines += [
                    f"  DSR after shrinking V[SR] by (1 - rho): "
                    f"{self.dsr_correlation_adjusted:.4f}",
                ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {note}" for note in self.notes]
        return "\n".join(lines)

    def _variance_term_comment(self) -> str:
        if self.variance_term <= 0.0:
            return "It is non-positive: the formula does not apply here."
        if self.variance_term < 0.5:
            return (
                "It is far below 1, so the non-normality correction is doing most "
                "of the work and the asymptotics are strained."
            )
        if self.variance_term < 1.0:
            return "Positive skewness is helping this strategy pass."
        return "Above 1, so non-normality is penalising the strategy."

    def _format_min_trl_normal(self) -> str:
        if not math.isfinite(self.min_track_record_periods_if_normal):
            return "infinite"
        periods = self.min_track_record_periods_if_normal
        return (
            f"{periods:.0f} periods (~{periods / self.periods_per_year:.1f} years) "
            f"-- the gap against the line above is what the fat tail is buying"
        )

    def _format_min_trl(self) -> str:
        if not math.isfinite(self.min_track_record_periods):
            return (
                "infinite (the observed Sharpe ratio is at or below the "
                "luck threshold, so no amount of data would clear it)"
            )
        periods = self.min_track_record_periods
        years = periods / self.periods_per_year
        have = "already met" if self.n_obs >= periods else f"have {self.n_obs}"
        return f"{periods:.0f} periods (~{years:.1f} years) -- {have}"

    def to_dict(self) -> dict:
        return {
            "min_track_record_periods": self.min_track_record_periods,
            "min_track_record_periods_if_normal": self.min_track_record_periods_if_normal,
            "variance_term": self.variance_term,
            "v_sr_source": self.v_sr_source,
            "mean_trial_correlation": self.mean_trial_correlation,
            "dsr_correlation_adjusted": self.dsr_correlation_adjusted,
            "sr_hat_per_period": self.sr_hat_per_period,
            "sr_hat_annual": self.sr_hat_annual,
            "periods_per_year": self.periods_per_year,
            "n_obs": self.n_obs,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "n_trials": self.n_trials,
            "sr_variance_across_trials": self.sr_variance_across_trials,
            "expected_max_sr_per_period": self.expected_max_sr_per_period,
            "expected_max_sr_annual": self.expected_max_sr_annual,
            "psr_vs_zero": self.psr_vs_zero,
            "dsr": self.dsr,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "n_trials_effective": self.n_trials_effective,
            "notes": list(self.notes),
        }


def deflated_sharpe_ratio(
    sr_hat_per_period: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    sr_variance_across_trials: float,
    *,
    strict_units: bool = True,
) -> float:
    """DSR = PSR evaluated against SR* = E[max SR] under the null.

    Returns a PROBABILITY in [0, 1]. Values below 0.95 mean the observed Sharpe
    ratio is not distinguishable from what N trials of luck would produce.
    """
    sr_star = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return psr(
        sr_hat_per_period=sr_hat_per_period,
        sr_benchmark_per_period=sr_star,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        strict_units=strict_units,
    )


def deflate(
    moments: SharpeMoments,
    n_trials: int,
    sr_variance_across_trials: float,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    mean_trial_correlation: float | None = None,
    v_sr_source: str = "empirical_cross_section",
    strict_units: bool = True,
) -> DeflationReport:
    """Run the full deflation and return every intermediate quantity.

    Parameters
    ----------
    sr_variance_across_trials : float
        V[SR], in per-period Sharpe units squared.
    v_sr_source : {"empirical_cross_section", "marginal"}
        How V[SR] was obtained, because it determines whether a correlation
        adjustment is legitimate:

        - ``"empirical_cross_section"``: the sample variance of the N estimated
          Sharpe ratios. Its expectation is already ``sigma**2 * (1 - rho)``,
          so trial correlation is ALREADY absorbed. `mean_trial_correlation` is
          then reported for context only and no adjustment is applied.
        - ``"marginal"``: a per-trial variance such as `sr_variance_under_null`.
          Correlation is NOT absorbed, so if `mean_trial_correlation` is given
          the variance is shrunk by ``(1 - rho)`` and a second DSR is reported.

    Getting this wrong in the permissive direction (adjusting an already
    adjusted variance) removes the deflation entirely.
    """
    if v_sr_source not in {"empirical_cross_section", "marginal"}:
        raise ValueError(
            f"v_sr_source must be 'empirical_cross_section' or 'marginal', "
            f"got {v_sr_source!r}."
        )

    sr_star = expected_max_sharpe(n_trials, sr_variance_across_trials)

    psr_zero = psr(
        moments.sr_per_period, 0.0, moments.n_obs, moments.skew, moments.kurtosis,
        strict_units=strict_units,
    )
    dsr = psr(
        moments.sr_per_period, sr_star, moments.n_obs, moments.skew, moments.kurtosis,
        strict_units=strict_units,
    )

    min_trl = min_track_record_length(
        moments.sr_per_period, sr_star, moments.skew, moments.kurtosis,
        confidence=confidence,
    )
    min_trl_normal = min_track_record_length(
        moments.sr_per_period, sr_star, 0.0, 3.0, confidence=confidence,
    )
    radicand = psr_radicand(moments.sr_per_period, moments.skew, moments.kurtosis)

    notes: list[str] = []
    n_eff = dsr_adjusted = None

    if mean_trial_correlation is not None:
        n_eff = effective_number_of_trials(n_trials, mean_trial_correlation)
        if v_sr_source == "empirical_cross_section":
            notes.append(
                f"V[SR] is a cross-sectional variance, whose expectation already "
                f"equals sigma^2 * (1 - rho). Trial correlation "
                f"(rho_bar={mean_trial_correlation:+.3f}) is therefore already "
                f"absorbed; N_eff={n_eff:.1f} is shown as a description of how much "
                f"independent exploration the grid contains, and is NOT applied."
            )
        else:
            shrunk = correlation_adjusted_variance(
                sr_variance_across_trials, mean_trial_correlation
            )
            sr_star_adj = expected_max_sharpe(n_trials, shrunk)
            dsr_adjusted = psr(
                moments.sr_per_period, sr_star_adj, moments.n_obs,
                moments.skew, moments.kurtosis, strict_units=strict_units,
            )
            notes.append(
                f"V[SR] is marginal, so it was shrunk by (1 - rho_bar) = "
                f"{1 - max(mean_trial_correlation, 0.0):.3f} before computing "
                f"E[max SR]."
            )

    if moments.n_obs < 60:
        notes.append(
            f"Only {moments.n_obs} observations: the asymptotic normality behind "
            f"PSR is fragile at this sample size."
        )
    if abs(moments.excess_kurtosis) > 20.0 or abs(moments.skew) > 3.0:
        notes.append(
            f"Moments are extreme (skew {moments.skew:+.2f}, kurtosis "
            f"{moments.kurtosis:.1f}). The PSR expansion assumes moderate "
            f"non-normality; at these values the probability should be read as an "
            f"ordering device, not as a calibrated number. Run the concentration "
            f"diagnostics before trusting it."
        )
    elif abs(moments.excess_kurtosis) > 3.0:
        notes.append(
            f"Excess kurtosis is {moments.excess_kurtosis:+.2f}: fat tails are "
            f"large, so the PSR correction is doing heavy lifting here."
        )
    if 0.0 < radicand < 0.5:
        notes.append(
            f"The PSR variance term is {radicand:.3f}, well below 1, which inflates "
            f"the z-score by {1 / math.sqrt(radicand):.2f}x. The reported "
            f"probability is being carried by the estimated skewness, which is "
            f"itself poorly determined on {moments.n_obs} observations."
        )

    return DeflationReport(
        sr_hat_per_period=moments.sr_per_period,
        sr_hat_annual=moments.sr_annual,
        periods_per_year=moments.periods_per_year,
        n_obs=moments.n_obs,
        skew=moments.skew,
        kurtosis=moments.kurtosis,
        n_trials=n_trials,
        sr_variance_across_trials=sr_variance_across_trials,
        expected_max_sr_per_period=sr_star,
        expected_max_sr_annual=to_annual(sr_star, moments.periods_per_year),
        psr_vs_zero=psr_zero,
        dsr=dsr,
        min_track_record_periods=min_trl,
        min_track_record_periods_if_normal=min_trl_normal,
        variance_term=radicand,
        v_sr_source=v_sr_source,
        confidence=confidence,
        mean_trial_correlation=mean_trial_correlation,
        n_trials_effective=n_eff,
        dsr_correlation_adjusted=dsr_adjusted,
        notes=notes,
    )


__all__ = [
    "EULER_MASCHERONI",
    "DEFAULT_CONFIDENCE",
    "SharpeMoments",
    "DeflationReport",
    "UnitError",
    "sharpe_moments",
    "sr_variance_under_null",
    "psr",
    "psr_radicand",
    "correlation_adjusted_variance",
    "min_track_record_length",
    "expected_max_sharpe",
    "effective_number_of_trials",
    "mean_offdiagonal_correlation",
    "deflated_sharpe_ratio",
    "deflate",
]
