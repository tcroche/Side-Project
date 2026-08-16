"""
The statistical half of the report, as data rather than as printed text.

`run_real_case.py` narrates the same computation to a console; this module
returns the objects so a renderer can lay them out. The two must not drift, so
`tests/test_report.py` recomputes the headline deflation the way
`run_real_case.analyse` does and asserts equality.

Unit discipline is the whole point of this project and is enforced here as
well: `deflate` is called with PER-PERIOD quantities, trial Sharpe ratios read
from the metadata are annualized and converted with `to_per_period` before use,
and every probability this module returns (PSR, DSR, PBO) is carried in a field
whose renderer labels it a probability in [0, 1]. Nothing here ever calls a
probability a Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.concentration import ConcentrationReport, concentration_report, shared_dependence
from core.cscv import PBOResult, cscv_pbo, pbo_null_distribution, pbo_percentile, suggest_n_blocks
from core.stats import (
    DeflationReport,
    deflate,
    mean_offdiagonal_correlation,
    sharpe_moments,
    sr_variance_under_null,
)
from core.units import TRADING_DAYS_PER_YEAR, to_per_period

#: The M2 dsr.py used plain moment ratios; matching it reproduces the published
#: figure rather than approximating it.
MOMENT_ESTIMATOR = "simple"
N_NULL_SIMULATIONS = 200


@dataclass(frozen=True)
class DeflationRow:
    """One line of the sensitivity table: DSR as a function of N and V[SR]."""

    n_trials: int
    v_sr_source_label: str
    v_sr: float
    expected_max_sr_annual: float
    dsr: float
    verdict: str
    dsr_correlation_adjusted: float | None = None


@dataclass
class DeflationSection:
    """Everything the statistical half of the report needs, already computed."""

    label: str
    selected_trial: str
    n_grid_trials: int
    n_all_trials: int
    periods_per_year: int
    moment_estimator: str
    sr_per_period: float
    sr_annual: float
    n_obs: int
    v_empirical_grid: float
    v_empirical_all: float
    v_null: float
    v_ratio: float
    v_ratio_reading: str
    mean_trial_correlation: float
    rows: tuple[DeflationRow, ...]
    headline: DeflationReport
    concentration: ConcentrationReport
    shared: pd.DataFrame
    pbo: PBOResult
    pbo_percentile: float
    pbo_null_mean: float
    pbo_null_sd: float
    pbo_null_lo: float
    pbo_null_hi: float
    pbo_reading: str
    notes: list[str] = field(default_factory=list)


def _v_ratio_reading(ratio: float) -> str:
    if ratio < 0.5:
        return (
            "Trials are far less dispersed than independent noise would be: they are "
            "highly correlated, so N overstates the number of independent bets."
        )
    if ratio > 2.0:
        return (
            "Trials are more dispersed than pure noise: configurations genuinely "
            "differ, so V[SR] is inflated and the DSR over-deflates."
        )
    return "Consistent with noise; the deflation is well specified."


def _pbo_reading(percentile: float) -> str:
    if percentile < 0.90:
        return (
            "Not distinguishable from what a no-skill dataset of this size produces. "
            "At this sample length PBO carries little information, and saying so is "
            "the honest conclusion."
        )
    return "Higher than 90% of no-skill datasets: evidence of selection overfitting."


def run_deflation(
    frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    label: str,
    sharpe_col: str,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    n_null_simulations: int = N_NULL_SIMULATIONS,
    seed: int = 0,
) -> DeflationSection:
    """Deflate one universe and return every quantity, computed, not printed."""
    grid_ids = [t for t in meta.loc[meta["trial_kind"] == "grid", "trial_id"] if t in frame.columns]
    all_ids = [t for t in meta["trial_id"] if t in frame.columns]
    if not grid_ids:
        raise ValueError("No grid trials found in the returns matrix.")

    frozen = meta.loc[meta["is_frozen_cell"], "trial_id"]
    if len(frozen):
        selected = frozen.iloc[0]
    else:
        selected = grid_ids[
            int(np.argmax([frame[t].mean() / frame[t].std(ddof=1) for t in grid_ids]))
        ]

    returns = frame[selected].dropna()
    moments = sharpe_moments(
        returns.to_numpy(dtype=float),
        periods_per_year=periods_per_year,
        moment_estimator=MOMENT_ESTIMATOR,
    )

    def trial_sr_per_period(trial_ids: list[str]) -> np.ndarray:
        return np.array(
            [
                to_per_period(
                    float(meta.loc[meta["trial_id"] == t, sharpe_col].iloc[0]),
                    periods_per_year,
                )
                for t in trial_ids
            ]
        )

    v_empirical_grid = float(np.var(trial_sr_per_period(grid_ids), ddof=1))
    v_empirical_all = float(np.var(trial_sr_per_period(all_ids), ddof=1))
    v_null = sr_variance_under_null(moments.n_obs)
    v_ratio = v_empirical_grid / v_null if v_null else float("nan")

    rho = float(
        mean_offdiagonal_correlation(frame[grid_ids].dropna().to_numpy(dtype=float))
    )

    specs = [
        (len(grid_ids), v_empirical_grid, "empirical, grid trials", "empirical_cross_section"),
        (len(all_ids), v_empirical_all, "empirical, all trials", "empirical_cross_section"),
        (30, v_empirical_grid, "empirical, grid trials", "empirical_cross_section"),
        (len(grid_ids), v_null, "marginal 1/T (raw)", "marginal"),
    ]
    rows = []
    for n_trials, v_sr, source_label, kind in specs:
        rep = deflate(
            moments,
            n_trials=n_trials,
            sr_variance_across_trials=v_sr,
            mean_trial_correlation=rho,
            v_sr_source=kind,
        )
        rows.append(
            DeflationRow(
                n_trials=n_trials,
                v_sr_source_label=source_label,
                v_sr=v_sr,
                expected_max_sr_annual=rep.expected_max_sr_annual,
                dsr=rep.dsr,
                verdict=rep.verdict,
                dsr_correlation_adjusted=rep.dsr_correlation_adjusted,
            )
        )

    # The headline is the same call run_real_case.py makes.
    headline = deflate(
        moments,
        n_trials=len(grid_ids),
        sr_variance_across_trials=v_empirical_grid,
        mean_trial_correlation=rho,
        v_sr_source="empirical_cross_section",
    )

    conc = concentration_report(returns, periods_per_year=periods_per_year)
    shared = shared_dependence(frame[grid_ids].dropna())

    matrix = frame[grid_ids].dropna().to_numpy(dtype=float)
    n_obs, n_configs = matrix.shape
    n_blocks = suggest_n_blocks(n_obs)
    pbo = cscv_pbo(matrix, n_blocks=n_blocks, strict=False)
    null = pbo_null_distribution(
        n_obs, n_configs, n_blocks, n_simulations=n_null_simulations, seed=seed
    )
    percentile = float(pbo_percentile(pbo.pbo, null))

    return DeflationSection(
        label=label,
        selected_trial=str(selected),
        n_grid_trials=len(grid_ids),
        n_all_trials=len(all_ids),
        periods_per_year=periods_per_year,
        moment_estimator=MOMENT_ESTIMATOR,
        sr_per_period=moments.sr_per_period,
        sr_annual=moments.sr_annual,
        n_obs=moments.n_obs,
        v_empirical_grid=v_empirical_grid,
        v_empirical_all=v_empirical_all,
        v_null=v_null,
        v_ratio=v_ratio,
        v_ratio_reading=_v_ratio_reading(v_ratio),
        mean_trial_correlation=rho,
        rows=tuple(rows),
        headline=headline,
        concentration=conc,
        shared=shared,
        pbo=pbo,
        pbo_percentile=percentile,
        pbo_null_mean=float(null.mean()),
        pbo_null_sd=float(null.std()),
        pbo_null_lo=float(np.percentile(null, 5)),
        pbo_null_hi=float(np.percentile(null, 95)),
        pbo_reading=_pbo_reading(percentile),
    )


__all__ = ["DeflationRow", "DeflationSection", "run_deflation", "MOMENT_ESTIMATOR"]
