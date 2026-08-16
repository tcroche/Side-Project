"""
run_real_case.py -- deflate and cross-validate the M2 intraday momentum backtest.

Prerequisite: run `export_trials.py` inside the M2 backtester repository, then
copy the three CSV files it writes into this project's data/ folder:

    data/trials_m2_with_rut.csv
    data/trials_m2_ex_rut.csv
    data/trials_m2_meta.csv

Then, from the project root:

    python run_real_case.py

To exercise the plumbing on the synthetic fixture instead (numbers meaningless,
files written by make_dry_run_fixture.py under distinct `dryrun_` names):

    python run_real_case.py --dry-run

What it produces, for each universe (with RUT / ex RUT):
  * the deflation report at N = 18 (grid only) and N = 21 (grid + gamma sweep)
  * both estimators of V[SR]: empirical across trials, and theoretical under
    the null, so the choice is visible instead of implicit
  * the PBO from CSCV, positioned inside its own simulated null distribution
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from core.concentration import concentration_report, shared_dependence
from core.cscv import cscv_pbo, pbo_null_distribution, pbo_percentile, suggest_n_blocks
from core.stats import (
    deflate,
    mean_offdiagonal_correlation,
    sharpe_moments,
    sr_variance_under_null,
)
from core.units import TRADING_DAYS_PER_YEAR, to_annual, to_per_period

DATA_DIR = "data"
ANN = TRADING_DAYS_PER_YEAR

#: The three files export_trials.py writes (real case) ...
REAL_FILES = {
    "meta": "trials_m2_meta.csv",
    "with_rut": "trials_m2_with_rut.csv",
    "ex_rut": "trials_m2_ex_rut.csv",
}
#: ... and the prefix make_dry_run_fixture.py puts in front of them, so the
#: fixture can never overwrite the export and a dry run must be asked for.
DRY_RUN_PREFIX = "dryrun_"


def input_files(dry_run: bool) -> dict[str, str]:
    """Names of the three input CSVs for a real run or a dry run."""
    prefix = DRY_RUN_PREFIX if dry_run else ""
    return {kind: prefix + name for kind, name in REAL_FILES.items()}


def is_synthetic(meta: pd.DataFrame) -> bool:
    """True when the metadata carries the fixture flag on any row."""
    if "is_synthetic" not in meta.columns:
        return False
    return bool(meta["is_synthetic"].astype(str).str.lower().isin({"true", "1"}).any())

#: The M2 dsr.py used the plain moment ratios; use the same estimator so the
#: published figure is reproduced rather than approximated.
MOMENT_ESTIMATOR = "simple"


def load_returns(name: str) -> pd.DataFrame:
    """Load a Date x trial_id matrix of daily returns."""
    frame = pd.read_csv(_path(name), index_col=0)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_meta(name: str) -> pd.DataFrame:
    """Load the trial metadata table (trial_id is a COLUMN, not the index)."""
    meta = pd.read_csv(_path(name))
    required = {"trial_id", "trial_kind", "is_frozen_cell"}
    missing = required - set(meta.columns)
    if missing:
        sys.exit(f"{name} is missing required columns: {sorted(missing)}")
    meta["is_frozen_cell"] = meta["is_frozen_cell"].astype(bool)
    return meta


def _path(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        if name.startswith(DRY_RUN_PREFIX):
            sys.exit(f"Missing {path}.\nRun `python make_dry_run_fixture.py` first.")
        sys.exit(
            f"Missing {path}.\n"
            f"Run export_trials.py inside the M2 repository first, then copy the "
            f"CSV files into {DATA_DIR}/."
        )
    return path


def analyse(frame: pd.DataFrame, meta: pd.DataFrame, label: str, sharpe_col: str) -> None:
    print()
    print("#" * 72)
    print(f"# {label}")
    print("#" * 72)

    grid_ids = meta.loc[meta["trial_kind"] == "grid", "trial_id"].tolist()
    all_ids = meta["trial_id"].tolist()
    grid_ids = [t for t in grid_ids if t in frame.columns]
    all_ids = [t for t in all_ids if t in frame.columns]

    frozen = meta.loc[meta["is_frozen_cell"], "trial_id"]
    selected = frozen.iloc[0] if len(frozen) else grid_ids[int(
        np.argmax([frame[t].mean() / frame[t].std(ddof=1) for t in grid_ids])
    )]

    returns = frame[selected].dropna().to_numpy(dtype=float)
    moments = sharpe_moments(
        returns, periods_per_year=ANN, moment_estimator=MOMENT_ESTIMATOR
    )

    print(f"Selected configuration : {selected} (the frozen cell)")
    print(
        f"In-sample Sharpe       : {moments.sr_per_period:.6f} per day "
        f"({moments.sr_annual:.4f} annualized) over {moments.n_obs} days"
    )

    # --- V[SR]: two estimators, both reported --------------------------------
    grid_sr_daily = np.array(
        [
            to_per_period(float(meta.loc[meta["trial_id"] == t, sharpe_col].iloc[0]), ANN)
            for t in grid_ids
        ]
    )
    all_sr_daily = np.array(
        [
            to_per_period(float(meta.loc[meta["trial_id"] == t, sharpe_col].iloc[0]), ANN)
            for t in all_ids
        ]
    )

    v_empirical_grid = float(np.var(grid_sr_daily, ddof=1))
    v_empirical_all = float(np.var(all_sr_daily, ddof=1))
    v_null = sr_variance_under_null(moments.n_obs)

    print()
    print("Variance of trial Sharpe ratios, per-day units:")
    print(f"  empirical across {len(grid_ids)} grid trials : {v_empirical_grid:.6e}")
    print(f"  empirical across {len(all_ids)} trials       : {v_empirical_all:.6e}")
    print(f"  theoretical under H0 (1/T, T={moments.n_obs})  : {v_null:.6e}")
    ratio = v_empirical_grid / v_null
    print(f"  ratio empirical / theoretical            : {ratio:.2f}")
    if ratio < 0.5:
        print(
            "  -> trials are far less dispersed than independent noise would be: "
            "they are highly correlated, so N overstates the independent bets."
        )
    elif ratio > 2.0:
        print(
            "  -> trials are more dispersed than pure noise: configurations "
            "genuinely differ, so V[SR] is inflated and the DSR over-deflates."
        )
    else:
        print("  -> consistent with noise; the deflation is well specified.")

    # --- Deflation at several trial counts -----------------------------------
    rho = mean_offdiagonal_correlation(frame[grid_ids].dropna().to_numpy(dtype=float))
    print(f"\nMean pairwise correlation between grid trials: {rho:+.3f}")

    print()
    print(f"{'N':>4}{'V[SR] source':>22}{'E[maxSR] ann':>15}{'DSR':>9}   verdict")
    print("-" * 68)
    for n_trials, v_sr, source, kind in (
        (len(grid_ids), v_empirical_grid, "empirical grid", "empirical_cross_section"),
        (len(all_ids), v_empirical_all, "empirical all", "empirical_cross_section"),
        (30, v_empirical_grid, "empirical grid", "empirical_cross_section"),
        (len(grid_ids), v_null, "marginal 1/T (raw)", "marginal"),
    ):
        report = deflate(
            moments, n_trials=n_trials, sr_variance_across_trials=v_sr,
            mean_trial_correlation=float(rho), v_sr_source=kind,
        )
        print(
            f"{n_trials:>4}{source:>22}"
            f"{report.expected_max_sr_annual:>15.4f}"
            f"{report.dsr:>9.4f}   {report.verdict}"
        )
        if kind == "marginal" and report.dsr_correlation_adjusted is not None:
            print(
                f"{'':>4}{'marginal, x(1-rho)':>22}"
                f"{'':>15}{report.dsr_correlation_adjusted:>9.4f}   "
                f"{'PASS' if report.dsr_correlation_adjusted >= 0.95 else 'REJECT'}"
            )

    headline = deflate(
        moments,
        n_trials=len(grid_ids),
        sr_variance_across_trials=v_empirical_grid,
        mean_trial_correlation=float(rho),
        v_sr_source="empirical_cross_section",
    )
    print()
    print(headline.to_text())

    # --- Concentration -------------------------------------------------------
    print()
    conc = concentration_report(frame[selected].dropna(), periods_per_year=ANN)
    print(conc.to_text())

    dependence = shared_dependence(frame[grid_ids].dropna())
    if len(dependence):
        print("\nObservations that a majority of the grid depends on:")
        for label, row in dependence.head(3).iterrows():
            stamp = getattr(label, "date", lambda: label)()
            print(
                f"  {stamp}: {int(row['n_trials_dependent'])}/{len(grid_ids)} "
                f"configurations draw more than half their P&L from this single day"
            )
        print(
            "  -> when every cell of a grid leans on the same session, the grid was\n"
            "     never exploring different strategies; it was re-expressing one event."
        )

    # --- CSCV / PBO ----------------------------------------------------------
    matrix = frame[grid_ids].dropna().to_numpy(dtype=float)
    n_obs, n_configs = matrix.shape
    n_blocks = suggest_n_blocks(n_obs)

    print()
    result = cscv_pbo(matrix, n_blocks=n_blocks, strict=False)
    print(result.to_text())

    print("\nPositioning the observed PBO inside its own null distribution")
    print("(pure noise, same T, N and S -- 200 simulations):")
    null = pbo_null_distribution(n_obs, n_configs, n_blocks, n_simulations=200, seed=0)
    pct = pbo_percentile(result.pbo, null)
    print(
        f"  null: mean={null.mean():.3f}  sd={null.std():.3f}  "
        f"90% interval=[{np.percentile(null, 5):.3f}, {np.percentile(null, 95):.3f}]"
    )
    print(f"  observed PBO {result.pbo:.3f} sits at percentile {pct:.2f} of that null.")
    if pct < 0.90:
        print(
            "  -> NOT distinguishable from what a no-skill dataset of this size "
            "produces. At this sample length PBO carries little information, and "
            "saying so is the honest conclusion."
        )
    else:
        print("  -> higher than 90% of no-skill datasets: evidence of overfitting.")


SYNTHETIC_BANNER = (
    "SYNTHETIC FIXTURE: the input carries is_synthetic=True; every number below "
    "is meaningless, only the plumbing is being exercised."
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deflate the M2 backtest (real case).")
    parser.add_argument("--dry-run", action="store_true",
                        help=f"read the {DRY_RUN_PREFIX}* fixture instead of the real export")
    args = parser.parse_args(argv)
    files = input_files(args.dry_run)

    meta = load_meta(files["meta"])
    synthetic = is_synthetic(meta)
    if synthetic:
        print(SYNTHETIC_BANNER)
    print("BACKTEST INTEGRITY AUDITOR -- real case: M2 intraday momentum backtest")
    print(f"trials exported: {len(meta)} "
          f"({int((meta['trial_kind'] == 'grid').sum())} grid + "
          f"{int((meta['trial_kind'] == 'gamma_sweep').sum())} gamma sweep)")
    print(f"moment estimator: {MOMENT_ESTIMATOR} (matches the M2 dsr.py)")

    with_tag = "SYNTHETIC FIXTURE, " if synthetic else ""
    analyse(load_returns(files["with_rut"]), meta, f"{with_tag}UNIVERSE WITH RUT", "sharpe_annual_with_rut")
    analyse(load_returns(files["ex_rut"]), meta, f"{with_tag}UNIVERSE EX RUT", "sharpe_annual_ex_rut")

    print()
    print("=" * 72)
    if synthetic:
        print(SYNTHETIC_BANNER)
    print(
        "Reading: PSR, DSR and PBO are all PROBABILITIES in [0,1], never Sharpe\n"
        "ratios. A DSR of 0.92 against a 0.95 threshold is a rejection, not a\n"
        "degraded Sharpe ratio."
    )


if __name__ == "__main__":
    main()
