"""
export_trials.py -- to be placed in the M2 backtester repository, alongside
calibrate_is.py, and run from there:

    python export_trials.py

Why this script exists
----------------------
`calibrate_is.eval_cell` returns only SCALAR Sharpe ratios per grid cell. That
is enough for the Deflated Sharpe Ratio, which needs V[SR] across trials, but
not for CSCV / PBO, which needs the full T x N matrix of daily returns, one
column per configuration. This script re-runs the same grid and persists both.

What it writes (into ./data/)
-----------------------------
  trials_m2_with_rut.csv   Date x trial_id, daily NET portfolio returns
  trials_m2_ex_rut.csv     same, with RUT excluded from the portfolio
  trials_m2_meta.csv       one row per trial: parameters and annualized Sharpe

Trial inventory
---------------
  18 grid cells  = WINDOWS (3) x K_ENTRYS (3) x STOPS (2), gamma fixed at 0.3
   3 gamma cells = the frozen cell re-evaluated at gamma in {0.2, 0.3, 0.5}
  --
  21 trials in total.

Note on the gamma sweep: calibrate_is.py demonstrates that the Sharpe ratio is
essentially invariant to gamma, because tanh sizing frozen at entry acts as a
pure leverage factor. Configurations that cannot change the ranking add almost
nothing to the selection room, so counting them as three extra trials is
conservative rather than accurate. The `trial_kind` column keeps them separable
so the deflation can be run at N = 18 and N = 21 and the difference reported.

This script does NOT re-tune anything and does NOT touch the out-of-sample
period: `end=IS_END` is passed exactly as in calibrate_is.py.
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from m2_backtester import backtester, portfolio
    import calibrate_is
    import config
    import data_loader
    import strategy_momentum
except ImportError as exc:  # pragma: no cover - depends on the host repo
    sys.exit(
        f"Import failed: {exc}\n"
        "Run this script from the root of the M2 backtester repository, the "
        "folder that contains calibrate_is.py and config.py."
    )

OUT_DIR = "data"
EXCLUDE_TICKER = "RUT"


def portfolio_returns_for_cell(
    series: dict,
    window: int,
    k_entry: float,
    stop_loss: float,
    gamma: float,
    exclude: tuple[str, ...] = (),
) -> pd.Series:
    """Daily equal-weight NET portfolio returns for one configuration, in-sample.

    Mirrors calibrate_is.eval_cell exactly, but returns the SERIES instead of
    collapsing it to a Sharpe ratio.
    """
    params = dict(
        window=window,
        k_entry=k_entry,
        k_exit=calibrate_is.K_EXIT,
        stop_loss=stop_loss,
        gamma=gamma,
    )
    backtester.strategy = strategy_momentum
    res = backtester.run_backtest(series, params=params, end=calibrate_is.IS_END)
    ret = portfolio.returns_matrix(res, "netRet")
    cols = [c for c in ret.columns if c not in exclude]
    if not cols:
        raise RuntimeError("No tickers left after exclusion.")
    return portfolio.portfolio_returns(ret[cols], portfolio.equal_weights(cols))


def build_trial_specs() -> list[dict]:
    """The 21 configurations, in a stable, documented order."""
    specs = []
    for window, k_entry, stop in itertools.product(
        calibrate_is.WINDOWS, calibrate_is.K_ENTRYS, calibrate_is.STOPS
    ):
        specs.append(
            dict(
                window=window,
                k_entry=k_entry,
                stop_loss=stop,
                gamma=calibrate_is.GAMMA,
                trial_kind="grid",
            )
        )

    frozen = calibrate_is.FROZEN
    for gamma in (0.2, 0.3, 0.5):
        specs.append(
            dict(
                window=frozen["window"],
                k_entry=frozen["k_entry"],
                stop_loss=frozen["stop_loss"],
                gamma=gamma,
                trial_kind="gamma_sweep",
            )
        )

    for i, spec in enumerate(specs):
        spec["trial_id"] = f"t{i:02d}"
        spec["is_frozen_cell"] = (
            spec["window"] == frozen["window"]
            and spec["k_entry"] == frozen["k_entry"]
            and spec["stop_loss"] == frozen["stop_loss"]
            and spec["trial_kind"] == "grid"
        )
    return specs


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading 1-minute series...")
    series = data_loader.load_ticker_series()
    print(f"  {len(series)} instruments: {sorted(series)}")
    print(f"  in-sample end (inclusive): {calibrate_is.IS_END}\n")

    specs = build_trial_specs()
    print(f"Running {len(specs)} configurations, in-sample only.")
    print("This re-runs the same grid as calibrate_is.py, so expect a similar runtime.\n")

    with_rut: dict[str, pd.Series] = {}
    ex_rut: dict[str, pd.Series] = {}
    started = time.time()

    for i, spec in enumerate(specs, start=1):
        tid = spec["trial_id"]
        with_rut[tid] = portfolio_returns_for_cell(
            series, spec["window"], spec["k_entry"], spec["stop_loss"], spec["gamma"]
        )
        ex_rut[tid] = portfolio_returns_for_cell(
            series,
            spec["window"],
            spec["k_entry"],
            spec["stop_loss"],
            spec["gamma"],
            exclude=(EXCLUDE_TICKER,),
        )
        elapsed = time.time() - started
        print(
            f"  [{i:>2}/{len(specs)}] {tid}  window={spec['window']:>3} "
            f"k_entry={spec['k_entry']:.1f} stop={spec['stop_loss']:.3f} "
            f"gamma={spec['gamma']:.1f}  ({elapsed:6.1f}s elapsed)"
        )

    frame_with = pd.DataFrame(with_rut).sort_index()
    frame_ex = pd.DataFrame(ex_rut).sort_index()

    for label, frame in (("with RUT", frame_with), ("ex RUT", frame_ex)):
        missing = int(frame.isna().sum().sum())
        if missing:
            print(
                f"\nWARNING ({label}): {missing} missing values. Configurations do "
                f"not share an identical set of trading days. Rows with any NaN "
                f"will be dropped so that CSCV sees a rectangular matrix."
            )
    frame_with = frame_with.dropna()
    frame_ex = frame_ex.dropna()

    meta = pd.DataFrame(specs)
    ann = float(np.sqrt(config.TRADING_DAYS_PER_YEAR))
    meta["sharpe_annual_with_rut"] = [
        _annual_sharpe(frame_with[t], ann) for t in meta["trial_id"]
    ]
    meta["sharpe_annual_ex_rut"] = [
        _annual_sharpe(frame_ex[t], ann) for t in meta["trial_id"]
    ]
    meta["net_return_pct_with_rut"] = [
        100.0 * frame_with[t].sum() for t in meta["trial_id"]
    ]

    meta = meta[
        [
            "trial_id",
            "trial_kind",
            "is_frozen_cell",
            "window",
            "k_entry",
            "stop_loss",
            "gamma",
            "sharpe_annual_with_rut",
            "sharpe_annual_ex_rut",
            "net_return_pct_with_rut",
        ]
    ]

    paths = {
        "with_rut": os.path.join(OUT_DIR, "trials_m2_with_rut.csv"),
        "ex_rut": os.path.join(OUT_DIR, "trials_m2_ex_rut.csv"),
        "meta": os.path.join(OUT_DIR, "trials_m2_meta.csv"),
    }
    frame_with.to_csv(paths["with_rut"], index_label="Date")
    frame_ex.to_csv(paths["ex_rut"], index_label="Date")
    meta.to_csv(paths["meta"], index=False)

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print(f"  {paths['with_rut']}  shape {frame_with.shape} (days x trials)")
    print(f"  {paths['ex_rut']}  shape {frame_ex.shape}")
    print(f"  {paths['meta']}  {len(meta)} trials")
    print(
        f"  in-sample window: {frame_with.index.min().date()} -> "
        f"{frame_with.index.max().date()}"
    )

    grid_only = meta[meta["trial_kind"] == "grid"]
    print("\nSanity checks for the deflation step:")
    print(
        f"  annualized Sharpe, grid cells, with RUT : "
        f"min={grid_only['sharpe_annual_with_rut'].min():+.2f} "
        f"median={grid_only['sharpe_annual_with_rut'].median():+.2f} "
        f"max={grid_only['sharpe_annual_with_rut'].max():+.2f}"
    )
    print(
        f"  annualized Sharpe, grid cells, ex RUT   : "
        f"min={grid_only['sharpe_annual_ex_rut'].min():+.2f} "
        f"median={grid_only['sharpe_annual_ex_rut'].median():+.2f} "
        f"max={grid_only['sharpe_annual_ex_rut'].max():+.2f}"
    )
    frozen_row = meta[meta["is_frozen_cell"]]
    if len(frozen_row):
        r = frozen_row.iloc[0]
        print(
            f"  frozen cell ({r['trial_id']}) Sharpe with RUT : "
            f"{r['sharpe_annual_with_rut']:+.2f} annualized  "
            f"-- this is the number the write-up calls 1.93"
        )

    gamma_rows = meta[meta["trial_kind"] == "gamma_sweep"]
    spread = gamma_rows["sharpe_annual_with_rut"].max() - gamma_rows[
        "sharpe_annual_with_rut"
    ].min()
    print(
        f"  gamma sweep Sharpe spread              : {spread:.4f} "
        f"(near zero confirms gamma is leverage, not edge)"
    )
    print(
        "\nCopy the three CSV files into the auditor repository under data/ and "
        "run:\n    python run_real_case.py"
    )


def _annual_sharpe(series: pd.Series, ann_factor: float) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if values.size < 2 or values.std(ddof=1) == 0.0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1) * ann_factor)


if __name__ == "__main__":
    main()
