"""
Known-truth validation of the statistical core.

Run this from the PROJECT ROOT (the folder containing this file):

    python run_deflation_demo.py

Four scenarios, in each of which the correct answer is known by construction:

  A. 30 strategies, NONE with a true edge. The best one looks excellent
     against SR* = 0. The DSR must reject it.
  B. 30 strategies, ONE with a true edge, but only 500 days of history.
     The DSR rejects it -- correctly, because the evidence is insufficient.
     MinTRL says how much history would be needed. This is the tool being
     honest about its own statistical power.
  C. The same true edge with 1000 days. The DSR now accepts it, which shows
     the tool is not merely a rejection machine.
  D. The unit trap: the same Sharpe ratio fed in annualized form. The guard
     must refuse to compute.

No external data, no API key, no network. Everything is generated from fixed
random seeds, so the output is reproducible line for line.
"""

from __future__ import annotations

import math

import numpy as np

from core.stats import deflate, mean_offdiagonal_correlation, psr, sharpe_moments
from core.units import TRADING_DAYS_PER_YEAR, UnitError, to_annual, to_per_period

N_TRIALS = 30
DAILY_VOL = 0.01
LUCKY_TRIAL = 7


def run_scenario(
    title: str,
    true_daily_sharpe: np.ndarray,
    n_obs: int,
    seed: int,
    expectation: str,
) -> bool:
    """Build a trial matrix, select the in-sample winner, deflate, and narrate."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(
        loc=true_daily_sharpe * DAILY_VOL, scale=DAILY_VOL, size=(n_obs, N_TRIALS)
    )

    srs = matrix.mean(axis=0) / matrix.std(axis=0, ddof=1)
    best = int(np.argmax(srs))

    moments = sharpe_moments(matrix[:, best], periods_per_year=TRADING_DAYS_PER_YEAR)
    report = deflate(
        moments,
        n_trials=N_TRIALS,
        sr_variance_across_trials=float(np.var(srs, ddof=1)),
        mean_trial_correlation=mean_offdiagonal_correlation(matrix),
    )

    print()
    print("#" * 72)
    print(f"# {title}")
    print("#" * 72)

    best_true = float(true_daily_sharpe.max())
    if best_true == 0.0:
        print("Ground truth : no trial has any edge. True Sharpe is 0 for all 30.")
    else:
        print(
            f"Ground truth : trial {int(np.argmax(true_daily_sharpe))} has a true daily "
            f"Sharpe of {best_true:.4f} "
            f"({to_annual(best_true, TRADING_DAYS_PER_YEAR):.2f} annualized). "
            f"The other 29 have none."
        )
    print(f"Researcher   : picked trial {best} on in-sample Sharpe, over {n_obs} days.")
    print(
        f"Naive report : annualized Sharpe {moments.sr_annual:.2f}, "
        f"PSR vs zero {report.psr_vs_zero:.1%}"
    )
    print(f"Expected     : {expectation}")
    print()
    print(report.to_text())
    return report.passes


def run_unit_trap() -> None:
    print()
    print("#" * 72)
    print("# SCENARIO D -- the unit trap")
    print("#" * 72)

    sr_annual = 1.93
    sr_daily = to_per_period(sr_annual, TRADING_DAYS_PER_YEAR)
    print(f"An annualized Sharpe of {sr_annual} is {sr_daily:.6f} per trading day.")
    print()

    correct = psr(sr_daily, 0.0, n_obs=500, skew=0.0, kurtosis=3.0)
    print(f"Correct   : psr(sr_per_period={sr_daily:.6f}, T=500) = {correct:.4f}")

    print("Incorrect : psr(sr_annual=1.93, T=500) ->", end=" ")
    try:
        psr(sr_annual, 0.0, n_obs=500, skew=0.0, kurtosis=3.0)
    except UnitError as exc:
        print("UnitError raised")
        print(f"            {exc}")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forced = psr(sr_annual, 0.0, 500, 0.0, 3.0, strict_units=False)
    print(f"Forced    : the same call with strict_units=False returns {forced:.6f}")
    print(
        "            -- a certainty, produced purely by inflating the z-score by "
        f"sqrt(252) = {math.sqrt(TRADING_DAYS_PER_YEAR):.2f}."
    )


def main() -> None:
    print("BACKTEST INTEGRITY AUDITOR -- statistical core, known-truth validation")
    print(f"trials per scenario = {N_TRIALS}, daily volatility = {DAILY_VOL}")

    no_edge = np.zeros(N_TRIALS)
    real_edge = np.zeros(N_TRIALS)
    real_edge[LUCKY_TRIAL] = 0.13  # daily Sharpe -> ~2.06 annualized

    a = run_scenario(
        "SCENARIO A -- 30 strategies, none of which has any edge",
        no_edge, n_obs=500, seed=2024,
        expectation="REJECT (there is nothing to find)",
    )
    b = run_scenario(
        "SCENARIO B -- one genuine edge, but only 500 days of history",
        real_edge, n_obs=500, seed=2025,
        expectation="REJECT (the edge is real but the evidence is too short)",
    )
    c = run_scenario(
        "SCENARIO C -- the same genuine edge, with 1000 days of history",
        real_edge, n_obs=1000, seed=2025,
        expectation="PASS (same edge, enough evidence)",
    )
    run_unit_trap()

    print()
    print("=" * 72)
    print("SUMMARY")
    print(f"  A (no edge,    T=500 ) -> {'PASS' if a else 'REJECT'}   expected REJECT")
    print(f"  B (real edge,  T=500 ) -> {'PASS' if b else 'REJECT'}   expected REJECT")
    print(f"  C (real edge,  T=1000) -> {'PASS' if c else 'REJECT'}   expected PASS")
    print()
    print(
        "B and C carry the argument. Same strategy, same true edge, different\n"
        "track record length. A tool that only ever rejects proves nothing; this\n"
        "one rejects luck, admits when it lacks power, and accepts a real edge\n"
        "once the evidence supports it."
    )


if __name__ == "__main__":
    main()
