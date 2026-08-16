# dsr.py

import math
import itertools
import datetime as dt
from statistics import NormalDist

import numpy as np

import config
import data_loader
from m2_backtester import backtester, portfolio
import strategy_momentum
import calibrate_is

OOS_START = dt.date.fromisoformat(config.OOS_START)
IS_END = OOS_START - dt.timedelta(days=1)        # IS = jusqu'au 2025-05-31 inclus
ANN = config.TRADING_DAYS_PER_YEAR
EULER = 0.5772156649015329
Z = NormalDist()


def daily_moments(returns):
    """SR journalier (non annualise), skewness, kurtosis (non-excess), T."""
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    T = x.size
    if T < 3 or x.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan"), T
    mu, sd = x.mean(), x.std(ddof=1)
    zc = (x - mu) / sd
    return mu / sd, float((zc ** 3).mean()), float((zc ** 4).mean()), T


def expected_max_sr0(var_sr_trials, n_trials):
    """Sharpe (journalier) attendu du MEILLEUR essai sous H0 (tous SR vrais = 0)."""
    s = math.sqrt(var_sr_trials)
    a = Z.inv_cdf(1.0 - 1.0 / n_trials)
    b = Z.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return s * ((1.0 - EULER) * a + EULER * b)


def deflated_sharpe(sr_daily, skew, kurt, T, sr0_daily):
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0_daily) * math.sqrt(T - 1) / denom
    return Z.cdf(z)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()

    # 1) Dispersion des Sharpe des 18 essais de la grille -> JOURNALIER (/sqrt(252))
    print("Calcul des essais (grille 18 cellules)... quelques secondes.\n")
    sr_ann_avec, sr_ann_sans = [], []
    for w, ke, sl in itertools.product(calibrate_is.WINDOWS,
                                       calibrate_is.K_ENTRYS, calibrate_is.STOPS):
        c = calibrate_is.eval_cell(series, w, ke, sl)
        sr_ann_avec.append(c["sharpe_RUT"])
        sr_ann_sans.append(c["sharpe_noRUT"])
    V_avec = float(np.var(np.array(sr_ann_avec) / math.sqrt(ANN), ddof=1))
    V_sans = float(np.var(np.array(sr_ann_sans) / math.sqrt(ANN), ddof=1))

    # 2) Rendements journaliers IS du portefeuille momentum FIGE (equal-weight)
    backtester.strategy = strategy_momentum
    res = backtester.run_backtest(series, end=IS_END)
    ret = portfolio.returns_matrix(res, "netRet")
    p_avec = portfolio.portfolio_returns(ret, portfolio.equal_weights(list(ret.columns)))
    cols = [c for c in ret.columns if c != "RUT"]
    p_sans = portfolio.portfolio_returns(ret[cols], portfolio.equal_weights(cols))

    print("=== DEFLATED SHARPE RATIO (IS, strategie figee, equal-weight) ===\n")
    for label, p, V in [("AVEC RUT", p_avec, V_avec), ("SANS RUT", p_sans, V_sans)]:
        sr_d, skew, kurt, T = daily_moments(p)
        print(f"--- {label} ---")
        print(f"T={T}   SR_jour={sr_d:+.4f}   SR_ann={sr_d*math.sqrt(ANN):+.2f}   "
              f"skew={skew:+.2f}   kurtosis={kurt:.2f}")
        print(f"Var(SR essais, journalier) = {V:.5f}   (estimee sur les 18 cellules)")
        print(f"{'N':>4}{'SR0_ann':>10}{'DSR':>8}   verdict (seuil 0,95)")
        for n_trials in (18, 21, 30):
            sr0 = expected_max_sr0(V, n_trials)
            d = deflated_sharpe(sr_d, skew, kurt, T, sr0)
            verdict = "significatif" if d > 0.95 else "NON significatif"
            print(f"{n_trials:>4}{sr0*math.sqrt(ANN):>10.2f}{d:>8.3f}   {verdict}")
        print()

    print("Lecture : DSR = P(Sharpe vrai > 0) apres correction du biais de selection, "
          "de la non-normalite et de T. DSR < 0,95 -> on ne peut pas conclure a un edge.")
