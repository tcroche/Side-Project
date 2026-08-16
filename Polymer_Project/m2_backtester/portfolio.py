# portfolio.py

import datetime as dt
import numpy as np
import pandas as pd

import config
import data_loader
from m2_backtester import backtester
import strategy_momentum

OOS_START = dt.date.fromisoformat(config.OOS_START)
IS_END = OOS_START - dt.timedelta(days=1)           # IS = jusqu'au 2025-05-31 inclus


# =========================================================================
#  1) MATRICE DE RENDEMENTS  (Date x Ticker)
# =========================================================================

def returns_matrix(res, col="netRet"):
    """Pivote le DataFrame long du backtester en matrice Date x Ticker.
    Les jours non tradees par un actif (N225 feries) restent NaN -> geres au
    moment de l'agregation par renormalisation des poids."""
    if res.empty:
        return pd.DataFrame()
    return res.pivot_table(index="Date", columns="Ticker", values=col).sort_index()


# =========================================================================
#  2) SCHEMAS DE PONDERATION
# =========================================================================

def equal_weights(tickers):
    """Baseline honnete : 1/N. Aucune estimation -> aucun risque de look-ahead."""
    tickers = list(tickers)
    return pd.Series(1.0 / len(tickers), index=tickers)


def inverse_vol_weights(ret_mat, end=IS_END):
    """Poids proportionnels a 1/sigma, sigma estime sur les rendements nets
    journaliers JUSQU'A 'end' (l'IS) -> figes ensuite. C'est anti-look-ahead :
    en OOS on reutilise ces memes poids IS, sans jamais regarder l'OOS."""
    sub = ret_mat.loc[ret_mat.index <= pd.Timestamp(end)] if end is not None else ret_mat
    vol = sub.std()
    inv = (1.0 / vol.replace(0.0, np.nan)).dropna()
    w = inv / inv.sum()
    return w.reindex(ret_mat.columns).fillna(0.0)


# =========================================================================
#  3) AGREGATION  (avec renormalisation par jour)
# =========================================================================

def portfolio_returns(ret_mat, weights):
    """Rendement net journalier du portefeuille. Chaque jour, on ne garde que les
    actifs disponibles et on RENORMALISE leurs poids pour qu'ils somment a 1
    -> un jour ferie au Japon n'introduit aucun biais, le N225 sort juste du
    melange ce jour-la."""
    w = pd.Series(weights).reindex(ret_mat.columns).fillna(0.0)
    out = {}
    for date, row in ret_mat.iterrows():
        avail = row.dropna()
        if avail.empty:
            continue
        wa = w.reindex(avail.index)
        s = wa.sum()
        if s <= 0:                              # garde-fou : repli equal-weight
            wa = pd.Series(1.0, index=avail.index)
            s = wa.sum()
        out[date] = float(((wa / s) * avail).sum())
    return pd.Series(out, name="portfolio").sort_index()


# =========================================================================
#  4) METRIQUES LEGERES
# =========================================================================

def ann_sharpe(daily, ann=config.TRADING_DAYS_PER_YEAR):
    a = np.asarray(daily, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 2 or a.std() == 0:
        return np.nan
    return a.mean() / a.std() * np.sqrt(ann)


def max_drawdown(daily):
    """Drawdown sur la courbe de capital ADDITIVE (somme des rendements nets),
    pour rester coherent avec le reste du projet. Renvoie une valeur <= 0."""
    eq = np.cumsum(np.asarray(daily, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min()) if eq.size else 0.0


# =========================================================================
#  5) RAPPORT IS 
# =========================================================================

def report(ret_mat, label):
    print(f"\n===== PORTEFEUILLE — {label} (IN-SAMPLE) =====")
    tickers = list(ret_mat.columns)

    # 5a) Par actif : signe du rendement net + Sharpe
    print(f"{'Ticker':8}{'jours':>7}{'netRet%':>11}{'netSharpe':>11}")
    per = {}
    for tk in tickers:
        col = ret_mat[tk].dropna()
        per[tk] = ann_sharpe(col)
        print(f"{tk:8}{len(col):>7}{100*col.sum():>11.2f}{per[tk]:>11.2f}")

    # 5b) Correlation (justifie -- ou non -- la diversification)
    print("\nCorrelation des rendements nets journaliers :")
    print(ret_mat.corr().round(2).to_string())

    # 5c) Portefeuilles equal-weight vs inverse-vol
    ew = equal_weights(tickers)
    iv = inverse_vol_weights(ret_mat, end=IS_END)
    p_ew = portfolio_returns(ret_mat, ew)
    p_iv = portfolio_returns(ret_mat, iv)
    print("\nPoids inverse-vol (estimes sur l'IS) :",
          {k: round(float(v), 2) for k, v in iv.items() if v > 0})
    print(f"\n{'Portefeuille':16}{'jours':>7}{'netRet%':>11}{'netSharpe':>11}{'maxDD%':>9}")
    for name, p in [("equal-weight", p_ew), ("inverse-vol", p_iv)]:
        print(f"{name:16}{len(p):>7}{100*p.sum():>11.2f}"
              f"{ann_sharpe(p):>11.2f}{100*max_drawdown(p):>9.2f}")

    # 5d) Punchline : gain de diversification = Sharpe portef. EW vs Sharpe median actif
    med = np.nanmedian(list(per.values()))
    print(f"\nSharpe median par actif : {med:>5.2f}")
    print(f"Sharpe portefeuille EW  : {ann_sharpe(p_ew):>5.2f}   "
          f"(gain de diversification = {ann_sharpe(p_ew) - med:+.2f})")
    print("Rappel : si netRet% <= 0, la diversification reduit le bruit mais "
          "n'inverse PAS l'edge.")


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    backtester.strategy = strategy_momentum          # injecte la strategie FIGEE
    res = backtester.run_backtest(series, end=IS_END) # IN-SAMPLE uniquement

    ret = returns_matrix(res, "netRet")
    print(f"=== {ret.shape[1]} actifs, {ret.shape[0]} jours IS "
          f"({ret.index.min().date()} -> {ret.index.max().date()}) ===")

    report(ret, "AVEC RUT")
    if "RUT" in ret.columns:
        report(ret.drop(columns=["RUT"]), "SANS RUT")
