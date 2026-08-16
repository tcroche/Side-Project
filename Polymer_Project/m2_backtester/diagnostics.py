# diagnostics.py
# Trois variantes a iso-moteur :
#   1) baseline      : parametres actuels (mean-reversion)
#   2) low_turnover  : seuil d'entree eleve + fenetre large -> beaucoup moins de trades
#   3) momentum      : MEME signal, position INVERSEE (test : a 1-min, ca trend ou ca revient ?)

import datetime as dt
import numpy as np
import pandas as pd

import config
import data_loader
import strategy

OOS_START = dt.date.fromisoformat(config.OOS_START)


def eval_day(close, sign=+1, params=None):
    """Miroir exact des formules de backtester.backtest_day, avec un facteur 'sign'
    (+1 = strategie telle quelle, -1 = position inversee). Renvoie (gross_ret, net_ret, entrees)."""
    close = close.dropna()
    n = len(close)
    if n < 3:
        return None
    pos = sign * strategy.compute_positions(close, **(params or {})).values
    P = close.values
    gross = float((pos[:-1] * np.diff(P)).sum())
    prev = np.concatenate([[0.0], pos[:-1]])
    fees = float((config.BASIS_POINT * np.abs(pos - prev) * P).sum())
    base = float(P[0])
    entries = int(((prev == 0) & (pos != 0)).sum())
    return gross / base, (gross - fees) / base, entries


def run_variant(series, sign=+1, params=None, end=OOS_START):
    rows = []
    for tk, df in series.items():
        df = df.sort_index()
        grets, nrets, trades = [], [], []
        for day_ts, day_df in df.groupby(df.index.normalize()):
            if day_ts.date() >= end:        # IN-SAMPLE uniquement
                continue
            r = eval_day(day_df["Close"], sign, params)
            if r is None:
                continue
            grets.append(r[0]); nrets.append(r[1]); trades.append(r[2])
        if not nrets:
            continue
        g, nn = np.array(grets), np.array(nrets)
        ann = config.TRADING_DAYS_PER_YEAR
        rows.append({
            "Ticker": tk, "jours": len(nn),
            "grossRet%": 100 * g.sum(), "netRet%": 100 * nn.sum(),
            "netSharpe": (nn.mean() / nn.std() * np.sqrt(ann)) if nn.std() > 0 else np.nan,
            "tr/j": float(np.mean(trades)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    variants = {
        "1) baseline       ": dict(sign=+1, params=None),
        "2) low_turnover   ": dict(sign=+1, params=dict(window=120, k_entry=3.0, k_exit=0.3)),
        "3) momentum (flip)": dict(sign=-1, params=None),
    }
    fmt = {c: "{:.2f}".format for c in ["grossRet%", "netRet%", "netSharpe", "tr/j"]}
    for name, kw in variants.items():
        d = run_variant(series, **kw)
        print(f"\n=== {name}  (IN-SAMPLE, dates < {config.OOS_START}) ===")
        print(d.to_string(index=False, formatters=fmt))
        print(f"   moyenne grossRet% = {d['grossRet%'].mean():6.2f}   |   "
              f"netSharpe median = {d['netSharpe'].median():5.2f}")
