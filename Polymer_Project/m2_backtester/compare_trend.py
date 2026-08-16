# compare_trend.py
# On regarde le netSharpe MEDIAN (robuste a un actif extreme), avec et sans RUT.

import datetime as dt
import numpy as np
import pandas as pd

import config
import data_loader
import strategy
import strategy_trend

OOS_START = dt.date.fromisoformat(config.OOS_START)
LOW = dict(window=120, k_entry=3.0, k_exit=0.3)   # faible rotation pour mean-rev / flip


def eval_day(close, posfn):
    close = close.dropna()
    if len(close) < 3:
        return None
    pos = np.asarray(posfn(close))
    P = close.values
    gross = float((pos[:-1] * np.diff(P)).sum())
    prev = np.concatenate([[0.0], pos[:-1]])
    fees = float((config.BASIS_POINT * np.abs(pos - prev) * P).sum())
    base = float(P[0])
    entries = int(((prev == 0) & (pos != 0)).sum())
    return (gross - fees) / base, entries


def evaluate(series, posfn, exclude=()):
    sharpes, nets, trs = [], [], []
    for tk, df in series.items():
        if tk in exclude:
            continue
        df = df.sort_index(); ns, tt = [], []
        for day_ts, day_df in df.groupby(df.index.normalize()):
            if day_ts.date() >= OOS_START:        # IN-SAMPLE uniquement
                continue
            r = eval_day(day_df["Close"], posfn)
            if r:
                ns.append(r[0]); tt.append(r[1])
        if not ns:
            continue
        a = np.array(ns)
        sh = a.mean() / a.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR) if a.std() > 0 else np.nan
        sharpes.append(sh); nets.append(100 * a.sum()); trs.append(np.mean(tt))
    return np.nanmedian(sharpes), np.mean(nets), np.mean(trs)


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    posfns = {
        "mean-rev low-trn ": lambda c: strategy.compute_positions(c, **LOW),
        "momentum flip    ": lambda c: -strategy.compute_positions(c, **LOW),
        "TREND (propre)   ": lambda c: strategy_trend.compute_positions(c),
    }
    for label, exc in [("AVEC RUT", ()), ("SANS RUT", ("RUT",))]:
        print(f"\n=== {label}  (IN-SAMPLE) ===")
        print(f"{'strategie':18}{'netSharpe med':>15}{'netRet moy%':>13}{'tr/j':>8}")
        for name, fn in posfns.items():
            sh, nr, tr = evaluate(series, fn, exclude=exc)
            print(f"{name:18}{sh:>15.2f}{nr:>13.2f}{tr:>8.1f}")
