# diagnostics2.py
# (A) Autopsie d'un actif suspect : jours extremes + barres a variation anormale.
# (B) Grille 2x2 : (mean-rev / momentum) x (rotation normale / faible) sur l'IN-SAMPLE,
#     avec et sans l'actif suspect, pour voir le signal "propre".

import datetime as dt
import numpy as np
import pandas as pd

import config
import data_loader
import strategy

OOS_START = dt.date.fromisoformat(config.OOS_START)
LOW_TURN = dict(window=120, k_entry=3.0, k_exit=0.3)


def day_iter(df, end=OOS_START):
    df = df.sort_index()
    for day_ts, day_df in df.groupby(df.index.normalize()):
        if day_ts.date() < end:                 # IN-SAMPLE uniquement
            yield day_ts.date(), day_df


def eval_day(close, sign=+1, params=None):
    close = close.dropna(); n = len(close)
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


def forensic(series, ticker, sign=+1, params=None, topk=6):
    df = series[ticker]
    rows, n_jumps = [], 0
    worst = (0.0, None)
    for d, day_df in day_iter(df):
        close = day_df["Close"].dropna()
        if len(close) < 3:
            continue
        P = close.values
        r1 = np.abs(np.diff(P) / P[:-1])         # |rendement| 1-minute
        if r1.size:
            n_jumps += int((r1 > 0.01).sum())    # barres > 1% en 1 min
            i = int(np.argmax(r1))
            if r1[i] > worst[0]:
                worst = (float(r1[i]), d)
        res = eval_day(close, sign, params)
        if res:
            rows.append((str(d), 100 * res[0], res[2], len(close)))
    dd = pd.DataFrame(rows, columns=["Date", "grossRet%", "trades", "bars"])
    print(f"\n--- AUTOPSIE {ticker} (IS, sign={sign}) ---")
    print(f"jours={len(dd)}   somme grossRet%={dd['grossRet%'].sum():.2f}")
    print(f"barres |ret 1-min| > 1% : {n_jumps}   |   pire barre : "
          f"{worst[0]*100:.2f}% le {worst[1]}")
    print("Jours les plus extremes (|grossRet%|) :")
    extreme = dd.reindex(dd["grossRet%"].abs().sort_values(ascending=False).index)
    print(extreme.head(topk).to_string(index=False))


def grid(series, exclude=()):
    cells = {
        "mean-rev | base   ": dict(sign=+1, params=None),
        "mean-rev | low-trn": dict(sign=+1, params=LOW_TURN),
        "momentum | base   ": dict(sign=-1, params=None),
        "momentum | low-trn": dict(sign=-1, params=LOW_TURN),
    }
    print(f"\n=== GRILLE 2x2 (IS) — exclus : {exclude or 'aucun'} ===")
    print(f"{'variante':20}{'netSharpe med':>15}{'netRet moy%':>13}{'tr/j':>8}")
    for name, kw in cells.items():
        sharpes, nets, trs = [], [], []
        for tk, df in series.items():
            if tk in exclude:
                continue
            ns, tt = [], []
            for d, day_df in day_iter(df):
                r = eval_day(day_df["Close"], **kw)
                if r:
                    ns.append(r[1]); tt.append(r[2])
            if not ns:
                continue
            a = np.array(ns)
            sh = (a.mean() / a.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
                  if a.std() > 0 else np.nan)
            sharpes.append(sh); nets.append(100 * a.sum()); trs.append(np.mean(tt))
        print(f"{name:20}{np.nanmedian(sharpes):>15.2f}"
              f"{np.mean(nets):>13.2f}{np.mean(trs):>8.1f}")


if __name__ == "__main__":
    series = data_loader.load_ticker_series()
    target = "RUT" if "RUT" in series else sorted(series)[0]
    forensic(series, target, sign=+1)
    grid(series, exclude=())
    grid(series, exclude=(target,))
