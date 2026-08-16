# strategy_intraday_momentum.py

import numpy as np
import pandas as pd

import config

FIRST_HALFHOUR = 30    # 1ere demi-heure = 30 barres 1-min (specification Gao et al.)
LAST_HALFHOUR  = 30    # derniere demi-heure = 30 barres 1-min


def compute_positions(close, first_window=FIRST_HALFHOUR, last_window=LAST_HALFHOUR):
    """Position (-1 / 0 / +1) pour UNE journee, indexee comme 'close'.
    0 partout SAUF pendant la derniere demi-heure, ou pos = sign(r1) avec r1 =
    rendement de la premiere demi-heure. Flat a la derniere barre (EOD)."""
    prices = close.values
    n = len(prices)
    pos = np.zeros(n)

    # Jour trop court pour une 1ere ET une derniere demi-heure disjointes -> flat
    if n < first_window + last_window + 1:
        return pd.Series(pos, index=close.index, name="position")

    # 1) Signal CAUSAL : signe du rendement de la premiere demi-heure (le SIGNE
    #    seul compte -> Gao = sign-following, aucune amplitude calibree)
    r1 = prices[first_window] - prices[0]
    direction = float(np.sign(r1))             # +1, -1 ou 0
    if direction == 0:
        return pd.Series(pos, index=close.index, name="position")

    # 2) En position pendant la DERNIERE demi-heure, puis flat a la cloture.
    #    Held sur [n-last_window, n-2] -> capture direction*(P[n-1]-P[n-last_window]),
    #    soit sign(r1) * rendement de la derniere demi-heure. pos[n-1]=0 -> exit/EOD.
    entry = n - last_window
    pos[entry:n - 1] = direction

    return pd.Series(pos, index=close.index, name="position")


if __name__ == "__main__":
    import data_loader

    series = data_loader.load_ticker_series()
    tk = "GSPC" if "GSPC" in series else sorted(series)[0]
    df = series[tk]
    day = df[df.index.normalize() == df.index.normalize().min()]
    close = day["Close"]
    n = len(close)

    pos = compute_positions(close)
    prev = pos.shift(1).fillna(0.0)
    entries = int(((prev == 0) & (pos != 0)).sum())
    nz = pos[pos != 0]

    r1 = close.values[FIRST_HALFHOUR] - close.values[0]
    print(f"Indice teste     : {tk}  ({day.index.min().date()})")
    print(f"Barres ce jour   : {n}")
    print(f"r1 (1ere 1/2h)   : {r1:+.3f}  -> direction attendue {int(np.sign(r1)):+d}")
    print(f"Position finale  : {pos.iloc[-1]:.0f}   (doit etre 0 -> flat EOD)")
    print(f"Entrees ce jour  : {entries}   (1 attendu si r1 != 0)")
    print(f"Barres en pos.   : {len(nz)}   (~{LAST_HALFHOUR - 1} attendu : derniere 1/2h)")
    if len(nz):
        print(f"Valeur position  : {sorted(set(nz.values))}   (doit etre {[int(np.sign(r1))]})")
        print(f"Fenetre tenue    : {nz.index.min().time()} -> {nz.index.max().time()}")
