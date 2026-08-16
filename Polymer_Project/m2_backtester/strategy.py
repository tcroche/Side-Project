# strategy.py
# Strategie mean-reversion par z-score, intraday, 100% CAUSALE.
# Principe : a la minute t, on ne regarde QUE les prix jusqu'a t (aucun futur).
# La position decidee en t est detenue de t a t+1 (le PnL est calcule ainsi
# dans le backtester) -> c'est la garantie anti-look-ahead.
# Position forcee a 0 sur la derniere barre -> aucune position overnight.

import numpy as np
import pandas as pd

import config


def rolling_zscore(close, window):
    """z_t = (P_t - moyenne) / ecart-type, sur la fenetre des 'window' dernieres
    minutes. pandas .rolling() est causal : a l'instant t il n'utilise que le passé."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


def compute_positions(close, window=config.STRAT_WINDOW,
                      k_entry=config.STRAT_K_ENTRY,
                      k_exit=config.STRAT_K_EXIT,
                      stop_loss=config.STRAT_STOP_LOSS):
    """Positions (-1 short / 0 flat / +1 long) pour UNE journée, indexées comme
    'close'. La boucle explicite rend la causalite evidente (a defendre dans le
    rapport)."""
    prices = close.values
    n = len(prices)
    pos = np.zeros(n)

    roll_mean = close.rolling(window).mean().values
    roll_std = close.rolling(window).std().values

    current = 0            # position courante : -1, 0, +1
    entry_price = np.nan   # prix d'entree (pour le stop-loss)

    for t in range(n):
        # 1) Flat force a la derniere barre -> pas de position overnight
        if t == n - 1:
            pos[t] = 0
            break
        # 2) Fenetre pas encore remplie / ecart-type nul -> rester flat
        if np.isnan(roll_mean[t]) or not roll_std[t] > 0:
            pos[t] = 0
            current = 0
            continue

        z = (prices[t] - roll_mean[t]) / roll_std[t]

        if current == 0:
            # 3) Entree : prix anormalement bas -> long ; haut -> short
            if z < -k_entry:
                current, entry_price = +1, prices[t]
            elif z > k_entry:
                current, entry_price = -1, prices[t]
        else:
            # 4) Sortie : retour vers la moyenne (|z| petit) ou stop-loss touche
            exit_signal = abs(z) < k_exit
            if stop_loss is not None and entry_price > 0:
                ret = current * (prices[t] - entry_price) / entry_price
                if ret < -stop_loss:
                    exit_signal = True
            if exit_signal:
                current, entry_price = 0, np.nan

        pos[t] = current

    return pd.Series(pos, index=close.index, name="position")


if __name__ == "__main__":
    # Test sur une vraie journee
    import data_loader

    series = data_loader.load_ticker_series()
    tk = "GSPC" if "GSPC" in series else sorted(series)[0]
    df = series[tk]

    first_day = df.index.normalize().min()
    day = df[df.index.normalize() == first_day]
    close = day["Close"]

    pos = compute_positions(close)
    trades = int((pos.diff().abs() > 0).sum())
    print(f"Indice teste    : {tk}  ({first_day.date()})")
    print(f"Barres ce jour  : {len(close)}")
    print(f"Position finale : {pos.iloc[-1]}  (doit etre 0 -> flat EOD)")
    print(f"Repartition pos : {pos.value_counts().to_dict()}")
    print(f"Nb changements  : {trades}  (entrees + sorties)")
