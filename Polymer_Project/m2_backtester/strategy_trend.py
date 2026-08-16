# strategy_trend.py

import numpy as np
import pandas as pd
import config

TREND_WINDOW = 120     # fenetre lente -> z-score lisse, peu de bascules
TREND_K_ENTRY = 1.0    # on entre quand |z| depasse ce seuil (tendance affirmee) -> LEVIER de rotation
TREND_K_EXIT = 0.0     # on sort quand z repasse ce seuil (tendance finie)
TREND_STOP = 0.01      # stop-loss 1% (essentiel en momentum ; None pour desactiver)


def rolling_zscore(close, window):
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


def compute_positions(close, window=TREND_WINDOW, k_entry=TREND_K_ENTRY,
                      k_exit=TREND_K_EXIT, stop_loss=TREND_STOP):
    prices = close.values
    n = len(prices)
    pos = np.zeros(n)
    roll_mean = close.rolling(window).mean().values
    roll_std = close.rolling(window).std().values
    current, entry_price = 0, np.nan
    for t in range(n):
        if t == n - 1:                       # flat EOD
            pos[t] = 0
            break
        if np.isnan(roll_mean[t]) or not roll_std[t] > 0:
            pos[t] = 0; current = 0; continue
        z = (prices[t] - roll_mean[t]) / roll_std[t]
        if current == 0:
            # ENTREE dans le sens de la tendance (momentum, pas contrarian)
            if z > k_entry:
                current, entry_price = +1, prices[t]
            elif z < -k_entry:
                current, entry_price = -1, prices[t]
        else:
            stopped = False
            if stop_loss is not None and entry_price > 0:
                ret = current * (prices[t] - entry_price) / entry_price
                stopped = ret < -stop_loss
            # SORTIE quand la tendance s'essouffle (z repasse la bande) ou stop
            if stopped or (current == +1 and z < k_exit) or (current == -1 and z > -k_exit):
                current, entry_price = 0, np.nan
        pos[t] = current
    return pd.Series(pos, index=close.index, name="position")


if __name__ == "__main__":
    import data_loader
    series = data_loader.load_ticker_series()
    tk = "GSPC" if "GSPC" in series else sorted(series)[0]
    df = series[tk]
    day = df[df.index.normalize() == df.index.normalize().min()]
    pos = compute_positions(day["Close"])
    print(f"{tk} {day.index.min().date()} : barres={len(pos)}  "
          f"pos finale={pos.iloc[-1]}  "
          f"repartition={pos.value_counts().to_dict()}  "
          f"changements={int((pos.diff().abs() > 0).sum())}")
