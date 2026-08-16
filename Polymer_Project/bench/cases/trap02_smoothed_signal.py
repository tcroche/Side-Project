"""SEEDED LEAK (R2): the smoothing window is centred.

A centred rolling mean at bar t averages bars t-k..t+k, so the "smooth trend"
already contains half a window of future prices.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series) -> pd.Series:
    smooth = close.rolling(21, center=True).mean()   # LEAK: window spans the future
    pos = (close > smooth).astype(float).fillna(0.0)
    pos.iloc[-1] = 0.0
    return pos


def backtest(close: pd.Series) -> float:
    pos = build_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(1).normal(0, 1, 500)))
    print(backtest(px))
