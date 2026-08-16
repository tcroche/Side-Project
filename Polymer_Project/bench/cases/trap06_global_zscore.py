"""SEEDED LEAK (R10): the signal is a whole-sample z-score.

The mean and standard deviation are computed once over the entire history, so
the z-score at the first bar already knows the level and the volatility of the
last one.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series) -> pd.Series:
    z = (close - close.mean()) / close.std()   # LEAK: full-sample statistics
    pos = (-z).clip(-1.0, 1.0)
    pos.iloc[-1] = 0.0
    return pos


def backtest(close: pd.Series) -> float:
    pos = build_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(5).normal(0, 1, 500)))
    print(backtest(px))
