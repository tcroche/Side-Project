"""SEEDED LEAK (R1): the momentum feature is tomorrow's close.

A one-bar negative shift moves the future back onto the current row; every
downstream decision then trades on information it could not have had.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series) -> pd.Series:
    feat = close.shift(-1) / close - 1.0          # LEAK: t sees P[t+1]
    pos = (feat > 0).astype(float)
    pos.iloc[-1] = 0.0
    return pos


def backtest(close: pd.Series) -> float:
    pos = build_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 500)))
    print(backtest(px))
