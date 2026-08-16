"""SEEDED SEMANTIC LEAK: the feature reads five bars ahead by plain indexing.

No shift(), no rolling, no fit: a loop quietly indexes t + 5 while building an
INPUT used at t. Invisible to every syntactic rule; obvious once read.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series, horizon: int = 5) -> pd.Series:
    prices = close.values
    n = len(prices)
    pos = np.zeros(n)
    for t in range(n - 1):
        if t + horizon >= n:
            break
        drift = prices[t + horizon] / prices[t] - 1.0   # LEAK: reads the future
        pos[t] = 1.0 if drift > 0 else -1.0
    return pd.Series(pos, index=close.index, name="position")


def backtest(close: pd.Series) -> float:
    pos = build_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(21).normal(0, 1, 500)))
    print(backtest(px))
