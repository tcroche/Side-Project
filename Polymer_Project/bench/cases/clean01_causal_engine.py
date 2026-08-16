"""CLEAN CONTROL: the canonical causal loop engine, self-contained.

The position decided at bar t is held over [t, t+1); the engine applies pos[t]
to P[t+1] - P[t]. Nothing here may fire.
"""
import numpy as np
import pandas as pd


def compute_positions(close: pd.Series, window: int = 30, k: float = 2.0) -> pd.Series:
    m = close.rolling(window).mean().values
    s = close.rolling(window).std().values
    prices = close.values
    pos = np.zeros(len(prices))
    for t in range(len(prices) - 1):
        if np.isnan(m[t]) or not s[t] > 0:
            continue
        z = (prices[t] - m[t]) / s[t]
        pos[t] = 1.0 if z > k else (-1.0 if z < -k else 0.0)
    return pd.Series(pos, index=close.index, name="position")


def backtest(close: pd.Series) -> float:
    # pnl_t = pos[t] * (P[t+1] - P[t]) : decided at t, earned over [t, t+1).
    pos = compute_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(10).normal(0, 1, 500)))
    print(backtest(px))
