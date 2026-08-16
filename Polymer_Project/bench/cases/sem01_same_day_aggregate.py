"""SEEDED SEMANTIC LEAK: each minute is scaled by the WHOLE session's mean.

groupby(day).transform("mean") hands every intraday bar the average of its
entire trading day -- morning bars see the afternoon. No banned keyword
appears: syntactically this is indistinguishable from a trailing window, which
is exactly why it belongs to the semantic pass.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series) -> pd.Series:
    day = close.index.normalize()
    day_mean = close.groupby(day).transform("mean")   # LEAK: full-day statistic
    rel = close / day_mean - 1.0                      # known only at the close
    pos = (-rel).clip(-1.0, 1.0)
    pos.iloc[-1] = 0.0
    return pos


def backtest(close: pd.Series) -> float:
    pos = build_positions(close).values
    dP = np.diff(close.values)
    return float((pos[:-1] * dP).sum())


if __name__ == "__main__":
    idx = pd.date_range("2025-01-06 09:30", periods=390 * 3, freq="1min")
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(20).normal(0, 0.05, len(idx))), index=idx)
    print(backtest(px))
