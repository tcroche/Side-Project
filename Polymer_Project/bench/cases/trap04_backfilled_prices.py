"""SEEDED LEAK (R8): missing prices are filled backwards.

bfill() writes the NEXT observed price onto every earlier gap, so any signal
computed on the filled series reacts to information before it existed.
"""
import numpy as np
import pandas as pd


def load_prices(raw: pd.Series) -> pd.Series:
    prices = raw.bfill()                 # LEAK: gaps take their FUTURE value
    return prices.dropna()


def build_positions(close: pd.Series) -> pd.Series:
    ma = close.rolling(30).mean()
    pos = (close > ma).astype(float).fillna(0.0)
    pos.iloc[-1] = 0.0
    return pos


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    raw = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 500)))
    raw[rng.integers(0, 500, 40)] = np.nan
    print(build_positions(load_prices(raw)).abs().sum())
