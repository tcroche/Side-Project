"""SEEDED LEAKS (R1 + R8): a future shift and a backward fill in one loader.

Two independent leaks; a detector must find both, not stop at the first.
"""
import numpy as np
import pandas as pd


def load(raw: pd.Series) -> pd.Series:
    px = raw.fillna(method="bfill")           # LEAK 1: gaps take future values
    return px.dropna()


def build_positions(close: pd.Series) -> pd.Series:
    sig = close.shift(-2)                     # LEAK 2: t sees P[t+2]
    pos = (sig > close).astype(float).fillna(0.0)
    pos.iloc[-1] = 0.0
    return pos


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    raw = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 500)))
    raw[rng.integers(0, 500, 25)] = np.nan
    print(build_positions(load(raw)).abs().sum())
