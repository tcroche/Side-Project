"""CLEAN CONTROL: properly lagged features, trailing windows, forward fill.

Positive shifts, trailing rolling statistics and ffill() are the causal
versions of the patterns the rules hunt. Nothing here may fire.
"""
import numpy as np
import pandas as pd


def make_features(close: pd.Series) -> pd.DataFrame:
    px = close.ffill()
    lag1 = px.shift(1)
    mom = px.pct_change(10).shift(1)
    vol = px.pct_change().rolling(20).std()
    return pd.DataFrame({"lag1": lag1, "mom": mom, "vol": vol})


if __name__ == "__main__":
    raw = pd.Series(100.0 + np.cumsum(np.random.default_rng(11).normal(0, 1, 400)))
    print(make_features(raw).notna().sum())
