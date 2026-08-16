"""CLEAN CONTROL: expanding normalisation and lagged execution.

An expanding window only ever looks backwards; combined with a one-bar lag on
the position this is the textbook causal pipeline. Nothing here may fire.
"""
import numpy as np
import pandas as pd


def build_positions(close: pd.Series) -> pd.Series:
    grow = close.expanding(50)
    z = (close - grow.mean()) / grow.std()
    pos = (-z).clip(-1.0, 1.0).shift(1).fillna(0.0)
    pos.iloc[-1] = 0.0
    return pos


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(14).normal(0, 1, 500)))
    print(build_positions(px).abs().mean())
