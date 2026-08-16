"""SEEDED LEAK (R5): a feature is built out of the label itself.

The label is a forward-looking quantity by construction; smoothing it does not
launder it. Any input touching the label's values imports the future.
"""
import numpy as np
import pandas as pd


def make_dataset(close: pd.Series):
    target = close.pct_change().shift(-1)     # forward return: the label
    edge = target.rolling(5).mean()           # LEAK: feature reads label values
    momentum = close.pct_change(10)
    return pd.DataFrame({"edge": edge, "momentum": momentum}), target


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(6).normal(0, 1, 500)))
    features, tgt = make_dataset(px)
    print(features.corrwith(tgt))
