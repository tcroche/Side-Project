"""DEPENDENT CONTROL: a signal-only module with no engine in sight.

Whether writing the bar-t signal into pos[t] is causal depends entirely on the
consumer's P&L convention, which lives outside this file. The correct semantic
answer is a QUESTION at severity "review" naming that convention -- not a
verdict, and not silence.
"""
import numpy as np
import pandas as pd


def compute_positions(close: pd.Series, window: int = 60, k: float = 1.5) -> pd.Series:
    m = close.rolling(window).mean().values
    s = close.rolling(window).std().values
    prices = close.values
    pos = np.zeros(len(prices))
    for t in range(len(prices)):
        if np.isnan(m[t]) or not s[t] > 0:
            continue
        z = (prices[t] - m[t]) / s[t]
        pos[t] = 1.0 if z > k else (-1.0 if z < -k else 0.0)
    return pd.Series(pos, index=close.index, name="position")


if __name__ == "__main__":
    px = pd.Series(100.0 + np.cumsum(np.random.default_rng(30).normal(0, 1, 400)))
    print(compute_positions(px).value_counts())
