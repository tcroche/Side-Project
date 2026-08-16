"""CLEAN CONTROL: the false-positive regression corpus, as one script.

Every construct below once fired a rule during the self-audit and was fixed:
alignment on y_test.index (metadata, not values), a Sharpe ratio written as
mean()/std() (a ratio of statistics, not a normalisation), and a z-score
through a bound rolling handle. Nothing here may fire.
"""
import numpy as np
import pandas as pd


def evaluate(predictions: np.ndarray, y_test: pd.Series) -> pd.Series:
    preds = pd.Series(predictions, index=y_test.index)   # alignment only
    return preds


def sharpe(rets: pd.Series) -> float:
    return float(rets.mean() / rets.std())               # ratio of statistics


def rolling_z(close: pd.Series, window: int = 30) -> pd.Series:
    roll = close.rolling(window)
    return (close - roll.mean()) / roll.std()            # trailing, causal


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    y = pd.Series(rng.normal(0, 1, 200))
    print(sharpe(y), evaluate(rng.normal(0, 1, 200), y).shape, rolling_z(y).std())
