"""SEEDED LEAK (R3): the scaler is fitted on the full sample before the split.

The train rows are then standardised with a mean and variance that include the
test period: the preprocessing itself has seen the future.
"""
import numpy as np
import pandas as pd


class Scaler:
    def fit(self, x):
        self.mu, self.sd = float(np.mean(x)), float(np.std(x))
        return self

    def transform(self, x):
        return (np.asarray(x) - self.mu) / self.sd


def prepare(features: pd.Series, split: int):
    scaler = Scaler()
    scaler.fit(features)                 # LEAK: fitted before the split
    x_train = scaler.transform(features.iloc[:split])
    x_out = scaler.transform(features.iloc[split:])
    return x_train, x_out


if __name__ == "__main__":
    f = pd.Series(np.random.default_rng(2).normal(0, 1, 400))
    a, b = prepare(f, 300)
    print(a.mean(), b.mean())
