"""CLEAN CONTROL: the scaler is fitted on the training slice only.

Same shape as the trapped version, with the one difference that matters: the
statistics come from x_train alone. Nothing here may fire.
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
    x_train = features.iloc[:split]
    x_out = features.iloc[split:]
    scaler = Scaler()
    scaler.fit(x_train)                  # statistics from the training slice only
    return scaler.transform(x_train), scaler.transform(x_out)


if __name__ == "__main__":
    f = pd.Series(np.random.default_rng(12).normal(0, 1, 400))
    a, b = prepare(f, 300)
    print(a.mean(), b.mean())
