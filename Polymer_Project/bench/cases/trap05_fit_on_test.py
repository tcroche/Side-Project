"""SEEDED LEAK (R9): the model is fitted on the held-out slice.

Whatever the metric says afterwards, it is an in-sample number wearing an
out-of-sample name.
"""
import numpy as np
import pandas as pd


class Ridge:
    def fit(self, x, resp):
        x = np.asarray(x, dtype=float)
        self.beta = float(np.dot(x, np.asarray(resp)) / (np.dot(x, x) + 1.0))
        return self

    def predict(self, x):
        return self.beta * np.asarray(x, dtype=float)


def evaluate(x_train, x_test, resp_train, resp_test) -> float:
    model = Ridge()
    model.fit(x_test, resp_test)     # LEAK: fitted on the test slice
    err = model.predict(x_test) - np.asarray(resp_test)
    return float(np.mean(err ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 400)
    y_all = 0.3 * x + rng.normal(0, 1, 400)
    print(evaluate(x[:300], x[300:], y_all[:300], y_all[300:]))
