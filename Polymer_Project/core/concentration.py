"""
Concentration and fragility diagnostics.

The Deflated Sharpe Ratio answers "is this distinguishable from luck?". It does
not answer "where did the profit actually come from?". Those are different
questions, and the second one is often the more damning.

A strategy can post a respectable Sharpe ratio while losing money on most days
and being rescued by one extreme session. Standard performance tables hide this
completely: the mean, the standard deviation and the Sharpe ratio are all
compatible with a single enormous outlier. Skewness and kurtosis hint at it, but
they are hard to read and, on a hundred observations, badly estimated.

This module makes the dependence explicit and countable:

  * what share of total P&L comes from the best 1, 3 and 5 days;
  * how many of the best days have to be removed before cumulative P&L turns
    negative;
  * the range of the Sharpe ratio under a leave-one-day-out jackknife;
  * the share of days that are actually profitable, and the median day.

These numbers require no statistics background to interpret, which makes them
the right material for a mixed audience, and they are exactly what a risk
reviewer asks for first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConcentrationReport:
    """Where a backtest's profit actually came from."""

    n_obs: int
    total_return: float
    top_day_label: str
    top_day_return: float
    top_1_share: float
    top_3_share: float
    top_5_share: float
    days_to_zero: int | None
    positive_day_share: float
    median_return: float
    sharpe_full: float
    sharpe_jackknife_min: float
    sharpe_jackknife_min_label: str
    sharpe_jackknife_max: float
    sharpe_ex_top_day: float
    herfindahl_of_gains: float
    periods_per_year: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_fragile(self) -> bool:
        """True when a single observation carries an implausible share of the result."""
        return self.top_1_share >= 0.4 or (self.days_to_zero is not None and self.days_to_zero <= 2)

    @property
    def verdict(self) -> str:
        if self.top_1_share >= 0.5:
            return "SINGLE-EVENT (one observation carries the result)"
        if self.is_fragile:
            return "FRAGILE"
        if self.top_3_share >= 0.6:
            return "CONCENTRATED"
        return "DISTRIBUTED"

    def to_text(self) -> str:
        dtz = (
            f"{self.days_to_zero}"
            if self.days_to_zero is not None
            else "n/a (cumulative P&L never turns negative)"
        )
        lines = [
            "CONCENTRATION AND FRAGILITY",
            "=" * 68,
            f"Observations             : {self.n_obs}",
            f"Total return (additive)  : {100 * self.total_return:+.3f}%",
            f"Profitable days          : {100 * self.positive_day_share:.1f}%",
            f"Median day               : {100 * self.median_return:+.4f}%",
            "",
            f"Best single day          : {self.top_day_label} at "
            f"{100 * self.top_day_return:+.3f}%",
            f"Share of total from top 1: {100 * self.top_1_share:.1f}%",
            f"Share of total from top 3: {100 * self.top_3_share:.1f}%",
            f"Share of total from top 5: {100 * self.top_5_share:.1f}%",
            f"Best days to remove before total turns negative: {dtz}",
            "",
            f"Annualized Sharpe, full  : {self.sharpe_full:+.3f}",
            f"  without the best day   : {self.sharpe_ex_top_day:+.3f}",
            f"  jackknife range        : [{self.sharpe_jackknife_min:+.3f}, "
            f"{self.sharpe_jackknife_max:+.3f}] "
            f"(minimum reached by dropping {self.sharpe_jackknife_min_label})",
            "",
            f"Herfindahl of daily gains: {self.herfindahl_of_gains:.4f} "
            f"(1.0 = a single winning day; 1/n = perfectly even)",
            f"VERDICT                  : {self.verdict}",
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "total_return": self.total_return,
            "top_day_label": self.top_day_label,
            "top_day_return": self.top_day_return,
            "top_1_share": self.top_1_share,
            "top_3_share": self.top_3_share,
            "top_5_share": self.top_5_share,
            "days_to_zero": self.days_to_zero,
            "positive_day_share": self.positive_day_share,
            "median_return": self.median_return,
            "sharpe_full": self.sharpe_full,
            "sharpe_ex_top_day": self.sharpe_ex_top_day,
            "sharpe_jackknife_min": self.sharpe_jackknife_min,
            "sharpe_jackknife_min_label": self.sharpe_jackknife_min_label,
            "sharpe_jackknife_max": self.sharpe_jackknife_max,
            "herfindahl_of_gains": self.herfindahl_of_gains,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def _annualized_sharpe(values: np.ndarray, periods_per_year: int) -> float:
    if values.size < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(np.mean(values) / sd * math.sqrt(periods_per_year))


def concentration_report(
    returns,
    *,
    periods_per_year: int = 252,
    labels=None,
) -> ConcentrationReport:
    """Analyse where a return series' profit is concentrated.

    Parameters
    ----------
    returns : sequence, ndarray or pandas Series
        Periodic returns. If a Series with a DatetimeIndex is passed, its index
        is used to label the extreme observations.
    periods_per_year : int
        Used only to annualize the Sharpe ratios that are reported.
    labels : sequence, optional
        Explicit labels for the observations, overriding a Series index.
    """
    if isinstance(returns, pd.Series):
        series = returns.dropna()
        values = series.to_numpy(dtype=float)
        index = [str(getattr(i, "date", lambda: i)()) for i in series.index]
    else:
        values = np.asarray(returns, dtype=float)
        values = values[np.isfinite(values)]
        index = [str(x) for x in range(len(values))]

    if labels is not None:
        index = [str(x) for x in labels][: len(values)]

    n_obs = values.size
    if n_obs < 5:
        raise ValueError(f"Need at least 5 observations, got {n_obs}.")

    total = float(values.sum())
    order = np.argsort(values)[::-1]  # descending

    def share(k: int) -> float:
        if total == 0.0:
            return float("nan")
        return float(values[order[:k]].sum() / total)

    # How many of the best observations must be removed before the total flips.
    days_to_zero: int | None = None
    if total > 0.0:
        running = total
        for k, idx in enumerate(order, start=1):
            running -= float(values[idx])
            if running <= 0.0:
                days_to_zero = k
                break

    # Leave-one-out jackknife on the Sharpe ratio.
    jack = np.empty(n_obs, dtype=float)
    for i in range(n_obs):
        jack[i] = _annualized_sharpe(np.delete(values, i), periods_per_year)
    j_min_idx = int(np.nanargmin(jack))
    j_max_idx = int(np.nanargmax(jack))

    gains = values[values > 0.0]
    if gains.size and gains.sum() > 0:
        weights = gains / gains.sum()
        herfindahl = float(np.sum(weights**2))
    else:
        herfindahl = float("nan")

    notes: list[str] = []
    top1 = share(1)
    if np.isfinite(top1) and top1 >= 0.5:
        notes.append(
            f"A single observation ({index[order[0]]}) accounts for "
            f"{100 * top1:.0f}% of the total. Any conclusion about this strategy "
            f"is really a conclusion about that one day."
        )
    positive_share = float(np.mean(values > 0.0))
    if positive_share < 0.5 and total > 0.0:
        notes.append(
            f"Only {100 * positive_share:.0f}% of observations are profitable, yet "
            f"the total is positive: the result rests on the size of a few winners, "
            f"not on their frequency."
        )
    if days_to_zero is not None and days_to_zero <= 3:
        notes.append(
            f"Removing the best {days_to_zero} observation(s) takes the cumulative "
            f"result below zero."
        )

    return ConcentrationReport(
        n_obs=n_obs,
        total_return=total,
        top_day_label=index[order[0]],
        top_day_return=float(values[order[0]]),
        top_1_share=top1,
        top_3_share=share(3),
        top_5_share=share(5),
        days_to_zero=days_to_zero,
        positive_day_share=positive_share,
        median_return=float(np.median(values)),
        sharpe_full=_annualized_sharpe(values, periods_per_year),
        sharpe_jackknife_min=float(jack[j_min_idx]),
        sharpe_jackknife_min_label=index[j_min_idx],
        sharpe_jackknife_max=float(jack[j_max_idx]),
        sharpe_ex_top_day=_annualized_sharpe(
            np.delete(values, order[0]), periods_per_year
        ),
        herfindahl_of_gains=herfindahl,
        periods_per_year=periods_per_year,
        notes=notes,
    )


def shared_dependence(
    matrix: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """For each observation, how many configurations depend on it.

    Takes a T x N frame of returns (one column per trial) and returns, for the
    most influential observations, the number of trials that draw at least
    `threshold` of their total P&L from that single observation.

    When every configuration in a grid leans on the same session, the grid was
    never exploring different strategies: it was re-expressing one event.
    """
    totals = matrix.sum(axis=0)
    usable = totals[totals > 0.0].index
    if len(usable) == 0:
        return pd.DataFrame(columns=["n_trials_dependent", "share_of_trials"])

    shares = matrix[usable].div(totals[usable], axis=1)
    counts = (shares >= threshold).sum(axis=1)
    counts = counts[counts > 0].sort_values(ascending=False)

    out = pd.DataFrame(
        {
            "n_trials_dependent": counts,
            "share_of_trials": counts / len(usable),
        }
    )
    out.index.name = matrix.index.name or "observation"
    return out


__all__ = ["ConcentrationReport", "concentration_report", "shared_dependence"]
