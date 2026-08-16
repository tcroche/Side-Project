"""
Unit discipline for Sharpe-ratio statistics.

The single most common error in applied PSR / DSR work is mixing an
ANNUALIZED Sharpe ratio into a formula that expects a PER-PERIOD one.

PSR, E[max SR] and DSR are all defined on the same time scale as the
returns used to compute T (number of observations), skewness and
kurtosis. If returns are daily, then T is a number of days and the
Sharpe ratio must be the daily one. Feeding an annualized Sharpe into
the formula while T counts days inflates the z-score by sqrt(252) and
turns any strategy into a certainty.

This module exists so that the conversion is always explicit, and so
that a suspicious value fails loudly instead of silently.

Convention used throughout the project
--------------------------------------
    sr_annual      = sr_per_period * sqrt(periods_per_year)
    sr_per_period  = sr_annual / sqrt(periods_per_year)

This is the standard IID scaling. It assumes serially uncorrelated
returns; under autocorrelation the correct factor is the Lo (2002)
adjustment, not sqrt(k). See `lo_annualization_factor` below.
"""

from __future__ import annotations

import math
import warnings

# --- Standard periodicities ------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

#: A per-period Sharpe above this magnitude is almost certainly an
#: annualized figure passed by mistake. A daily Sharpe of 1.0 corresponds
#: to an annualized Sharpe of 15.9; a monthly Sharpe of 1.0 corresponds to
#: an annualized Sharpe of 3.46.
IMPLAUSIBLE_PER_PERIOD_SHARPE = 1.0


class UnitError(ValueError):
    """Raised when a value appears to be expressed in the wrong time unit."""


def to_per_period(sr_annual: float, periods_per_year: int) -> float:
    """Convert an annualized Sharpe ratio to a per-period Sharpe ratio.

    >>> round(to_per_period(1.93, TRADING_DAYS_PER_YEAR), 6)
    0.121578
    """
    _check_periods_per_year(periods_per_year)
    return sr_annual / math.sqrt(periods_per_year)


def to_annual(sr_per_period: float, periods_per_year: int) -> float:
    """Convert a per-period Sharpe ratio to an annualized Sharpe ratio.

    >>> round(to_annual(0.121578, TRADING_DAYS_PER_YEAR), 4)
    1.93
    """
    _check_periods_per_year(periods_per_year)
    return sr_per_period * math.sqrt(periods_per_year)


def check_per_period_sharpe(
    sr_per_period: float,
    *,
    name: str = "sharpe",
    strict: bool = True,
) -> float:
    """Guard against an annualized Sharpe being passed where a per-period one is expected.

    Parameters
    ----------
    sr_per_period : float
        The value to check.
    name : str
        Name used in the error message, so the caller knows which argument failed.
    strict : bool
        If True (default) an implausible value raises `UnitError`. If False it
        only emits a warning. Set to False only if you genuinely work at a
        periodicity where a per-period Sharpe above 1.0 is possible, and say so
        in writing.

    Returns
    -------
    float
        The value unchanged, so the guard can be used inline.
    """
    if not math.isfinite(sr_per_period):
        raise UnitError(f"{name} must be finite, got {sr_per_period!r}.")

    if abs(sr_per_period) > IMPLAUSIBLE_PER_PERIOD_SHARPE:
        annual_equivalent_daily = to_annual(sr_per_period, TRADING_DAYS_PER_YEAR)
        message = (
            f"{name}={sr_per_period:.4f} looks like an ANNUALIZED Sharpe ratio, "
            f"not a per-period one (it would imply an annualized Sharpe of "
            f"{annual_equivalent_daily:.2f} on daily data). "
            f"Use units.to_per_period(sr_annual, periods_per_year) first, or pass "
            f"strict=False if this periodicity really allows it."
        )
        if strict:
            raise UnitError(message)
        warnings.warn(message, stacklevel=2)

    return sr_per_period


def lo_annualization_factor(rho1: float, periods: int) -> float:
    """Lo (2002) annualization factor under first-order autocorrelation.

    Under IID returns the factor is sqrt(k). With AR(1) autocorrelation rho1
    the correct scaling of a k-period Sharpe ratio is

        k / sqrt( k + 2 * sum_{i=1}^{k-1} (k - i) * rho1**i )

    which reduces to sqrt(k) when rho1 = 0. Positive autocorrelation makes the
    naive sqrt(k) scaling OPTIMISTIC.

    This is provided for the report's caveat section: intraday strategies are
    autocorrelated, so the naive annualization overstates the Sharpe ratio.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1.")
    if not -1.0 < rho1 < 1.0:
        raise ValueError("rho1 must lie strictly between -1 and 1.")
    if periods == 1:
        return 1.0
    tail = sum((periods - i) * rho1**i for i in range(1, periods))
    return periods / math.sqrt(periods + 2.0 * tail)


def _check_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year!r}.")


if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod()
    print(f"doctest failures: {failures}")

    print("\n--- Unit conversion demo ---")
    sr_annual = 1.93
    sr_daily = to_per_period(sr_annual, TRADING_DAYS_PER_YEAR)
    print(f"annualized Sharpe : {sr_annual:.4f}  (per year)")
    print(f"daily Sharpe      : {sr_daily:.6f} (per trading day)")
    print(f"round trip        : {to_annual(sr_daily, TRADING_DAYS_PER_YEAR):.4f}")

    print("\n--- Guard demo ---")
    try:
        check_per_period_sharpe(sr_annual, name="sr_hat_per_period")
    except UnitError as exc:
        print(f"UnitError raised as expected:\n  {exc}")

    print("\n--- Lo (2002) autocorrelation factor ---")
    for rho in (0.0, 0.1, 0.3):
        f = lo_annualization_factor(rho, TRADING_DAYS_PER_YEAR)
        print(f"rho1={rho:>4}: factor={f:8.4f}  (naive sqrt(252)={math.sqrt(252):.4f})")
