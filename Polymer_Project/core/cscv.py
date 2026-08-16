"""
Combinatorially Symmetric Cross-Validation (CSCV) and the Probability of
Backtest Overfitting (PBO).

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2017).
    The Probability of Backtest Overfitting. Journal of Computational
    Finance, 20(4), 39-69.

The question PBO answers
------------------------
DSR asks "is this Sharpe ratio distinguishable from luck given N trials?".
PBO asks a different and complementary question: "if I select the best
configuration in-sample, how often does it end up below median out-of-sample?".

Algorithm
---------
1. Split the T x N performance matrix into S disjoint blocks of consecutive
   rows (S even).
2. For every combination of S/2 blocks (in-sample) versus its complement
   (out-of-sample) -- there are C(S, S/2) of them:
     a. compute the performance of each of the N configurations in-sample;
     b. take n*, the in-sample winner;
     c. compute the out-of-sample performance of all N configurations;
     d. take omega, the relative rank of n* out-of-sample, in (0, 1);
     e. compute the logit lambda = ln(omega / (1 - omega)).
3. PBO = the fraction of combinations where lambda < 0, i.e. where the
   in-sample winner lands below the out-of-sample median.

Why blocks and not a single split
---------------------------------
A single split gives one observation of the selection outcome, which is
noise. Combinations of blocks give C(S, S/2) observations while keeping the
in-sample and out-of-sample sets the same size (symmetry), so the comparison
is not biased by differing sample lengths.

Known limitation
----------------
Blocks are contiguous, so within-block serial dependence is preserved but
dependence ACROSS block boundaries is destroyed by the recombination. With
strongly autocorrelated returns, use fewer and therefore longer blocks. With
short histories this is a real constraint: T/S is the block length, and a
block of a handful of observations makes the Sharpe ratio within that block
almost meaningless.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

#: Below this many rows per block, block-level performance is too noisy to mean much.
MIN_ROWS_PER_BLOCK = 8

#: Above this many combinations the computation gets heavy; warn the caller.
MAX_COMBINATIONS_WARN = 200_000


@dataclass
class PBOResult:
    """Outcome of a CSCV run, with the diagnostics that make it interpretable."""

    pbo: float
    n_blocks: int
    n_combinations: int
    n_configs: int
    n_obs_used: int
    n_obs_dropped: int
    rows_per_block: int
    logits: np.ndarray = field(repr=False)
    is_performance_of_winner: np.ndarray = field(repr=False)
    oos_performance_of_winner: np.ndarray = field(repr=False)
    winner_indices: np.ndarray = field(repr=False)
    degradation_slope: float = float("nan")
    degradation_intercept: float = float("nan")
    probability_of_loss: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def median_oos_rank(self) -> float:
        """Median relative rank of the in-sample winner, out-of-sample, in (0, 1).

        0.5 means the winner is a coin flip out-of-sample. Below 0.5 means
        selection actively hurts.
        """
        omega = 1.0 / (1.0 + np.exp(-self.logits))
        return float(np.median(omega))

    @property
    def verdict(self) -> str:
        if self.pbo >= 0.5:
            return "SEVERE (selection is worse than a coin flip)"
        if self.pbo >= 0.25:
            return "ELEVATED"
        return "LOW"

    def to_text(self) -> str:
        lines = [
            "PROBABILITY OF BACKTEST OVERFITTING (CSCV)",
            "=" * 68,
            f"Configurations N         : {self.n_configs}",
            f"Observations used        : {self.n_obs_used} "
            f"({self.n_obs_dropped} dropped so that T divides evenly by S)",
            f"Blocks S                 : {self.n_blocks} "
            f"({self.rows_per_block} observations per block)",
            f"Combinations C(S, S/2)   : {self.n_combinations}",
            "",
            f"PBO                      : {self.pbo:.4f} "
            f"(probability, in [0,1] -- NOT a performance figure)",
            f"    -> the in-sample winner falls below the out-of-sample median "
            f"in {self.pbo:.1%} of splits.",
            f"Median OOS rank of winner: {self.median_oos_rank:.3f} "
            f"(0.5 = indistinguishable from a coin flip)",
            f"Probability of loss      : {self.probability_of_loss:.4f} "
            f"(winner has negative OOS performance)",
            f"Performance degradation  : OOS = {self.degradation_slope:+.3f} * IS "
            f"{self.degradation_intercept:+.3f}",
            f"    -> a negative slope means better in-sample implies WORSE "
            f"out-of-sample.",
            f"VERDICT                  : {self.verdict}",
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "pbo": self.pbo,
            "n_blocks": self.n_blocks,
            "n_combinations": self.n_combinations,
            "n_configs": self.n_configs,
            "n_obs_used": self.n_obs_used,
            "n_obs_dropped": self.n_obs_dropped,
            "rows_per_block": self.rows_per_block,
            "median_oos_rank": self.median_oos_rank,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "probability_of_loss": self.probability_of_loss,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def _block_moments(
    matrix: np.ndarray, n_blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block count, sum and sum of squares, for each configuration.

    Sharpe ratios depend only on the first two moments, so a combination's
    in-sample statistics are obtained by ADDING the moments of its blocks.
    That turns C(S, S/2) recomputations into three matrix products.
    """
    n_obs, n_configs = matrix.shape
    rows_per_block = n_obs // n_blocks
    trimmed = matrix[: rows_per_block * n_blocks]
    blocks = trimmed.reshape(n_blocks, rows_per_block, n_configs)

    counts = np.full(n_blocks, rows_per_block, dtype=float)
    sums = blocks.sum(axis=1)
    sumsq = (blocks**2).sum(axis=1)
    return counts, sums, sumsq


def _sharpe_from_moments(
    n: np.ndarray, s: np.ndarray, ss: np.ndarray
) -> np.ndarray:
    """Per-period Sharpe ratio from aggregated moments. Shapes: n (C,), s/ss (C, N)."""
    n_col = n[:, None]
    mean = s / n_col
    var = (ss - n_col * mean**2) / (n_col - 1.0)
    var = np.where(var > 0.0, var, np.nan)
    return mean / np.sqrt(var)


def cscv_pbo(
    returns_matrix: np.ndarray,
    n_blocks: int = 16,
    *,
    strict: bool = True,
) -> PBOResult:
    """Run CSCV and return the probability of backtest overfitting.

    Parameters
    ----------
    returns_matrix : array of shape (T, N)
        One column per configuration, one row per period. These must be
        RETURNS, not cumulative performance: the Sharpe ratio is computed
        within each block combination.
    n_blocks : int
        S, the number of blocks. Must be even and at least 4. The default of
        16 follows the paper; lower it when the history is short so that each
        block still holds enough observations.
    strict : bool
        If True, raise when blocks would be shorter than MIN_ROWS_PER_BLOCK.
        If False, warn and proceed.

    Returns
    -------
    PBOResult
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D, got shape {matrix.shape}.")

    n_obs, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError(f"Need at least 2 configurations, got {n_configs}.")
    if n_blocks < 4 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and >= 4, got {n_blocks}.")
    if n_obs < n_blocks:
        raise ValueError(f"Need at least {n_blocks} observations, got {n_obs}.")
    if not np.isfinite(matrix).all():
        raise ValueError("returns_matrix contains NaN or inf; clean it first.")

    notes: list[str] = []
    rows_per_block = n_obs // n_blocks
    dropped = n_obs - rows_per_block * n_blocks

    if rows_per_block < MIN_ROWS_PER_BLOCK:
        message = (
            f"{rows_per_block} observations per block (S={n_blocks}, T={n_obs}). "
            f"Below {MIN_ROWS_PER_BLOCK}, a within-block Sharpe ratio is mostly "
            f"noise and PBO becomes unreliable. Reduce n_blocks."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
        notes.append(message)

    if dropped:
        notes.append(
            f"{dropped} trailing observation(s) dropped so that T divides evenly "
            f"by S; this is the standard CSCV trimming."
        )

    n_comb = math.comb(n_blocks, n_blocks // 2)
    if n_comb > MAX_COMBINATIONS_WARN:
        warnings.warn(
            f"C({n_blocks}, {n_blocks // 2}) = {n_comb} combinations; this will be "
            f"slow and memory-hungry.",
            stacklevel=2,
        )

    counts, sums, sumsq = _block_moments(matrix, n_blocks)

    # Boolean masks over blocks: one row per combination.
    masks = np.zeros((n_comb, n_blocks), dtype=float)
    for i, combo in enumerate(combinations(range(n_blocks), n_blocks // 2)):
        masks[i, list(combo)] = 1.0
    complement = 1.0 - masks

    sharpe_is = _sharpe_from_moments(masks @ counts, masks @ sums, masks @ sumsq)
    sharpe_oos = _sharpe_from_moments(
        complement @ counts, complement @ sums, complement @ sumsq
    )

    if np.isnan(sharpe_is).any() or np.isnan(sharpe_oos).any():
        notes.append(
            "Some block combinations produced a zero-variance configuration; "
            "those entries are treated as the worst rank."
        )
        sharpe_is = np.nan_to_num(sharpe_is, nan=-np.inf)
        sharpe_oos = np.nan_to_num(sharpe_oos, nan=-np.inf)

    winners = np.argmax(sharpe_is, axis=1)
    rows = np.arange(n_comb)
    winner_oos = sharpe_oos[rows, winners]
    winner_is = sharpe_is[rows, winners]

    # Relative rank of the winner out-of-sample, mapped into the open (0, 1).
    ranks = (sharpe_oos < winner_oos[:, None]).sum(axis=1) + 1
    omega = ranks / (n_configs + 1.0)
    logits = np.log(omega / (1.0 - omega))

    pbo = float(np.mean(logits < 0.0))

    # Degradation: does a better in-sample rank predict a better out-of-sample one?
    finite = np.isfinite(winner_is) & np.isfinite(winner_oos)
    if finite.sum() >= 2 and np.std(winner_is[finite]) > 0:
        slope, intercept = np.polyfit(winner_is[finite], winner_oos[finite], 1)
    else:
        slope = intercept = float("nan")

    prob_loss = float(np.mean(winner_oos[finite] < 0.0)) if finite.any() else float("nan")

    if rows_per_block < 20:
        notes.append(
            f"Blocks hold {rows_per_block} observations. Serial dependence across "
            f"block boundaries is destroyed by recombination, so PBO is optimistic "
            f"if the returns are strongly autocorrelated."
        )

    return PBOResult(
        pbo=pbo,
        n_blocks=n_blocks,
        n_combinations=n_comb,
        n_configs=n_configs,
        n_obs_used=rows_per_block * n_blocks,
        n_obs_dropped=dropped,
        rows_per_block=rows_per_block,
        logits=logits,
        is_performance_of_winner=winner_is,
        oos_performance_of_winner=winner_oos,
        winner_indices=winners,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        probability_of_loss=prob_loss,
        notes=notes,
    )


def pbo_null_distribution(
    n_obs: int,
    n_configs: int,
    n_blocks: int,
    *,
    n_simulations: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Monte-Carlo distribution of PBO under IID noise with no skill.

    PBO from a single dataset is a very noisy statistic. At T=103, N=18, S=8
    the null has a mean near 0.49 but a standard deviation near 0.21, so a
    90% interval of roughly [0.16, 0.87]. An observed PBO of 0.70 is therefore
    unremarkable at those dimensions, even though it sounds alarming.

    Simulating the null at the caller's own (T, N, S) turns a bare number into
    a percentile. Report both.

    Returns
    -------
    ndarray of shape (n_simulations,)
        PBO values obtained from pure-noise matrices of the same dimensions.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_simulations, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_simulations):
            noise = rng.normal(0.0, 0.01, size=(n_obs, n_configs))
            out[i] = cscv_pbo(noise, n_blocks=n_blocks, strict=False).pbo
    return out


def pbo_percentile(observed_pbo: float, null_sample: np.ndarray) -> float:
    """Where the observed PBO sits in its own null distribution, in [0, 1].

    0.95 means only 5% of no-skill datasets of the same shape would produce a
    PBO this high: evidence of genuine overfitting. 0.50 means the observed
    value is exactly what noise produces.
    """
    null = np.asarray(null_sample, dtype=float)
    return float(np.mean(null <= observed_pbo))


def suggest_n_blocks(n_obs: int, *, target_rows_per_block: int = 12) -> int:
    """Largest even S >= 4 such that each block holds at least the target rows.

    With 103 in-sample days and a target of 12 rows per block, this returns 8,
    giving C(8, 4) = 70 combinations. Fewer combinations than the paper's 16
    blocks, but blocks that actually mean something.
    """
    if n_obs < 4 * target_rows_per_block:
        return 4
    s = n_obs // target_rows_per_block
    s = min(s, 16)
    if s % 2 == 1:
        s -= 1
    return max(s, 4)


__all__ = [
    "PBOResult",
    "cscv_pbo",
    "pbo_null_distribution",
    "pbo_percentile",
    "suggest_n_blocks",
    "MIN_ROWS_PER_BLOCK",
]
