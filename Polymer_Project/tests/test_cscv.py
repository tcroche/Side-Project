"""Tests for core.cscv.

Known-truth constructions where the correct PBO is known by design, plus
structural invariants. Run from the project root: pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.cscv import (
    MIN_ROWS_PER_BLOCK,
    cscv_pbo,
    pbo_null_distribution,
    pbo_percentile,
    suggest_n_blocks,
)


def _noise(n_obs: int, n_configs: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(n_obs, n_configs))


# ---------------------------------------------------------------------------
# Structure and guards
# ---------------------------------------------------------------------------


def test_rejects_odd_block_count():
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(_noise(480, 10, 0), n_blocks=15)


def test_rejects_too_few_configs():
    with pytest.raises(ValueError, match="at least 2 configurations"):
        cscv_pbo(_noise(480, 1, 0), n_blocks=8)


def test_rejects_non_finite_values():
    matrix = _noise(480, 10, 0)
    matrix[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cscv_pbo(matrix, n_blocks=8)


def test_rejects_blocks_that_are_too_short_by_default():
    """S=16 on 100 observations gives 6 rows per block: refuse loudly."""
    with pytest.raises(ValueError, match="per block"):
        cscv_pbo(_noise(100, 10, 0), n_blocks=16)


def test_short_blocks_allowed_when_strict_is_off():
    result = cscv_pbo(_noise(100, 10, 0), n_blocks=16, strict=False)
    assert result.rows_per_block < MIN_ROWS_PER_BLOCK
    assert any("per block" in note for note in result.notes)


def test_combination_count_matches_binomial_coefficient():
    result = cscv_pbo(_noise(480, 10, 0), n_blocks=8)
    assert result.n_combinations == math.comb(8, 4)


def test_trailing_observations_are_trimmed_and_reported():
    result = cscv_pbo(_noise(485, 10, 0), n_blocks=8)
    assert result.n_obs_used == 480
    assert result.n_obs_dropped == 5
    assert any("trimmed" in note or "dropped" in note for note in result.notes)


def test_pbo_is_a_probability():
    result = cscv_pbo(_noise(480, 12, 3), n_blocks=8)
    assert 0.0 <= result.pbo <= 1.0
    assert 0.0 <= result.median_oos_rank <= 1.0


def test_result_serialises_and_narrates():
    result = cscv_pbo(_noise(480, 12, 3), n_blocks=8)
    text = result.to_text()
    assert "NOT a performance figure" in text
    assert "PBO" in text
    payload = result.to_dict()
    assert payload["pbo"] == pytest.approx(result.pbo)
    assert payload["verdict"] == result.verdict


# ---------------------------------------------------------------------------
# Known-truth cases
# ---------------------------------------------------------------------------


def test_genuine_edge_gives_low_pbo():
    """A configuration whose edge dominates the selection noise is stable.

    The drift is deliberately large (per-period Sharpe 0.40). At a realistic
    per-period Sharpe of 0.15 the in-sample winner is only about two standard
    errors above the best of nineteen noise competitors, and mean PBO is 0.35
    rather than 0: CSCV has limited power when the edge is comparable to the
    cross-sectional selection noise. That limitation is real and is stated in
    the report rather than hidden behind a lucky seed.
    """
    for seed in range(6):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 0.01, size=(480, 20))
        matrix[:, 5] += 0.004
        result = cscv_pbo(matrix, n_blocks=16)
        assert result.pbo < 0.15, f"seed {seed}: PBO={result.pbo}"
        assert result.median_oos_rank > 0.85
        assert result.probability_of_loss < 0.2


def test_cscv_power_degrades_with_a_marginal_edge():
    """Documents where CSCV stops being able to tell skill from luck."""
    strong, marginal = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 0.01, size=(480, 20))
        strong.append(cscv_pbo(base + np.eye(20)[5] * 0.004, n_blocks=16).pbo)
        marginal.append(cscv_pbo(base + np.eye(20)[5] * 0.0015, n_blocks=16).pbo)
    assert float(np.mean(strong)) < float(np.mean(marginal))
    assert float(np.mean(marginal)) > 0.15


def test_alternating_regime_gives_high_pbo():
    """Loadings that flip sign between blocks: the in-sample winner is the
    out-of-sample loser by construction."""
    n_blocks, rows_per_block, n_configs = 16, 30, 20
    rng = np.random.default_rng(5)
    flip = np.repeat(
        np.where(np.arange(n_blocks) % 2 == 0, 1.0, -1.0), rows_per_block
    )
    loadings = rng.normal(0.0, 1.0, n_configs)
    matrix = rng.normal(0.0, 0.01, size=(n_blocks * rows_per_block, n_configs))
    matrix += 0.012 * np.outer(flip, loadings)

    result = cscv_pbo(matrix, n_blocks=n_blocks)
    assert result.pbo > 0.70
    assert result.median_oos_rank < 0.30
    assert result.degradation_slope < 0.0


def test_noise_gives_pbo_near_one_half_on_average():
    """A single PBO is noisy; averaged over datasets it must centre on 0.5.

    This is the test that justifies reporting a null distribution alongside
    any single observed PBO.
    """
    values = [cscv_pbo(_noise(480, 20, seed), n_blocks=16).pbo for seed in range(12)]
    assert 0.35 < float(np.mean(values)) < 0.65


def test_single_pbo_is_highly_dispersed_under_the_null():
    """Documents the limitation: the null has a standard deviation near 0.2."""
    null = pbo_null_distribution(480, 20, 16, n_simulations=60, seed=7)
    assert null.std() > 0.10, "if this ever gets small, the caveat can be relaxed"
    assert 0.35 < null.mean() < 0.65


# ---------------------------------------------------------------------------
# Null distribution helpers
# ---------------------------------------------------------------------------


def test_null_distribution_has_requested_shape_and_range():
    null = pbo_null_distribution(240, 10, 8, n_simulations=25, seed=1)
    assert null.shape == (25,)
    assert np.all((null >= 0.0) & (null <= 1.0))


def test_percentile_is_monotone_in_observed_value():
    null = pbo_null_distribution(240, 10, 8, n_simulations=60, seed=2)
    low = pbo_percentile(0.2, null)
    high = pbo_percentile(0.9, null)
    assert 0.0 <= low <= high <= 1.0
    assert high > low


def test_percentile_of_an_impossible_value_is_one():
    null = pbo_null_distribution(240, 10, 8, n_simulations=20, seed=3)
    assert pbo_percentile(1.01, null) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Block-count heuristic
# ---------------------------------------------------------------------------


def test_suggest_n_blocks_is_even_and_at_least_four():
    for n_obs in (20, 50, 103, 250, 1000, 5000):
        s = suggest_n_blocks(n_obs)
        assert s >= 4 and s % 2 == 0


def test_suggest_n_blocks_caps_at_sixteen():
    assert suggest_n_blocks(100_000) == 16


def test_suggest_n_blocks_on_the_m2_in_sample_length():
    """103 in-sample days: 8 blocks of 12 observations, C(8,4) = 70 splits."""
    assert suggest_n_blocks(103) == 8
    result = cscv_pbo(_noise(103, 18, 0), n_blocks=8)
    assert result.rows_per_block == 12
    assert result.n_combinations == 70
