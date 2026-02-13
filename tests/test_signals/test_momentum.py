"""Tests for the momentum signal generator."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    Generate synthetic price data for 5 tickers over 300 trading days.
    Ticker A trends up strongly, E trends down, others are flat-ish.
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = ["A", "B", "C", "D", "E"]

    # Cumulative returns with drift.
    drifts = [0.001, 0.0002, 0.0001, -0.0001, -0.0008]
    frames = {}
    for t, drift in zip(tickers, drifts):
        log_ret = np.random.normal(drift, 0.015, n_days)
        price = 100 * np.exp(np.cumsum(log_ret))
        frames[t] = price

    return pd.DataFrame(frames, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMomentum:
    def test_shape_matches_input(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum

        mom = compute_momentum(sample_prices, slow_period=252, skip_period=21)
        assert mom.shape == sample_prices.shape

    def test_early_rows_are_nan(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum

        mom = compute_momentum(sample_prices, slow_period=252, skip_period=21)
        # First 252 rows should be NaN (not enough lookback).
        assert mom.iloc[:252].isna().all().all()

    def test_strong_uptrend_has_positive_momentum(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum

        mom = compute_momentum(sample_prices, slow_period=200, skip_period=21)
        # Ticker A (strong uptrend) should have positive momentum in later rows.
        valid = mom["A"].dropna()
        assert valid.iloc[-1] > 0

    def test_strong_downtrend_has_negative_momentum(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum

        mom = compute_momentum(sample_prices, slow_period=200, skip_period=21)
        valid = mom["E"].dropna()
        assert valid.iloc[-1] < 0


class TestRankAndSelect:
    def test_selects_correct_count(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum, rank_and_select

        mom = compute_momentum(sample_prices, slow_period=200, skip_period=21)
        selected = rank_and_select(mom, top_quantile=0.40)
        # With 5 tickers and top 40%, we expect ~2 selected per row.
        last_row = selected.iloc[-1]
        n_selected = (last_row > 0).sum()
        assert 1 <= n_selected <= 3

    def test_output_is_binary(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_momentum, rank_and_select

        mom = compute_momentum(sample_prices, slow_period=200, skip_period=21)
        selected = rank_and_select(mom, top_quantile=0.20)
        unique_vals = set(selected.values.flatten())
        assert unique_vals.issubset({0.0, 1.0, np.nan})


class TestVolatility:
    def test_volatility_is_positive(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_volatility

        vol = compute_volatility(sample_prices, lookback=63)
        valid = vol.dropna()
        assert (valid > 0).all().all()

    def test_annualised_magnitude(self, sample_prices: pd.DataFrame):
        from hedge.signals.momentum import compute_volatility

        vol = compute_volatility(sample_prices, lookback=63)
        valid = vol.dropna()
        # With daily sigma ~1.5%, annualised should be ~24%.
        # Allow wide range to account for randomness.
        assert valid.mean().mean() < 1.0  # sanity: not >100%
        assert valid.mean().mean() > 0.05  # sanity: not near zero
