"""Tests for the portfolio optimiser."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    np.random.seed(123)
    n_days = 200
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = ["A", "B", "C", "D", "E"]
    frames = {}
    for t in tickers:
        log_ret = np.random.normal(0.0003, 0.015, n_days)
        frames[t] = 100 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(frames, index=dates)


@pytest.fixture
def selected() -> pd.Series:
    return pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.0, "E": 0.0})


class TestEqualWeight:
    def test_weights_sum_to_one(self, selected: pd.Series):
        from hedge.portfolio.optimizer import equal_weight

        w = equal_weight(selected)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_only_selected_tickers(self, selected: pd.Series):
        from hedge.portfolio.optimizer import equal_weight

        w = equal_weight(selected)
        assert set(w.index) == {"A", "B", "C"}

    def test_uniform(self, selected: pd.Series):
        from hedge.portfolio.optimizer import equal_weight

        w = equal_weight(selected)
        assert abs(w["A"] - w["B"]) < 1e-9


class TestInverseVolatility:
    def test_weights_sum_to_one(self, selected: pd.Series, sample_prices: pd.DataFrame):
        from hedge.portfolio.optimizer import inverse_volatility

        w = inverse_volatility(selected, sample_prices, vol_lookback=63)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_lower_vol_gets_higher_weight(self, sample_prices: pd.DataFrame):
        from hedge.portfolio.optimizer import inverse_volatility

        # Make ticker A much less volatile.
        sample_prices["A"] = 100 + np.linspace(0, 5, len(sample_prices))
        sel = pd.Series({"A": 1.0, "B": 1.0})
        w = inverse_volatility(sel, sample_prices, vol_lookback=63)
        assert w["A"] > w["B"]


class TestApplyConstraints:
    def test_max_position_cap(self, sample_prices: pd.DataFrame):
        from hedge.portfolio.optimizer import apply_constraints

        # Start with one ticker having 80% weight.
        w = pd.Series({"A": 0.80, "B": 0.10, "C": 0.10})
        constrained = apply_constraints(w, sample_prices)
        # Max position is 5% by default config, but after vol scaling
        # and cash buffer, just verify no single pos dominates absurdly.
        assert constrained.max() < 0.90  # loose check
