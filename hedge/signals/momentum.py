"""
Momentum signal generator.

This is the heart of the strategy. The core factor is **12-1 momentum**:
the total return over the past 12 months, *excluding* the most recent month
(which empirically shows short-term mean-reversion). We then optionally
adjust by trailing volatility to get a Sharpe-like signal, rank
cross-sectionally, and keep the top quantile.

References:
  Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers
  Moskowitz, Ooi & Pedersen (2012) — Time-Series Momentum
  Daniel & Moskowitz (2016) — Momentum Crashes

CRITICAL GOTCHAS:
  1. All calculations use strictly past data relative to the signal date.
     The skip_period ensures we don't use the most recent month's return.
  2. Returns are computed from adjusted close prices that already account
     for splits and dividends. Using raw close would be wrong.
  3. NaNs in early rows are expected (insufficient history) and are NOT
     forward-filled — they propagate as NaN in the signal, meaning those
     tickers are excluded from ranking on that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Core factor: 12-1 Momentum
# ======================================================================

def compute_momentum(
    prices: pd.DataFrame,
    slow_period: int | None = None,
    skip_period: int | None = None,
) -> pd.DataFrame:
    """
    Compute the 12-1 momentum factor for each ticker.

    momentum_t = price_{t - skip} / price_{t - slow} - 1

    This gives the return from (t - slow) to (t - skip), so the most
    recent *skip_period* days of return are excluded.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close prices. Columns = tickers, index = trading dates.
    slow_period : int
        Total lookback in trading days (~252 for 12 months).
    skip_period : int
        Days to skip at the end (~21 for 1 month).

    Returns
    -------
    DataFrame of same shape as *prices* with momentum values.
    """
    cfg = get_settings()
    slow = slow_period or cfg.signals.momentum.slow_period
    skip = skip_period or cfg.signals.momentum.skip_period

    lagged_skip = prices.shift(skip)
    lagged_slow = prices.shift(slow)

    mom = lagged_skip / lagged_slow - 1.0
    logger.debug(
        "Computed 12-1 momentum: slow=%d skip=%d  shape=%s",
        slow,
        skip,
        mom.shape,
    )
    return mom


def compute_volatility(
    prices: pd.DataFrame,
    lookback: int | None = None,
) -> pd.DataFrame:
    """
    Annualised trailing volatility of daily log-returns.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close prices.
    lookback : int
        Rolling window in trading days.

    Returns
    -------
    DataFrame of annualised volatilities.
    """
    cfg = get_settings()
    lb = lookback or cfg.signals.vol_lookback
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(lb, min_periods=lb).std() * np.sqrt(252)
    return vol


def compute_risk_adjusted_momentum(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Momentum divided by trailing volatility — a Sharpe-like signal.

    This dampens high-momentum names that achieved their return through
    wild swings, and boosts steady compounders.
    """
    mom = compute_momentum(prices)
    vol = compute_volatility(prices)

    # Avoid division by near-zero vol.
    vol_clipped = vol.clip(lower=0.05)
    risk_adj = mom / vol_clipped
    return risk_adj


# ======================================================================
# Cross-sectional ranking & selection
# ======================================================================

def rank_and_select(
    signal: pd.DataFrame,
    top_quantile: float | None = None,
    long_only: bool | None = None,
) -> pd.DataFrame:
    """
    On each date, rank all tickers by *signal* and flag the top quantile
    as selected (1.0), the rest as 0.0.

    Parameters
    ----------
    signal : DataFrame
        Raw signal values (higher = stronger momentum).
    top_quantile : float
        Fraction of tickers to keep (e.g., 0.10 for top decile).

    Returns
    -------
    DataFrame with 1.0 for selected tickers, 0.0 otherwise.
    """
    cfg = get_settings()
    q = top_quantile or cfg.signals.top_quantile
    lo = long_only if long_only is not None else cfg.signals.long_only

    def _select_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return pd.Series(0.0, index=row.index)
        threshold = valid.quantile(1.0 - q)
        selected = (row >= threshold).astype(float)
        # NaN positions stay 0 (excluded).
        selected = selected.fillna(0.0)
        return selected

    selected = signal.apply(_select_row, axis=1)
    avg_selected = selected.sum(axis=1).mean()
    logger.info(
        "Cross-sectional selection: top %.0f%% → avg %.1f names per day",
        q * 100,
        avg_selected,
    )
    return selected


def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Full signal pipeline: compute factor → adjust → rank → select.

    Returns a DataFrame with 1.0 for longs, 0.0 for excluded.
    """
    cfg = get_settings()

    if cfg.signals.vol_adjust:
        signal = compute_risk_adjusted_momentum(prices)
    else:
        signal = compute_momentum(prices)

    selected = rank_and_select(signal)
    return selected
