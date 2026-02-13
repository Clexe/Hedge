"""
Risk management module.

Provides pre-trade and live risk checks:
  - Max drawdown circuit breaker
  - Per-position stop loss
  - Correlation guard (don't add highly correlated names)

These checks run BEFORE orders are submitted and can veto the entire
rebalance or individual positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


def check_drawdown(equity_curve: pd.Series) -> bool:
    """
    Return True if the portfolio has breached the max drawdown limit.

    If breached, the pipeline should flatten all positions.
    """
    cfg = get_settings()
    max_dd = cfg.risk.max_drawdown_pct

    if equity_curve.empty or len(equity_curve) < 2:
        return False

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    current_dd = dd.iloc[-1]

    if current_dd < -max_dd:
        logger.critical(
            "DRAWDOWN BREACHED: current=%.2f%% limit=%.2f%%",
            current_dd * 100,
            -max_dd * 100,
        )
        return True
    return False


def apply_stop_losses(
    weights: pd.Series,
    prices: pd.DataFrame,
    entry_prices: pd.Series,
) -> pd.Series:
    """
    Zero out positions that have fallen below the stop-loss threshold
    from their entry price.

    Parameters
    ----------
    weights : Series
        Current target weights.
    prices : DataFrame
        Full price history.
    entry_prices : Series
        Price at which each position was entered.

    Returns
    -------
    Modified weights with stopped-out positions set to 0.
    """
    cfg = get_settings()
    stop_pct = cfg.risk.stop_loss_pct

    if prices.empty or entry_prices.empty:
        return weights

    current = prices.iloc[-1]
    for sym in weights.index:
        if sym not in entry_prices or weights[sym] <= 0:
            continue
        entry = entry_prices[sym]
        if entry <= 0:
            continue
        drawdown = (current.get(sym, entry) - entry) / entry
        if drawdown < -stop_pct:
            logger.warning(
                "STOP LOSS: %s dropped %.2f%% from entry $%.2f",
                sym,
                drawdown * 100,
                entry,
            )
            weights[sym] = 0.0

    # Re-normalise if anything was stopped out.
    total = weights.sum()
    if total > 0:
        weights = weights / total * min(total, 1.0)

    return weights


def correlation_guard(
    candidate_tickers: list[str],
    prices: pd.DataFrame,
    lookback: int = 63,
) -> list[str]:
    """
    Remove tickers whose average pairwise correlation with the existing
    selection exceeds the configured threshold.

    This prevents the portfolio from loading up on a single sector or
    theme that happens to have high momentum simultaneously.
    """
    cfg = get_settings()
    max_corr = cfg.risk.max_avg_correlation

    if len(candidate_tickers) < 2:
        return candidate_tickers

    subset = prices[candidate_tickers].iloc[-lookback:]
    rets = np.log(subset / subset.shift(1)).dropna()
    corr_matrix = rets.corr()

    survivors = []
    for t in candidate_tickers:
        if t not in corr_matrix.columns:
            continue
        avg_corr = corr_matrix[t].drop(t).mean()
        if avg_corr <= max_corr:
            survivors.append(t)
        else:
            logger.info(
                "Correlation guard: dropping %s (avg_corr=%.2f > %.2f)",
                t,
                avg_corr,
                max_corr,
            )

    return survivors
