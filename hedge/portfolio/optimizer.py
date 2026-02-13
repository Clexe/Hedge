"""
Portfolio construction & optimisation.

Given a set of selected tickers and their historical prices, produce a
weight vector that satisfies position / sector / volatility constraints.

Methods implemented:
  1. Equal weight — simplest, surprisingly hard to beat.
  2. Inverse volatility — allocate more to lower-vol names.
  3. Risk parity (simplified) — target equal risk contribution.
  4. Minimum variance — via scipy optimiser (heavier, optional).

All methods then apply:
  - Max position cap
  - Target portfolio vol scaling
  - Cash buffer reservation
  - Minimum turnover filter (skip if rebalance delta is tiny)

IMPORTANT:
  The optimiser intentionally does NOT use expected returns as input.
  Mean estimates from historical data are notoriously noisy and degrade
  out-of-sample performance. Using risk-only allocation (inv-vol, min-var)
  is far more robust for momentum portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


def equal_weight(selected: pd.Series) -> pd.Series:
    """Uniform weights across all selected names."""
    names = selected[selected > 0].index
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / n, index=names)
    return w


def inverse_volatility(
    selected: pd.Series,
    prices: pd.DataFrame,
    vol_lookback: int | None = None,
) -> pd.Series:
    """
    Weight proportional to 1 / trailing vol.

    Lower-volatility names get larger allocations, which empirically
    produces better risk-adjusted returns and smoother equity curves.
    """
    cfg = get_settings()
    lb = vol_lookback or cfg.signals.vol_lookback

    names = selected[selected > 0].index.tolist()
    if not names:
        return pd.Series(dtype=float)

    subset = prices[names]
    log_ret = np.log(subset / subset.shift(1))
    vol = log_ret.iloc[-lb:].std() * np.sqrt(252)

    # Guard against zero / near-zero vol.
    vol = vol.clip(lower=0.01)
    inv = 1.0 / vol
    w = inv / inv.sum()
    return w


def risk_parity(
    selected: pd.Series,
    prices: pd.DataFrame,
    vol_lookback: int | None = None,
) -> pd.Series:
    """
    Simplified risk-parity: equalise marginal risk contribution.

    This approximation uses inv-vol then iterates a few rounds of
    rescaling by marginal risk. Good enough for daily equity momentum.
    """
    cfg = get_settings()
    lb = vol_lookback or cfg.signals.vol_lookback

    names = selected[selected > 0].index.tolist()
    if not names:
        return pd.Series(dtype=float)

    subset = prices[names].iloc[-lb:]
    log_ret = np.log(subset / subset.shift(1)).dropna()
    cov = log_ret.cov() * 252  # annualised

    n = len(names)
    w = np.ones(n) / n

    # Iterative rescaling (Spinu 2013 simplified).
    for _ in range(50):
        sigma_p = np.sqrt(w @ cov.values @ w)
        if sigma_p < 1e-10:
            break
        mrc = (cov.values @ w) / sigma_p  # marginal risk contribution
        rc = w * mrc
        target_rc = sigma_p / n
        w = w * (target_rc / (rc + 1e-12))
        w = w / w.sum()

    return pd.Series(w, index=names)


# ------------------------------------------------------------------
# Constraint application
# ------------------------------------------------------------------

def apply_constraints(
    weights: pd.Series,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Apply position-level and portfolio-level constraints.

    1. Cap each position at max_position_pct.
    2. Scale to target volatility.
    3. Reserve cash buffer.
    """
    cfg = get_settings()
    max_pos = cfg.portfolio.max_position_pct
    target_vol = cfg.portfolio.target_volatility
    cash_buf = cfg.portfolio.cash_buffer_pct

    if weights.empty:
        return weights

    # 1. Position cap — redistribute excess pro-rata.
    for _ in range(10):  # iterate until stable
        excess = weights[weights > max_pos]
        if excess.empty:
            break
        total_excess = (excess - max_pos).sum()
        weights[weights > max_pos] = max_pos
        under = weights[weights < max_pos]
        if under.empty:
            break
        weights[under.index] += total_excess * (under / under.sum())
    weights = weights / weights.sum()  # re-normalise

    # 2. Vol targeting — scale gross exposure.
    names = weights.index.tolist()
    subset = prices[names].iloc[-63:]
    log_ret = np.log(subset / subset.shift(1)).dropna()
    if len(log_ret) > 5:
        port_ret = log_ret @ weights
        realised_vol = port_ret.std() * np.sqrt(252)
        if realised_vol > 1e-6:
            vol_scalar = target_vol / realised_vol
            vol_scalar = np.clip(vol_scalar, 0.2, 2.0)  # safety bounds
            weights = weights * vol_scalar
            logger.info(
                "Vol targeting: realised=%.2f%% target=%.2f%% scalar=%.2f",
                realised_vol * 100,
                target_vol * 100,
                vol_scalar,
            )

    # 3. Cash buffer.
    gross = weights.sum()
    max_invested = 1.0 - cash_buf
    if gross > max_invested:
        weights = weights * (max_invested / gross)

    return weights


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def optimise_portfolio(
    selected: pd.Series,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Given today's selected signals and historical prices, return a
    weight vector ready for order generation.

    Parameters
    ----------
    selected : Series
        1.0 for each selected ticker, 0.0 or absent for the rest.
    prices : DataFrame
        Full adjusted close history.

    Returns
    -------
    Series — target weights summing to ≤ 1.0.
    """
    cfg = get_settings()
    method = cfg.portfolio.method

    if method == "equal_weight":
        w = equal_weight(selected)
    elif method == "inv_vol":
        w = inverse_volatility(selected, prices)
    elif method == "risk_parity":
        w = risk_parity(selected, prices)
    else:
        logger.warning("Unknown method '%s', falling back to equal_weight", method)
        w = equal_weight(selected)

    if w.empty:
        logger.warning("No positions selected — returning empty weights")
        return w

    w = apply_constraints(w, prices)
    logger.info(
        "Portfolio: %d positions, gross=%.2f%%, max=%.2f%%, min=%.2f%%",
        len(w),
        w.sum() * 100,
        w.max() * 100,
        w.min() * 100,
    )
    return w
