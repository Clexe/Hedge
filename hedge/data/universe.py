"""
Universe construction: determine which tickers are eligible for trading.

The default universe is the current S&P 500 constituents (scraped from
Wikipedia). A liquidity filter removes names below a configurable average
dollar-volume threshold.

IMPORTANT survivorship-bias note:
  Fetching *today's* S&P 500 list and applying it historically introduces
  look-ahead / survivorship bias in backtests. For research-grade results you
  need point-in-time constituent snapshots (available from Sharadar, Compustat,
  or archived CSVs).  This module is fine for a Phase-1 MVP but should be
  replaced with point-in-time data before drawing return conclusions.
"""

from __future__ import annotations

import pandas as pd

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    return sorted(tickers)


def get_universe() -> list[str]:
    """Return the trading universe based on config."""
    cfg = get_settings()
    source = cfg.universe.source

    if source == "sp500":
        return get_sp500_tickers()
    elif source == "custom":
        symbols = list(cfg.universe.custom_symbols)
        logger.info("Using custom universe: %d symbols", len(symbols))
        return sorted(symbols)
    else:
        raise ValueError(f"Unknown universe source: {source}")


def filter_by_liquidity(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_dollar_volume: float | None = None,
    lookback_days: int | None = None,
) -> list[str]:
    """
    Remove tickers whose average daily dollar volume over the trailing
    *lookback_days* is below *min_dollar_volume*.

    Parameters
    ----------
    prices : DataFrame
        Close prices, columns = tickers, index = dates.
    volumes : DataFrame
        Share volumes, same shape as *prices*.
    """
    cfg = get_settings()
    min_dv = min_dollar_volume or cfg.universe.min_dollar_volume
    lb = lookback_days or cfg.universe.lookback_days

    dollar_vol = (prices * volumes).iloc[-lb:]
    avg_dv = dollar_vol.mean()
    survivors = avg_dv[avg_dv >= min_dv].index.tolist()
    removed = len(prices.columns) - len(survivors)
    logger.info(
        "Liquidity filter: kept %d / %d tickers (removed %d)",
        len(survivors),
        len(prices.columns),
        removed,
    )
    return sorted(survivors)
