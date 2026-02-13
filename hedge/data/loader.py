"""
Market data loader: download, cache, and serve OHLCV bars.

The default provider is yfinance (free, no key required, rate-limited).
Data is persisted as Parquet files so repeat runs read from disk.

Design decisions
----------------
* One Parquet file per ticker (columnar, fast reads, easy incremental update).
* Adjusted close is the default price series. Stock splits and dividends are
  handled by yfinance's auto_adjust=True.
* Never return future data to callers — the API always requires an explicit
  as_of date so that backtests cannot accidentally peek ahead.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataLoader:
    """Download, cache, and serve daily OHLCV data."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._parquet_dir = Path(cfg.data.parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._history_years: int = cfg.data.history_years
        self._api_delay: float = cfg.data.api_delay_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_ticker(
        self,
        ticker: str,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV for *ticker* and persist to Parquet.

        Returns the full DataFrame (all dates on disk after the update).
        """
        path = self._ticker_path(ticker)
        existing = self._read_parquet(path)

        if start is None:
            start = date.today() - timedelta(days=self._history_years * 365)
        if end is None:
            end = date.today()

        # Only download what we don't already have.
        if existing is not None and not existing.empty:
            last_date = existing.index.max().date()
            if last_date >= (date.today() - timedelta(days=1)):
                logger.debug("%s: already up to date", ticker)
                return existing
            start = last_date + timedelta(days=1)

        logger.info("Downloading %s  %s → %s", ticker, start, end)
        try:
            df = yf.download(
                ticker,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            logger.exception("Failed to download %s", ticker)
            return existing if existing is not None else pd.DataFrame()

        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return existing if existing is not None else df

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]]

        if existing is not None and not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        df.to_parquet(path)
        time.sleep(self._api_delay)
        return df

    def download_universe(self, tickers: list[str]) -> None:
        """Download / refresh data for every ticker in *tickers*."""
        logger.info("Downloading universe (%d tickers) ...", len(tickers))
        for i, t in enumerate(tickers):
            if (i + 1) % 50 == 0:
                logger.info("  progress: %d / %d", i + 1, len(tickers))
            self.download_ticker(t)

    def load_close_prices(
        self,
        tickers: list[str],
        as_of: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of adjusted close prices (columns = tickers).

        If *as_of* is given, rows after that date are dropped — this is the
        primary defence against look-ahead bias during backtests.
        """
        frames: dict[str, pd.Series] = {}
        for t in tickers:
            path = self._ticker_path(t)
            df = self._read_parquet(path)
            if df is not None and not df.empty:
                frames[t] = df["Close"]

        if not frames:
            return pd.DataFrame()

        prices = pd.DataFrame(frames)
        prices.sort_index(inplace=True)

        if as_of is not None:
            prices = prices.loc[:str(as_of)]

        return prices

    def load_volumes(
        self,
        tickers: list[str],
        as_of: str | date | None = None,
    ) -> pd.DataFrame:
        """Same as load_close_prices but for volume."""
        frames: dict[str, pd.Series] = {}
        for t in tickers:
            path = self._ticker_path(t)
            df = self._read_parquet(path)
            if df is not None and not df.empty:
                frames[t] = df["Volume"]

        if not frames:
            return pd.DataFrame()

        vols = pd.DataFrame(frames)
        vols.sort_index(inplace=True)

        if as_of is not None:
            vols = vols.loc[:str(as_of)]

        return vols

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ticker_path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace(".", "_")
        return self._parquet_dir / f"{safe}.parquet"

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_parquet(path)
        return None
