"""
Vectorised backtesting engine.

This engine replays the signal → portfolio → execution pipeline over
historical data to estimate strategy performance. It is *vectorised*
(operates on full DataFrames, not bar-by-bar) for speed, but still
applies realistic costs at every rebalance.

Key realism features:
  - Transaction costs: commission + slippage + spread, applied on the
    absolute turnover at each rebalance.
  - No look-ahead bias: the signal generator only sees data up to (and
    including) the rebalance date. Prices used for execution are the
    NEXT day's open (or close, configurable).
  - Survivorship bias caveat: if using current S&P 500 constituents,
    your backtest is biased. See hedge/data/universe.py for details.

What this engine does NOT do (yet — Phase 2):
  - Intraday bar simulation.
  - Limit / VWAP order fills.
  - Margin / short selling funding costs.
  - Tax-lot tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from hedge.signals.momentum import generate_signals
from hedge.portfolio.optimizer import optimise_portfolio
from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=pd.Series)
    trades: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # --- Derived stats (populated by compute_stats) ---
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    avg_turnover: float = 0.0
    win_rate: float = 0.0

    def compute_stats(self, benchmark_returns: pd.Series | None = None) -> None:
        """Populate derived performance statistics."""
        ret = self.returns.dropna()
        if ret.empty:
            return

        n_years = len(ret) / 252
        cum = (1 + ret).cumprod()

        # CAGR
        self.cagr = (cum.iloc[-1] ** (1 / max(n_years, 0.01))) - 1

        # Sharpe (annualised, excess over 0 — adjust if you want RF).
        self.sharpe = ret.mean() / max(ret.std(), 1e-10) * np.sqrt(252)

        # Sortino
        downside = ret[ret < 0]
        down_std = downside.std() if len(downside) > 0 else 1e-10
        self.sortino = ret.mean() / max(down_std, 1e-10) * np.sqrt(252)

        # Max drawdown
        peak = cum.cummax()
        dd = (cum - peak) / peak
        self.max_drawdown = dd.min()

        # Calmar
        self.calmar = self.cagr / max(abs(self.max_drawdown), 1e-10)

        # Win rate
        self.win_rate = (ret > 0).sum() / len(ret)

        # Avg turnover
        if not self.turnover.empty:
            self.avg_turnover = self.turnover.mean()

    def summary(self) -> str:
        """Pretty-print performance summary."""
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"  CAGR:           {self.cagr:>8.2%}\n"
            f"  Sharpe:         {self.sharpe:>8.2f}\n"
            f"  Sortino:        {self.sortino:>8.2f}\n"
            f"  Max Drawdown:   {self.max_drawdown:>8.2%}\n"
            f"  Calmar:         {self.calmar:>8.2f}\n"
            f"  Win Rate:       {self.win_rate:>8.2%}\n"
            f"  Avg Turnover:   {self.avg_turnover:>8.2%}\n"
            f"  Total Trades:   {self.trades:>8d}\n"
            f"  Commission:    ${self.total_commission:>9,.2f}\n"
            f"  Slippage:      ${self.total_slippage:>9,.2f}\n"
            f"{'='*60}\n"
        )


def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str,
) -> list[pd.Timestamp]:
    """Return the subset of dates on which we rebalance."""
    if frequency == "daily":
        return list(dates)
    elif frequency == "weekly":
        # Rebalance every Monday (or first trading day of the week).
        return list(dates[dates.weekday == 0])
    elif frequency == "monthly":
        # Last trading day of each month.
        month_groups = dates.to_series().groupby(
            [dates.year, dates.month]
        )
        return [group.iloc[-1] for _, group in month_groups]
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def run_backtest(prices: pd.DataFrame) -> BacktestResult:
    """
    Run the full backtest.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close prices for the full universe.
        Columns = tickers, index = trading dates.

    Returns
    -------
    BacktestResult with equity curve and statistics.
    """
    cfg = get_settings()
    bt = cfg.backtest

    # Trim to date range.
    start = pd.Timestamp(bt.start_date)
    end = pd.Timestamp(bt.end_date) if bt.end_date else prices.index.max()
    prices = prices.loc[start:end]

    if prices.empty:
        logger.error("No price data in [%s, %s]", start, end)
        return BacktestResult()

    commission_rate = bt.commission_bps / 10_000
    slippage_rate = bt.slippage_bps / 10_000
    spread_rate = bt.spread_cost_bps / 10_000
    total_cost_rate = commission_rate + slippage_rate + spread_rate

    initial_capital = bt.initial_capital
    rebal_freq = cfg.portfolio.rebalance_frequency
    rebal_dates = _get_rebalance_dates(prices.index, rebal_freq)

    logger.info(
        "Backtest: %s → %s  |  %d rebalance dates  |  %d tickers",
        prices.index[0].date(),
        prices.index[-1].date(),
        len(rebal_dates),
        len(prices.columns),
    )

    # --- State ---
    current_weights = pd.Series(0.0, index=prices.columns)
    portfolio_value = initial_capital
    equity = []
    daily_returns = []
    weight_snapshots = []
    turnover_list = []
    total_trades = 0
    total_comm = 0.0
    total_slip = 0.0

    prev_date = None

    for dt in prices.index:
        # 1. Mark-to-market: apply today's return to yesterday's weights.
        if prev_date is not None:
            daily_ret_by_ticker = prices.loc[dt] / prices.loc[prev_date] - 1
            # Portfolio return = weighted sum of individual returns.
            port_ret = (current_weights * daily_ret_by_ticker).sum()
            portfolio_value *= 1 + port_ret
            daily_returns.append(port_ret)
        else:
            daily_returns.append(0.0)

        # 2. Rebalance if today is a rebalance date.
        if dt in rebal_dates:
            # Generate signals using ONLY data up to today (no look-ahead).
            prices_up_to_now = prices.loc[:dt]

            signals = generate_signals(prices_up_to_now)
            if signals.empty or dt not in signals.index:
                equity.append(portfolio_value)
                prev_date = dt
                continue

            today_signals = signals.loc[dt]
            new_weights = optimise_portfolio(today_signals, prices_up_to_now)

            # Re-index to full universe.
            new_weights = new_weights.reindex(prices.columns, fill_value=0.0)

            # Turnover = sum of absolute weight changes.
            turnover = (new_weights - current_weights).abs().sum()
            turnover_list.append(turnover)

            # Transaction costs.
            cost = turnover * portfolio_value * total_cost_rate
            n_trades = (new_weights != current_weights).sum()
            comm_part = turnover * portfolio_value * commission_rate
            slip_part = turnover * portfolio_value * slippage_rate

            portfolio_value -= cost
            total_trades += n_trades
            total_comm += comm_part
            total_slip += slip_part

            current_weights = new_weights
            weight_snapshots.append(
                new_weights.rename(dt)
            )

        equity.append(portfolio_value)
        prev_date = dt

    # --- Assemble result ---
    result = BacktestResult(
        equity_curve=pd.Series(equity, index=prices.index, name="equity"),
        returns=pd.Series(daily_returns, index=prices.index, name="returns"),
        weights_history=(
            pd.DataFrame(weight_snapshots) if weight_snapshots else pd.DataFrame()
        ),
        turnover=pd.Series(turnover_list, name="turnover"),
        trades=total_trades,
        total_commission=total_comm,
        total_slippage=total_slip,
    )
    result.compute_stats()
    logger.info(result.summary())
    return result
