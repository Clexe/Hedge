"""
Demo mode for Hedge.

Generates synthetic but realistic-looking price data for a small universe
of well-known tickers, then runs the full pipeline (signals, portfolio,
backtest) so you can see how everything works — no API keys, no internet,
no downloads required.

Usage:
    python -m hedge demo              # Quick demo with summary
    python -m hedge demo --full       # Full demo with step-by-step walkthrough
    python -m hedge demo --output x   # Save equity curve to CSV
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from hedge.utils.logging import get_logger

logger = get_logger(__name__)

# A small, recognisable universe for the demo.
DEMO_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "MA",
    "DIS", "NFLX", "ADBE", "CRM", "PYPL",
    "INTC", "CSCO", "PEP", "KO", "MRK",
    "ABT", "TMO", "COST", "AVGO", "NKE",
]

# Per-ticker parameters to make the synthetic data feel distinct.
# (annual_drift, annual_vol) — loosely inspired by real characteristics.
_TICKER_PARAMS: dict[str, tuple[float, float]] = {
    "AAPL": (0.18, 0.28), "MSFT": (0.20, 0.25), "GOOGL": (0.14, 0.27),
    "AMZN": (0.16, 0.32), "NVDA": (0.30, 0.45), "META": (0.15, 0.38),
    "TSLA": (0.22, 0.55), "JPM": (0.10, 0.24), "V": (0.14, 0.22),
    "JNJ": (0.06, 0.16), "WMT": (0.08, 0.18), "PG": (0.07, 0.15),
    "UNH": (0.16, 0.23), "HD": (0.13, 0.24), "MA": (0.15, 0.23),
    "DIS": (0.04, 0.30), "NFLX": (0.20, 0.42), "ADBE": (0.17, 0.30),
    "CRM": (0.14, 0.32), "PYPL": (0.05, 0.38), "INTC": (0.02, 0.30),
    "CSCO": (0.08, 0.22), "PEP": (0.08, 0.16), "KO": (0.07, 0.15),
    "MRK": (0.10, 0.20), "ABT": (0.11, 0.20), "TMO": (0.14, 0.22),
    "COST": (0.15, 0.20), "AVGO": (0.22, 0.32), "NKE": (0.09, 0.26),
}

# Rough starting prices (order of magnitude correct).
_START_PRICES: dict[str, float] = {
    "AAPL": 130, "MSFT": 250, "GOOGL": 90, "AMZN": 100, "NVDA": 200,
    "META": 180, "TSLA": 200, "JPM": 140, "V": 220, "JNJ": 160,
    "WMT": 140, "PG": 145, "UNH": 480, "HD": 300, "MA": 350,
    "DIS": 100, "NFLX": 350, "ADBE": 400, "CRM": 170, "PYPL": 80,
    "INTC": 30, "CSCO": 48, "PEP": 170, "KO": 58, "MRK": 105,
    "ABT": 110, "TMO": 550, "COST": 500, "AVGO": 600, "NKE": 110,
}


def generate_synthetic_prices(
    tickers: list[str] | None = None,
    n_years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily close prices using geometric Brownian motion.

    Each ticker gets its own drift and volatility so the momentum strategy
    has meaningful cross-sectional dispersion to work with.

    Parameters
    ----------
    tickers : list[str]
        Tickers to generate. Default: DEMO_TICKERS.
    n_years : int
        Number of years of history.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns = tickers, index = business-day dates.
    """
    tickers = tickers or DEMO_TICKERS
    rng = np.random.default_rng(seed)

    n_days = n_years * 252
    end = date.today() - timedelta(days=1)
    dates = pd.bdate_range(end=end, periods=n_days, freq="B")

    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        mu, sigma = _TICKER_PARAMS.get(ticker, (0.10, 0.25))
        p0 = _START_PRICES.get(ticker, 100.0)

        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)
        shocks = rng.normal(daily_mu, daily_sigma, n_days)

        # Add mild mean-reversion to keep prices from exploding.
        log_prices = np.zeros(n_days)
        log_prices[0] = np.log(p0)
        for i in range(1, n_days):
            log_prices[i] = log_prices[i - 1] + shocks[i]

        prices[ticker] = pd.Series(np.exp(log_prices), index=dates, name=ticker)

    df = pd.DataFrame(prices)
    return df


def run_demo(full: bool = False, output: str | None = None) -> None:
    """
    Run the demo: generate synthetic data, compute signals, run backtest.

    Parameters
    ----------
    full : bool
        If True, print step-by-step walkthrough with explanations.
    output : str | None
        If given, save the equity curve CSV to this path.
    """
    sep = "=" * 60

    print(f"\n{sep}")
    print("  HEDGE — DEMO MODE")
    print(f"  Practice with synthetic data (no API keys needed)")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic prices
    # ------------------------------------------------------------------
    print("[1/5] Generating synthetic market data ...")
    prices = generate_synthetic_prices()
    print(f"       {len(DEMO_TICKERS)} tickers, "
          f"{len(prices)} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    if full:
        print("\n  How it works: We simulate daily stock prices using geometric")
        print("  Brownian motion — each ticker has its own drift (expected return)")
        print("  and volatility, modelled after real-world characteristics.")
        print("  This gives the momentum strategy meaningful dispersion to work with.\n")

    # ------------------------------------------------------------------
    # Step 2: Generate momentum signals
    # ------------------------------------------------------------------
    print("[2/5] Computing momentum signals ...")
    from hedge.signals.momentum import generate_signals
    signals = generate_signals(prices)

    if signals.empty:
        print("       ERROR: No signals generated. Something went wrong.")
        sys.exit(1)

    today_signals = signals.iloc[-1]
    selected = today_signals[today_signals > 0].index.tolist()
    print(f"       Selected {len(selected)} tickers (top momentum decile)")

    if full:
        print("\n  Selected tickers (strongest momentum):")
        for i, sym in enumerate(selected[:10], 1):
            print(f"    {i:2d}. {sym}")
        if len(selected) > 10:
            print(f"    ... and {len(selected) - 10} more")
        print()

    # ------------------------------------------------------------------
    # Step 3: Optimise portfolio
    # ------------------------------------------------------------------
    print("[3/5] Constructing portfolio (inverse-volatility weighting) ...")
    from hedge.portfolio.optimizer import optimise_portfolio
    weights = optimise_portfolio(today_signals, prices)

    if weights.empty:
        print("       WARNING: Empty portfolio. Continuing to backtest anyway.")
    else:
        print(f"       {len(weights)} positions, "
              f"gross exposure: {weights.sum():.1%}")

        if full:
            print("\n  Top 10 positions by weight:")
            for sym, w in weights.nlargest(10).items():
                bar = "#" * int(w * 200)
                print(f"    {sym:6s} {w:6.2%}  {bar}")
            print()

    # ------------------------------------------------------------------
    # Step 4: Run backtest
    # ------------------------------------------------------------------
    print("[4/5] Running backtest ...")
    from hedge.backtest.engine import run_backtest
    result = run_backtest(prices)

    print(result.summary())

    if output:
        result.equity_curve.to_csv(output)
        print(f"  Equity curve saved to: {output}\n")

    # ------------------------------------------------------------------
    # Step 5: Demo order generation (dry run)
    # ------------------------------------------------------------------
    print("[5/5] Generating demo orders (paper broker, dry run) ...")
    from hedge.execution.broker import PaperBroker
    from hedge.execution.order_manager import generate_orders

    broker = PaperBroker(initial_cash=100_000)
    current_prices_dict = {t: float(prices[t].iloc[-1]) for t in prices.columns}
    broker.update_prices(current_prices_dict)

    if not weights.empty:
        current_prices_series = prices.iloc[-1]
        orders = generate_orders(weights, current_prices_series, broker)
        buys = [o for o in orders if o.side == "buy"]
        sells = [o for o in orders if o.side == "sell"]
        print(f"       {len(orders)} orders generated: "
              f"{len(buys)} buys, {len(sells)} sells")

        if full and orders:
            print("\n  Sample orders:")
            for o in orders[:8]:
                usd = o.qty * current_prices_dict.get(o.symbol, 0)
                print(f"    {o.side.upper():4s}  {o.symbol:6s}  "
                      f"{o.qty:8.2f} shares  (~${usd:,.0f})")
            if len(orders) > 8:
                print(f"    ... and {len(orders) - 8} more")
    else:
        print("       No orders (empty portfolio)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  DEMO COMPLETE")
    print(f"{sep}")
    print()
    print("  What you just saw:")
    print("    1. Synthetic price data generated (no download needed)")
    print("    2. 12-1 momentum signals computed")
    print("    3. Inverse-volatility portfolio constructed")
    print("    4. Full historical backtest with realistic costs")
    print("    5. Demo order generation (paper broker)")
    print()
    print("  Next steps to use real data:")
    print("    python -m hedge download    # Download real S&P 500 data")
    print("    python -m hedge signals     # See today's real signals")
    print("    python -m hedge backtest    # Backtest on real data")
    print("    python -m hedge run --dry-run  # Full pipeline, no orders")
    print()
