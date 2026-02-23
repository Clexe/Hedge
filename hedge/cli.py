"""
Command-line interface for Hedge.

Usage:
    python -m hedge demo              # Demo mode — practice with synthetic data
    python -m hedge demo --full       # Demo with step-by-step walkthrough
    python -m hedge download          # Download / refresh all market data
    python -m hedge signals           # Generate today's momentum signals
    python -m hedge backtest          # Run full historical backtest
    python -m hedge run               # Run the full nightly pipeline (paper)
    python -m hedge run --dry-run     # Generate orders but don't submit
    python -m hedge schedule          # Start the APScheduler cron loop
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_download(args: argparse.Namespace) -> None:
    """Download / refresh market data for the full universe."""
    from hedge.data.universe import get_universe
    from hedge.data.loader import MarketDataLoader

    tickers = get_universe()
    loader = MarketDataLoader()
    loader.download_universe(tickers)
    print(f"Downloaded data for {len(tickers)} tickers.")


def cmd_signals(args: argparse.Namespace) -> None:
    """Generate and display today's signals."""
    from hedge.data.universe import get_universe
    from hedge.data.loader import MarketDataLoader
    from hedge.signals.momentum import generate_signals

    tickers = get_universe()
    loader = MarketDataLoader()
    prices = loader.load_close_prices(tickers)

    if prices.empty:
        print("No price data available. Run 'download' first.")
        sys.exit(1)

    signals = generate_signals(prices)
    today = signals.iloc[-1]
    selected = today[today > 0].sort_values(ascending=False)

    print(f"\nMomentum signals for {prices.index[-1].date()}")
    print(f"Selected {len(selected)} tickers (top quantile):\n")
    for sym in selected.index:
        print(f"  {sym}")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run the vectorised backtest."""
    from hedge.data.universe import get_universe
    from hedge.data.loader import MarketDataLoader
    from hedge.backtest.engine import run_backtest

    tickers = get_universe()
    loader = MarketDataLoader()
    prices = loader.load_close_prices(tickers)

    if prices.empty:
        print("No price data available. Run 'download' first.")
        sys.exit(1)

    result = run_backtest(prices)
    print(result.summary())

    if args.output:
        result.equity_curve.to_csv(args.output)
        print(f"Equity curve saved to {args.output}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full nightly pipeline."""
    from hedge.pipeline.orchestrator import run_pipeline

    result = run_pipeline(dry_run=args.dry_run)

    if result["error"]:
        print(f"\nPipeline failed: {result['error']}")
        sys.exit(1)

    weights = result.get("weights")
    if weights is not None and not weights.empty:
        print(f"\nTarget portfolio ({len(weights)} positions):")
        for sym, w in weights.nlargest(10).items():
            print(f"  {sym}: {w:.2%}")
    else:
        print("\nNo positions generated.")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run demo mode with synthetic data."""
    from hedge.demo import run_demo
    run_demo(full=args.full, output=args.output)


def cmd_schedule(args: argparse.Namespace) -> None:
    """Start the APScheduler cron loop."""
    from hedge.pipeline.orchestrator import schedule_pipeline
    schedule_pipeline()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hedge",
        description="Hedge — Automated Momentum Trading System",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # demo
    demo_parser = sub.add_parser(
        "demo",
        help="Demo mode — practice with synthetic data (no API keys needed)",
    )
    demo_parser.add_argument(
        "--full", action="store_true",
        help="Step-by-step walkthrough with explanations",
    )
    demo_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save demo equity curve CSV",
    )

    # download
    sub.add_parser("download", help="Download / refresh market data")

    # signals
    sub.add_parser("signals", help="Generate today's momentum signals")

    # backtest
    bt_parser = sub.add_parser("backtest", help="Run historical backtest")
    bt_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save equity curve CSV",
    )

    # run
    run_parser = sub.add_parser("run", help="Run the full nightly pipeline")
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate orders but don't submit",
    )

    # schedule
    sub.add_parser("schedule", help="Start the APScheduler cron loop")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "demo": cmd_demo,
        "download": cmd_download,
        "signals": cmd_signals,
        "backtest": cmd_backtest,
        "run": cmd_run,
        "schedule": cmd_schedule,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
