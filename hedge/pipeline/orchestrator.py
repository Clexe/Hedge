"""
Nightly pipeline orchestrator.

Runs the complete ETL → Signal → Portfolio → Execute pipeline as a
single function call. Designed to be triggered by:
  - APScheduler (in-process cron)
  - System crontab
  - Prefect / Airflow (for more complex workflows)
  - Manual invocation via CLI

The pipeline is idempotent: running it twice on the same day will
detect that data is already current and skip the download, then
re-compute signals (cheap) and only generate orders if the target
weights differ from current positions.

Error handling philosophy:
  Data download failures for individual tickers are logged and skipped.
  Signal / portfolio errors abort the pipeline (fail-fast).
  Order submission errors are retried per the broker module.
  All exceptions are logged with full tracebacks.
"""

from __future__ import annotations

import traceback
from datetime import date, datetime

import pandas as pd

from hedge.data.loader import MarketDataLoader
from hedge.data.universe import get_universe, filter_by_liquidity
from hedge.signals.momentum import generate_signals
from hedge.portfolio.optimizer import optimise_portfolio
from hedge.execution.order_manager import generate_orders, execute_orders, ExecutionReport
from hedge.execution.broker import get_broker
from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline(
    as_of: date | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute the full nightly pipeline.

    Parameters
    ----------
    as_of : date, optional
        Override "today" for testing. Default is date.today().
    dry_run : bool
        If True, generate orders but don't submit them.

    Returns
    -------
    dict with keys: tickers, signals, weights, orders, report.
    """
    cfg = get_settings()
    as_of = as_of or date.today()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("PIPELINE START  |  run_id=%s  |  as_of=%s", run_id, as_of)
    logger.info("=" * 60)

    result = {
        "run_id": run_id,
        "as_of": as_of,
        "tickers": [],
        "signals": None,
        "weights": None,
        "orders": [],
        "report": None,
        "error": None,
    }

    try:
        # ==============================================================
        # STEP 1: Universe
        # ==============================================================
        logger.info("STEP 1: Resolving universe")
        tickers = get_universe()
        result["tickers"] = tickers

        # ==============================================================
        # STEP 2: Data download / refresh
        # ==============================================================
        logger.info("STEP 2: Downloading / refreshing market data")
        loader = MarketDataLoader()
        loader.download_universe(tickers)

        prices = loader.load_close_prices(tickers, as_of=as_of)
        volumes = loader.load_volumes(tickers, as_of=as_of)

        if prices.empty:
            raise RuntimeError("No price data loaded — aborting")

        # ==============================================================
        # STEP 3: Liquidity filter
        # ==============================================================
        logger.info("STEP 3: Applying liquidity filter")
        liquid = filter_by_liquidity(prices, volumes)
        prices = prices[liquid]

        # ==============================================================
        # STEP 4: Signal generation
        # ==============================================================
        logger.info("STEP 4: Generating momentum signals")
        signals = generate_signals(prices)
        result["signals"] = signals

        if signals.empty:
            logger.warning("No signals generated — nothing to trade")
            return result

        # Use the latest row of signals.
        today_signals = signals.iloc[-1]

        # ==============================================================
        # STEP 5: Portfolio optimisation
        # ==============================================================
        logger.info("STEP 5: Optimising portfolio")
        weights = optimise_portfolio(today_signals, prices)
        result["weights"] = weights

        if weights.empty:
            logger.warning("Empty weight vector — no positions")
            return result

        logger.info("Target portfolio (%d names):", len(weights))
        for sym, w in weights.nlargest(10).items():
            logger.info("  %s: %.2f%%", sym, w * 100)

        # ==============================================================
        # STEP 6: Order generation & execution
        # ==============================================================
        logger.info("STEP 6: Generating orders")
        current_prices = prices.iloc[-1]
        broker = get_broker()
        orders = generate_orders(weights, current_prices, broker)
        result["orders"] = orders

        if dry_run:
            logger.info("DRY RUN: %d orders generated but NOT submitted", len(orders))
            for o in orders:
                logger.info("  %s %s %.2f %s", o.side, o.symbol, o.qty, o.order_type)
            return result

        logger.info("Submitting %d orders ...", len(orders))
        report = execute_orders(orders, broker)
        result["report"] = report

        logger.info(
            "Execution report: %d filled, %d rejected",
            report.orders_filled,
            report.orders_rejected,
        )

    except Exception as e:
        logger.error("PIPELINE FAILED: %s", e)
        logger.error(traceback.format_exc())
        result["error"] = str(e)

    logger.info("PIPELINE END  |  run_id=%s", run_id)
    return result


def schedule_pipeline() -> None:
    """
    Start the APScheduler cron loop.

    This blocks the main thread and runs run_pipeline() on the
    schedule defined in config (default: 5 AM UTC, weekdays only).
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    cfg = get_settings()
    cron_expr = cfg.schedule.cron
    tz = cfg.schedule.timezone

    scheduler = BlockingScheduler()

    # Parse "minute hour day month day_of_week" from standard cron.
    parts = cron_expr.split()
    trigger = CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
        timezone=tz,
    )

    scheduler.add_job(run_pipeline, trigger=trigger, id="nightly_pipeline")
    logger.info("Scheduler started: %s (%s)", cron_expr, tz)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
