"""
Order manager: convert target weights into orders, validate, submit.

This is the bridge between the portfolio optimiser output (a weight
vector) and the broker API (individual buy/sell orders).

The flow:
  1. Fetch current positions & account value from broker.
  2. Compute target dollar allocation per ticker.
  3. Diff against current holdings → determine buys and sells.
  4. Apply safety guardrails (max order size, paper-only check).
  5. Submit sells first (free up cash), then buys.
  6. Return execution report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from hedge.execution.broker import BaseBroker, Order, get_broker
from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionReport:
    """Summary of an order batch."""

    orders_submitted: list[Order] = field(default_factory=list)
    orders_filled: int = 0
    orders_rejected: int = 0
    total_buy_usd: float = 0.0
    total_sell_usd: float = 0.0


def generate_orders(
    target_weights: pd.Series,
    current_prices: pd.Series,
    broker: BaseBroker | None = None,
) -> list[Order]:
    """
    Generate a list of Order objects to move from current positions
    to target_weights.

    Parameters
    ----------
    target_weights : Series
        Target portfolio weight per ticker (sums to ≤ 1.0).
    current_prices : Series
        Latest price per ticker (used to compute share counts).
    broker : BaseBroker
        Broker instance to read current positions from.

    Returns
    -------
    List of Order objects (not yet submitted).
    """
    cfg = get_settings()

    if broker is None:
        broker = get_broker()

    # Safety check.
    if cfg.execution.paper_only and cfg.project.environment != "paper":
        raise RuntimeError(
            "SAFETY: paper_only=True but environment is not 'paper'. "
            "Refusing to generate orders. Change config deliberately."
        )

    account_value = broker.get_account_value()
    current_positions = broker.get_positions()  # symbol → qty
    max_order_usd = cfg.execution.max_order_usd
    order_type = cfg.execution.order_type

    orders: list[Order] = []

    # All tickers we care about (union of target + current).
    all_tickers = set(target_weights.index) | set(current_positions.keys())

    for symbol in sorted(all_tickers):
        target_w = target_weights.get(symbol, 0.0)
        target_usd = account_value * target_w

        price = current_prices.get(symbol)
        if price is None or price <= 0:
            logger.warning("No price for %s, skipping", symbol)
            continue

        target_qty = target_usd / price
        current_qty = current_positions.get(symbol, 0.0)
        delta_qty = target_qty - current_qty

        if abs(delta_qty) < 0.01:  # not worth trading
            continue

        # Cap order size.
        order_usd = abs(delta_qty * price)
        if order_usd > max_order_usd:
            logger.warning(
                "%s: capping order from $%.0f to $%.0f",
                symbol,
                order_usd,
                max_order_usd,
            )
            delta_qty = (max_order_usd / price) * (1 if delta_qty > 0 else -1)

        limit_price = None
        if order_type == "limit":
            offset = cfg.execution.limit_offset_bps / 10_000
            if delta_qty > 0:
                limit_price = round(price * (1 + offset), 2)
            else:
                limit_price = round(price * (1 - offset), 2)

        order = Order(
            symbol=symbol,
            side="buy" if delta_qty > 0 else "sell",
            qty=round(abs(delta_qty), 4),
            order_type=order_type,
            limit_price=limit_price,
        )
        orders.append(order)

    logger.info(
        "Generated %d orders (account=$%.0f)",
        len(orders),
        account_value,
    )
    return orders


def execute_orders(
    orders: list[Order],
    broker: BaseBroker | None = None,
) -> ExecutionReport:
    """
    Submit orders to broker. Sells first (to free cash), then buys.

    Parameters
    ----------
    orders : list[Order]
        Orders from generate_orders().
    broker : BaseBroker
        Broker instance.

    Returns
    -------
    ExecutionReport summarising what happened.
    """
    cfg = get_settings()

    if broker is None:
        broker = get_broker()

    report = ExecutionReport()

    # Sort: sells first.
    sells = [o for o in orders if o.side == "sell"]
    buys = [o for o in orders if o.side == "buy"]
    ordered = sells + buys

    for order in ordered:
        for attempt in range(cfg.execution.max_retries):
            try:
                result = broker.submit_order(order)
                report.orders_submitted.append(result)
                if result.status == "filled" or result.status == "accepted":
                    report.orders_filled += 1
                    if result.side == "buy":
                        report.total_buy_usd += result.qty * (
                            result.limit_price or 0
                        )
                    else:
                        report.total_sell_usd += result.qty * (
                            result.limit_price or 0
                        )
                else:
                    report.orders_rejected += 1
                break
            except Exception:
                logger.exception(
                    "Order failed for %s (attempt %d/%d)",
                    order.symbol,
                    attempt + 1,
                    cfg.execution.max_retries,
                )
                if attempt == cfg.execution.max_retries - 1:
                    order.status = "error"
                    report.orders_submitted.append(order)
                    report.orders_rejected += 1

    logger.info(
        "Execution done: %d filled, %d rejected out of %d",
        report.orders_filled,
        report.orders_rejected,
        len(orders),
    )
    return report
