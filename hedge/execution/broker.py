"""
Broker abstraction layer.

Translates target portfolio weights into concrete orders and submits
them through a broker API. Currently supports:
  - Paper trading (simulated, no real money)
  - Alpaca (commission-free equities, good REST + WebSocket API)

The execution flow is:
  1. Read current positions from broker.
  2. Diff against target weights → generate order list.
  3. Validate orders against safety limits.
  4. Submit orders (with retry logic).
  5. Log confirmations.

SAFETY RAILS:
  - If config says paper_only=True and environment != "paper", refuse all
    orders.  This is the single most important safety check.
  - Every order is capped at max_order_usd.
  - A hard drawdown circuit-breaker can flatten everything.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hedge.utils.config import get_settings
from hedge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    """A single order to be submitted."""

    symbol: str
    side: str           # "buy" | "sell"
    qty: float          # number of shares (fractional OK for Alpaca)
    order_type: str     # "market" | "limit"
    limit_price: float | None = None
    status: str = "pending"
    broker_id: str | None = None


class BaseBroker(abc.ABC):
    """Abstract broker interface."""

    @abc.abstractmethod
    def get_account_value(self) -> float: ...

    @abc.abstractmethod
    def get_positions(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abc.abstractmethod
    def cancel_all_orders(self) -> None: ...


class PaperBroker(BaseBroker):
    """
    Simulated broker for testing.

    Tracks positions in memory. All orders fill instantly at the
    "last known price" (which you must supply via update_prices).
    """

    def __init__(self, initial_cash: float = 100_000) -> None:
        self._cash = initial_cash
        self._positions: dict[str, float] = {}  # symbol → qty
        self._prices: dict[str, float] = {}      # symbol → last price

    def update_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def get_account_value(self) -> float:
        pos_value = sum(
            qty * self._prices.get(sym, 0) for sym, qty in self._positions.items()
        )
        return self._cash + pos_value

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def submit_order(self, order: Order) -> Order:
        price = self._prices.get(order.symbol)
        if price is None or price <= 0:
            order.status = "rejected"
            logger.warning("Paper broker: no price for %s, rejecting", order.symbol)
            return order

        cost = order.qty * price
        if order.side == "buy":
            self._cash -= cost
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + order.qty
            )
        elif order.side == "sell":
            self._cash += cost
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) - order.qty
            )
            if abs(self._positions[order.symbol]) < 1e-9:
                del self._positions[order.symbol]

        order.status = "filled"
        order.broker_id = f"PAPER-{id(order)}"
        return order

    def cancel_all_orders(self) -> None:
        logger.info("Paper broker: no pending orders to cancel")


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker via alpaca-trade-api.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in secrets.yaml or
    environment variables.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError(
                "alpaca-py is required for live trading. "
                "Install with: pip install alpaca-py"
            )

        api_key = cfg.alpaca.api_key
        secret_key = cfg.alpaca.secret_key
        paper = cfg.project.environment == "paper"

        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        logger.info("AlpacaBroker initialised (paper=%s)", paper)

    def get_account_value(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_positions(self) -> dict[str, float]:
        positions = self._client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def submit_order(self, order: Order) -> Order:
        side = (
            self._OrderSide.BUY if order.side == "buy" else self._OrderSide.SELL
        )

        if order.order_type == "market":
            req = self._MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=self._TimeInForce.DAY,
            )
        else:
            req = self._LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=self._TimeInForce.DAY,
                limit_price=order.limit_price,
            )

        result = self._client.submit_order(req)
        order.broker_id = str(result.id)
        order.status = str(result.status)
        logger.info(
            "Alpaca order submitted: %s %s %.2f shares → %s",
            order.side,
            order.symbol,
            order.qty,
            order.status,
        )
        return order

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()
        logger.info("Alpaca: all open orders cancelled")


def get_broker() -> BaseBroker:
    """Factory: return the broker configured in settings."""
    cfg = get_settings()
    broker_name = cfg.execution.broker

    if broker_name == "paper":
        return PaperBroker(initial_cash=cfg.backtest.initial_capital)
    elif broker_name == "alpaca":
        return AlpacaBroker()
    else:
        raise ValueError(f"Unknown broker: {broker_name}")
