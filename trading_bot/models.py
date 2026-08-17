"""Core value types shared by the feed, strategy, risk and broker layers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class RejectionKind(str, Enum):
    """Why an order was refused, and therefore who should react to it.

    The distinction drives the circuit breaker. A venue refusing our orders
    means something is wrong that waiting will not fix — bad credentials, a
    restricted account, a halted symbol — so it should stop the bot. Our own
    risk layer declining an order is the system working correctly, and must
    never be mistaken for a fault: a long-only account refusing short signals
    would otherwise halt itself after a handful of bars.
    """

    #: The counterparty refused. Counts toward the circuit breaker.
    VENUE = "venue"
    #: Our own guard, instrument rules, leverage or policy declined it.
    RISK = "risk"
    #: A normal market outcome, e.g. a limit that was never reached.
    MARKET = "market"
    #: The broker was already halted, so nothing was attempted.
    HALTED = "halted"

    @property
    def counts_toward_breaker(self) -> bool:
        return self is RejectionKind.VENUE


class OrderStatus(str, Enum):
    """Terminal state of a submitted order."""

    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"

    @property
    def is_fill(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar.

    ``timestamp`` identifies the bar. Every feed here labels a bar by its
    opening time, which is what the data providers return; what matters is that
    a feed is internally consistent, since the engine only ever acts on a bar
    that has already completed.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high {self.high} below low {self.low} at {self.timestamp}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [{self.low}, {self.high}] at {self.timestamp}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [{self.low}, {self.high}] at {self.timestamp}")

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True)
class Order:
    """An order request.

    ``client_order_id`` is assigned here rather than by the venue, so a retry
    after an ambiguous failure carries the same identity as the original and
    can be recognised as a duplicate instead of becoming a second position.
    """

    symbol: str
    side: Side
    quantity: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reason: str = ""
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require a limit_price")
        if not self.client_order_id:
            raise ValueError("client_order_id must not be empty")


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    reason: str = ""
    client_order_id: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def cash_delta(self) -> float:
        """Change in cash balance caused by this fill, commission included."""
        return -self.side.sign * self.notional - self.commission


@dataclass(frozen=True)
class OrderResult:
    """What became of a submitted order.

    Every submission returns one of these. A caller that ignores the status
    cannot tell a full fill from a partial one, which is how a bot ends up
    believing it holds a position it only partly has.
    """

    order: Order
    status: OrderStatus
    filled_quantity: float = 0.0
    fill: Fill | None = None
    reason: str = ""
    #: Set when status is REJECTED, so callers can tell an external refusal
    #: from our own risk layer declining to trade.
    rejection_kind: RejectionKind | None = None

    @property
    def requested_quantity(self) -> float:
        return self.order.quantity

    @property
    def unfilled_quantity(self) -> float:
        return max(self.order.quantity - self.filled_quantity, 0.0)

    @property
    def filled(self) -> bool:
        return self.status.is_fill

    @property
    def complete(self) -> bool:
        return self.status is OrderStatus.FILLED

    def __bool__(self) -> bool:
        return self.filled

    def describe(self) -> str:
        text = (
            f"{self.order.client_order_id} {self.order.side.value} "
            f"{self.filled_quantity:.8f}/{self.order.quantity:.8f} "
            f"{self.order.symbol} {self.status.value}"
        )
        return f"{text} ({self.reason})" if self.reason else text


@dataclass
class Position:
    """A long/short position tracked at average cost.

    ``quantity`` is signed: positive is long, negative is short.
    """

    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    @property
    def is_long(self) -> bool:
        return self.quantity > 1e-12

    @property
    def is_short(self) -> bool:
        return self.quantity < -1e-12

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.average_price) * self.quantity

    def apply(self, side: Side, quantity: float, price: float) -> float:
        """Apply a fill and return the realized P&L it produced.

        Increasing exposure re-averages the cost basis; reducing it realizes
        P&L against that basis. A fill that crosses through zero closes the old
        side first, then opens the remainder at the fill price.
        """
        signed = side.sign * quantity
        realized = 0.0

        if self.is_flat or (signed > 0) == (self.quantity > 0):
            new_quantity = self.quantity + signed
            if abs(new_quantity) > 1e-12:
                self.average_price = (
                    self.average_price * self.quantity + price * signed
                ) / new_quantity
            self.quantity = new_quantity
        else:
            closing = min(abs(signed), abs(self.quantity))
            realized = (price - self.average_price) * closing * (1 if self.quantity > 0 else -1)
            self.realized_pnl += realized
            remaining = abs(signed) - closing
            self.quantity += signed
            if remaining > 1e-12:
                self.average_price = price
            elif self.is_flat:
                self.quantity = 0.0
                self.average_price = 0.0

        return realized


@dataclass
class Trade:
    """A completed round trip, from first entry to the fill that flattened it."""

    symbol: str
    side: Side
    quantity: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    commission: float
    reason: str = ""

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.pnl / cost if cost else 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class EquityPoint:
    timestamp: datetime
    cash: float
    equity: float
    exposure: float = 0.0


@dataclass
class Signal:
    """A strategy's desired exposure, expressed as a fraction of equity.

    ``target`` of 1.0 means fully long, -1.0 fully short, 0.0 flat. The risk
    layer decides how much of that target it is willing to fund.
    """

    target: float
    reason: str = ""
    stop_price: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.target <= 1.0:
            raise ValueError(f"signal target must be within [-1, 1], got {self.target}")
