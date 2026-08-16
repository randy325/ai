"""Paper broker: order execution, cash accounting and trade bookkeeping.

Fills are simulated against the bar the order was submitted on, with
configurable commission and slippage. Nothing here talks to a real exchange —
see ``docs/going-live.md`` for what a live adapter would have to add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Candle, Fill, Order, OrderType, Position, Side, Trade


@dataclass
class Commission:
    """Commission model: a per-share rate, a rate on notional, and a floor."""

    per_share: float = 0.0
    percent: float = 0.0
    minimum: float = 0.0

    def charge(self, quantity: float, price: float) -> float:
        fee = quantity * self.per_share + quantity * price * self.percent
        return max(fee, self.minimum) if quantity > 0 else 0.0


@dataclass
class SlippageModel:
    """Moves the fill price against the trader by a fraction of the price.

    ``percent`` is a fixed cost; ``spread_fraction`` adds a cost proportional to
    the bar's own range, so thin, volatile bars are modelled as more expensive.
    """

    percent: float = 0.0005
    spread_fraction: float = 0.0

    def fill_price(self, side: Side, reference: float, candle: Candle | None = None) -> float:
        cost = reference * self.percent
        if candle is not None and self.spread_fraction:
            cost += (candle.high - candle.low) * self.spread_fraction
        return max(reference + side.sign * cost, 0.01)


class InsufficientFunds(Exception):
    """Raised when an order would overdraw the account beyond its leverage."""


@dataclass
class _OpenTrade:
    side: Side
    quantity: float
    entry_time: datetime
    entry_price: float
    commission: float
    reason: str


@dataclass
class PaperBroker:
    """Simulated broker holding cash and at most one position per symbol.

    Commission and slippage default to non-zero. A frictionless default would
    quietly flatter every strategy that never sets them, and a strategy that
    only works at zero cost does not work.
    """

    starting_cash: float = 100_000.0
    commission: Commission = field(default_factory=lambda: Commission(percent=0.001))
    slippage: SlippageModel = field(default_factory=SlippageModel)
    allow_short: bool = True
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if self.max_leverage < 1.0:
            raise ValueError("max_leverage must be at least 1.0")
        self.cash: float = self.starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self.total_commission: float = 0.0
        self.rejections: list[tuple[datetime, str]] = []
        self._marks: dict[str, float] = {}
        self._open_trades: dict[str, _OpenTrade] = {}

    # -- state ---------------------------------------------------------------

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def mark(self, candle: Candle) -> None:
        """Record the latest price, used for equity and buying-power checks."""
        self._marks[candle.symbol] = candle.close

    def mark_price(self, symbol: str, price: float) -> None:
        """Mark a symbol at an explicit price, e.g. a bar's open."""
        self._marks[symbol] = price

    def market_value(self) -> float:
        return sum(
            position.market_value(self._marks.get(symbol, position.average_price))
            for symbol, position in self.positions.items()
        )

    def equity(self) -> float:
        return self.cash + self.market_value()

    def exposure(self) -> float:
        """Gross exposure as a fraction of equity."""
        equity = self.equity()
        if equity <= 0:
            return 0.0
        gross = sum(
            abs(position.market_value(self._marks.get(symbol, position.average_price)))
            for symbol, position in self.positions.items()
        )
        return gross / equity

    # -- execution -----------------------------------------------------------

    def submit(
        self, order: Order, candle: Candle, reference_price: float | None = None
    ) -> Fill | None:
        """Execute ``order`` against ``candle``. Returns ``None`` if rejected.

        ``reference_price`` overrides the bar's close as the pre-slippage fill
        price, which is how the engine fills on the next bar's open.
        """
        if order.quantity <= 0:
            return None

        reference = candle.close if reference_price is None else reference_price
        if order.type is OrderType.LIMIT:
            # A limit order only fills if the bar traded through its price.
            if order.side is Side.BUY and candle.low > order.limit_price:
                self.rejections.append((candle.timestamp, "buy limit not reached"))
                return None
            if order.side is Side.SELL and candle.high < order.limit_price:
                self.rejections.append((candle.timestamp, "sell limit not reached"))
                return None
            reference = order.limit_price

        price = self.slippage.fill_price(order.side, reference, candle)
        position = self.position(order.symbol)

        if not self.allow_short:
            resulting = position.quantity + order.side.sign * order.quantity
            if resulting < -1e-9:
                self.rejections.append((candle.timestamp, "shorting disabled"))
                return None

        quantity = self._affordable_quantity(order, position, price, candle)
        if quantity <= 1e-9:
            self.rejections.append((candle.timestamp, "insufficient buying power"))
            return None

        commission = self.commission.charge(quantity, price)
        fill = Fill(
            timestamp=candle.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            reason=order.reason,
        )

        self.cash += fill.cash_delta
        self.total_commission += commission
        self.fills.append(fill)
        position.apply(order.side, quantity, price)
        self._record_trade(fill, position)
        return fill

    def _affordable_quantity(
        self, order: Order, position: Position, price: float, candle: Candle
    ) -> float:
        """Trim an order down to what buying power allows.

        Reducing an existing position always frees capital, so only the portion
        that increases exposure is checked against the leverage cap.
        """
        signed = order.side.sign * order.quantity
        reducing = position.quantity != 0 and (signed > 0) != (position.quantity > 0)
        closing_quantity = min(order.quantity, abs(position.quantity)) if reducing else 0.0
        increasing_quantity = order.quantity - closing_quantity
        if increasing_quantity <= 1e-9:
            return order.quantity

        equity = self.equity()
        if equity <= 0:
            return 0.0

        gross_after_close = sum(
            abs(
                (p.quantity - (order.side.sign * closing_quantity if s == order.symbol else 0.0))
                * self._marks.get(s, p.average_price)
            )
            for s, p in self.positions.items()
        )
        headroom = self.max_leverage * equity - gross_after_close
        if headroom <= 0:
            return closing_quantity

        # Leave room for commission so a max-size order can't overdraw.
        unit_cost = price * (1.0 + self.commission.percent) + self.commission.per_share
        affordable = headroom / unit_cost if unit_cost > 0 else 0.0
        return closing_quantity + min(increasing_quantity, max(affordable, 0.0))

    def _record_trade(self, fill: Fill, position: Position) -> None:
        """Track round trips so per-trade statistics can be computed later."""
        open_trade = self._open_trades.get(fill.symbol)

        if open_trade is None:
            if not position.is_flat:
                self._open_trades[fill.symbol] = _OpenTrade(
                    side=fill.side,
                    quantity=fill.quantity,
                    entry_time=fill.timestamp,
                    entry_price=fill.price,
                    commission=fill.commission,
                    reason=fill.reason,
                )
            return

        if fill.side is open_trade.side:
            total = open_trade.quantity + fill.quantity
            open_trade.entry_price = (
                open_trade.entry_price * open_trade.quantity + fill.price * fill.quantity
            ) / total
            open_trade.quantity = total
            open_trade.commission += fill.commission
            return

        closed = min(fill.quantity, open_trade.quantity)
        gross = (fill.price - open_trade.entry_price) * closed * open_trade.side.sign
        share = closed / open_trade.quantity
        commission = open_trade.commission * share + fill.commission

        self.trades.append(
            Trade(
                symbol=fill.symbol,
                side=open_trade.side,
                quantity=closed,
                entry_time=open_trade.entry_time,
                entry_price=open_trade.entry_price,
                exit_time=fill.timestamp,
                exit_price=fill.price,
                pnl=gross - commission,
                commission=commission,
                reason=fill.reason,
            )
        )

        remaining = open_trade.quantity - closed
        if remaining > 1e-9:
            open_trade.quantity = remaining
            open_trade.commission *= 1.0 - share
        elif not position.is_flat:
            # The fill flipped the position; the excess opens a new trade.
            self._open_trades[fill.symbol] = _OpenTrade(
                side=fill.side,
                quantity=abs(position.quantity),
                entry_time=fill.timestamp,
                entry_price=fill.price,
                commission=0.0,
                reason=fill.reason,
            )
        else:
            del self._open_trades[fill.symbol]

    def close_all(self, candle: Candle, reference_price: float | None = None) -> list[Fill]:
        """Flatten every open position at the given bar."""
        fills = []
        for symbol, position in list(self.positions.items()):
            if position.is_flat:
                continue
            side = Side.SELL if position.is_long else Side.BUY
            order = Order(
                symbol=symbol,
                side=side,
                quantity=abs(position.quantity),
                reason="close-all",
            )
            fill = self.submit(order, candle, reference_price)
            if fill is not None:
                fills.append(fill)
        return fills
