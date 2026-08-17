"""Paper broker: order execution, cash accounting and trade bookkeeping.

Fills are simulated against the bar the order was submitted on, with
configurable commission and slippage. Nothing here talks to a real exchange —
see ``docs/going-live.md`` for what a live adapter would have to add.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime

from .instruments import EQUITY, InstrumentSpec
from .models import (
    Candle,
    Fill,
    Order,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    RejectionKind,
    Side,
    Trade,
)

logger = logging.getLogger("trading_bot.broker")


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


class ReconciliationError(Exception):
    """Raised when the broker's books do not agree with themselves."""


@dataclass
class OrderGuard:
    """Pre-trade sanity checks applied to every order.

    These are the cheap checks that catch a strategy or a data feed having gone
    wrong — a price spike from a bad tick, a sizing bug asking for the account
    ten times over. They are deliberately independent of the risk manager: a
    fat-finger check that shares state with the thing it is checking is not a
    check.
    """

    #: Reject if the fill price is more than this fraction from the last trade.
    max_price_deviation: float = 0.10
    #: Reject if a single order's notional exceeds this fraction of equity.
    max_order_fraction: float = 1.5
    #: Reject an order that would open a position beyond this count.
    max_open_positions: int = 5
    enabled: bool = True

    def check(
        self, order: Order, price: float, candle: Candle, broker: "PaperBroker"
    ) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""

        if price <= 0:
            return False, f"non-positive price {price}"

        last = broker.last_price(order.symbol)
        if last and last > 0:
            deviation = abs(price - last) / last
            if deviation > self.max_price_deviation:
                return False, (
                    f"price {price:.4f} is {deviation:.1%} from last {last:.4f}, "
                    f"limit {self.max_price_deviation:.0%}"
                )

        equity = broker.equity()
        if equity > 0:
            notional = order.quantity * price
            if notional > equity * self.max_order_fraction:
                return False, (
                    f"order notional {notional:,.2f} exceeds "
                    f"{self.max_order_fraction:.1f}x equity {equity:,.2f}"
                )

        position = broker.position(order.symbol)
        if position.is_flat:
            open_positions = sum(1 for p in broker.positions.values() if not p.is_flat)
            if open_positions >= self.max_open_positions:
                return False, f"already holding {open_positions} positions, limit {self.max_open_positions}"

        return True, ""


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
    spec: InstrumentSpec = field(default_factory=lambda: EQUITY)
    guard: OrderGuard = field(default_factory=OrderGuard)

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if self.max_leverage < 1.0:
            raise ValueError("max_leverage must be at least 1.0")
        if self.guard.enabled and self.guard.max_order_fraction < self.max_leverage:
            # A guard tighter than the leverage the account is configured for
            # rejects every order, which looks like a dead strategy rather than
            # a misconfiguration. Raise it to match and say so.
            self.guard = replace(self.guard, max_order_fraction=self.max_leverage)
            logger.info(
                "raised guard max_order_fraction to %.2f to match max_leverage",
                self.max_leverage,
            )
        self.cash: float = self.starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self.total_commission: float = 0.0
        self.rejections: list[tuple[datetime, str]] = []
        self.halted: bool = False
        self.halt_reason: str = ""
        self._marks: dict[str, float] = {}
        self._open_trades: dict[str, _OpenTrade] = {}
        self._results: dict[str, OrderResult] = {}

    # -- kill switch ---------------------------------------------------------

    def kill(self, reason: str, candle: Candle | None = None) -> list[OrderResult]:
        """Stop trading immediately, flattening if a price is available.

        After this the broker rejects every order, so a caller that keeps
        submitting cannot quietly resume. There are no resting orders to cancel
        in this simulator — fills are synchronous — so flattening is the whole
        of it; a live adapter would cancel working orders here first.
        """
        results: list[OrderResult] = []
        if candle is not None and not self.halted:
            results = self.close_all(candle)
        self.halted = True
        self.halt_reason = reason
        logger.error("KILL SWITCH: %s", reason)
        return results

    def resume(self) -> None:
        """Clear a halt. Deliberately explicit — nothing auto-resumes."""
        self.halted = False
        self.halt_reason = ""

    # -- reconciliation ------------------------------------------------------

    def reconcile(self, tolerance: float = 1e-6) -> list[str]:
        """Check the books against themselves and report any disagreement.

        With no exchange to query, the authority is the fill ledger: positions
        and cash are both derivable from it, so replaying the fills and
        comparing against the running state catches accounting drift that would
        otherwise surface as a position the bot does not know it has.

        A live adapter would replace this with a query to the venue and treat
        *that* as the authority — never local state.
        """
        problems: list[str] = []

        expected_quantities: dict[str, float] = {}
        expected_cash = self.starting_cash
        for fill in self.fills:
            expected_quantities[fill.symbol] = (
                expected_quantities.get(fill.symbol, 0.0) + fill.side.sign * fill.quantity
            )
            expected_cash += fill.cash_delta

        for symbol, expected in expected_quantities.items():
            actual = self.position(symbol).quantity
            if abs(actual - expected) > tolerance:
                problems.append(
                    f"{symbol}: position {actual:.8f} but fills imply {expected:.8f}"
                )

        for symbol, position in self.positions.items():
            if symbol not in expected_quantities and not position.is_flat:
                problems.append(f"{symbol}: holding {position.quantity:.8f} with no fills")

        if abs(self.cash - expected_cash) > tolerance:
            problems.append(f"cash {self.cash:.8f} but fills imply {expected_cash:.8f}")

        equity = self.equity()
        if abs(equity - (self.cash + self.market_value())) > tolerance:
            problems.append("equity does not equal cash plus market value")

        commission = sum(f.commission for f in self.fills)
        if abs(commission - self.total_commission) > tolerance:
            problems.append(
                f"commission total {self.total_commission:.8f} but fills sum to {commission:.8f}"
            )

        return problems

    def assert_reconciled(self, tolerance: float = 1e-6) -> None:
        """Reconcile and raise if the books disagree."""
        problems = self.reconcile(tolerance)
        if problems:
            raise ReconciliationError("; ".join(problems))

    # -- state ---------------------------------------------------------------

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def mark(self, candle: Candle) -> None:
        """Record the latest price, used for equity and buying-power checks."""
        self._marks[candle.symbol] = candle.close

    def mark_price(self, symbol: str, price: float) -> None:
        """Mark a symbol at an explicit price, e.g. a bar's open."""
        self._marks[symbol] = price

    def last_price(self, symbol: str) -> float | None:
        """The most recent mark, used by the pre-trade price sanity check."""
        return self._marks.get(symbol)

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

    def _reject(
        self,
        order: Order,
        candle: Candle,
        reason: str,
        kind: RejectionKind = RejectionKind.RISK,
    ) -> OrderResult:
        self.rejections.append((candle.timestamp, reason))
        result = OrderResult(
            order, OrderStatus.REJECTED, reason=reason, rejection_kind=kind
        )
        self._results[order.client_order_id] = result
        # Rejections after a halt are the kill switch working as intended, so
        # they log at debug; anything else is a real refusal worth surfacing.
        logger.log(
            logging.DEBUG if self.halted else logging.WARNING,
            "order rejected: %s",
            result.describe(),
        )
        return result

    def submit(
        self, order: Order, candle: Candle, reference_price: float | None = None
    ) -> OrderResult:
        """Execute ``order`` against ``candle`` and report what happened.

        Always returns an :class:`OrderResult`; check ``status`` rather than
        assuming a fill. An order trimmed by buying power comes back
        ``PARTIALLY_FILLED`` with the shortfall in ``unfilled_quantity``, which
        the previous version silently reported as a complete fill.

        ``reference_price`` overrides the bar's close as the pre-slippage fill
        price, which is how the engine fills on the next bar's open.
        """
        if self.halted:
            return self._reject(
                order, candle, f"broker halted: {self.halt_reason}", RejectionKind.HALTED
            )

        # Idempotency: a resubmitted client_order_id returns the original
        # outcome instead of opening a second position. This is what makes a
        # retry after an ambiguous failure safe.
        previous = self._results.get(order.client_order_id)
        if previous is not None:
            logger.warning("duplicate submission of %s ignored", order.client_order_id)
            return OrderResult(
                order,
                OrderStatus.DUPLICATE,
                filled_quantity=previous.filled_quantity,
                fill=previous.fill,
                reason=f"already {previous.status.value}",
            )

        if order.quantity <= 0:
            return self._reject(order, candle, "non-positive quantity")

        reference = candle.close if reference_price is None else reference_price
        if order.type is OrderType.LIMIT:
            # A limit order only fills if the bar traded through its price.
            if order.side is Side.BUY and candle.low > order.limit_price:
                return self._reject(
                    order, candle, "buy limit not reached", RejectionKind.MARKET
                )
            if order.side is Side.SELL and candle.high < order.limit_price:
                return self._reject(
                    order, candle, "sell limit not reached", RejectionKind.MARKET
                )
            reference = order.limit_price

        price = float(self.spec.round_price(self.slippage.fill_price(order.side, reference, candle)))
        position = self.position(order.symbol)

        if not self.allow_short:
            resulting = position.quantity + order.side.sign * order.quantity
            if resulting < -1e-9:
                return self._reject(order, candle, "shorting disabled")

        allowed, guard_reason = self.guard.check(order, price, candle, self)
        if not allowed:
            return self._reject(order, candle, guard_reason)

        affordable = self._affordable_quantity(order, position, price, candle)
        quantity = float(self.spec.round_quantity(affordable))
        tradable, spec_reason = self.spec.is_tradable(quantity, price)
        if not tradable:
            return self._reject(order, candle, spec_reason)

        commission = self.commission.charge(quantity, price)
        fill = Fill(
            timestamp=candle.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            reason=order.reason,
            client_order_id=order.client_order_id,
        )

        self.cash += fill.cash_delta
        self.total_commission += commission
        self.fills.append(fill)
        position.apply(order.side, quantity, price)
        self._record_trade(fill, position)

        # Partial means "less than the venue could have filled", so compare
        # against the lot-rounded request. Rounding 10.7 down to a 1-share lot
        # is quantisation, not a shortfall: no venue would ever fill the 0.7.
        requested = float(self.spec.round_quantity(order.quantity))
        shortfall = requested - quantity
        partial = shortfall > float(self.spec.lot_size) / 2
        result = OrderResult(
            order,
            OrderStatus.PARTIALLY_FILLED if partial else OrderStatus.FILLED,
            filled_quantity=quantity,
            fill=fill,
            reason=f"{shortfall:.8f} unfilled" if partial else "",
        )
        self._results[order.client_order_id] = result
        if partial:
            logger.warning("partial fill: %s", result.describe())
        return result

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

    def close_all(self, candle: Candle, reference_price: float | None = None) -> list[OrderResult]:
        """Flatten every open position at the given bar."""
        results = []
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
            result = self.submit(order, candle, reference_price)
            results.append(result)
            if not result.filled:
                logger.error("could not flatten %s: %s", symbol, result.reason)
        return results
