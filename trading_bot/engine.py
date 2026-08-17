"""The event loop that drives a strategy over a data feed.

Ordering on each bar matters more than anything else here. By default a signal
generated from a bar's close is executed on the *next* bar's open, because
filling at the same close the signal was derived from is lookahead bias — the
single easiest way to produce a backtest that cannot be traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .broker import PaperBroker
from .metrics import Metrics, compute_metrics
from .models import (
    Candle,
    EquityPoint,
    Fill,
    Order,
    OrderResult,
    OrderStatus,
    Side,
    Signal,
    Trade,
)
from .risk import RiskManager
from .strategy import Strategy

logger = logging.getLogger("trading_bot")

MIN_REBALANCE_FRACTION = 0.10
"""Ignore rebalances smaller than this fraction of equity. Without a band this
wide, holding a fixed-fraction target re-trades on ordinary daily moves, and
the commission on that churn swamps the strategy's edge. Exits are exempt."""


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: list[EquityPoint]
    trades: list[Trade]
    fills: list[Fill]
    symbol: str = ""
    strategy: str = ""
    halted: bool = False
    halt_reason: str = ""
    bars: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        title = f"{self.strategy} on {self.symbol}" if self.symbol else self.strategy
        report = self.metrics.format_report(title or "Backtest results")
        if self.halted:
            report += f"\n\n!! Trading halted: {self.halt_reason}"
        for warning in self.warnings:
            report += f"\n!! {warning}"
        return report


class TradingEngine:
    """Runs one strategy over one symbol's feed."""

    def __init__(
        self,
        strategy: Strategy,
        broker: PaperBroker,
        risk: RiskManager,
        execute_on: str = "next_open",
        close_at_end: bool = True,
        warmup_bars: int = 0,
        min_rebalance_fraction: float = MIN_REBALANCE_FRACTION,
        on_fill: Callable[[Fill], None] | None = None,
        on_bar: Callable[[Candle, Signal, float], None] | None = None,
        on_order: Callable[[OrderResult], None] | None = None,
        reconcile_every: int = 0,
        max_consecutive_rejections: int = 5,
        audit=None,
    ) -> None:
        if execute_on not in {"next_open", "close"}:
            raise ValueError("execute_on must be 'next_open' or 'close'")
        self.strategy = strategy
        self.broker = broker
        self.risk = risk
        self.execute_on = execute_on
        self.close_at_end = close_at_end
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        #: Leading bars fed to the strategy without trading, and left out of the
        #: equity curve, so a live session's metrics describe the live part only.
        self.warmup_bars = warmup_bars
        self.min_rebalance_fraction = min_rebalance_fraction
        self.on_fill = on_fill
        self.on_bar = on_bar
        self.on_order = on_order
        #: Reconcile the books every N bars; 0 checks only at start and end.
        self.reconcile_every = reconcile_every
        #: Trip the kill switch after this many rejections in a row. A venue
        #: refusing everything means something is wrong with us, the account or
        #: the venue; retrying once a bar forever is how a bot spends a day
        #: hammering an endpoint that was never going to say yes. 0 disables.
        self.max_consecutive_rejections = max_consecutive_rejections
        self._consecutive_rejections = 0
        #: Optional AuditLogger; every order, fill, error and equity change
        #: goes through it when set, independent of the console output.
        self.audit = audit
        self.equity_curve: list[EquityPoint] = []
        self.reconciliation_problems: list[str] = []

    def _rebalance_order(
        self, symbol: str, target_notional: float, price: float, reason: str
    ) -> Order | None:
        """Build the order that moves the current position to ``target_notional``."""
        if price <= 0:
            return None

        position = self.broker.position(symbol)
        target_quantity = target_notional / price
        delta = target_quantity - position.quantity
        if abs(delta) < 1e-9:
            return None

        equity = self.broker.equity()
        flattening = abs(target_quantity) < 1e-9 and not position.is_flat
        flipping = (
            not position.is_flat
            and abs(target_quantity) > 1e-9
            and (target_quantity > 0) != (position.quantity > 0)
        )
        # Full exits and reversals always go through. Everything else — including
        # trimming a position the strategy still wants — must clear the band, or
        # a fixed-fraction target sells a sliver every time price ticks up.
        if not (flattening or flipping):
            if equity > 0 and abs(delta * price) < equity * self.min_rebalance_fraction:
                return None

        side = Side.BUY if delta > 0 else Side.SELL
        return Order(symbol=symbol, side=side, quantity=abs(delta), reason=reason)

    def run(self, feed: Iterable[Candle]) -> BacktestResult:
        """Consume ``feed`` bar by bar and return the result."""
        pending: Order | None = None
        last_candle: Candle | None = None
        bars = 0
        warnings: list[str] = []
        symbol = ""

        # Never trust local state blindly: check the books before starting, so
        # a run does not begin on top of an inconsistency inherited from a
        # previous session or a partially applied recovery.
        self.reconciliation_problems = self.broker.reconcile()
        if self.audit:
            self.audit.reconciliation(self.reconciliation_problems, phase="startup")
        if self.reconciliation_problems:
            warnings.extend(f"startup reconciliation: {p}" for p in self.reconciliation_problems)
            logger.error("startup reconciliation failed: %s", "; ".join(self.reconciliation_problems))

        for candle in feed:
            bars += 1
            symbol = symbol or candle.symbol

            if pending is not None:
                # Fill last bar's decision at this bar's open.
                self.broker.mark_price(candle.symbol, candle.open)
                closed_before = len(self.broker.trades)
                if self.audit:
                    self.audit.order_submitted(pending)
                result = self.broker.submit(pending, candle, reference_price=candle.open)
                if self.audit:
                    self.audit.order_result(result)
                fill = result.fill
                if result.status is OrderStatus.PARTIALLY_FILLED:
                    # Surfaced, not swallowed: the position is smaller than the
                    # strategy asked for and the next rebalance must see that.
                    warnings.append(
                        f"partial fill on {candle.timestamp:%Y-%m-%d}: "
                        f"{result.unfilled_quantity:.6f} of {result.requested_quantity:.6f} unfilled"
                    )
                elif result.status is OrderStatus.REJECTED:
                    warnings.append(f"order rejected on {candle.timestamp:%Y-%m-%d}: {result.reason}")
                    # Only an external refusal counts. Our own risk layer saying
                    # no is the system working: a long-only account declining
                    # short signals, or a rebalance below the venue's minimum,
                    # would otherwise halt a perfectly healthy bot within a few
                    # bars.
                    kind = result.rejection_kind
                    if kind is not None and kind.counts_toward_breaker:
                        self._consecutive_rejections += 1
                    if (
                        self.max_consecutive_rejections
                        and self._consecutive_rejections >= self.max_consecutive_rejections
                        and not self.broker.halted
                    ):
                        reason = (
                            f"{self._consecutive_rejections} consecutive rejections; "
                            f"last: {result.reason}"
                        )
                        if self.audit:
                            self.audit.kill_switch(reason, bar=bars)
                        self.broker.kill(reason, candle)
                        warnings.append(f"circuit breaker: {reason}")
                if result.filled:
                    self._consecutive_rejections = 0
                if fill is not None:
                    self.risk.record_trade()
                    # An adaptive risk layer needs the result of each round
                    # trip, not just that a fill happened.
                    if hasattr(self.risk, "record_result"):
                        for trade in self.broker.trades[closed_before:]:
                            self.risk.record_result(trade.pnl)
                    logger.info(
                        "%s %s %.4f %s @ %.4f (%s)",
                        fill.timestamp.date(),
                        fill.side.value.upper(),
                        fill.quantity,
                        fill.symbol,
                        fill.price,
                        fill.reason,
                    )
                    if self.audit:
                        self.audit.fill(fill)
                    if self.on_fill:
                        self.on_fill(fill)
                if self.on_order:
                    self.on_order(result)
                pending = None

            self.broker.mark(candle)
            equity = self.broker.equity()

            if equity <= 0:
                if self.audit:
                    self.audit.error("account wiped out", timestamp=candle.timestamp)
                warnings.append(f"account wiped out on {candle.timestamp:%Y-%m-%d}")
                self.equity_curve.append(
                    EquityPoint(candle.timestamp, self.broker.cash, equity, 0.0)
                )
                last_candle = candle
                break

            self.risk.observe(candle, equity)
            if self.audit:
                self.audit.equity(candle.timestamp, equity, self.broker.cash, self.broker.exposure())
            signal = self.strategy.on_candle(candle)

            if bars <= self.warmup_bars:
                # Replayed history: prime the indicators, but do not trade it.
                # Filling these bars would open a position at prices that have
                # already passed, and credit the session with P&L it never made.
                if self.on_bar:
                    self.on_bar(candle, signal, equity)
                last_candle = candle
                continue

            allowed, block_reason = self.risk.can_trade()
            target = self.risk.target_notional(signal, equity, candle)
            if not allowed and self.broker.position(candle.symbol).is_flat:
                # Blocked and flat: nothing to do. Blocked while holding still
                # allows the exit, since target_notional is already zero.
                target = 0.0

            order = self._rebalance_order(
                candle.symbol,
                target,
                candle.close,
                signal.reason if allowed else block_reason,
            )

            if order is not None:
                if self.execute_on == "close":
                    closed_before = len(self.broker.trades)
                    result = self.broker.submit(order, candle)
                    if result.fill is not None:
                        self.risk.record_trade()
                        if hasattr(self.risk, "record_result"):
                            for trade in self.broker.trades[closed_before:]:
                                self.risk.record_result(trade.pnl)
                        if self.on_fill:
                            self.on_fill(result.fill)
                    if self.on_order:
                        self.on_order(result)
                else:
                    pending = order

            self.equity_curve.append(
                EquityPoint(
                    timestamp=candle.timestamp,
                    cash=self.broker.cash,
                    equity=self.broker.equity(),
                    exposure=self.broker.exposure(),
                )
            )

            if self.on_bar:
                self.on_bar(candle, signal, self.broker.equity())

            if self.reconcile_every and bars % self.reconcile_every == 0:
                problems = self.broker.reconcile()
                if self.audit:
                    self.audit.reconciliation(problems, phase="periodic", bar=bars)
                if problems:
                    # Books disagreeing mid-run means the position is unknown.
                    # Stop rather than keep trading against a bad picture.
                    reason = f"reconciliation failed: {'; '.join(problems)}"
                    if self.audit:
                        self.audit.kill_switch(reason, bar=bars)
                    self.broker.kill(reason, candle)
                    warnings.extend(f"reconciliation: {p}" for p in problems)
                    last_candle = candle
                    break

            last_candle = candle

        if pending is not None:
            warnings.append("final bar's order was never filled (no subsequent bar)")

        if self.close_at_end and last_candle is not None:
            fills = [r for r in self.broker.close_all(last_candle) if r.filled]
            if fills and self.equity_curve:
                self.equity_curve[-1] = EquityPoint(
                    timestamp=last_candle.timestamp,
                    cash=self.broker.cash,
                    equity=self.broker.equity(),
                    exposure=self.broker.exposure(),
                )

        # Final check: whatever the run did, the books must still add up.
        closing_problems = self.broker.reconcile()
        if self.audit:
            self.audit.reconciliation(closing_problems, phase="final")
        if closing_problems:
            self.reconciliation_problems.extend(closing_problems)
            warnings.extend(f"final reconciliation: {p}" for p in closing_problems)
            logger.error("final reconciliation failed: %s", "; ".join(closing_problems))

        metrics = compute_metrics(
            self.equity_curve, self.broker.trades, self.broker.total_commission
        )
        metrics.extras["rejected_orders"] = len(self.broker.rejections)

        return BacktestResult(
            metrics=metrics,
            equity_curve=self.equity_curve,
            trades=self.broker.trades,
            fills=self.broker.fills,
            symbol=symbol,
            strategy=self.strategy.describe(),
            halted=self.risk.halted,
            halt_reason=self.risk.halt_reason,
            bars=bars,
            warnings=warnings,
        )
