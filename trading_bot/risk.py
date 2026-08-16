"""Risk management: position sizing and the limits that halt trading.

The strategy says which direction it wants; this layer decides how large that
position may be and when to stop trading altogether. Keeping the two separate
means a risk rule can be changed without touching strategy logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .indicators import ATR
from .models import Candle, Signal


@dataclass
class RiskLimits:
    """Hard limits enforced on every bar.

    max_position_fraction: cap on a single position as a fraction of equity.
    max_drawdown: peak-to-trough equity loss that flattens and halts the bot.
    daily_loss_limit: intraday loss fraction that pauses trading until the next day.
    risk_per_trade: equity fraction risked between entry and stop, for ATR sizing.
    max_trades_per_day: throttle to keep a noisy signal from churning commission.
    """

    max_position_fraction: float = 1.0
    max_drawdown: float | None = 0.25
    daily_loss_limit: float | None = None
    risk_per_trade: float = 0.02
    max_trades_per_day: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.max_position_fraction <= 10:
            raise ValueError("max_position_fraction must be in (0, 10]")
        if self.max_drawdown is not None and not 0 < self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be in (0, 1]")
        if self.daily_loss_limit is not None and not 0 < self.daily_loss_limit <= 1:
            raise ValueError("daily_loss_limit must be in (0, 1]")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")


class PositionSizer:
    """Turns a signal's target exposure into a target notional value."""

    def target_notional(self, signal: Signal, equity: float, candle: Candle) -> float:
        raise NotImplementedError  # pragma: no cover - abstract

    def observe(self, candle: Candle) -> None:
        """Hook for sizers that maintain their own indicators."""


class FixedFractionSizer(PositionSizer):
    """Allocates a fixed fraction of equity, scaled by the signal strength."""

    def __init__(self, fraction: float = 1.0) -> None:
        if not 0 < fraction <= 10:
            raise ValueError("fraction must be in (0, 10]")
        self.fraction = fraction

    def target_notional(self, signal: Signal, equity: float, candle: Candle) -> float:
        return equity * self.fraction * signal.target


class VolatilitySizer(PositionSizer):
    """Sizes so a stop-distance move costs ``risk_per_trade`` of equity.

    Distance to the stop is the signal's own stop price when it supplies one,
    otherwise an ATR multiple. Quiet markets get larger positions, volatile
    ones smaller, which keeps risk per trade roughly constant.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        atr_period: int = 14,
        atr_multiple: float = 2.0,
        max_fraction: float = 1.0,
    ) -> None:
        if not 0 < risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        self.risk_per_trade = risk_per_trade
        self.atr_multiple = atr_multiple
        self.max_fraction = max_fraction
        self._atr = ATR(atr_period)

    def observe(self, candle: Candle) -> None:
        self._atr.update_candle(candle)

    def target_notional(self, signal: Signal, equity: float, candle: Candle) -> float:
        if signal.target == 0.0:
            return 0.0

        price = candle.close
        if signal.stop_price is not None:
            stop_distance = abs(price - signal.stop_price)
        elif self._atr.ready:
            stop_distance = self.atr_multiple * self._atr.value
        else:
            stop_distance = 0.0

        if stop_distance <= 1e-9:
            # Not enough information to size on risk; fall back to the cap.
            return equity * self.max_fraction * signal.target

        risk_capital = equity * self.risk_per_trade
        quantity = risk_capital / stop_distance
        notional = min(quantity * price, equity * self.max_fraction)
        return notional * (1 if signal.target > 0 else -1) * min(abs(signal.target), 1.0)


class RiskManager:
    """Applies limits to a signal and reports when trading must stop."""

    def __init__(self, limits: RiskLimits, sizer: PositionSizer | None = None) -> None:
        self.limits = limits
        self.sizer = sizer or FixedFractionSizer()
        self.peak_equity: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""
        self._paused_until: date | None = None
        self._day: date | None = None
        self._day_open_equity: float = 0.0
        self._last_equity: float | None = None
        self._trades_today: int = 0
        self.pause_count: int = 0

    def observe(self, candle: Candle, equity: float) -> None:
        """Update drawdown and daily state before the signal is evaluated."""
        self.sizer.observe(candle)
        self.peak_equity = max(self.peak_equity, equity)

        day = candle.timestamp.date()
        if self._day != day:
            self._day = day
            # Daily P&L runs from the prior session's close, not from the first
            # bar of the new day — otherwise the limit is dead on daily bars,
            # where the day rolls before the move can ever be measured.
            self._day_open_equity = equity if self._last_equity is None else self._last_equity
            self._trades_today = 0
            if self._paused_until is not None and day > self._paused_until:
                self._paused_until = None
        self._last_equity = equity

        if self.limits.max_drawdown is not None and self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.limits.max_drawdown and not self.halted:
                self.halted = True
                self.halt_reason = (
                    f"max drawdown {drawdown:.1%} breached limit "
                    f"{self.limits.max_drawdown:.1%} on {candle.timestamp:%Y-%m-%d}"
                )

        if (
            self.limits.daily_loss_limit is not None
            and self._day_open_equity > 0
            and self._paused_until is None
        ):
            daily_loss = (self._day_open_equity - equity) / self._day_open_equity
            if daily_loss >= self.limits.daily_loss_limit:
                self._paused_until = day
                self.pause_count += 1

    def record_trade(self) -> None:
        self._trades_today += 1

    @property
    def paused(self) -> bool:
        return self._paused_until is not None

    def can_trade(self) -> tuple[bool, str]:
        if self.halted:
            return False, self.halt_reason
        if self.paused:
            return False, "daily loss limit reached; paused until next session"
        if (
            self.limits.max_trades_per_day is not None
            and self._trades_today >= self.limits.max_trades_per_day
        ):
            return False, "daily trade limit reached"
        return True, ""

    def target_notional(self, signal: Signal, equity: float, candle: Candle) -> float:
        """The position value to hold, after sizing and the position cap.

        A halted or paused bot targets zero, so open positions are flattened
        rather than merely frozen.
        """
        if self.halted or self.paused:
            return 0.0
        notional = self.sizer.target_notional(signal, equity, candle)
        cap = equity * self.limits.max_position_fraction
        return max(min(notional, cap), -cap)
