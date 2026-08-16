"""Streaming technical indicators.

Each indicator is fed one value at a time and returns its current reading, or
``None`` until it has seen enough data to be meaningful. Keeping them
incremental means a backtest and a live session run identical code paths.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

from .models import Candle


class Indicator:
    """Base class: ``update`` pushes a new value, ``value`` reads the last one."""

    def __init__(self) -> None:
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def update(self, value: float) -> float | None:  # pragma: no cover - abstract
        raise NotImplementedError


class SMA(Indicator):
    """Simple moving average over a fixed window."""

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("SMA period must be >= 1")
        self.period = period
        self._window: Deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, value: float) -> float | None:
        if len(self._window) == self.period:
            self._sum -= self._window[0]
        self._window.append(value)
        self._sum += value
        if len(self._window) == self.period:
            self._value = self._sum / self.period
        return self._value


class EMA(Indicator):
    """Exponential moving average, seeded with an SMA of the first ``period`` values."""

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self.period = period
        self._alpha = 2.0 / (period + 1.0)
        self._seed: list[float] = []

    def update(self, value: float) -> float | None:
        if self._value is None:
            self._seed.append(value)
            if len(self._seed) == self.period:
                self._value = sum(self._seed) / self.period
                self._seed = []
            return self._value
        self._value = value * self._alpha + self._value * (1.0 - self._alpha)
        return self._value


class RollingStdDev(Indicator):
    """Sample standard deviation over a fixed window."""

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 2:
            raise ValueError("standard deviation period must be >= 2")
        self.period = period
        self._window: Deque[float] = deque(maxlen=period)

    def update(self, value: float) -> float | None:
        self._window.append(value)
        if len(self._window) == self.period:
            mean = sum(self._window) / self.period
            variance = sum((x - mean) ** 2 for x in self._window) / (self.period - 1)
            self._value = math.sqrt(variance)
        return self._value


class RSI(Indicator):
    """Wilder's relative strength index."""

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("RSI period must be >= 1")
        self.period = period
        self._previous: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []

    def update(self, value: float) -> float | None:
        if self._previous is None:
            self._previous = value
            return None

        change = value - self._previous
        self._previous = value
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self._avg_gain is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < self.period:
                return None
            self._avg_gain = sum(self._gains) / self.period
            self._avg_loss = sum(self._losses) / self.period
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            self._value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._value = 100.0 - (100.0 / (1.0 + rs))
        return self._value


class ATR(Indicator):
    """Average true range. Fed candles rather than scalars."""

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("ATR period must be >= 1")
        self.period = period
        self._previous_close: float | None = None
        self._seed: list[float] = []

    def update(self, value: float) -> float | None:  # pragma: no cover - use update_candle
        raise TypeError("ATR consumes candles; call update_candle instead")

    def update_candle(self, candle: Candle) -> float | None:
        if self._previous_close is None:
            true_range = candle.high - candle.low
        else:
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - self._previous_close),
                abs(candle.low - self._previous_close),
            )
        self._previous_close = candle.close

        if self._value is None:
            self._seed.append(true_range)
            if len(self._seed) == self.period:
                self._value = sum(self._seed) / self.period
                self._seed = []
        else:
            self._value = (self._value * (self.period - 1) + true_range) / self.period
        return self._value


class BollingerBands:
    """Moving average with standard-deviation bands above and below."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self.period = period
        self.num_std = num_std
        self._sma = SMA(period)
        self._std = RollingStdDev(period)

    @property
    def ready(self) -> bool:
        return self._sma.ready and self._std.ready

    @property
    def middle(self) -> float | None:
        return self._sma.value

    @property
    def upper(self) -> float | None:
        if not self.ready:
            return None
        return self._sma.value + self.num_std * self._std.value

    @property
    def lower(self) -> float | None:
        if not self.ready:
            return None
        return self._sma.value - self.num_std * self._std.value

    def update(self, value: float) -> float | None:
        self._sma.update(value)
        self._std.update(value)
        return self.middle

    def percent_b(self, value: float) -> float | None:
        """Where ``value`` sits in the band: 0 at the lower band, 1 at the upper."""
        if not self.ready:
            return None
        width = self.upper - self.lower
        if width <= 0:
            return 0.5
        return (value - self.lower) / width


class MACD:
    """Moving average convergence/divergence with its signal line."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("MACD fast period must be shorter than slow period")
        self._fast = EMA(fast)
        self._slow = EMA(slow)
        self._signal = EMA(signal)
        self.macd: float | None = None
        self.signal: float | None = None

    @property
    def ready(self) -> bool:
        return self.signal is not None

    @property
    def histogram(self) -> float | None:
        if self.macd is None or self.signal is None:
            return None
        return self.macd - self.signal

    def update(self, value: float) -> float | None:
        self._fast.update(value)
        self._slow.update(value)
        if self._fast.ready and self._slow.ready:
            self.macd = self._fast.value - self._slow.value
            self._signal.update(self.macd)
            self.signal = self._signal.value
        return self.macd


class RollingHighLow:
    """Highest high and lowest low over a window, for breakout logic."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._highs: Deque[float] = deque(maxlen=period)
        self._lows: Deque[float] = deque(maxlen=period)

    @property
    def ready(self) -> bool:
        return len(self._highs) == self.period

    @property
    def highest(self) -> float | None:
        return max(self._highs) if self.ready else None

    @property
    def lowest(self) -> float | None:
        return min(self._lows) if self.ready else None

    def update_candle(self, candle: Candle) -> None:
        self._highs.append(candle.high)
        self._lows.append(candle.low)
