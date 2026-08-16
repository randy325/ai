"""Trading strategies.

A strategy sees one candle at a time and returns a :class:`Signal` describing
the exposure it wants, as a fraction of equity. It never sizes orders or
touches cash — the risk layer and broker own that, so a strategy stays a pure
function of the price history it has seen.
"""

from __future__ import annotations

from typing import Callable

from .indicators import ATR, EMA, MACD, RSI, SMA, BollingerBands, RollingHighLow
from .models import Candle, Signal

FLAT = Signal(0.0, reason="flat")


class Strategy:
    """Base class for all strategies."""

    name: str = "strategy"

    def on_candle(self, candle: Candle) -> Signal:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class BuyAndHold(Strategy):
    """Goes long on the first bar and stays there. The benchmark to beat."""

    name = "buy-and-hold"

    def on_candle(self, candle: Candle) -> Signal:
        return Signal(1.0, reason="hold")


class SMACrossover(Strategy):
    """Long while the fast average is above the slow one.

    With ``allow_short`` the strategy flips to short on the down-cross instead
    of going flat.
    """

    name = "sma-crossover"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False) -> None:
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow period")
        self.fast = SMA(fast)
        self.slow = SMA(slow)
        self.allow_short = allow_short
        self._fast_period = fast
        self._slow_period = slow

    def describe(self) -> str:
        mode = "long/short" if self.allow_short else "long-only"
        return f"{self.name}(fast={self._fast_period}, slow={self._slow_period}, {mode})"

    def on_candle(self, candle: Candle) -> Signal:
        self.fast.update(candle.close)
        self.slow.update(candle.close)
        if not (self.fast.ready and self.slow.ready):
            return FLAT

        spread = self.fast.value - self.slow.value
        if spread > 0:
            return Signal(1.0, reason=f"fast {self.fast.value:.2f} > slow {self.slow.value:.2f}")
        if self.allow_short:
            return Signal(-1.0, reason=f"fast {self.fast.value:.2f} < slow {self.slow.value:.2f}")
        return Signal(0.0, reason="fast below slow")


class RSIMeanReversion(Strategy):
    """Buys oversold readings and exits once the RSI mean-reverts.

    Entry is at ``oversold``; the position is held until the RSI recovers past
    ``exit_level``, which avoids the whipsaw of exiting the moment it ticks
    back above the entry threshold.
    """

    name = "rsi-mean-reversion"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
        allow_short: bool = False,
    ) -> None:
        if not 0 < oversold < exit_level < overbought < 100:
            raise ValueError("thresholds must satisfy 0 < oversold < exit_level < overbought < 100")
        self.rsi = RSI(period)
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.allow_short = allow_short
        self._target = 0.0

    def describe(self) -> str:
        return (
            f"{self.name}(period={self.rsi.period}, oversold={self.oversold}, "
            f"overbought={self.overbought}, exit={self.exit_level})"
        )

    def on_candle(self, candle: Candle) -> Signal:
        rsi = self.rsi.update(candle.close)
        if rsi is None:
            return FLAT

        if self._target == 0.0:
            if rsi <= self.oversold:
                self._target = 1.0
                return Signal(1.0, reason=f"RSI {rsi:.1f} oversold")
            if self.allow_short and rsi >= self.overbought:
                self._target = -1.0
                return Signal(-1.0, reason=f"RSI {rsi:.1f} overbought")
            return FLAT

        if self._target > 0 and rsi >= self.exit_level:
            self._target = 0.0
            return Signal(0.0, reason=f"RSI {rsi:.1f} reverted")
        if self._target < 0 and rsi <= self.exit_level:
            self._target = 0.0
            return Signal(0.0, reason=f"RSI {rsi:.1f} reverted")

        return Signal(self._target, reason=f"holding, RSI {rsi:.1f}")


class Breakout(Strategy):
    """Donchian-style breakout with an ATR trailing stop.

    Enters long when price closes above the highest high of the lookback
    window, and exits when price falls an ATR multiple below the running peak.
    """

    name = "breakout"

    def __init__(
        self,
        lookback: int = 20,
        exit_lookback: int = 10,
        atr_period: int = 14,
        atr_stop: float = 2.5,
    ) -> None:
        if lookback < 2 or exit_lookback < 2:
            raise ValueError("lookback windows must be >= 2")
        self.channel = RollingHighLow(lookback)
        self.exit_channel = RollingHighLow(exit_lookback)
        self.atr = ATR(atr_period)
        self.atr_stop = atr_stop
        self._in_position = False
        self._peak = 0.0

    def describe(self) -> str:
        return (
            f"{self.name}(lookback={self.channel.period}, "
            f"exit={self.exit_channel.period}, atr_stop={self.atr_stop})"
        )

    def on_candle(self, candle: Candle) -> Signal:
        # Read the channel before this bar updates it, so a breakout compares
        # against prior bars rather than including itself.
        breakout_level = self.channel.highest
        exit_level = self.exit_channel.lowest
        atr = self.atr.update_candle(candle)
        self.channel.update_candle(candle)
        self.exit_channel.update_candle(candle)

        if breakout_level is None or atr is None:
            return FLAT

        if not self._in_position:
            if candle.close > breakout_level:
                self._in_position = True
                self._peak = candle.close
                stop = candle.close - self.atr_stop * atr
                return Signal(1.0, reason=f"broke {breakout_level:.2f}", stop_price=stop)
            return FLAT

        self._peak = max(self._peak, candle.high)
        trailing_stop = self._peak - self.atr_stop * atr

        if candle.close < trailing_stop:
            self._in_position = False
            return Signal(0.0, reason=f"trailing stop {trailing_stop:.2f}")
        if exit_level is not None and candle.close < exit_level:
            self._in_position = False
            return Signal(0.0, reason=f"lost channel {exit_level:.2f}")

        return Signal(1.0, reason="riding trend", stop_price=trailing_stop)


class MACDTrend(Strategy):
    """Long while the MACD line is above zero; optionally short when below.

    Direction comes from the MACD line's sign — the fast EMA above or below the
    slow one — not from the histogram. In a sustained steady trend the signal
    line catches up and the histogram decays to zero, so it reads as "no trend"
    exactly when the trend is cleanest; worse, on a decaying series the EMA
    spread shrinks with price and the histogram turns positive in a downtrend.
    """

    name = "macd-trend"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        allow_short: bool = False,
    ) -> None:
        self.macd = MACD(fast, slow, signal)
        self.allow_short = allow_short

    def describe(self) -> str:
        mode = "long/short" if self.allow_short else "long-only"
        return f"{self.name}({mode})"

    def on_candle(self, candle: Candle) -> Signal:
        self.macd.update(candle.close)
        line = self.macd.macd
        if line is None:
            return FLAT
        if line > 0:
            return Signal(1.0, reason=f"MACD {line:+.3f} above zero")
        if self.allow_short:
            return Signal(-1.0, reason=f"MACD {line:+.3f} below zero")
        return Signal(0.0, reason=f"MACD {line:+.3f} below zero")


class BollingerReversion(Strategy):
    """Fades moves outside the Bollinger bands, exiting at the middle band."""

    name = "bollinger-reversion"

    def __init__(self, period: int = 20, num_std: float = 2.0, allow_short: bool = False) -> None:
        self.bands = BollingerBands(period, num_std)
        self.allow_short = allow_short
        self._target = 0.0

    def describe(self) -> str:
        return f"{self.name}(period={self.bands.period}, std={self.bands.num_std})"

    def on_candle(self, candle: Candle) -> Signal:
        self.bands.update(candle.close)
        if not self.bands.ready:
            return FLAT

        price = candle.close
        if self._target == 0.0:
            if price < self.bands.lower:
                self._target = 1.0
                return Signal(1.0, reason=f"below lower band {self.bands.lower:.2f}")
            if self.allow_short and price > self.bands.upper:
                self._target = -1.0
                return Signal(-1.0, reason=f"above upper band {self.bands.upper:.2f}")
            return FLAT

        if self._target > 0 and price >= self.bands.middle:
            self._target = 0.0
            return Signal(0.0, reason="reverted to mean")
        if self._target < 0 and price <= self.bands.middle:
            self._target = 0.0
            return Signal(0.0, reason="reverted to mean")

        return Signal(self._target, reason="holding reversion")


class TrendFilter(Strategy):
    """Wraps another strategy and suppresses longs below a long-term average.

    A cheap way to keep a mean-reversion strategy from buying every dip of a
    sustained downtrend.
    """

    name = "trend-filter"

    def __init__(self, inner: Strategy, period: int = 200) -> None:
        self.inner = inner
        self.trend = EMA(period)
        self._period = period

    def describe(self) -> str:
        return f"{self.name}({self.inner.describe()}, ema={self._period})"

    def on_candle(self, candle: Candle) -> Signal:
        self.trend.update(candle.close)
        signal = self.inner.on_candle(candle)
        if not self.trend.ready:
            return signal
        if signal.target > 0 and candle.close < self.trend.value:
            return Signal(0.0, reason=f"below {self._period}-EMA, long suppressed")
        if signal.target < 0 and candle.close > self.trend.value:
            return Signal(0.0, reason=f"above {self._period}-EMA, short suppressed")
        return signal


STRATEGIES: dict[str, Callable[..., Strategy]] = {
    BuyAndHold.name: BuyAndHold,
    SMACrossover.name: SMACrossover,
    RSIMeanReversion.name: RSIMeanReversion,
    Breakout.name: Breakout,
    MACDTrend.name: MACDTrend,
    BollingerReversion.name: BollingerReversion,
}


def build_strategy(name: str, **params) -> Strategy:
    """Instantiate a registered strategy by name."""
    try:
        factory = STRATEGIES[name]
    except KeyError:
        available = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown strategy {name!r}; available: {available}") from None
    return factory(**params)
