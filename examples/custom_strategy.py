"""Write a strategy, register it, and benchmark it against buy-and-hold.

    python examples/custom_strategy.py
"""

import sys
from pathlib import Path

# Run from a clone without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import PaperBroker, RunConfig, Signal, Strategy, SyntheticFeed, TradingEngine  # noqa: E402
from trading_bot.indicators import EMA, RSI  # noqa: E402
from trading_bot.risk import RiskLimits, RiskManager, VolatilitySizer  # noqa: E402
from trading_bot.strategy import STRATEGIES, BuyAndHold  # noqa: E402


class PullbackBuyer(Strategy):
    """Buys dips, but only while the long-term trend is still up.

    An uptrend is the EMA rising; a dip is the RSI dropping below ``entry``.
    The position is held until momentum recovers past ``exit_level``.
    """

    name = "pullback-buyer"

    def __init__(self, trend_period: int = 100, rsi_period: int = 10,
                 entry: float = 35.0, exit_level: float = 60.0) -> None:
        self.trend = EMA(trend_period)
        self.rsi = RSI(rsi_period)
        self.entry = entry
        self.exit_level = exit_level
        self._previous_trend: float | None = None
        self._holding = False

    def describe(self) -> str:
        return f"{self.name}(entry={self.entry}, exit={self.exit_level})"

    def on_candle(self, candle):
        self.trend.update(candle.close)
        rsi = self.rsi.update(candle.close)
        if not self.trend.ready or rsi is None:
            return Signal(0.0)

        rising = self._previous_trend is not None and self.trend.value > self._previous_trend
        self._previous_trend = self.trend.value

        if self._holding:
            if rsi >= self.exit_level or not rising:
                self._holding = False
                return Signal(0.0, reason=f"exit, RSI {rsi:.0f}")
            return Signal(1.0, reason="holding the dip")

        if rising and rsi <= self.entry:
            self._holding = True
            return Signal(1.0, reason=f"dip in an uptrend, RSI {rsi:.0f}")
        return Signal(0.0, reason="waiting for a dip")


def main() -> None:
    STRATEGIES[PullbackBuyer.name] = PullbackBuyer

    # The same series for both runs, so the comparison is apples to apples.
    bars = list(SyntheticFeed(symbol="DEMO", bars=1500, drift=0.10, volatility=0.28, seed=21))

    def run(strategy):
        engine = TradingEngine(
            strategy=strategy,
            broker=PaperBroker(starting_cash=100_000, allow_short=False),
            risk=RiskManager(
                RiskLimits(max_position_fraction=0.95, max_drawdown=0.30),
                VolatilitySizer(risk_per_trade=0.015, max_fraction=0.95),
            ),
        )
        return engine.run(bars)

    strategy_result = run(PullbackBuyer())
    benchmark = run(BuyAndHold())

    print(strategy_result.summary())
    print()
    print(benchmark.summary())

    delta = strategy_result.metrics.total_return - benchmark.metrics.total_return
    verdict = "beat" if delta > 0 else "lost to"
    print(f"\n{PullbackBuyer.name} {verdict} buy-and-hold by {abs(delta):.1%} on this series.")
    print("One series is one sample — try other seeds before believing it.")

    # The CLI reaches registered strategies too, via a config file.
    config = RunConfig(strategy="pullback-buyer", bars=1500, seed=21)
    print(f"\nEquivalent config: {config.strategy} over {config.bars} bars")


if __name__ == "__main__":
    main()
