import unittest
from datetime import datetime, timedelta

from trading_bot.models import Candle
from trading_bot.strategy import (
    STRATEGIES,
    BollingerReversion,
    Breakout,
    BuyAndHold,
    MACDTrend,
    RSIMeanReversion,
    SMACrossover,
    TrendFilter,
    build_strategy,
)


def series(closes, symbol="X"):
    start = datetime(2020, 1, 1)
    bars = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i else close
        bars.append(
            Candle(
                timestamp=start + timedelta(days=i),
                open=open_,
                high=max(open_, close) * 1.005,
                low=min(open_, close) * 0.995,
                close=close,
                symbol=symbol,
            )
        )
    return bars


def run(strategy, closes):
    return [strategy.on_candle(bar) for bar in series(closes)]


class TestRegistry(unittest.TestCase):
    def test_every_registered_strategy_builds_and_runs(self):
        for name in STRATEGIES:
            strategy = build_strategy(name)
            signals = run(strategy, [100 + (i % 11) - 5 for i in range(300)])
            self.assertEqual(len(signals), 300, name)
            self.assertTrue(all(-1.0 <= s.target <= 1.0 for s in signals), name)
            self.assertTrue(strategy.describe(), name)

    def test_unknown_strategy_lists_the_alternatives(self):
        with self.assertRaises(KeyError) as ctx:
            build_strategy("does-not-exist")
        self.assertIn("sma-crossover", str(ctx.exception))

    def test_parameters_are_forwarded(self):
        strategy = build_strategy("sma-crossover", fast=5, slow=15)
        self.assertIn("fast=5", strategy.describe())


class TestBuyAndHold(unittest.TestCase):
    def test_always_fully_long(self):
        self.assertTrue(all(s.target == 1.0 for s in run(BuyAndHold(), [100] * 20)))


class TestSMACrossover(unittest.TestCase):
    def test_rejects_fast_slower_than_slow(self):
        with self.assertRaises(ValueError):
            SMACrossover(fast=50, slow=20)

    def test_flat_until_both_averages_are_ready(self):
        signals = run(SMACrossover(3, 6), [100 + i for i in range(10)])
        self.assertTrue(all(s.target == 0.0 for s in signals[:5]))

    def test_long_in_an_uptrend(self):
        signals = run(SMACrossover(3, 6), [100 + i for i in range(40)])
        self.assertEqual(signals[-1].target, 1.0)

    def test_flat_in_a_downtrend_when_long_only(self):
        signals = run(SMACrossover(3, 6), [200 - i for i in range(40)])
        self.assertEqual(signals[-1].target, 0.0)

    def test_short_in_a_downtrend_when_permitted(self):
        signals = run(SMACrossover(3, 6, allow_short=True), [200 - i for i in range(40)])
        self.assertEqual(signals[-1].target, -1.0)


class TestRSIMeanReversion(unittest.TestCase):
    def test_rejects_inconsistent_thresholds(self):
        with self.assertRaises(ValueError):
            RSIMeanReversion(oversold=60, exit_level=50, overbought=70)

    def test_buys_after_a_sustained_selloff(self):
        signals = run(RSIMeanReversion(period=5), [200 - i * 2 for i in range(30)])
        self.assertTrue(any(s.target == 1.0 for s in signals))

    def test_exits_once_the_rsi_recovers(self):
        prices = [200 - i * 2 for i in range(25)] + [150 + i * 3 for i in range(25)]
        signals = run(RSIMeanReversion(period=5), prices)
        entered = next(i for i, s in enumerate(signals) if s.target == 1.0)
        self.assertTrue(any(s.target == 0.0 for s in signals[entered:]))

    def test_shorts_an_overbought_market_when_permitted(self):
        signals = run(RSIMeanReversion(period=5, allow_short=True), [100 + i * 2 for i in range(30)])
        self.assertTrue(any(s.target == -1.0 for s in signals))


class TestBreakout(unittest.TestCase):
    def test_rejects_tiny_windows(self):
        with self.assertRaises(ValueError):
            Breakout(lookback=1)

    def test_enters_on_a_new_high(self):
        prices = [100 + (i % 3) for i in range(30)] + [100 + i * 4 for i in range(20)]
        signals = run(Breakout(lookback=10, exit_lookback=5, atr_period=5), prices)
        self.assertTrue(any(s.target == 1.0 for s in signals))

    def test_entry_signal_carries_a_stop(self):
        prices = [100 + (i % 3) for i in range(30)] + [100 + i * 4 for i in range(20)]
        signals = run(Breakout(lookback=10, exit_lookback=5, atr_period=5), prices)
        entry = next(s for s in signals if s.target == 1.0)
        self.assertIsNotNone(entry.stop_price)

    def test_exits_when_the_trend_breaks(self):
        prices = (
            [100 + (i % 3) for i in range(30)]
            + [100 + i * 4 for i in range(20)]
            + [180 - i * 6 for i in range(20)]
        )
        signals = run(Breakout(lookback=10, exit_lookback=5, atr_period=5), prices)
        entered = next(i for i, s in enumerate(signals) if s.target == 1.0)
        self.assertTrue(any(s.target == 0.0 for s in signals[entered:]))

    def test_never_goes_short(self):
        signals = run(Breakout(lookback=5, exit_lookback=3, atr_period=3), [200 - i for i in range(60)])
        self.assertTrue(all(s.target >= 0 for s in signals))


class TestMACDTrend(unittest.TestCase):
    def test_long_in_an_uptrend(self):
        signals = run(MACDTrend(3, 6, 3), [100 * 1.02**i for i in range(60)])
        self.assertEqual(signals[-1].target, 1.0)

    def test_flat_in_a_downtrend_when_long_only(self):
        signals = run(MACDTrend(3, 6, 3), [100 * 0.98**i for i in range(60)])
        self.assertEqual(signals[-1].target, 0.0)

    def test_short_in_a_downtrend_when_permitted(self):
        signals = run(MACDTrend(3, 6, 3, allow_short=True), [100 * 0.98**i for i in range(60)])
        self.assertEqual(signals[-1].target, -1.0)

    def test_stays_long_through_a_steady_linear_uptrend(self):
        # The histogram decays to zero once the signal line catches up, so a
        # histogram-based rule would read this clean uptrend as "no trend".
        signals = run(MACDTrend(3, 6, 3), [100 + i for i in range(80)])
        self.assertEqual(signals[-1].target, 1.0)

    def test_does_not_turn_bullish_on_a_decaying_series(self):
        # As price decays the absolute EMA spread shrinks, which lifts the
        # histogram above zero even though the trend is down.
        signals = run(MACDTrend(3, 6, 3), [100 * 0.98**i for i in range(80)])
        self.assertTrue(all(s.target <= 0 for s in signals))


class TestBollingerReversion(unittest.TestCase):
    def test_buys_below_the_lower_band(self):
        prices = [100 + (i % 5) for i in range(40)] + [80, 78, 76]
        signals = run(BollingerReversion(period=10), prices)
        self.assertTrue(any(s.target == 1.0 for s in signals))

    def test_exits_back_at_the_mean(self):
        prices = [100 + (i % 5) for i in range(40)] + [80, 78, 76] + [100] * 15
        signals = run(BollingerReversion(period=10), prices)
        entered = next(i for i, s in enumerate(signals) if s.target == 1.0)
        self.assertTrue(any(s.target == 0.0 for s in signals[entered:]))


class TestTrendFilter(unittest.TestCase):
    def test_suppresses_longs_below_the_trend(self):
        inner = BuyAndHold()
        filtered = TrendFilter(inner, period=10)
        signals = run(filtered, [200 - i * 2 for i in range(60)])
        self.assertEqual(signals[-1].target, 0.0)

    def test_allows_longs_above_the_trend(self):
        filtered = TrendFilter(BuyAndHold(), period=10)
        signals = run(filtered, [100 + i * 2 for i in range(60)])
        self.assertEqual(signals[-1].target, 1.0)

    def test_passes_signals_through_before_the_trend_is_ready(self):
        filtered = TrendFilter(BuyAndHold(), period=50)
        self.assertEqual(run(filtered, [100] * 5)[-1].target, 1.0)

    def test_describe_mentions_the_inner_strategy(self):
        filtered = TrendFilter(SMACrossover(3, 6), period=20)
        self.assertIn("sma-crossover", filtered.describe())


if __name__ == "__main__":
    unittest.main()
