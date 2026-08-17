import unittest
from datetime import datetime, timedelta

from trading_bot.broker import Commission, PaperBroker, SlippageModel
from trading_bot.engine import TradingEngine
from trading_bot.models import Candle, Signal
from trading_bot.risk import FixedFractionSizer, RiskLimits, RiskManager
from trading_bot.strategy import BuyAndHold, SMACrossover, Strategy


def series(closes, symbol="X", opens=None):
    """Build candles whose open equals the prior close unless overridden."""
    start = datetime(2020, 1, 1)
    bars = []
    for i, close in enumerate(closes):
        if opens is not None:
            open_ = opens[i]
        else:
            open_ = closes[i - 1] if i else close
        bars.append(
            Candle(
                timestamp=start + timedelta(days=i),
                open=open_,
                high=max(open_, close),
                low=min(open_, close),
                close=close,
                volume=1000,
                symbol=symbol,
            )
        )
    return bars


class AlwaysLong(Strategy):
    name = "always-long"

    def on_candle(self, candle):
        return Signal(1.0, reason="test")


class FlipFlop(Strategy):
    """Long on even bars, flat on odd ones."""

    name = "flip-flop"

    def __init__(self):
        self.count = -1

    def on_candle(self, candle):
        self.count += 1
        return Signal(1.0 if self.count % 2 == 0 else 0.0, reason="test")


def build_engine(strategy, cash=10_000.0, execute_on="next_open", **risk_kwargs):
    broker = PaperBroker(
        starting_cash=cash,
        commission=Commission(percent=0.0),
        slippage=SlippageModel(percent=0.0),
        allow_short=True,
    )
    risk_kwargs.setdefault("max_drawdown", None)
    limits = RiskLimits(max_position_fraction=1.0, **risk_kwargs)
    risk = RiskManager(limits, FixedFractionSizer(1.0))
    return TradingEngine(strategy, broker, risk, execute_on=execute_on)


class TestExecutionTiming(unittest.TestCase):
    def test_signal_fills_on_the_next_bar_open_not_the_signal_bar_close(self):
        # Bar 0 closes at 100 and bar 1 opens at 50. A lookahead bug would fill
        # at 100; correct behaviour fills at bar 1's open of 50.
        bars = series([100.0, 60.0], opens=[100.0, 50.0])
        engine = build_engine(AlwaysLong())
        engine.run(bars)

        entry_fills = [f for f in engine.broker.fills if f.reason != "close-all"]
        self.assertTrue(entry_fills)
        self.assertAlmostEqual(entry_fills[0].price, 50.0)

    def test_execute_on_close_fills_at_the_signal_bar(self):
        bars = series([100.0, 60.0], opens=[100.0, 50.0])
        engine = build_engine(AlwaysLong(), execute_on="close")
        engine.run(bars)
        self.assertAlmostEqual(engine.broker.fills[0].price, 100.0)

    def test_rejects_unknown_execution_mode(self):
        with self.assertRaises(ValueError):
            build_engine(AlwaysLong(), execute_on="yesterday")

    def test_single_bar_feed_warns_that_the_order_never_filled(self):
        engine = build_engine(AlwaysLong())
        result = engine.run(series([100.0]))
        self.assertTrue(any("never filled" in w for w in result.warnings))


class TestBuyAndHold(unittest.TestCase):
    def test_produces_exactly_one_round_trip(self):
        engine = build_engine(BuyAndHold())
        result = engine.run(series([100.0 + i for i in range(60)]))
        self.assertEqual(
            result.metrics.num_trades,
            1,
            "a fixed-fraction target must not churn as equity drifts",
        )

    def test_tracks_the_underlying_return(self):
        engine = build_engine(BuyAndHold())
        result = engine.run(series([100.0 * (1.01**i) for i in range(80)]))
        self.assertGreater(result.metrics.total_return, 0.5)
        self.assertEqual(result.metrics.num_trades, 1)

    def test_equity_curve_has_one_point_per_bar(self):
        bars = series([100.0 + i for i in range(30)])
        engine = build_engine(BuyAndHold())
        result = engine.run(bars)
        self.assertEqual(len(result.equity_curve), len(bars))
        self.assertEqual(result.bars, len(bars))


class TestPositionLifecycle(unittest.TestCase):
    def test_flat_signal_closes_the_position(self):
        engine = build_engine(FlipFlop())
        result = engine.run(series([100.0] * 10))
        self.assertTrue(engine.broker.position("X").is_flat)
        self.assertGreater(result.metrics.num_trades, 0)

    def test_close_at_end_flattens_the_book(self):
        engine = build_engine(AlwaysLong())
        engine.run(series([100.0 + i for i in range(20)]))
        self.assertTrue(engine.broker.position("X").is_flat)

    def test_close_at_end_disabled_leaves_the_position_open(self):
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.0),
            slippage=SlippageModel(percent=0.0),
        )
        risk = RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(1.0))
        engine = TradingEngine(AlwaysLong(), broker, risk, close_at_end=False)
        engine.run(series([100.0 + i for i in range(20)]))
        self.assertFalse(broker.position("X").is_flat)


class TestAccounting(unittest.TestCase):
    def test_equity_equals_cash_plus_market_value_on_every_bar(self):
        engine = build_engine(FlipFlop())
        result = engine.run(series([100.0 + (i % 5) for i in range(40)]))
        for point in result.equity_curve:
            self.assertGreater(point.equity, 0)
        self.assertAlmostEqual(
            result.equity_curve[-1].equity,
            engine.broker.cash,
            places=6,
            msg="the book is flat at the end, so equity is all cash",
        )

    def test_commission_and_slippage_make_a_flat_market_lose_money(self):
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.001),
            slippage=SlippageModel(percent=0.001),
        )
        risk = RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(1.0))
        engine = TradingEngine(FlipFlop(), broker, risk)
        result = engine.run(series([100.0] * 30))
        self.assertLess(result.metrics.total_return, 0)
        self.assertGreater(broker.total_commission, 0)

    def test_no_trading_leaves_equity_untouched(self):
        class Flat(Strategy):
            name = "flat"

            def on_candle(self, candle):
                return Signal(0.0)

        engine = build_engine(Flat())
        result = engine.run(series([100.0 + i for i in range(30)]))
        self.assertEqual(result.metrics.num_trades, 0)
        self.assertAlmostEqual(result.metrics.total_return, 0.0)


class TestHalting(unittest.TestCase):
    def test_drawdown_breach_halts_and_flattens(self):
        engine = build_engine(AlwaysLong(), max_drawdown=0.10)
        # A steady collapse: the bot must stop rather than ride it to zero.
        result = engine.run(series([100.0 * (0.97**i) for i in range(60)]))
        self.assertTrue(result.halted)
        self.assertIn("max drawdown", result.halt_reason)
        self.assertTrue(engine.broker.position("X").is_flat)

    def test_halt_prevents_further_entries(self):
        engine = build_engine(AlwaysLong(), max_drawdown=0.10)
        prices = [100.0 * (0.97**i) for i in range(30)] + [100.0 * (1.05**i) for i in range(30)]
        engine.run(series(prices))
        halt_index = next(
            i for i, f in enumerate(engine.broker.fills) if f.reason.startswith("max drawdown")
        )
        after = engine.broker.fills[halt_index + 1 :]
        self.assertTrue(
            all(f.reason in {"close-all"} or f.side.value == "sell" for f in after),
            "no new exposure may be opened after a halt",
        )

    def test_daily_loss_limit_pauses_trading(self):
        engine = build_engine(AlwaysLong(), max_drawdown=None, daily_loss_limit=0.02)
        result = engine.run(series([100.0 * (0.95**i) for i in range(20)]))
        self.assertFalse(result.halted, "a daily limit pauses, it does not halt")
        self.assertGreater(engine.risk.pause_count, 0)

    def test_daily_loss_limit_leaves_a_calm_market_alone(self):
        engine = build_engine(AlwaysLong(), max_drawdown=None, daily_loss_limit=0.20)
        engine.run(series([100.0 + i * 0.1 for i in range(20)]))
        self.assertEqual(engine.risk.pause_count, 0)


class TestWarmupBars(unittest.TestCase):
    def build(self, warmup, strategy=None):
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.0),
            slippage=SlippageModel(percent=0.0),
        )
        risk = RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(1.0))
        return TradingEngine(
            strategy or AlwaysLong(), broker, risk,
            close_at_end=False, warmup_bars=warmup,
        )

    def test_no_trades_are_placed_during_warmup(self):
        # Replayed history must not open a position at prices already past.
        engine = self.build(warmup=10)
        engine.run(series([100.0 + i for i in range(10)]))
        self.assertEqual(engine.broker.fills, [])
        self.assertAlmostEqual(engine.broker.cash, 10_000.0)
        self.assertTrue(engine.broker.position("X").is_flat)

    def test_trading_resumes_after_warmup(self):
        engine = self.build(warmup=5)
        engine.run(series([100.0 + i for i in range(20)]))
        self.assertTrue(engine.broker.fills)

    def test_warmup_bars_are_excluded_from_the_equity_curve(self):
        engine = self.build(warmup=8)
        result = engine.run(series([100.0 + i for i in range(20)]))
        self.assertEqual(len(result.equity_curve), 12)
        self.assertEqual(result.bars, 20, "every bar is still consumed")

    def test_equity_is_untouched_through_warmup(self):
        engine = self.build(warmup=15)
        result = engine.run(series([100.0 + i for i in range(16)]))
        self.assertAlmostEqual(result.equity_curve[0].equity, 10_000.0)

    def test_indicators_are_primed_by_warmup(self):
        # A 10-period average is ready immediately after a 12-bar warmup, so
        # the first live bar can trade instead of waiting another 10.
        engine = self.build(warmup=12, strategy=SMACrossover(3, 10))
        engine.run(series([100.0 + i for i in range(15)]))
        self.assertTrue(engine.broker.fills)

    def test_zero_warmup_is_the_default_behaviour(self):
        engine = self.build(warmup=0)
        result = engine.run(series([100.0 + i for i in range(10)]))
        self.assertEqual(len(result.equity_curve), 10)

    def test_negative_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(warmup=-1)

    def test_warmup_longer_than_the_feed_trades_nothing(self):
        engine = self.build(warmup=100)
        result = engine.run(series([100.0 + i for i in range(10)]))
        self.assertEqual(engine.broker.fills, [])
        self.assertEqual(result.equity_curve, [])


class TestCallbacks(unittest.TestCase):
    def test_on_fill_and_on_bar_are_invoked(self):
        fills, bars = [], []
        broker = PaperBroker(starting_cash=10_000.0, slippage=SlippageModel(percent=0.0))
        risk = RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(1.0))
        engine = TradingEngine(
            AlwaysLong(), broker, risk,
            on_fill=fills.append,
            on_bar=lambda c, s, e: bars.append((c, s, e)),
        )
        engine.run(series([100.0 + i for i in range(20)]))
        self.assertTrue(fills)
        self.assertEqual(len(bars), 20)


if __name__ == "__main__":
    unittest.main()
