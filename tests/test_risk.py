import unittest
from datetime import datetime

from trading_bot.models import Candle, Signal
from trading_bot.risk import (
    FixedFractionSizer,
    RiskLimits,
    RiskManager,
    VolatilitySizer,
)


def candle(price=100.0, day=1, high=None, low=None):
    return Candle(
        timestamp=datetime(2020, 1, day),
        open=price,
        high=high if high is not None else price * 1.02,
        low=low if low is not None else price * 0.98,
        close=price,
        symbol="X",
    )


class TestRiskLimits(unittest.TestCase):
    def test_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            RiskLimits(max_position_fraction=0)
        with self.assertRaises(ValueError):
            RiskLimits(max_drawdown=1.5)
        with self.assertRaises(ValueError):
            RiskLimits(risk_per_trade=0)
        with self.assertRaises(ValueError):
            RiskLimits(daily_loss_limit=2.0)

    def test_drawdown_may_be_disabled(self):
        self.assertIsNone(RiskLimits(max_drawdown=None).max_drawdown)


class TestFixedFractionSizer(unittest.TestCase):
    def test_scales_with_equity_and_signal(self):
        sizer = FixedFractionSizer(0.5)
        self.assertAlmostEqual(sizer.target_notional(Signal(1.0), 10_000, candle()), 5_000)
        self.assertAlmostEqual(sizer.target_notional(Signal(0.5), 10_000, candle()), 2_500)
        self.assertAlmostEqual(sizer.target_notional(Signal(-1.0), 10_000, candle()), -5_000)
        self.assertAlmostEqual(sizer.target_notional(Signal(0.0), 10_000, candle()), 0.0)

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            FixedFractionSizer(0)


class TestVolatilitySizer(unittest.TestCase):
    def test_stop_distance_determines_size(self):
        sizer = VolatilitySizer(risk_per_trade=0.01, max_fraction=10.0)
        bar = candle(100.0)
        # Risking 1% of 100k = $1,000 over a $5 stop distance buys 200 shares.
        signal = Signal(1.0, stop_price=95.0)
        notional = sizer.target_notional(signal, 100_000, bar)
        self.assertAlmostEqual(notional, 200 * 100.0)

    def test_wider_stops_produce_smaller_positions(self):
        sizer = VolatilitySizer(risk_per_trade=0.01, max_fraction=10.0)
        bar = candle(100.0)
        tight = sizer.target_notional(Signal(1.0, stop_price=98.0), 100_000, bar)
        wide = sizer.target_notional(Signal(1.0, stop_price=90.0), 100_000, bar)
        self.assertGreater(tight, wide)

    def test_capped_by_max_fraction(self):
        sizer = VolatilitySizer(risk_per_trade=0.5, max_fraction=1.0)
        notional = sizer.target_notional(Signal(1.0, stop_price=99.9), 100_000, candle(100.0))
        self.assertLessEqual(notional, 100_000)

    def test_falls_back_to_the_cap_before_atr_is_ready(self):
        sizer = VolatilitySizer(risk_per_trade=0.01, atr_period=14, max_fraction=0.5)
        notional = sizer.target_notional(Signal(1.0), 100_000, candle(100.0))
        self.assertAlmostEqual(notional, 50_000)

    def test_uses_atr_once_warmed_up(self):
        sizer = VolatilitySizer(risk_per_trade=0.01, atr_period=3, atr_multiple=2.0, max_fraction=10.0)
        for day in range(1, 6):
            sizer.observe(candle(100.0, day=day, high=105.0, low=95.0))
        notional = sizer.target_notional(Signal(1.0), 100_000, candle(100.0, day=6))
        # ATR is 10, so the stop sits 20 away; $1,000 of risk buys 50 shares.
        self.assertAlmostEqual(notional, 50 * 100.0, places=4)

    def test_short_signal_produces_negative_notional(self):
        sizer = VolatilitySizer(risk_per_trade=0.01, max_fraction=10.0)
        notional = sizer.target_notional(Signal(-1.0, stop_price=105.0), 100_000, candle(100.0))
        self.assertLess(notional, 0)

    def test_flat_signal_is_zero(self):
        sizer = VolatilitySizer()
        self.assertEqual(sizer.target_notional(Signal(0.0), 100_000, candle()), 0.0)


class TestRiskManager(unittest.TestCase):
    def test_position_cap_is_enforced(self):
        manager = RiskManager(
            RiskLimits(max_position_fraction=0.5, max_drawdown=None),
            FixedFractionSizer(1.0),
        )
        bar = candle(100.0)
        manager.observe(bar, 10_000)
        self.assertAlmostEqual(manager.target_notional(Signal(1.0), 10_000, bar), 5_000)

    def test_short_target_is_capped_symmetrically(self):
        manager = RiskManager(
            RiskLimits(max_position_fraction=0.5, max_drawdown=None),
            FixedFractionSizer(1.0),
        )
        bar = candle(100.0)
        manager.observe(bar, 10_000)
        self.assertAlmostEqual(manager.target_notional(Signal(-1.0), 10_000, bar), -5_000)

    def test_halts_once_drawdown_breaches_the_limit(self):
        manager = RiskManager(RiskLimits(max_drawdown=0.20), FixedFractionSizer(1.0))
        manager.observe(candle(day=1), 100_000)
        self.assertFalse(manager.halted)
        manager.observe(candle(day=2), 75_000)
        self.assertTrue(manager.halted)
        self.assertIn("max drawdown", manager.halt_reason)

    def test_halt_forces_a_zero_target(self):
        manager = RiskManager(RiskLimits(max_drawdown=0.20), FixedFractionSizer(1.0))
        manager.observe(candle(day=1), 100_000)
        manager.observe(candle(day=2), 75_000)
        bar = candle(day=3)
        self.assertEqual(manager.target_notional(Signal(1.0), 75_000, bar), 0.0)
        allowed, reason = manager.can_trade()
        self.assertFalse(allowed)
        self.assertTrue(reason)

    def test_halt_is_permanent_even_after_recovery(self):
        manager = RiskManager(RiskLimits(max_drawdown=0.20), FixedFractionSizer(1.0))
        manager.observe(candle(day=1), 100_000)
        manager.observe(candle(day=2), 70_000)
        manager.observe(candle(day=3), 200_000)
        self.assertTrue(manager.halted)

    def test_peak_tracks_the_high_water_mark(self):
        manager = RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(1.0))
        for day, equity in enumerate([100.0, 150.0, 120.0], start=1):
            manager.observe(candle(day=day), equity)
        self.assertAlmostEqual(manager.peak_equity, 150.0)

    def test_daily_trade_limit_blocks_further_trades(self):
        manager = RiskManager(
            RiskLimits(max_drawdown=None, max_trades_per_day=2), FixedFractionSizer(1.0)
        )
        manager.observe(candle(day=1), 100_000)
        self.assertTrue(manager.can_trade()[0])
        manager.record_trade()
        manager.record_trade()
        self.assertFalse(manager.can_trade()[0])

    def test_daily_trade_limit_resets_the_next_day(self):
        manager = RiskManager(
            RiskLimits(max_drawdown=None, max_trades_per_day=1), FixedFractionSizer(1.0)
        )
        manager.observe(candle(day=1), 100_000)
        manager.record_trade()
        self.assertFalse(manager.can_trade()[0])
        manager.observe(candle(day=2), 100_000)
        self.assertTrue(manager.can_trade()[0])

    def test_daily_loss_limit_measures_from_the_prior_close(self):
        manager = RiskManager(
            RiskLimits(max_drawdown=None, daily_loss_limit=0.05), FixedFractionSizer(1.0)
        )
        manager.observe(candle(day=1), 100_000)
        self.assertFalse(manager.paused)
        manager.observe(candle(day=2), 90_000)
        self.assertTrue(manager.paused)
        self.assertEqual(manager.pause_count, 1)

    def test_pause_lifts_on_the_following_day(self):
        manager = RiskManager(
            RiskLimits(max_drawdown=None, daily_loss_limit=0.05), FixedFractionSizer(1.0)
        )
        manager.observe(candle(day=1), 100_000)
        manager.observe(candle(day=2), 90_000)
        self.assertTrue(manager.paused)
        manager.observe(candle(day=3), 90_000)
        self.assertFalse(manager.paused)


if __name__ == "__main__":
    unittest.main()
