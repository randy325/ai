import unittest
from datetime import datetime, timedelta

from trading_bot.indicators import (
    ATR,
    EMA,
    MACD,
    RSI,
    SMA,
    BollingerBands,
    RollingHighLow,
    RollingStdDev,
)
from trading_bot.models import Candle


def candles(rows):
    start = datetime(2020, 1, 1)
    return [
        Candle(start + timedelta(days=i), open=o, high=h, low=l, close=c)
        for i, (o, h, l, c) in enumerate(rows)
    ]


class TestSMA(unittest.TestCase):
    def test_not_ready_until_window_is_full(self):
        sma = SMA(3)
        self.assertIsNone(sma.update(1))
        self.assertIsNone(sma.update(2))
        self.assertEqual(sma.update(3), 2.0)

    def test_window_rolls(self):
        sma = SMA(3)
        for value in (1, 2, 3, 4):
            sma.update(value)
        self.assertAlmostEqual(sma.value, 3.0)

    def test_rejects_invalid_period(self):
        with self.assertRaises(ValueError):
            SMA(0)


class TestEMA(unittest.TestCase):
    def test_seeds_with_sma_then_smooths(self):
        ema = EMA(3)
        for value in (1, 2, 3):
            ema.update(value)
        self.assertAlmostEqual(ema.value, 2.0)
        ema.update(10)
        # alpha = 2/(3+1) = 0.5 -> 10*0.5 + 2*0.5
        self.assertAlmostEqual(ema.value, 6.0)

    def test_constant_series_converges_to_the_constant(self):
        ema = EMA(5)
        for _ in range(50):
            ema.update(42.0)
        self.assertAlmostEqual(ema.value, 42.0, places=6)


class TestRollingStdDev(unittest.TestCase):
    def test_sample_standard_deviation(self):
        std = RollingStdDev(4)
        for value in (2, 4, 4, 6):
            std.update(value)
        # mean 4, sum sq dev 8, sample variance 8/3
        self.assertAlmostEqual(std.value, (8 / 3) ** 0.5)

    def test_constant_series_has_zero_deviation(self):
        std = RollingStdDev(3)
        for _ in range(5):
            std.update(7.0)
        self.assertAlmostEqual(std.value, 0.0)


class TestRSI(unittest.TestCase):
    def test_monotonic_rise_pins_at_100(self):
        rsi = RSI(5)
        for value in range(1, 20):
            rsi.update(float(value))
        self.assertAlmostEqual(rsi.value, 100.0)

    def test_monotonic_fall_pins_at_zero(self):
        rsi = RSI(5)
        for value in range(20, 1, -1):
            rsi.update(float(value))
        self.assertAlmostEqual(rsi.value, 0.0)

    def test_stays_within_bounds_on_mixed_series(self):
        rsi = RSI(14)
        prices = [100 + (i % 7) - 3 for i in range(80)]
        for price in prices:
            rsi.update(float(price))
        self.assertTrue(0.0 <= rsi.value <= 100.0)

    def test_needs_period_plus_one_observations(self):
        rsi = RSI(3)
        self.assertIsNone(rsi.update(10.0))
        self.assertIsNone(rsi.update(11.0))
        self.assertIsNone(rsi.update(12.0))
        self.assertIsNotNone(rsi.update(13.0))


class TestATR(unittest.TestCase):
    def test_true_range_includes_the_gap_from_prior_close(self):
        atr = ATR(2)
        bars = candles([(10, 11, 9, 10), (20, 21, 19, 20)])
        atr.update_candle(bars[0])
        atr.update_candle(bars[1])
        # First TR = 2; second TR = max(2, |21-10|, |19-10|) = 11 -> mean 6.5
        self.assertAlmostEqual(atr.value, 6.5)

    def test_rejects_scalar_updates(self):
        with self.assertRaises(TypeError):
            ATR(3).update(1.0)


class TestBollingerBands(unittest.TestCase):
    def test_bands_straddle_the_middle(self):
        bands = BollingerBands(5, 2.0)
        for value in (10, 12, 11, 13, 12):
            bands.update(value)
        self.assertTrue(bands.ready)
        self.assertLess(bands.lower, bands.middle)
        self.assertGreater(bands.upper, bands.middle)

    def test_percent_b_is_zero_at_lower_band(self):
        bands = BollingerBands(5, 2.0)
        for value in (10, 12, 11, 13, 12):
            bands.update(value)
        self.assertAlmostEqual(bands.percent_b(bands.lower), 0.0)
        self.assertAlmostEqual(bands.percent_b(bands.upper), 1.0)

    def test_flat_series_reports_midpoint(self):
        bands = BollingerBands(3, 2.0)
        for _ in range(5):
            bands.update(100.0)
        self.assertAlmostEqual(bands.percent_b(100.0), 0.5)


class TestMACD(unittest.TestCase):
    def test_rejects_fast_slower_than_slow(self):
        with self.assertRaises(ValueError):
            MACD(fast=26, slow=12)

    def test_histogram_positive_in_an_uptrend(self):
        macd = MACD(3, 6, 3)
        for value in range(1, 60):
            macd.update(float(value))
        self.assertTrue(macd.ready)
        self.assertGreater(macd.macd, 0)


class TestRollingHighLow(unittest.TestCase):
    def test_tracks_window_extremes(self):
        channel = RollingHighLow(3)
        for bar in candles([(3, 5, 0, 3), (3, 7, 2, 4), (5, 6, 1, 5), (3, 4, 3, 3)]):
            channel.update_candle(bar)
        self.assertEqual(channel.highest, 7)
        self.assertEqual(channel.lowest, 1)

    def test_not_ready_before_window_fills(self):
        channel = RollingHighLow(3)
        channel.update_candle(candles([(1, 5, 0, 3)])[0])
        self.assertFalse(channel.ready)
        self.assertIsNone(channel.highest)


if __name__ == "__main__":
    unittest.main()
