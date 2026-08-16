import unittest
from datetime import datetime

from trading_bot.models import Candle, Order, OrderType, Position, Side, Signal


class TestCandle(unittest.TestCase):
    def test_rejects_inverted_high_low(self):
        with self.assertRaises(ValueError):
            Candle(datetime(2020, 1, 1), open=10, high=9, low=11, close=10)

    def test_rejects_close_outside_range(self):
        with self.assertRaises(ValueError):
            Candle(datetime(2020, 1, 1), open=10, high=11, low=9, close=12)

    def test_typical_price(self):
        candle = Candle(datetime(2020, 1, 1), open=10, high=12, low=9, close=11)
        self.assertAlmostEqual(candle.typical_price, (12 + 9 + 11) / 3)


class TestOrder(unittest.TestCase):
    def test_rejects_non_positive_quantity(self):
        with self.assertRaises(ValueError):
            Order("X", Side.BUY, 0)

    def test_limit_order_requires_price(self):
        with self.assertRaises(ValueError):
            Order("X", Side.BUY, 10, type=OrderType.LIMIT)


class TestSignal(unittest.TestCase):
    def test_rejects_out_of_range_target(self):
        with self.assertRaises(ValueError):
            Signal(1.5)


class TestPosition(unittest.TestCase):
    def test_opening_long_sets_basis(self):
        position = Position("X")
        realized = position.apply(Side.BUY, 100, 10.0)
        self.assertEqual(realized, 0.0)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.average_price, 10.0)

    def test_adding_averages_the_basis(self):
        position = Position("X")
        position.apply(Side.BUY, 100, 10.0)
        position.apply(Side.BUY, 100, 20.0)
        self.assertEqual(position.quantity, 200)
        self.assertAlmostEqual(position.average_price, 15.0)

    def test_partial_close_realizes_proportional_pnl(self):
        position = Position("X")
        position.apply(Side.BUY, 100, 10.0)
        realized = position.apply(Side.SELL, 40, 15.0)
        self.assertAlmostEqual(realized, 200.0)
        self.assertAlmostEqual(position.quantity, 60)
        self.assertAlmostEqual(position.average_price, 10.0, msg="basis must survive a partial exit")

    def test_full_close_flattens(self):
        position = Position("X")
        position.apply(Side.BUY, 100, 10.0)
        realized = position.apply(Side.SELL, 100, 12.0)
        self.assertAlmostEqual(realized, 200.0)
        self.assertTrue(position.is_flat)
        self.assertEqual(position.average_price, 0.0)

    def test_flip_closes_then_reopens_at_fill_price(self):
        position = Position("X")
        position.apply(Side.BUY, 100, 10.0)
        realized = position.apply(Side.SELL, 150, 12.0)
        self.assertAlmostEqual(realized, 200.0, msg="only the 100 long shares realize P&L")
        self.assertAlmostEqual(position.quantity, -50)
        self.assertAlmostEqual(position.average_price, 12.0)

    def test_short_position_profits_when_price_falls(self):
        position = Position("X")
        position.apply(Side.SELL, 100, 20.0)
        self.assertTrue(position.is_short)
        self.assertAlmostEqual(position.unrealized_pnl(15.0), 500.0)
        realized = position.apply(Side.BUY, 100, 15.0)
        self.assertAlmostEqual(realized, 500.0)

    def test_unrealized_pnl_tracks_price(self):
        position = Position("X")
        position.apply(Side.BUY, 10, 100.0)
        self.assertAlmostEqual(position.unrealized_pnl(110.0), 100.0)
        self.assertAlmostEqual(position.market_value(110.0), 1100.0)


if __name__ == "__main__":
    unittest.main()
