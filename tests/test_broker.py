import unittest
from datetime import datetime

from trading_bot.broker import Commission, PaperBroker, SlippageModel
from trading_bot.models import Candle, Order, OrderType, Side


def candle(price=100.0, day=1, symbol="X", high=None, low=None):
    return Candle(
        timestamp=datetime(2020, 1, day),
        open=price,
        high=high if high is not None else price * 1.01,
        low=low if low is not None else price * 0.99,
        close=price,
        volume=1000,
        symbol=symbol,
    )


class TestCommission(unittest.TestCase):
    def test_percent_of_notional(self):
        self.assertAlmostEqual(Commission(percent=0.001).charge(100, 50.0), 5.0)

    def test_minimum_applies(self):
        self.assertAlmostEqual(Commission(percent=0.001, minimum=1.0).charge(1, 10.0), 1.0)

    def test_per_share(self):
        self.assertAlmostEqual(Commission(per_share=0.005).charge(200, 10.0), 1.0)


class TestSlippage(unittest.TestCase):
    def test_buys_fill_above_and_sells_below(self):
        model = SlippageModel(percent=0.01)
        self.assertAlmostEqual(model.fill_price(Side.BUY, 100.0), 101.0)
        self.assertAlmostEqual(model.fill_price(Side.SELL, 100.0), 99.0)

    def test_spread_component_scales_with_bar_range(self):
        model = SlippageModel(percent=0.0, spread_fraction=0.5)
        bar = candle(100.0, high=110.0, low=90.0)
        self.assertAlmostEqual(model.fill_price(Side.BUY, 100.0, bar), 110.0)


class TestPaperBroker(unittest.TestCase):
    def setUp(self):
        self.broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.0),
            slippage=SlippageModel(percent=0.0),
        )

    def test_buy_moves_cash_into_position(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        fill = self.broker.submit(Order("X", Side.BUY, 50), bar)
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(self.broker.cash, 5_000.0)
        self.assertAlmostEqual(self.broker.position("X").quantity, 50)
        self.assertAlmostEqual(self.broker.equity(), 10_000.0)

    def test_commission_reduces_equity(self):
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.01),
            slippage=SlippageModel(percent=0.0),
        )
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 50), bar)
        self.assertAlmostEqual(broker.total_commission, 50.0)
        self.assertAlmostEqual(broker.equity(), 9_950.0)

    def test_round_trip_records_a_trade(self):
        entry = candle(100.0, day=1)
        self.broker.mark(entry)
        self.broker.submit(Order("X", Side.BUY, 50), entry)

        exit_bar = candle(120.0, day=2)
        self.broker.mark(exit_bar)
        self.broker.submit(Order("X", Side.SELL, 50), exit_bar)

        self.assertEqual(len(self.broker.trades), 1)
        trade = self.broker.trades[0]
        self.assertAlmostEqual(trade.pnl, 1_000.0)
        self.assertTrue(trade.is_win)
        self.assertAlmostEqual(trade.return_pct, 0.2)
        self.assertTrue(self.broker.position("X").is_flat)
        self.assertAlmostEqual(self.broker.cash, 11_000.0)

    def test_partial_exit_leaves_the_rest_open(self):
        entry = candle(100.0, day=1)
        self.broker.mark(entry)
        self.broker.submit(Order("X", Side.BUY, 50), entry)

        exit_bar = candle(110.0, day=2)
        self.broker.mark(exit_bar)
        self.broker.submit(Order("X", Side.SELL, 20), exit_bar)

        self.assertEqual(len(self.broker.trades), 1)
        self.assertAlmostEqual(self.broker.trades[0].pnl, 200.0)
        self.assertAlmostEqual(self.broker.position("X").quantity, 30)

    def test_scaling_in_averages_the_entry_of_the_recorded_trade(self):
        first = candle(100.0, day=1)
        self.broker.mark(first)
        self.broker.submit(Order("X", Side.BUY, 10), first)

        second = candle(200.0, day=2)
        self.broker.mark(second)
        self.broker.submit(Order("X", Side.BUY, 10), second)

        third = candle(200.0, day=3)
        self.broker.mark(third)
        self.broker.submit(Order("X", Side.SELL, 20), third)

        self.assertEqual(len(self.broker.trades), 1)
        self.assertAlmostEqual(self.broker.trades[0].entry_price, 150.0)
        self.assertAlmostEqual(self.broker.trades[0].pnl, 1_000.0)

    def test_losing_trade_is_not_a_win(self):
        entry = candle(100.0, day=1)
        self.broker.mark(entry)
        self.broker.submit(Order("X", Side.BUY, 10), entry)
        exit_bar = candle(90.0, day=2)
        self.broker.mark(exit_bar)
        self.broker.submit(Order("X", Side.SELL, 10), exit_bar)
        self.assertFalse(self.broker.trades[0].is_win)
        self.assertAlmostEqual(self.broker.trades[0].pnl, -100.0)

    def test_short_sale_credits_cash_and_profits_on_a_fall(self):
        entry = candle(100.0, day=1)
        self.broker.mark(entry)
        self.broker.submit(Order("X", Side.SELL, 20), entry)
        self.assertAlmostEqual(self.broker.cash, 12_000.0)
        self.assertTrue(self.broker.position("X").is_short)

        cover = candle(80.0, day=2)
        self.broker.mark(cover)
        self.broker.submit(Order("X", Side.BUY, 20), cover)
        self.assertAlmostEqual(self.broker.trades[0].pnl, 400.0)
        self.assertAlmostEqual(self.broker.equity(), 10_400.0)

    def test_shorting_rejected_when_disabled(self):
        broker = PaperBroker(starting_cash=10_000.0, allow_short=False)
        bar = candle(100.0)
        broker.mark(bar)
        self.assertIsNone(broker.submit(Order("X", Side.SELL, 10), bar))
        self.assertEqual(len(broker.rejections), 1)

    def test_order_is_trimmed_to_available_buying_power(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        fill = self.broker.submit(Order("X", Side.BUY, 500), bar)
        # 10k of equity at 1x leverage buys 100 shares at $100, not 500.
        self.assertAlmostEqual(fill.quantity, 100.0, places=6)
        self.assertGreaterEqual(self.broker.cash, -1e-6)

    def test_leverage_cap_is_respected(self):
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(),
            slippage=SlippageModel(percent=0.0),
            max_leverage=2.0,
        )
        bar = candle(100.0)
        broker.mark(bar)
        fill = broker.submit(Order("X", Side.BUY, 1000), bar)
        self.assertAlmostEqual(fill.quantity, 200.0, places=6)
        self.assertAlmostEqual(broker.exposure(), 2.0, places=6)

    def test_exit_is_allowed_even_with_no_buying_power(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        self.broker.submit(Order("X", Side.BUY, 100), bar)
        self.assertAlmostEqual(self.broker.cash, 0.0, places=6)

        exit_bar = candle(100.0, day=2)
        self.broker.mark(exit_bar)
        fill = self.broker.submit(Order("X", Side.SELL, 100), exit_bar)
        self.assertIsNotNone(fill, "closing a position must never be blocked on buying power")
        self.assertAlmostEqual(fill.quantity, 100.0)

    def test_limit_buy_does_not_fill_above_the_bar(self):
        bar = candle(100.0, high=101.0, low=99.0)
        self.broker.mark(bar)
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=95.0)
        self.assertIsNone(self.broker.submit(order, bar))

    def test_limit_buy_fills_when_the_bar_trades_through(self):
        bar = candle(100.0, high=101.0, low=94.0)
        self.broker.mark(bar)
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=95.0)
        fill = self.broker.submit(order, bar)
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.price, 95.0)

    def test_close_all_flattens_every_position(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        self.broker.submit(Order("X", Side.BUY, 20), bar)
        self.broker.close_all(candle(105.0, day=2))
        self.assertTrue(self.broker.position("X").is_flat)
        self.assertEqual(len(self.broker.trades), 1)

    def test_reference_price_overrides_the_close(self):
        bar = candle(100.0, high=120.0, low=80.0)
        self.broker.mark(bar)
        fill = self.broker.submit(Order("X", Side.BUY, 10), bar, reference_price=85.0)
        self.assertAlmostEqual(fill.price, 85.0)

    def test_default_broker_charges_costs(self):
        # A frictionless default would silently flatter every strategy that
        # forgets to set one.
        broker = PaperBroker(starting_cash=10_000.0)
        bar = candle(100.0)
        broker.mark(bar)
        fill = broker.submit(Order("X", Side.BUY, 10), bar)
        self.assertGreater(fill.commission, 0)
        self.assertGreater(fill.price, 100.0, "buys must fill above the reference price")

    def test_rejects_invalid_construction(self):
        with self.assertRaises(ValueError):
            PaperBroker(starting_cash=0)
        with self.assertRaises(ValueError):
            PaperBroker(max_leverage=0.5)


if __name__ == "__main__":
    unittest.main()
