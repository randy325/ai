import unittest
from datetime import datetime

from trading_bot.broker import Commission, PaperBroker, SlippageModel
from trading_bot.models import Candle, Order, OrderStatus, OrderType, Side


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
        result = self.broker.submit(Order("X", Side.BUY, 50), bar)
        self.assertTrue(result.complete)
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
        result = broker.submit(Order("X", Side.SELL, 10), bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("shorting disabled", result.reason)
        self.assertEqual(len(broker.rejections), 1)

    def test_order_is_trimmed_to_available_buying_power(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        # 120 shares is within the fat-finger guard but beyond buying power:
        # 10k of equity at 1x leverage buys 100 shares at $100.
        result = self.broker.submit(Order("X", Side.BUY, 120), bar)
        self.assertAlmostEqual(result.filled_quantity, 100.0, places=6)
        self.assertEqual(result.status, OrderStatus.PARTIALLY_FILLED,
                         "a trimmed order is a partial fill, not a complete one")
        self.assertAlmostEqual(result.unfilled_quantity, 20.0, places=6)
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
        # 200 shares at $100 is exactly the 2x-leverage notional the guard now
        # matches to max_leverage; buying power caps it there too.
        result = broker.submit(Order("X", Side.BUY, 200), bar)
        self.assertAlmostEqual(result.filled_quantity, 200.0, places=6)
        self.assertAlmostEqual(broker.exposure(), 2.0, places=6)

    def test_exit_is_allowed_even_with_no_buying_power(self):
        bar = candle(100.0)
        self.broker.mark(bar)
        self.broker.submit(Order("X", Side.BUY, 100), bar)
        self.assertAlmostEqual(self.broker.cash, 0.0, places=6)

        exit_bar = candle(100.0, day=2)
        self.broker.mark(exit_bar)
        result = self.broker.submit(Order("X", Side.SELL, 100), exit_bar)
        self.assertTrue(result.filled, "closing a position must never be blocked on buying power")
        self.assertAlmostEqual(result.filled_quantity, 100.0)

    def test_limit_buy_does_not_fill_above_the_bar(self):
        bar = candle(100.0, high=101.0, low=99.0)
        self.broker.mark(bar)
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=95.0)
        result = self.broker.submit(order, bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("limit not reached", result.reason)

    def test_limit_buy_fills_when_the_bar_trades_through(self):
        bar = candle(100.0, high=101.0, low=94.0)
        self.broker.mark(bar)
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=95.0)
        result = self.broker.submit(order, bar)
        self.assertTrue(result.filled)
        self.assertAlmostEqual(result.fill.price, 95.0)

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
        result = self.broker.submit(Order("X", Side.BUY, 10), bar, reference_price=95.0)
        self.assertAlmostEqual(result.fill.price, 95.0)

    def test_default_broker_charges_costs(self):
        # A frictionless default would silently flatter every strategy that
        # forgets to set one.
        broker = PaperBroker(starting_cash=10_000.0)
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 10), bar)
        self.assertGreater(result.fill.commission, 0)
        self.assertGreater(result.fill.price, 100.0, "buys must fill above the reference price")

    def test_guard_is_widened_to_match_configured_leverage(self):
        # The guard's default 1.5x cap would reject every order on an account
        # deliberately configured for 3x, which reads as a dead strategy
        # rather than the misconfiguration it actually is.
        broker = PaperBroker(
            starting_cash=10_000.0,
            commission=Commission(percent=0.0),
            slippage=SlippageModel(percent=0.0),
            max_leverage=3.0,
        )
        self.assertGreaterEqual(broker.guard.max_order_fraction, 3.0)
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 300), bar)
        self.assertAlmostEqual(result.filled_quantity, 300.0, places=6)

    def test_rejects_invalid_construction(self):
        with self.assertRaises(ValueError):
            PaperBroker(starting_cash=0)
        with self.assertRaises(ValueError):
            PaperBroker(max_leverage=0.5)


if __name__ == "__main__":
    unittest.main()
