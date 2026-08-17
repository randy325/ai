"""Order lifecycle, precision, guards, kill switch and reconciliation.

These cover the robustness work rather than the strategy logic: what happens
when an order is rejected, partly filled, or submitted twice, and whether the
books still agree afterwards.
"""

import json
import logging
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.audit_log import AuditLogger, configure_audit_log
from trading_bot.broker import (
    Commission,
    OrderGuard,
    PaperBroker,
    ReconciliationError,
    SlippageModel,
)
from trading_bot.config import RunConfig
from trading_bot.instruments import CRYPTO, EQUITY, WHOLE_SHARE, InstrumentSpec, build_spec
from trading_bot.models import Candle, Fill, Order, OrderStatus, Side


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


def clean_broker(**kwargs):
    kwargs.setdefault("starting_cash", 10_000.0)
    kwargs.setdefault("commission", Commission(percent=0.0))
    kwargs.setdefault("slippage", SlippageModel(percent=0.0))
    return PaperBroker(**kwargs)


class TestInstrumentSpec(unittest.TestCase):
    def test_price_snaps_to_the_tick(self):
        spec = InstrumentSpec.create(tick_size="0.01")
        self.assertEqual(spec.round_price(100.017), Decimal("100.02"))
        self.assertEqual(spec.round_price(100.014), Decimal("100.01"))

    def test_quantity_rounds_down_to_the_lot(self):
        # Rounding up would turn an affordable order into an unaffordable one.
        spec = InstrumentSpec.create(lot_size="0.001")
        self.assertEqual(spec.round_quantity(1.23456), Decimal("1.234"))
        self.assertEqual(spec.round_quantity(0.0009), Decimal("0.000"))

    def test_negative_quantities_round_toward_zero(self):
        spec = InstrumentSpec.create(lot_size="0.001")
        self.assertEqual(spec.round_quantity(-1.23456), Decimal("-1.234"))

    def test_whole_share_spec_discards_fractions(self):
        self.assertEqual(WHOLE_SHARE.round_quantity(10.99), Decimal("10"))

    def test_decimal_conversion_avoids_binary_float_error(self):
        # Decimal(0.1) is 0.1000000000000000055...; via str it is exactly 0.1.
        spec = InstrumentSpec.create(tick_size="0.1")
        self.assertEqual(spec.round_price(0.3), Decimal("0.3"))

    def test_min_quantity_is_enforced(self):
        ok, reason = CRYPTO.is_tradable(Decimal("0.000001"), Decimal("50000"))
        self.assertFalse(ok)
        self.assertIn("below minimum", reason)

    def test_min_notional_is_enforced(self):
        ok, reason = CRYPTO.is_tradable(Decimal("0.0001"), Decimal("1"))
        self.assertFalse(ok)
        self.assertIn("notional", reason)

    def test_zero_quantity_is_not_tradable(self):
        ok, reason = EQUITY.is_tradable(Decimal("0"), Decimal("100"))
        self.assertFalse(ok)
        self.assertIn("rounds to zero", reason)

    def test_valid_order_passes(self):
        ok, reason = EQUITY.is_tradable(Decimal("10"), Decimal("100"))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_max_quantity_caps(self):
        spec = InstrumentSpec.create(lot_size="1", max_quantity="100")
        self.assertEqual(spec.round_quantity(500), Decimal("100"))

    def test_rejects_invalid_specs(self):
        with self.assertRaises(ValueError):
            InstrumentSpec.create(tick_size="0")
        with self.assertRaises(ValueError):
            InstrumentSpec.create(lot_size="0")

    def test_build_spec_by_name(self):
        self.assertEqual(build_spec("crypto").min_notional, CRYPTO.min_notional)
        with self.assertRaises(KeyError):
            build_spec("forex")


class TestPrecisionInTheBroker(unittest.TestCase):
    def test_fill_price_lands_on_a_tick(self):
        broker = clean_broker(slippage=SlippageModel(percent=0.0007))
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 10), bar)
        price = Decimal(str(result.fill.price))
        self.assertEqual(price % EQUITY.tick_size, 0, f"{price} is not a whole tick")

    def test_quantity_lands_on_a_lot(self):
        broker = clean_broker(spec=WHOLE_SHARE)
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 10.7), bar)
        self.assertEqual(result.filled_quantity, 10.0)
        self.assertEqual(
            result.status, OrderStatus.FILLED,
            "lot quantisation is not a partial fill: no venue could fill the 0.7",
        )

    def test_order_below_the_minimum_is_rejected(self):
        broker = clean_broker(spec=CRYPTO, starting_cash=1_000_000.0)
        bar = candle(50_000.0, symbol="BTC")
        broker.mark(bar)
        result = broker.submit(Order("BTC", Side.BUY, 0.0000001), bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)

    def test_sub_lot_order_rejected_rather_than_silently_zero(self):
        broker = clean_broker(spec=WHOLE_SHARE)
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 0.4), bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertEqual(len(broker.fills), 0)


class TestOrderIdentityAndIdempotency(unittest.TestCase):
    def test_orders_get_distinct_ids(self):
        first, second = Order("X", Side.BUY, 1), Order("X", Side.BUY, 1)
        self.assertNotEqual(first.client_order_id, second.client_order_id)

    def test_explicit_id_is_kept(self):
        self.assertEqual(Order("X", Side.BUY, 1, client_order_id="abc").client_order_id, "abc")

    def test_empty_id_is_rejected(self):
        with self.assertRaises(ValueError):
            Order("X", Side.BUY, 1, client_order_id="")

    def test_resubmitting_the_same_id_does_not_trade_twice(self):
        # This is the whole point of a client order ID: a retry after an
        # ambiguous failure must not open a second position.
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        order = Order("X", Side.BUY, 10, client_order_id="retry-me")

        first = broker.submit(order, bar)
        second = broker.submit(order, bar)

        self.assertEqual(first.status, OrderStatus.FILLED)
        self.assertEqual(second.status, OrderStatus.DUPLICATE)
        self.assertEqual(len(broker.fills), 1)
        self.assertAlmostEqual(broker.position("X").quantity, 10.0)

    def test_duplicate_reports_the_original_outcome(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        order = Order("X", Side.BUY, 10, client_order_id="dup")
        broker.submit(order, bar)
        second = broker.submit(order, bar)
        self.assertAlmostEqual(second.filled_quantity, 10.0)
        self.assertIn("already filled", second.reason)

    def test_a_rejected_id_is_also_remembered(self):
        broker = clean_broker(allow_short=False)
        bar = candle(100.0)
        broker.mark(bar)
        order = Order("X", Side.SELL, 5, client_order_id="nope")
        self.assertEqual(broker.submit(order, bar).status, OrderStatus.REJECTED)
        self.assertEqual(broker.submit(order, bar).status, OrderStatus.DUPLICATE)

    def test_fill_carries_the_order_id(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 5, client_order_id="tagged"), bar)
        self.assertEqual(result.fill.client_order_id, "tagged")


class TestPartialFills(unittest.TestCase):
    def test_a_trimmed_order_reports_as_partial(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 120), bar)
        self.assertEqual(result.status, OrderStatus.PARTIALLY_FILLED)
        self.assertAlmostEqual(result.filled_quantity, 100.0, places=6)
        self.assertAlmostEqual(result.unfilled_quantity, 20.0, places=6)
        self.assertFalse(result.complete)
        self.assertTrue(result.filled)

    def test_a_full_fill_reports_complete(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 10), bar)
        self.assertTrue(result.complete)
        self.assertAlmostEqual(result.unfilled_quantity, 0.0)

    def test_partial_fills_still_reconcile(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 120), bar)
        self.assertEqual(broker.reconcile(), [])


class TestOrderGuard(unittest.TestCase):
    def test_a_price_spike_is_rejected(self):
        broker = clean_broker(guard=OrderGuard(max_price_deviation=0.05))
        broker.mark(candle(100.0))
        spike = candle(200.0, day=2)
        result = broker.submit(Order("X", Side.BUY, 1), spike, reference_price=200.0)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("from last", result.reason)

    def test_a_price_inside_the_band_passes(self):
        broker = clean_broker(guard=OrderGuard(max_price_deviation=0.10))
        broker.mark(candle(100.0))
        result = broker.submit(Order("X", Side.BUY, 1), candle(104.0, day=2), reference_price=104.0)
        self.assertTrue(result.filled)

    def test_an_oversized_order_is_rejected_not_trimmed(self):
        # A sizing bug asking for 10x should be refused outright; trimming it
        # to a maximum position still leaves the bug fully invested.
        broker = clean_broker(guard=OrderGuard(max_order_fraction=1.5))
        bar = candle(100.0)
        broker.mark(bar)
        result = broker.submit(Order("X", Side.BUY, 1000), bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("exceeds", result.reason)
        self.assertEqual(len(broker.fills), 0)

    def test_max_open_positions_blocks_a_new_symbol(self):
        broker = clean_broker(starting_cash=1_000_000.0,
                              guard=OrderGuard(max_open_positions=2))
        for index, symbol in enumerate("ABC"):
            bar = candle(100.0, day=index + 1, symbol=symbol)
            broker.mark(bar)
            broker.submit(Order(symbol, Side.BUY, 10), bar)
        self.assertEqual(len([p for p in broker.positions.values() if not p.is_flat]), 2)
        self.assertTrue(any("limit 2" in reason for _, reason in broker.rejections))

    def test_adding_to_an_existing_position_is_not_a_new_position(self):
        broker = clean_broker(starting_cash=1_000_000.0,
                              guard=OrderGuard(max_open_positions=1))
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        second = broker.submit(Order("X", Side.BUY, 10), candle(100.0, day=2))
        self.assertTrue(second.filled)

    def test_guard_can_be_disabled(self):
        broker = clean_broker(guard=OrderGuard(enabled=False), max_leverage=1.0)
        broker.mark(candle(100.0))
        result = broker.submit(Order("X", Side.BUY, 1), candle(500.0, day=2), reference_price=500.0)
        self.assertTrue(result.filled, "a disabled guard must not block anything")


class TestKillSwitch(unittest.TestCase):
    def test_kill_flattens_and_halts(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 50), bar)

        broker.kill("operator stop", candle(101.0, day=2))
        self.assertTrue(broker.halted)
        self.assertTrue(broker.position("X").is_flat)

    def test_a_halted_broker_refuses_new_orders(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.kill("stop")
        result = broker.submit(Order("X", Side.BUY, 1), bar)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("halted", result.reason)

    def test_kill_without_a_candle_still_halts(self):
        broker = clean_broker()
        broker.kill("no price available")
        self.assertTrue(broker.halted)

    def test_kill_is_idempotent(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        broker.kill("first", candle(100.0, day=2))
        fills_after_first = len(broker.fills)
        broker.kill("second", candle(100.0, day=3))
        self.assertEqual(len(broker.fills), fills_after_first)

    def test_resume_is_explicit(self):
        broker = clean_broker()
        broker.kill("stop")
        broker.resume()
        self.assertFalse(broker.halted)
        bar = candle(100.0)
        broker.mark(bar)
        self.assertTrue(broker.submit(Order("X", Side.BUY, 1), bar).filled)

    def test_books_reconcile_after_a_kill(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 50), bar)
        broker.kill("stop", candle(105.0, day=2))
        self.assertEqual(broker.reconcile(), [])


class TestReconciliation(unittest.TestCase):
    def test_clean_books_report_nothing(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        self.assertEqual(broker.reconcile(), [])
        broker.assert_reconciled()

    def test_a_tampered_position_is_detected(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        broker.position("X").quantity = 999.0
        problems = broker.reconcile()
        self.assertTrue(any("fills imply" in p for p in problems))

    def test_a_tampered_cash_balance_is_detected(self):
        broker = clean_broker()
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        broker.cash += 5_000.0
        self.assertTrue(any("cash" in p for p in broker.reconcile()))

    def test_a_position_with_no_fills_is_detected(self):
        broker = clean_broker()
        broker.position("GHOST").quantity = 10.0
        self.assertTrue(any("no fills" in p for p in broker.reconcile()))

    def test_assert_reconciled_raises(self):
        broker = clean_broker()
        broker.position("GHOST").quantity = 10.0
        with self.assertRaises(ReconciliationError):
            broker.assert_reconciled()

    def test_commission_drift_is_detected(self):
        broker = PaperBroker(starting_cash=10_000.0, slippage=SlippageModel(percent=0.0))
        bar = candle(100.0)
        broker.mark(bar)
        broker.submit(Order("X", Side.BUY, 10), bar)
        broker.total_commission += 1.0
        self.assertTrue(any("commission" in p for p in broker.reconcile()))

    def test_engine_reconciles_at_start_and_end(self):
        config = RunConfig(strategy="sma-crossover", bars=200, cache_dir=None)
        engine = config.build_engine()
        result = engine.run(config.build_feed())
        self.assertEqual(engine.reconciliation_problems, [])
        self.assertFalse(any("reconciliation" in w for w in result.warnings))

    def test_engine_halts_when_periodic_reconciliation_fails(self):
        config = RunConfig(strategy="buy-and-hold", bars=200, cache_dir=None)
        engine = config.build_engine()
        engine.reconcile_every = 10
        candles = list(config.build_feed())

        # Corrupt the books mid-run, the way a partially applied recovery or a
        # bug in position tracking would.
        def corrupt(candle, signal, equity):
            if len(engine.equity_curve) == 25:
                engine.broker.cash += 1_000.0

        engine.on_bar = corrupt
        result = engine.run(candles)
        self.assertTrue(engine.broker.halted)
        self.assertTrue(any("reconciliation" in w for w in result.warnings))


class TestRejectionCircuitBreaker(unittest.TestCase):
    """A venue refusing everything must stop the bot, not be retried forever."""

    def build(self, **kwargs):
        config = RunConfig(strategy="buy-and-hold", bars=200, cache_dir=None,
                           starting_cash=10_000.0)
        engine = config.build_engine()
        for key, value in kwargs.items():
            setattr(engine, key, value)
        return engine, list(config.build_feed())

    def reject_everything(self, broker):
        original = broker.submit

        def rejecting(order, candle, reference_price=None):
            if order.side is Side.BUY and not broker.halted:
                return broker._reject(order, candle, "simulated venue rejection")
            return original(order, candle, reference_price)

        broker.submit = rejecting

    def test_repeated_rejections_trip_the_kill_switch(self):
        engine, candles = self.build(max_consecutive_rejections=3)
        self.reject_everything(engine.broker)
        result = engine.run(candles)
        self.assertTrue(engine.broker.halted)
        self.assertIn("consecutive rejections", engine.broker.halt_reason)
        self.assertTrue(any("circuit breaker" in w for w in result.warnings))

    def test_books_still_reconcile_after_the_breaker_trips(self):
        engine, candles = self.build(max_consecutive_rejections=3)
        self.reject_everything(engine.broker)
        engine.run(candles)
        self.assertEqual(engine.broker.reconcile(), [])

    def test_the_breaker_can_be_disabled(self):
        engine, candles = self.build(max_consecutive_rejections=0)
        self.reject_everything(engine.broker)
        engine.run(candles)
        self.assertFalse(engine.broker.halted)

    def test_a_successful_fill_resets_the_counter(self):
        engine, candles = self.build(max_consecutive_rejections=3)
        broker = engine.broker
        original = broker.submit
        state = {"n": 0}

        def sometimes(order, candle, reference_price=None):
            state["n"] += 1
            # Reject two, allow one, forever: never three in a row.
            if state["n"] % 3 != 0 and not broker.halted:
                return broker._reject(order, candle, "intermittent rejection")
            return original(order, candle, reference_price)

        broker.submit = sometimes
        engine.run(candles)
        self.assertFalse(broker.halted, "an intermittent rejection is not an outage")

    def test_a_healthy_run_never_trips_it(self):
        engine, candles = self.build(max_consecutive_rejections=3)
        engine.run(candles)
        self.assertFalse(engine.broker.halted)


class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "audit.jsonl"

    def tearDown(self):
        logger = logging.getLogger("trading_bot.audit")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        self._tmp.cleanup()

    def lines(self):
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def test_every_line_is_valid_json(self):
        audit = configure_audit_log(self.path)
        order = Order("X", Side.BUY, 1.5)
        audit.order_submitted(order)
        audit.fill(Fill(datetime(2020, 1, 1), "X", Side.BUY, 1.5, 100.0, 0.15))
        audit.error("boom", context="test")
        for entry in self.lines():
            self.assertIn("timestamp", entry)
            self.assertIn("event", entry)

    def test_events_are_named(self):
        audit = configure_audit_log(self.path)
        audit.order_submitted(Order("X", Side.BUY, 1))
        audit.kill_switch("stop")
        events = [e["event"] for e in self.lines()]
        self.assertIn("order.submitted", events)
        self.assertIn("kill_switch", events)

    def test_enums_serialise_as_values(self):
        audit = configure_audit_log(self.path)
        audit.order_submitted(Order("X", Side.BUY, 1))
        self.assertEqual(self.lines()[0]["fields"]["side"], "buy")

    def test_equity_only_logs_on_change(self):
        audit = configure_audit_log(self.path)
        for _ in range(3):
            audit.equity(datetime(2020, 1, 1), 10_000.0, 10_000.0)
        audit.equity(datetime(2020, 1, 2), 10_500.0, 10_500.0)
        equity_lines = [e for e in self.lines() if e["event"] == "equity"]
        self.assertEqual(len(equity_lines), 2)
        self.assertAlmostEqual(equity_lines[1]["fields"]["change"], 500.0)

    def test_configure_is_idempotent(self):
        configure_audit_log(self.path)
        configure_audit_log(self.path)
        AuditLogger(logging.getLogger("trading_bot.audit")).error("once")
        self.assertEqual(len(self.lines()), 1, "a second configure must not duplicate every line")

    def test_engine_writes_orders_and_fills(self):
        config = RunConfig(strategy="buy-and-hold", bars=120, cache_dir=None,
                           log_file=str(self.path))
        engine = config.build_engine()
        engine.run(config.build_feed())
        events = [e["event"] for e in self.lines()]
        self.assertIn("order.submitted", events)
        self.assertIn("fill", events)
        self.assertIn("reconciliation", events)


class TestModeGate(unittest.TestCase):
    def test_paper_is_the_default(self):
        self.assertEqual(RunConfig().mode, "paper")
        self.assertIsNotNone(RunConfig().build_broker())

    def test_live_mode_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            RunConfig(mode="live").build_broker()
        self.assertIn("does not implement", str(ctx.exception))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            RunConfig(mode="turbo").build_broker()


if __name__ == "__main__":
    unittest.main()
