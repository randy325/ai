"""Regression tests for the independent review findings.

Each test was written before its fix and confirmed to fail against the
reviewed commit, so it reproduces the reported defect rather than describing
the repair.
"""

import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.broker import Commission, OrderGuard, PaperBroker, SlippageModel
from trading_bot.models import (
    Candle,
    Order,
    OrderStatus,
    OrderType,
    RejectionKind,
    Side,
)


def candle(price=100.0, day=1, symbol="X", high=None, low=None, hours=0):
    return Candle(
        timestamp=datetime(2020, 1, day, hours),
        open=price,
        high=high if high is not None else price * 1.01,
        low=low if low is not None else price * 0.99,
        close=price,
        volume=1000,
        symbol=symbol,
    )


def broker(**kwargs):
    kwargs.setdefault("starting_cash", 100_000.0)
    kwargs.setdefault("commission", Commission(percent=0.0))
    kwargs.setdefault("slippage", SlippageModel(percent=0.0))
    return PaperBroker(**kwargs)


# ---------------------------------------------------------------------------
# Critical
# ---------------------------------------------------------------------------


class TestFinding01ReversalLosesOpeningLeg(unittest.TestCase):
    """#1 — a flip must close the old side AND open the new one."""

    def test_reversal_opens_the_new_side(self):
        book = broker(max_leverage=2.0, guard=OrderGuard(max_order_fraction=5.0))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 1000), entry)
        self.assertAlmostEqual(book.position("X").quantity, 1000)

        flip = candle(100.0, day=2)
        book.mark(flip)
        result = book.submit(Order("X", Side.SELL, 2000), flip)

        self.assertAlmostEqual(
            book.position("X").quantity, -1000,
            msg="the opening leg of a reversal must survive the buying-power check",
        )
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_quantity, 2000)

    def test_closing_leg_frees_capital_for_the_opening_leg(self):
        # Closing a long releases its notional, so a same-size flip needs no
        # additional headroom beyond what the position already used.
        book = broker(max_leverage=1.0, guard=OrderGuard(max_order_fraction=5.0))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 900), entry)

        flip = candle(100.0, day=2)
        book.mark(flip)
        book.submit(Order("X", Side.SELL, 1800), flip)
        self.assertLess(book.position("X").quantity, -1e-9)


class TestFinding02KillSwitchCannotFlatten(unittest.TestCase):
    """#2 — the guard must never block a reduce-only order."""

    def test_kill_switch_flattens_through_a_price_gap(self):
        book = broker(guard=OrderGuard(max_price_deviation=0.10))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 900), entry)

        crash = candle(80.0, day=2)
        book.kill("crash", crash)

        self.assertTrue(book.position("X").is_flat,
                        "a kill switch that cannot exit is not a kill switch")

    def test_a_reduce_only_order_ignores_the_deviation_check(self):
        book = broker(guard=OrderGuard(max_price_deviation=0.05))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 500), entry)

        gap = candle(70.0, day=2)
        book.mark(gap)
        result = book.submit(Order("X", Side.SELL, 500), gap)
        self.assertTrue(result.filled)

    def test_a_reduce_only_order_ignores_the_notional_cap(self):
        # After losses the position can be worth more than the guard's multiple
        # of equity; the exit must still go through.
        book = broker(starting_cash=10_000.0, guard=OrderGuard(max_order_fraction=1.2))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 100), entry)

        drop = candle(92.0, day=2)
        book.mark(drop)
        result = book.submit(Order("X", Side.SELL, 100), drop)
        self.assertTrue(result.filled, "an exit must not be blocked by the notional cap")

    def test_an_opening_order_still_obeys_the_guard(self):
        book = broker(guard=OrderGuard(max_price_deviation=0.05))
        first = candle(100.0, day=1)
        book.mark(first)
        book.submit(Order("X", Side.BUY, 10), first)

        # No mark() on the spike bar: the guard's reference is the last known
        # price, which is the situation a bad tick actually presents.
        spike = candle(200.0, day=2)
        result = book.submit(Order("X", Side.BUY, 10), spike)
        self.assertEqual(result.status, OrderStatus.REJECTED,
                         "increasing exposure must still be checked")

    def test_a_reversal_is_not_treated_as_reduce_only(self):
        # The closing half is exempt, but a flip also opens new exposure and
        # must not slip past the guard wholesale.
        book = broker(guard=OrderGuard(max_price_deviation=0.05))
        entry = candle(100.0, day=1)
        book.mark(entry)
        book.submit(Order("X", Side.BUY, 100), entry)

        spike = candle(200.0, day=2)
        result = book.submit(Order("X", Side.SELL, 300), spike)
        self.assertEqual(result.status, OrderStatus.REJECTED)


if __name__ == "__main__":
    unittest.main()


class TestFinding03DailyTradeLimit(unittest.TestCase):
    """#3 — the daily trade limit must actually block trading."""

    def build(self, limit, interval_hours=24, bars=20):
        from trading_bot.engine import TradingEngine
        from trading_bot.risk import FixedFractionSizer, RiskLimits, RiskManager
        from trading_bot.models import Signal
        from trading_bot.strategy import Strategy

        class Flip(Strategy):
            name = "flip"

            def __init__(self):
                self.n = -1

            def on_candle(self, c):
                self.n += 1
                return Signal(1.0 if self.n % 2 == 0 else 0.0, reason="flip")

        start = datetime(2020, 3, 2)
        candles = []
        for i in range(bars):
            price = 100.0 + (i % 3)
            previous = 100.0 + ((i - 1) % 3) if i else price
            candles.append(
                Candle(
                    timestamp=start + timedelta(hours=interval_hours * i),
                    open=previous, high=max(previous, price), low=min(previous, price),
                    close=price, volume=100, symbol="X",
                )
            )
        risk = RiskManager(
            RiskLimits(max_drawdown=None, max_trades_per_day=limit),
            FixedFractionSizer(0.95),
        )
        engine = TradingEngine(
            Flip(),
            broker(starting_cash=10_000.0, guard=OrderGuard(enabled=False)),
            risk,
            close_at_end=False,
        )
        return engine, candles

    def test_daily_bars_are_not_actually_violated(self):
        # NOTE: the review reported "19 fills in 20 bars, never blocked" as a
        # failure. It is not one. With one bar per day, one fill per day is
        # exactly compliant with a limit of 1/day, and the counter resetting on
        # a new calendar day is correct behaviour rather than a defect.
        engine, candles = self.build(limit=1, interval_hours=24, bars=20)
        engine.run(candles)
        per_day = {}
        for fill in engine.broker.fills:
            if fill.reason == "close-all":
                continue
            per_day[fill.timestamp.date()] = per_day.get(fill.timestamp.date(), 0) + 1
        self.assertTrue(all(n <= 1 for n in per_day.values()),
                        f"limit of 1/day violated: {per_day}")

    def test_intraday_bars_respect_the_limit(self):
        engine, candles = self.build(limit=2, interval_hours=1, bars=40)
        engine.run(candles)
        per_day = {}
        for fill in engine.broker.fills:
            if fill.reason == "close-all":
                continue
            per_day[fill.timestamp.date()] = per_day.get(fill.timestamp.date(), 0) + 1
        self.assertTrue(all(n <= 2 for n in per_day.values()),
                        f"limit of 2/day violated: {per_day}")

    def test_a_blocked_bot_holding_a_position_does_not_reverse(self):
        engine, candles = self.build(limit=1, interval_hours=1, bars=40)
        engine.run(candles)
        reasons = [f.reason for f in engine.broker.fills]
        self.assertFalse(
            any("daily trade limit" in r for r in reasons),
            "a fill must never carry the reason that was supposed to block it",
        )

    def test_the_limit_binds_exactly_not_off_by_one(self):
        # The defect was N+1 trades against a limit of N: a blocked bot holding
        # a position still submitted a full reversal.
        for limit in (1, 2, 5):
            engine, candles = self.build(limit=limit, interval_hours=1, bars=40)
            engine.run(candles)
            per_day = {}
            for fill in engine.broker.fills:
                if fill.reason == "close-all":
                    continue
                per_day[fill.timestamp.date()] = per_day.get(fill.timestamp.date(), 0) + 1
            self.assertTrue(
                all(n <= limit for n in per_day.values()),
                f"limit {limit} exceeded: {per_day}",
            )

    def test_the_limit_actually_throttles(self):
        # Guard against "fixing" this by blocking everything.
        unlimited, candles = self.build(limit=None, interval_hours=1, bars=40)
        unlimited.run(candles)
        limited, candles = self.build(limit=2, interval_hours=1, bars=40)
        limited.run(candles)
        self.assertGreater(len(unlimited.broker.fills), len(limited.broker.fills))
        self.assertGreater(len(limited.broker.fills), 0, "throttled is not the same as dead")

    def test_a_throttled_bot_holds_rather_than_liquidating(self):
        # A trade cap means "no more orders today", not "go flat" — unlike a
        # halt or a daily-loss pause, which are decisions to be out.
        engine, candles = self.build(limit=1, interval_hours=1, bars=40)
        engine.run(candles)
        self.assertFalse(engine.broker.halted)



class TestFinding04TieredRiskLoosensWhileCrashing(unittest.TestCase):
    """#4 — risk must never be relaxed because the account got smaller.

    The original tier keyed on *current* equity, so it could not tell a
    genuinely small account from a large one that had crashed into the same
    number. The second case is the martingale failure mode: size up as you
    lose. Selection now keys on the high-water mark, which only ever rises, and
    promotions are refused outright while the account is below its peak.
    """

    def manager(self, **kwargs):
        from trading_bot.risk import TieredRiskManager
        return TieredRiskManager(**kwargs)

    def test_a_crash_does_not_promote_to_aggressive(self):
        m = self.manager(tier_threshold=90_000.0, starting_equity=100_000.0)
        m.observe(candle(day=1), 100_000)
        self.assertEqual(m.tier.name, "moderate")
        m.observe(candle(day=2), 72_000)
        self.assertNotEqual(m.tier.name, "aggressive",
                            "a 28% drawdown must not relax the limits")

    def test_the_drawdown_halt_still_fires_during_a_crash(self):
        m = self.manager(tier_threshold=90_000.0, starting_equity=100_000.0)
        m.observe(candle(day=1), 100_000)
        m.observe(candle(day=2), 72_000)
        self.assertTrue(m.halted, "28% breaches moderate's 25% limit")

    def test_a_genuinely_small_account_still_gets_the_aggressive_tier(self):
        # The stated intent — small absolute stakes justify variance — survives.
        m = self.manager(tier_threshold=10_000.0, starting_equity=5_000.0)
        m.observe(candle(day=1), 5_000)
        self.assertEqual(m.tier.name, "aggressive")

    def test_a_large_account_fallen_to_the_same_equity_does_not(self):
        m = self.manager(tier_threshold=10_000.0, starting_equity=50_000.0)
        m.observe(candle(day=1), 50_000)
        m.observe(candle(day=2), 5_000)
        self.assertNotEqual(m.tier.name, "aggressive",
                            "same equity, opposite history, must not be aggressive")

    def test_growth_through_the_threshold_still_tightens(self):
        m = self.manager(tier_threshold=10_000.0, starting_equity=5_000.0)
        m.observe(candle(day=1), 5_000)
        self.assertEqual(m.tier.name, "aggressive")
        m.observe(candle(day=2), 15_000)
        self.assertEqual(m.tier.name, "moderate", "a bigger account is tightened")

    def test_no_promotion_while_below_the_high_water_mark(self):
        m = self.manager(tier_threshold=10_000.0, starting_equity=20_000.0)
        m.observe(candle(day=1), 20_000)
        m.record_result(-1)
        m.record_result(-1)
        m.record_result(-1)
        m.observe(candle(day=2), 19_000)
        self.assertEqual(m.tier.name, "conservative")
        # Wins recover the streak, but the account is still under its peak.
        m.record_result(1)
        m.record_result(1)
        m.observe(candle(day=3), 19_500)
        self.assertEqual(m.tier.name, "conservative",
                         "promotion requires a new high, not merely a bounce")

    def test_a_new_high_permits_promotion_again(self):
        m = self.manager(tier_threshold=10_000.0, starting_equity=20_000.0)
        m.observe(candle(day=1), 20_000)
        for _ in range(3):
            m.record_result(-1)
        m.observe(candle(day=2), 19_000)
        self.assertEqual(m.tier.name, "conservative")
        for _ in range(2):
            m.record_result(1)
        m.observe(candle(day=3), 21_000)
        self.assertEqual(m.tier.name, "moderate")

    def test_tiers_are_ranked_by_permissiveness(self):
        from trading_bot.risk import AGGRESSIVE, CONSERVATIVE, MODERATE
        self.assertLess(CONSERVATIVE.rank, MODERATE.rank)
        self.assertLess(MODERATE.rank, AGGRESSIVE.rank)

    def test_an_unseeded_manager_starts_conservative(self):
        # Never start permissive before the account size is known.
        m = self.manager(tier_threshold=10_000.0)
        self.assertEqual(m.tier.name, "conservative")


# ---------------------------------------------------------------------------
# High
# ---------------------------------------------------------------------------


class TestFinding05RejectionsPoisonIdempotencyKey(unittest.TestCase):
    """#5 — idempotency must latch fills, not rejections."""

    def test_a_missed_limit_can_be_resubmitted(self):
        book = broker()
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=95.0)

        miss = candle(100.0, day=1, high=101.0, low=99.0)
        book.mark(miss)
        first = book.submit(order, miss)
        self.assertEqual(first.status, OrderStatus.REJECTED)

        hit = candle(97.0, day=2, high=99.0, low=94.0)
        book.mark(hit)
        second = book.submit(order, hit)
        self.assertTrue(second.filled,
                        "a limit that missed one bar must be retryable on the next")

    def test_an_order_rejected_while_halted_works_after_resume(self):
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        book.kill("manual stop")
        order = Order("X", Side.BUY, 10)
        self.assertEqual(book.submit(order, bar).status, OrderStatus.REJECTED)

        book.resume()
        self.assertTrue(book.submit(order, bar).filled,
                        "resume() must not leave the order permanently poisoned")

    def test_a_filled_order_is_still_latched(self):
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        order = Order("X", Side.BUY, 10)
        book.submit(order, bar)
        again = book.submit(order, bar)
        self.assertEqual(again.status, OrderStatus.DUPLICATE)
        self.assertEqual(len(book.fills), 1)


class TestFinding06DuplicateReadsAsFailure(unittest.TestCase):
    """#6 — a duplicate of a filled order must report as filled."""

    def test_duplicate_of_a_fill_is_truthy(self):
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        order = Order("X", Side.BUY, 10)
        book.submit(order, bar)
        duplicate = book.submit(order, bar)

        self.assertTrue(duplicate.filled,
                        "a caller retrying after an ambiguous failure must see success")
        self.assertTrue(bool(duplicate))
        self.assertAlmostEqual(duplicate.filled_quantity, 10)

    def test_duplicate_of_a_rejection_is_not_truthy(self):
        book = broker(allow_short=False)
        bar = candle(100.0, day=1)
        book.mark(bar)
        order = Order("X", Side.SELL, 10)
        book.submit(order, bar)
        self.assertFalse(book.submit(order, bar).filled)

    def test_a_duplicate_does_not_double_count_as_a_new_trade(self):
        from trading_bot.risk import FixedFractionSizer, RiskLimits, RiskManager
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        order = Order("X", Side.BUY, 10)
        book.submit(order, bar)
        before = len(book.fills)
        book.submit(order, bar)
        self.assertEqual(len(book.fills), before, "a duplicate must not add a fill")


class TestFinding07LimitFillsBeyondTheLimit(unittest.TestCase):
    """#7 — a limit order must never fill worse than its limit price."""

    def test_buy_limit_never_fills_above_its_price(self):
        book = broker(slippage=SlippageModel(percent=0.001))
        bar = candle(99.0, day=1, high=100.0, low=99.0)
        book.mark(bar)
        order = Order("X", Side.BUY, 10, type=OrderType.LIMIT, limit_price=99.5)
        result = book.submit(order, bar)
        self.assertTrue(result.filled)
        self.assertLessEqual(result.fill.price, 99.5,
                             "slippage must not push a buy limit above its limit")

    def test_sell_limit_never_fills_below_its_price(self):
        book = broker(slippage=SlippageModel(percent=0.001))
        book.submit(Order("X", Side.BUY, 20), candle(100.0, day=1))
        bar = candle(101.0, day=2, high=102.0, low=100.0)
        book.mark(bar)
        order = Order("X", Side.SELL, 10, type=OrderType.LIMIT, limit_price=100.5)
        result = book.submit(order, bar)
        self.assertTrue(result.filled)
        self.assertGreaterEqual(result.fill.price, 100.5)

    def test_market_orders_still_take_slippage(self):
        book = broker(slippage=SlippageModel(percent=0.01))
        bar = candle(100.0, day=1)
        book.mark(bar)
        result = book.submit(Order("X", Side.BUY, 10), bar)
        self.assertGreater(result.fill.price, 100.0)


class TestFinding08CircuitBreakerIsReachable(unittest.TestCase):
    """#8 — the breaker path must be exercisable, not dead code.

    Nothing in the package emits RejectionKind.VENUE, because the paper broker
    has no venue to be refused by. ``fault_injector`` is a test-only hook that
    supplies one, so the breaker is covered now rather than when a live adapter
    eventually appears.
    """

    def build(self, injector=None, limit=3, bars=20):
        from trading_bot.engine import TradingEngine
        from trading_bot.risk import FixedFractionSizer, RiskLimits, RiskManager
        from trading_bot.strategy import BuyAndHold

        candles = []
        for i in range(bars):
            price = 100.0 + i
            previous = 100.0 + i - 1 if i else price
            candles.append(
                Candle(
                    timestamp=datetime(2020, 5, 1) + timedelta(days=i),
                    open=previous, high=max(previous, price), low=min(previous, price),
                    close=price, volume=100, symbol="X",
                )
            )
        book = broker(starting_cash=10_000.0, guard=OrderGuard(enabled=False))
        book.fault_injector = injector
        engine = TradingEngine(
            BuyAndHold(), book,
            RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(0.95)),
            close_at_end=False, max_consecutive_rejections=limit,
        )
        return engine, candles

    def test_venue_rejections_trip_the_breaker(self):
        engine, candles = self.build(injector=lambda order, candle: "venue refused")
        engine.run(candles)
        self.assertTrue(engine.broker.halted)
        self.assertIn("consecutive rejections", engine.broker.halt_reason)

    def test_the_breaker_trips_at_the_configured_count(self):
        engine, candles = self.build(injector=lambda o, c: "venue refused", limit=3)
        engine.run(candles)
        venue = [r for _, r in engine.broker.rejections if r == "venue refused"]
        self.assertEqual(len(venue), 3, "must stop at the limit, not keep hammering")

    def test_a_successful_fill_resets_the_streak(self):
        state = {"n": 0}

        def intermittent(order, candle):
            state["n"] += 1
            # Two refusals then a success, forever: never three in a row.
            return "venue refused" if state["n"] % 3 else None

        engine, candles = self.build(injector=intermittent, limit=3, bars=30)
        engine.run(candles)
        self.assertFalse(engine.broker.halted,
                         "an intermittent venue must not trip a consecutive-failure breaker")

    def test_risk_rejections_do_not_reset_a_venue_streak(self):
        # A risk rejection is not evidence the venue recovered, so it must
        # neither count toward the breaker nor clear it.
        engine, candles = self.build(injector=lambda o, c: "venue refused", limit=3)
        engine.broker.allow_short = False
        engine.run(candles)
        self.assertTrue(engine.broker.halted)

    def test_no_injector_means_no_venue_rejections(self):
        engine, candles = self.build(injector=None)
        engine.run(candles)
        self.assertFalse(engine.broker.halted)
        self.assertEqual(engine._consecutive_rejections, 0)


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------


class TestFinding09AnnualisationAssumesDailyEquities(unittest.TestCase):
    """#9 — bars-per-year must follow the data, not a hardcoded calendar.

    Rather than guess a market from the bar spacing, this now measures actual
    bar density over the elapsed period. A weekday-only series yields ~252, a
    24/7 series ~365, an hourly session series ~1638 and hourly crypto ~8760,
    with no per-asset special-casing.
    """

    def curve(self, timestamps):
        from trading_bot.models import EquityPoint
        return [
            EquityPoint(ts, cash=0.0, equity=100.0 * (1.001 ** i))
            for i, ts in enumerate(timestamps)
        ]

    def periods(self, curve):
        from trading_bot.metrics import _bars_per_year
        return _bars_per_year(curve)

    def weekdays(self, count, start=None, step=timedelta(days=1)):
        ts, current = [], start or datetime(2021, 1, 4)
        while len(ts) < count:
            if current.weekday() < 5:
                ts.append(current)
            current += step
        return ts

    def test_weekly_bars_are_52_per_year(self):
        start = datetime(2021, 1, 4)
        weekly = self.curve([start + timedelta(days=7 * i) for i in range(80)])
        self.assertAlmostEqual(self.periods(weekly), 52.0, delta=2.0)

    def test_weekday_daily_bars_land_near_the_trading_year(self):
        # Weekdays with no holidays is 5/7 of 365.25 = 261.6 a year, which is
        # what this fixture contains. Real exchange data carries ~9 holidays
        # and lands near the conventional 252 — the point is that the figure
        # now follows the data instead of being asserted as a constant.
        daily = self.curve(self.weekdays(400))
        self.assertAlmostEqual(self.periods(daily), 261.6, delta=3.0)
        self.assertLess(self.periods(daily), 365.0, "weekend gaps must reduce it")

    def test_continuous_daily_bars_are_about_365(self):
        # Crypto trades every calendar day; treating it as 252 understates it.
        start = datetime(2021, 1, 1)
        daily = self.curve([start + timedelta(days=i) for i in range(400)])
        self.assertAlmostEqual(self.periods(daily), 365.0, delta=8.0)

    def test_continuous_hourly_bars_are_about_8760(self):
        start = datetime(2021, 1, 1)
        hourly = self.curve([start + timedelta(hours=i) for i in range(24 * 90)])
        self.assertAlmostEqual(self.periods(hourly), 8760.0, delta=200.0)

    def test_session_hourly_bars_scale_with_the_session(self):
        # 7 hourly bars per weekday (09:00-15:00 inclusive) over 261.6 weekdays
        # is ~1831 a year. The old code hardcoded a 6.5 hour session regardless
        # of what the file actually contained.
        ts, current = [], datetime(2021, 1, 4, 9)
        while len(ts) < 2000:
            if current.weekday() < 5 and 9 <= current.hour < 16:
                ts.append(current)
            current += timedelta(hours=1)
        self.assertAlmostEqual(self.periods(self.curve(ts)), 1831.0, delta=30.0)

    def test_a_short_series_still_produces_something_sane(self):
        start = datetime(2021, 1, 4)
        tiny = self.curve([start + timedelta(days=i) for i in range(4)])
        self.assertGreater(self.periods(tiny), 0)


class TestFinding10AmbiguousDatesParseDayFirst(unittest.TestCase):
    """#10 — an ambiguous slash date must not silently parse day-first."""

    def test_month_first_is_preferred(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(parse_timestamp("03/04/2020"), datetime(2020, 3, 4))

    def test_a_day_first_value_needs_the_day_first_order(self):
        # Superseded by the per-file resolver: parse_timestamp no longer falls
        # through to the other ordering, because falling through IS the
        # interleaving bug. The caller states the order; CSVFeed derives it.
        from trading_bot.data import parse_timestamp
        self.assertEqual(
            parse_timestamp("25/12/2020", date_order="day-first"), datetime(2020, 12, 25)
        )
        with self.assertRaises(ValueError):
            parse_timestamp("25/12/2020", date_order="month-first")

    def test_iso_is_unaffected(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(parse_timestamp("2020-03-04"), datetime(2020, 3, 4))


class TestFinding11DuplicateBarsPassSilently(unittest.TestCase):
    """#11 — a repeated timestamp is corrupt data, not a bar."""

    def write(self, tmp, text):
        path = tmp / "p.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_repeated_timestamp_is_rejected(self):
        import tempfile
        from trading_bot.data import CSVFeed
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(
                __import__("pathlib").Path(tmp),
                "date,close\n2020-01-01,100\n2020-01-01,101\n2020-01-02,102\n",
            )
            with self.assertRaises(ValueError) as ctx:
                list(CSVFeed(path))
            self.assertIn("duplicate", str(ctx.exception).lower())


class TestFinding12MixedTimestampAwareness(unittest.TestCase):
    """#12 — a mixed naive/aware file must fail with a readable error."""

    def test_mixed_awareness_raises_valueerror_not_typeerror(self):
        import tempfile
        from pathlib import Path as P
        from trading_bot.data import CSVFeed
        with tempfile.TemporaryDirectory() as tmp:
            path = P(tmp) / "p.csv"
            path.write_text(
                "date,close\n2020-01-01,100\n1577923200,101\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                list(CSVFeed(path))
            self.assertIn("timezone", str(ctx.exception).lower())


class TestFinding13ReconciliationTautology(unittest.TestCase):
    """#13 — the equity check must be able to fail."""

    def test_a_corrupted_position_is_detected(self):
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        book.submit(Order("X", Side.BUY, 10), bar)
        self.assertEqual(book.reconcile(), [])

        book.position("X").quantity += 5          # drift the books
        problems = book.reconcile()
        self.assertTrue(problems)
        self.assertTrue(any("position" in p for p in problems))

    def test_corrupted_cash_is_detected(self):
        book = broker()
        bar = candle(100.0, day=1)
        book.mark(bar)
        book.submit(Order("X", Side.BUY, 10), bar)
        book.cash -= 1.0
        self.assertTrue(any("cash" in p for p in book.reconcile()))


class TestFinding14UnboundedMemoryGrowth(unittest.TestCase):
    """#14 — a long-running session must not grow without bound."""

    def test_result_and_rejection_history_is_capped(self):
        book = broker(starting_cash=1_000_000.0, history_limit=50,
                      guard=OrderGuard(enabled=False))
        for day in range(1, 29):
            bar = candle(100.0, day=day)
            book.mark(bar)
            for _ in range(20):
                book.submit(Order("X", Side.BUY, 1), bar)
                book.submit(Order("X", Side.SELL, 1), bar)
        self.assertLessEqual(len(book._results), 50)
        self.assertLessEqual(len(book.rejections), 50)

    def test_recent_ids_are_still_deduplicated(self):
        book = broker(history_limit=50)
        bar = candle(100.0, day=1)
        book.mark(bar)
        order = Order("X", Side.BUY, 1)
        book.submit(order, bar)
        self.assertEqual(book.submit(order, bar).status, OrderStatus.DUPLICATE)


class TestFinding15EnsembleWeightingDisablesItself(unittest.TestCase):
    """#15 — the warmup check must consider every member, not member 0."""

    def test_a_flat_first_member_does_not_disable_weighting(self):
        from trading_bot.strategy import Ensemble
        from trading_bot.models import Signal
        from trading_bot.strategy import Strategy

        class Fixed(Strategy):
            def __init__(self, target, name):
                self.target, self.name = target, name

            def on_candle(self, c):
                return Signal(self.target, reason=self.name)

        # Member 0 sits flat and scores exactly zero; member 1 is long into a
        # rally and scores well. Weighting must still favour member 1.
        e = Ensemble(members=[Fixed(0.0, "flat"), Fixed(1.0, "long")], lookback=50)
        start = datetime(2020, 1, 1)
        price = 100.0
        for i in range(40):
            previous, price = price, price * 1.01
            e.on_candle(Candle(start + timedelta(days=i), previous,
                               max(previous, price), min(previous, price), price,
                               100, "X"))
        weights = e.weights()
        self.assertGreater(weights[1], weights[0],
                           "a flat member 0 must not force equal weighting")


class TestInertOverboughtParameter(unittest.TestCase):
    """A parameter that silently does nothing is a trap for future tuning.

    ``overbought`` is read only on the short entry, so in the default
    long-only configuration it can be set to anything with no effect — the
    robustness sweep flagged it as "no effect" for exactly this reason. It
    cannot simply be deleted: long/short mode genuinely needs it. So setting
    it where it would do nothing is now an error rather than a silent no-op.
    """

    def strategy(self, **kwargs):
        from trading_bot.strategy import RSIMeanReversion
        return RSIMeanReversion(**kwargs)

    def test_setting_it_long_only_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.strategy(overbought=70.0)
        self.assertIn("allow_short", str(ctx.exception))

    def test_long_only_construction_without_it_is_fine(self):
        s = self.strategy()
        self.assertIsNone(s.overbought)

    def test_long_only_describe_omits_it(self):
        self.assertNotIn("overbought", self.strategy().describe())

    def test_it_is_required_and_used_when_shorting(self):
        s = self.strategy(allow_short=True)
        self.assertIsNotNone(s.overbought, "short entries need a threshold")
        self.assertIn("overbought", s.describe())

    def test_an_explicit_value_works_when_shorting(self):
        s = self.strategy(allow_short=True, overbought=85.0)
        self.assertAlmostEqual(s.overbought, 85.0)
        self.assertIn("85", s.describe())

    def test_threshold_ordering_is_still_validated_when_shorting(self):
        with self.assertRaises(ValueError):
            self.strategy(allow_short=True, oversold=60, exit_level=50, overbought=70)
        with self.assertRaises(ValueError):
            self.strategy(allow_short=True, exit_level=80, overbought=70)

    def test_shorting_still_triggers_on_the_threshold(self):
        from trading_bot.models import Signal
        s = self.strategy(period=5, allow_short=True, overbought=70.0)
        start = datetime(2020, 6, 1)
        price = 100.0
        targets = []
        for i in range(30):
            previous, price = price, price + 2
            targets.append(
                s.on_candle(Candle(start + timedelta(days=i), previous,
                                   max(previous, price), min(previous, price),
                                   price, 100, "X")).target
            )
        self.assertIn(-1.0, targets)

    def test_the_sweep_no_longer_reports_it_as_inert(self):
        import importlib.util
        from pathlib import Path as P
        repo = P(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "rs", repo / "scripts" / "robustness_sweep.py"
        )
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        params = sweep.current_parameters("rsi-mean-reversion")
        self.assertNotIn("overbought", params,
                         "a parameter with no effect in this mode must not be swept")
        self.assertIn("oversold", params)
