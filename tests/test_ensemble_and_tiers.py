"""The ensemble strategy, tiered risk, and the provider fallback chain."""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.config import RunConfig
from trading_bot.models import Candle, Signal
from trading_bot.providers import DataFeedError, FallbackProvider, Provider, SymbolNotFound
from trading_bot.risk import (
    AGGRESSIVE,
    CONSERVATIVE,
    MODERATE,
    RiskTier,
    TieredRiskManager,
)
from trading_bot.strategy import Ensemble, Strategy, build_strategy

REPO = Path(__file__).resolve().parent.parent


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


class Fixed(Strategy):
    """A member that always wants the same exposure."""

    def __init__(self, target, name="fixed", stop=None):
        self.target = target
        self.name = name
        self.stop = stop

    def on_candle(self, candle):
        return Signal(self.target, reason=self.name, stop_price=self.stop)


def candle(price=100.0, day=1):
    return Candle(datetime(2020, 1, day), price, price * 1.02, price * 0.98, price, symbol="X")


class TestEnsembleConstruction(unittest.TestCase):
    def test_builds_named_members(self):
        e = build_strategy("ensemble")
        self.assertEqual([m.name for m in e.members],
                         ["breakout", "rsi-breakout", "sma-crossover"])

    def test_member_params_are_forwarded(self):
        e = Ensemble(members=["sma-crossover"], member_params={"sma-crossover": {"fast": 5, "slow": 9}})
        self.assertIn("fast=5", e.members[0].describe())

    def test_rejects_bad_configuration(self):
        with self.assertRaises(ValueError):
            Ensemble(mode="telepathy")
        with self.assertRaises(ValueError):
            Ensemble(members=[])
        with self.assertRaises(ValueError):
            Ensemble(lookback=1)

    def test_describe_lists_members(self):
        self.assertIn("breakout", build_strategy("ensemble").describe())


class TestEnsembleBlending(unittest.TestCase):
    def test_equal_members_average_to_their_shared_target(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(1.0, "b")])
        signals = [e.on_candle(c) for c in series([100 + i for i in range(10)])]
        self.assertAlmostEqual(signals[-1].target, 1.0)

    def test_disagreement_lands_between_the_votes(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")])
        signals = [e.on_candle(c) for c in series([100] * 5)]
        self.assertGreater(signals[-1].target, 0.0)
        self.assertLess(signals[-1].target, 1.0)

    def test_target_is_clamped_to_the_legal_range(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(1.0, "b"), Fixed(1.0, "c")])
        for c in series([100 + i for i in range(20)]):
            signal = e.on_candle(c)
            self.assertLessEqual(abs(signal.target), 1.0)

    def test_a_winning_member_gains_weight(self):
        # "a" is long into a rising market; "b" sits flat and earns nothing.
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], lookback=50)
        for c in series([100 * 1.01**i for i in range(40)]):
            e.on_candle(c)
        weights = e.weights()
        self.assertGreater(weights[0], weights[1])
        self.assertEqual(e.best_member, "a")

    def test_a_losing_member_keeps_a_floor(self):
        # Zeroing a member out makes the ensemble a lagging copy of whatever
        # just worked, so a loser is down-weighted rather than dropped.
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(-1.0, "b")], lookback=50)
        for c in series([100 * 1.01**i for i in range(40)]):
            e.on_candle(c)
        self.assertGreater(e.weights()[1], 0.0)

    def test_equal_weights_before_any_history(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")])
        self.assertEqual(e.weights(), [0.5, 0.5])

    def test_scores_track_member_performance(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], lookback=50)
        for c in series([100 * 1.01**i for i in range(30)]):
            e.on_candle(c)
        scores = e.scores
        self.assertGreater(scores[0], 0)
        self.assertAlmostEqual(scores[1], 0.0)

    def test_reason_names_the_leader(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], lookback=50)
        signals = [e.on_candle(c) for c in series([100 * 1.01**i for i in range(30)])]
        self.assertIn("leader", signals[-1].reason)


class TestEnsembleModes(unittest.TestCase):
    def test_best_mode_follows_the_leader_outright(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], mode="best", lookback=50)
        signals = [e.on_candle(c) for c in series([100 * 1.01**i for i in range(30)])]
        self.assertAlmostEqual(signals[-1].target, 1.0)
        self.assertIn("a", signals[-1].reason)

    def test_unanimous_mode_trades_only_on_agreement(self):
        agree = Ensemble(members=[Fixed(1.0, "a"), Fixed(1.0, "b")], mode="unanimous")
        split = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], mode="unanimous")
        bars = series([100] * 5)
        self.assertAlmostEqual([agree.on_candle(c) for c in bars][-1].target, 1.0)
        self.assertAlmostEqual([split.on_candle(c) for c in bars][-1].target, 0.0)

    def test_unanimous_mode_explains_a_disagreement(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(0.0, "b")], mode="unanimous")
        self.assertIn("disagree", [e.on_candle(c) for c in series([100] * 3)][-1].reason)


class TestEnsembleStops(unittest.TestCase):
    def test_the_safest_stop_wins_for_a_long(self):
        # Sizing off the most optimistic stop would overstate position size.
        e = Ensemble(members=[Fixed(1.0, "a", stop=90.0), Fixed(1.0, "b", stop=95.0)])
        signal = [e.on_candle(c) for c in series([100] * 5)][-1]
        self.assertAlmostEqual(signal.stop_price, 95.0)

    def test_no_stop_when_members_supply_none(self):
        e = Ensemble(members=[Fixed(1.0, "a"), Fixed(1.0, "b")])
        self.assertIsNone([e.on_candle(c) for c in series([100] * 3)][-1].stop_price)


class TestRiskTiers(unittest.TestCase):
    def test_tier_produces_matching_limits(self):
        limits = RiskTier("x", 0.5, 0.2, risk_per_trade=0.01).to_limits()
        self.assertAlmostEqual(limits.max_position_fraction, 0.5)
        self.assertAlmostEqual(limits.max_drawdown, 0.2)

    def test_shipped_tiers_are_ordered_by_risk(self):
        self.assertGreater(AGGRESSIVE.max_position_fraction, CONSERVATIVE.max_position_fraction)
        self.assertGreater(AGGRESSIVE.max_drawdown, CONSERVATIVE.max_drawdown)
        self.assertGreater(AGGRESSIVE.risk_per_trade, MODERATE.risk_per_trade)
        self.assertGreater(MODERATE.risk_per_trade, CONSERVATIVE.risk_per_trade)


class TestTieredRiskManager(unittest.TestCase):
    def make(self, **kwargs):
        return TieredRiskManager(tier_threshold=10_000.0, **kwargs)

    def test_small_account_runs_aggressive(self):
        m = self.make()
        m.observe(candle(day=1), 5_000)
        self.assertEqual(m.tier.name, "aggressive")

    def test_crossing_the_threshold_switches_to_moderate(self):
        m = self.make()
        m.observe(candle(day=1), 5_000)
        m.observe(candle(day=2), 12_000)
        self.assertEqual(m.tier.name, "moderate")

    def test_exactly_at_the_threshold_is_moderate(self):
        m = self.make()
        m.observe(candle(day=1), 10_000)
        self.assertEqual(m.tier.name, "moderate")

    def test_a_losing_streak_demotes_to_conservative(self):
        m = self.make(losing_streak=3)
        m.observe(candle(day=1), 5_000)
        for _ in range(3):
            m.record_result(-50)
        m.observe(candle(day=2), 5_000)
        self.assertEqual(m.tier.name, "conservative")
        self.assertTrue(m.demoted)

    def test_a_shorter_streak_does_not_demote(self):
        m = self.make(losing_streak=3)
        m.observe(candle(day=1), 5_000)
        m.record_result(-50)
        m.record_result(-50)
        m.observe(candle(day=2), 5_000)
        self.assertEqual(m.tier.name, "aggressive")

    def test_a_win_resets_the_streak(self):
        m = self.make(losing_streak=3)
        m.observe(candle(day=1), 5_000)
        m.record_result(-50)
        m.record_result(-50)
        m.record_result(+10)
        m.record_result(-50)
        m.observe(candle(day=2), 5_000)
        self.assertFalse(m.demoted)

    def test_recovery_needs_several_wins(self):
        m = self.make(losing_streak=2, recovery_wins=2)
        m.observe(candle(day=1), 5_000)
        m.record_result(-50)
        m.record_result(-50)
        m.observe(candle(day=2), 5_000)
        self.assertTrue(m.demoted)

        m.record_result(+50)
        m.observe(candle(day=3), 5_000)
        self.assertTrue(m.demoted, "one win is not a recovery")

        m.record_result(+50)
        m.observe(candle(day=4), 5_000)
        self.assertFalse(m.demoted)
        self.assertEqual(m.tier.name, "aggressive")

    def test_demotion_overrides_account_size(self):
        m = self.make(losing_streak=2)
        m.observe(candle(day=1), 50_000)
        self.assertEqual(m.tier.name, "moderate")
        m.record_result(-50)
        m.record_result(-50)
        m.observe(candle(day=2), 50_000)
        self.assertEqual(m.tier.name, "conservative")

    def test_limits_actually_change_with_the_tier(self):
        m = self.make(losing_streak=1)
        m.observe(candle(day=1), 5_000)
        before = m.limits.max_position_fraction
        m.record_result(-50)
        m.observe(candle(day=2), 5_000)
        self.assertLess(m.limits.max_position_fraction, before)

    def test_a_halt_survives_a_tier_change(self):
        # The high-water mark and any halt belong to the account, not to the
        # posture it happens to be in.
        m = self.make(losing_streak=1)
        m.observe(candle(day=1), 20_000)
        m.observe(candle(day=2), 12_000)
        self.assertTrue(m.halted)
        m.record_result(-50)
        m.observe(candle(day=3), 12_000)
        self.assertTrue(m.halted)
        self.assertAlmostEqual(m.peak_equity, 20_000)

    def test_tier_changes_are_recorded(self):
        m = self.make()
        m.observe(candle(day=1), 5_000)
        m.observe(candle(day=2), 20_000)
        self.assertEqual(m.tier_changes, [("aggressive", "moderate")])

    def test_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            TieredRiskManager(tier_threshold=0)
        with self.assertRaises(ValueError):
            TieredRiskManager(losing_streak=0)


class TestTieredRiskInTheEngine(unittest.TestCase):
    def test_closed_trades_reach_the_risk_layer(self):
        config = RunConfig(
            strategy="ensemble", providers=["mock"], symbol="MOCK", interval="1m",
            limit=800, cache_dir=None, starting_cash=5_000.0, risk_profile="tiered",
        )
        engine = config.build_engine()
        engine.run(config.build_feed())
        # Something must have driven the tier machinery over 800 bars.
        self.assertIsInstance(engine.risk, TieredRiskManager)
        self.assertGreater(len(engine.broker.trades), 0)


class TestFallbackProvider(unittest.TestCase):
    def make(self, *behaviours):
        providers = []
        for index, behaviour in enumerate(behaviours):
            class Member(Provider):
                name = f"p{index}"
                intervals = ("1d", "1w")

                def __init__(self, behaviour=behaviour):
                    super().__init__()
                    self.behaviour = behaviour
                    self.symbols_seen = []

                def fetch(self, symbol, interval="1d", limit=500):
                    self.symbols_seen.append(symbol)
                    if isinstance(self.behaviour, Exception):
                        raise self.behaviour
                    close = self.behaviour
                    return [
                        Candle(datetime(2024, 1, 1, tzinfo=timezone.utc),
                               close, close * 1.01, close * 0.99, close, symbol=symbol)
                    ]

            providers.append(Member())
        return providers

    def test_first_working_provider_wins(self):
        members = self.make(100.0, 200.0)
        chain = FallbackProvider(members)
        self.assertAlmostEqual(chain.fetch("X")[0].close, 100.0)
        self.assertEqual(chain.used, "p0")
        self.assertEqual(members[1].symbols_seen, [], "the second must not be called")

    def test_a_failure_falls_through(self):
        members = self.make(SymbolNotFound("down"), 200.0)
        chain = FallbackProvider(members)
        self.assertAlmostEqual(chain.fetch("X")[0].close, 200.0)
        self.assertEqual(chain.used, "p1")

    def test_all_failing_reports_every_reason(self):
        chain = FallbackProvider(self.make(SymbolNotFound("a down"), SymbolNotFound("b down")))
        with self.assertRaises(DataFeedError) as ctx:
            chain.fetch("X")
        self.assertIn("a down", str(ctx.exception))
        self.assertIn("b down", str(ctx.exception))

    def test_symbol_overrides_are_applied_per_provider(self):
        members = self.make(SymbolNotFound("down"), 200.0)
        chain = FallbackProvider(members, {"p0": "aapl.us", "p1": "AAPL"})
        chain.fetch("ignored")
        self.assertEqual(members[0].symbols_seen, ["aapl.us"])
        self.assertEqual(members[1].symbols_seen, ["AAPL"])

    def test_intervals_are_the_intersection(self):
        members = self.make(100.0, 200.0)
        members[1].intervals = ("1d",)
        self.assertEqual(FallbackProvider(members).intervals, ("1d",))

    def test_a_member_lacking_the_interval_is_skipped(self):
        members = self.make(100.0, 200.0)
        members[0].intervals = ("1w",)
        chain = FallbackProvider(members)
        self.assertAlmostEqual(chain.fetch("X", "1d")[0].close, 200.0)

    def test_empty_chain_is_rejected(self):
        with self.assertRaises(ValueError):
            FallbackProvider([])

    def test_describe_shows_the_order(self):
        self.assertEqual(FallbackProvider(self.make(1.0, 2.0)).describe(), "p0 -> p1")


class TestShippedConfigs(unittest.TestCase):
    def test_every_shipped_config_loads_and_builds(self):
        paths = sorted((REPO / "configs").glob("*.json"))
        self.assertTrue(paths, "no configs found")
        for path in paths:
            config = RunConfig.from_file(path)
            self.assertIsNotNone(config.build_strategy(), path.name)
            self.assertIsNotNone(config.build_risk(), path.name)
            self.assertIsNotNone(config.build_broker(), path.name)

    def test_preferred_config_encodes_the_stated_preferences(self):
        config = RunConfig.from_file(REPO / "configs" / "preferred.json")
        self.assertEqual(config.strategy, "ensemble")
        self.assertEqual(len(config.strategy_params["members"]), 3)
        self.assertGreater(len(config.providers), 1, "more than one data source")
        self.assertEqual(config.risk_profile, "tiered")
        self.assertAlmostEqual(config.tier_threshold, 10_000.0)

    def test_preferred_config_interval_is_supported_by_the_whole_chain(self):
        config = RunConfig.from_file(REPO / "configs" / "preferred.json")
        self.assertIn(config.interval, config.build_provider().intervals)

    def test_mock_config_runs_offline(self):
        config = RunConfig.from_file(REPO / "configs" / "preferred-mock.json")
        result = config.build_engine().run(config.build_feed())
        self.assertGreater(result.bars, 0)

    def test_configs_are_valid_json_with_known_keys(self):
        known = set(RunConfig.__dataclass_fields__)
        for path in sorted((REPO / "configs").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data) - known, set(), path.name)


if __name__ == "__main__":
    unittest.main()
