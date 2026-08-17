import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from trading_bot.cli import main
from trading_bot.config import RunConfig
from trading_bot.risk import FixedFractionSizer, VolatilitySizer


class TestRunConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_build_every_component(self):
        config = RunConfig()
        self.assertIsNotNone(config.build_strategy())
        self.assertIsNotNone(config.build_broker())
        self.assertIsNotNone(config.build_risk())
        self.assertIsNotNone(config.build_engine())
        self.assertIsNotNone(config.build_feed())

    def test_file_round_trip(self):
        original = RunConfig(strategy="breakout", bars=200, starting_cash=50_000.0)
        path = original.to_file(self.dir / "config.json")
        restored = RunConfig.from_file(path)
        self.assertEqual(restored.strategy, "breakout")
        self.assertEqual(restored.bars, 200)
        self.assertEqual(restored.starting_cash, 50_000.0)

    def test_unknown_keys_are_rejected(self):
        path = self.dir / "bad.json"
        path.write_text(json.dumps({"stratgey": "typo"}), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            RunConfig.from_file(path)
        self.assertIn("stratgey", str(ctx.exception))

    def test_sizer_selection(self):
        self.assertIsInstance(RunConfig(sizer="fixed").build_sizer(), FixedFractionSizer)
        self.assertIsInstance(RunConfig(sizer="volatility").build_sizer(), VolatilitySizer)
        with self.assertRaises(ValueError):
            RunConfig(sizer="magic").build_sizer()

    def test_allow_short_propagates_to_strategy_and_broker(self):
        config = RunConfig(strategy="sma-crossover", allow_short=True)
        self.assertIn("long/short", config.build_strategy().describe())
        self.assertTrue(config.build_broker().allow_short)

    def test_buy_and_hold_ignores_allow_short(self):
        # BuyAndHold takes no allow_short parameter; it must not be passed one.
        RunConfig(strategy="buy-and-hold", allow_short=True).build_strategy()

    def test_trend_filter_wraps_the_strategy(self):
        config = RunConfig(strategy="rsi-mean-reversion", trend_filter=50)
        self.assertIn("trend-filter", config.build_strategy().describe())

    def test_data_file_selects_a_csv_feed(self):
        path = self.dir / "p.csv"
        path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")
        feed = RunConfig(data_file=str(path)).build_feed()
        self.assertEqual(len(list(feed)), 1)

    def test_end_to_end_run_produces_metrics(self):
        config = RunConfig(strategy="sma-crossover", bars=300, seed=3)
        result = config.build_engine().run(config.build_feed())
        self.assertEqual(result.bars, 300)
        self.assertGreater(result.metrics.starting_equity, 0)
        self.assertIn("sma-crossover", result.summary())


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_strategies_command_lists_them(self):
        code, out, _ = self.run_cli(["strategies"])
        self.assertEqual(code, 0)
        self.assertIn("sma-crossover", out)
        self.assertIn("breakout", out)

    def test_backtest_prints_a_report(self):
        code, out, _ = self.run_cli(["backtest", "--bars", "200"])
        self.assertEqual(code, 0)
        self.assertIn("Total return", out)
        self.assertIn("Sharpe ratio", out)

    def test_backtest_json_is_parseable(self):
        code, out, _ = self.run_cli(["backtest", "--bars", "200", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("metrics", payload)
        self.assertIn("total_return", payload["metrics"])

    def test_backtest_show_trades(self):
        code, out, _ = self.run_cli(
            ["backtest", "--strategy", "macd-trend", "--bars", "400", "--show-trades", "5"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Trades", out)

    def test_json_stdout_stays_parseable_alongside_exports(self):
        # Export notices are status, not data; on stdout they corrupt --json.
        equity = self.dir / "e.csv"
        trades = self.dir / "t.csv"
        code, out, err = self.run_cli(
            ["backtest", "--bars", "300", "--strategy", "macd-trend", "--json",
             "--export-equity", str(equity), "--export-trades", str(trades)]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("metrics", payload)
        self.assertIn("written to", err)
        self.assertTrue(equity.exists() and trades.exists())

    def test_compare_ranks_strategies(self):
        code, out, _ = self.run_cli(["compare", "--bars", "300"])
        self.assertEqual(code, 0)
        self.assertIn("Ranked by Sharpe", out)
        self.assertIn("buy-and-hold", out)

    def test_compare_accepts_a_subset(self):
        code, out, _ = self.run_cli(
            ["compare", "--bars", "200", "--strategies", "buy-and-hold", "breakout"]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("macd-trend", out)

    def test_optimize_reports_a_grid(self):
        code, out, _ = self.run_cli(
            ["optimize", "--strategy", "sma-crossover", "--grid", "fast=5,10",
             "--grid", "slow=30,60", "--bars", "200"]
        )
        self.assertEqual(code, 0)
        self.assertIn("combinations", out)
        self.assertIn("overfit" if "overfit" in out else "out of sample", out)

    def test_optimize_skips_invalid_combinations(self):
        # fast=50 with slow=30 is invalid and must not abort the sweep.
        code, out, _ = self.run_cli(
            ["optimize", "--strategy", "sma-crossover", "--grid", "fast=5,50",
             "--grid", "slow=30", "--bars", "200"]
        )
        self.assertEqual(code, 0)

    def test_demo_data_writes_a_readable_file(self):
        path = self.dir / "demo.csv"
        code, out, _ = self.run_cli(["demo-data", "--out", str(path), "--bars", "50"])
        self.assertEqual(code, 0)
        self.assertTrue(path.exists())
        code, _, _ = self.run_cli(["backtest", "--data", str(path), "--bars", "50"])
        self.assertEqual(code, 0)

    def test_exports_produce_csv_files(self):
        equity = self.dir / "equity.csv"
        trades = self.dir / "trades.csv"
        code, _, _ = self.run_cli(
            ["backtest", "--strategy", "macd-trend", "--bars", "300",
             "--export-equity", str(equity), "--export-trades", str(trades)]
        )
        self.assertEqual(code, 0)
        self.assertTrue(equity.exists())
        self.assertTrue(trades.exists())
        self.assertIn("timestamp", equity.read_text(encoding="utf-8").splitlines()[0])

    def test_param_flag_reaches_the_strategy(self):
        code, out, _ = self.run_cli(
            ["backtest", "--strategy", "sma-crossover", "--param", "fast=5",
             "--param", "slow=15", "--bars", "200"]
        )
        self.assertEqual(code, 0)
        self.assertIn("fast=5", out)

    def test_malformed_param_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["backtest", "--param", "nonsense", "--bars", "100"])

    def test_missing_data_file_reports_an_error(self):
        code, _, err = self.run_cli(["backtest", "--data", str(self.dir / "nope.csv")])
        self.assertEqual(code, 2)
        self.assertIn("error", err)

    def test_config_file_is_honoured(self):
        config = RunConfig(strategy="breakout", bars=250)
        path = config.to_file(self.dir / "cfg.json")
        code, out, _ = self.run_cli(["backtest", "--config", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("breakout", out)

    def test_cli_flag_overrides_the_config_file(self):
        path = RunConfig(strategy="breakout", starting_cash=1_000.0).to_file(self.dir / "c.json")
        code, out, _ = self.run_cli(["backtest", "--config", str(path), "--cash", "250000"])
        self.assertEqual(code, 0)
        self.assertIn("250,000", out)

    def test_zero_max_drawdown_disables_the_halt(self):
        code, out, _ = self.run_cli(
            ["backtest", "--bars", "400", "--max-drawdown", "0", "--seed", "1"]
        )
        self.assertEqual(code, 0)
        self.assertNotIn("Trading halted", out)


if __name__ == "__main__":
    unittest.main()
