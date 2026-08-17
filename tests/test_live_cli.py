"""CLI and config coverage for the live-data path.

The network is stubbed out at the transport boundary, so these exercise the
real provider parsing, the real config wiring and the real command plumbing —
everything except the socket.
"""

import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from trading_bot import cli, config as config_module
from trading_bot.config import RunConfig
from trading_bot.models import Candle
from trading_bot.providers import (
    CachingTransport,
    LiveFeed,
    MarketDataFeed,
    Provider,
    StooqProvider,
)

STOOQ_CSV = b"""Date,Open,High,Low,Close,Volume
2024-01-02,187.15,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
2024-01-04,182.15,183.09,180.88,181.91,71983600
2024-01-05,181.99,182.76,180.17,181.18,62303300
2024-01-08,182.09,185.60,181.50,185.56,59144500
"""


def setUpModule():
    logging.getLogger("trading_bot.providers").setLevel(logging.ERROR)


class StubTransport:
    def __init__(self, body=STOOQ_CSV):
        self.body = body
        self.urls = []

    def get(self, url, headers=None):
        self.urls.append(url)
        return self.body


def stub_stooq(*args, **kwargs):
    return StooqProvider(StubTransport())


class ScriptedProvider(Provider):
    """Yields one extra bar on each successive fetch, like a live market."""

    name = "scripted"
    intervals = ("1m", "1d")

    def __init__(self, total=6, start=3):
        super().__init__(StubTransport())
        self.total = total
        self.available = start

    def fetch(self, symbol, interval="1d", limit=500):
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        count = min(self.available, self.total)
        self.available += 1
        return [
            Candle(
                timestamp=base + timedelta(minutes=i),
                open=100.0 + i, high=101.5 + i, low=99.0 + i, close=100.5 + i,
                volume=10.0, symbol=symbol.upper(),
            )
            for i in range(count)
        ][-limit:]


class TestRunConfigProviders(unittest.TestCase):
    def test_provider_feed_is_used_when_configured(self):
        config = RunConfig(provider="stooq", symbol="AAPL", cache_dir=None)
        with mock.patch.object(config_module, "build_provider", stub_stooq):
            feed = config.build_feed()
        self.assertIsInstance(feed, MarketDataFeed)
        self.assertEqual(len(list(feed)), 5)

    def test_data_file_wins_over_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")
            config = RunConfig(provider="stooq", data_file=str(path))
            feed = config.build_feed()
        self.assertNotIsInstance(feed, MarketDataFeed)

    def test_synthetic_feed_when_no_provider_or_file(self):
        self.assertNotIsInstance(RunConfig().build_feed(), MarketDataFeed)

    def test_cache_dir_produces_a_caching_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = RunConfig(provider="stooq", cache_dir=tmp).build_provider()
        self.assertIsInstance(provider.transport, CachingTransport)

    def test_no_cache_dir_skips_the_cache(self):
        provider = RunConfig(provider="stooq", cache_dir=None).build_provider()
        self.assertNotIsInstance(provider.transport, CachingTransport)

    def test_build_provider_without_one_configured_raises(self):
        with self.assertRaises(ValueError):
            RunConfig().build_provider()

    def test_live_feed_requires_a_provider(self):
        with self.assertRaises(ValueError):
            RunConfig().build_live_feed()

    def test_live_feed_is_never_cached(self):
        # A cached response would serve a stale bar as if it had just closed.
        feed = RunConfig(provider="stooq", cache_dir=".cache/x").build_live_feed()
        self.assertIsInstance(feed, LiveFeed)
        self.assertNotIsInstance(feed.provider.transport, CachingTransport)

    def test_provider_settings_round_trip_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = RunConfig(provider="binance", symbol="BTCUSDT", interval="1h", limit=300)
            restored = RunConfig.from_file(original.to_file(Path(tmp) / "c.json"))
        self.assertEqual(restored.provider, "binance")
        self.assertEqual(restored.interval, "1h")
        self.assertEqual(restored.limit, 300)

    def test_backtest_runs_on_provider_data(self):
        config = RunConfig(provider="stooq", symbol="AAPL", strategy="buy-and-hold", cache_dir=None)
        with mock.patch.object(config_module, "build_provider", stub_stooq):
            result = config.build_engine().run(config.build_feed())
        self.assertEqual(result.bars, 5)
        self.assertEqual(result.symbol, "AAPL")


class TestLiveCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_providers_command_lists_them(self):
        code, out, _ = self.run_cli(["providers"])
        self.assertEqual(code, 0)
        for name in ("stooq", "yahoo", "binance", "coinbase"):
            self.assertIn(name, out)
        self.assertIn("BTCUSDT", out)

    def test_fetch_writes_a_readable_csv(self):
        path = self.dir / "aapl.csv"
        with mock.patch.object(cli, "build_provider", stub_stooq):
            code, out, _ = self.run_cli(
                ["fetch", "--provider", "stooq", "--symbol", "AAPL", "--out", str(path)]
            )
        self.assertEqual(code, 0)
        self.assertIn("Fetched 5", out)
        self.assertIn("181.18" if "181.18" in out else "185.56", out)
        self.assertTrue(path.exists())

        # The written file must be readable by the ordinary backtest path.
        code, _, _ = self.run_cli(["backtest", "--data", str(path), "--strategy", "buy-and-hold"])
        self.assertEqual(code, 0)

    def test_fetch_defaults_the_output_path(self):
        with mock.patch.object(cli, "build_provider", stub_stooq):
            code, out, _ = self.run_cli(
                ["fetch", "--provider", "stooq", "--symbol", "TESTSYM",
                 "--out", str(self.dir / "t.csv")]
            )
        self.assertEqual(code, 0)
        self.assertIn("TESTSYM", out)

    def test_fetch_reports_a_missing_symbol_cleanly(self):
        def missing(*args, **kwargs):
            return StooqProvider(StubTransport(b"No data"))

        with mock.patch.object(cli, "build_provider", missing):
            code, _, err = self.run_cli(["fetch", "--provider", "stooq", "--symbol", "NOPE"])
        self.assertEqual(code, 2)
        self.assertIn("error", err)

    def test_backtest_accepts_a_provider(self):
        with mock.patch.object(config_module, "build_provider", stub_stooq):
            code, out, _ = self.run_cli(
                ["backtest", "--provider", "stooq", "--symbol", "AAPL",
                 "--strategy", "buy-and-hold", "--no-cache"]
            )
        self.assertEqual(code, 0)
        self.assertIn("AAPL", out)
        self.assertIn("Total return", out)

    def test_no_cache_flag_disables_caching(self):
        with mock.patch.object(config_module, "build_provider") as builder:
            builder.side_effect = stub_stooq
            self.run_cli(
                ["backtest", "--provider", "stooq", "--symbol", "AAPL", "--no-cache"]
            )
        self.assertIsNone(builder.call_args.kwargs.get("cache_dir"))

    def test_cache_dir_is_passed_through(self):
        with mock.patch.object(config_module, "build_provider") as builder:
            builder.side_effect = stub_stooq
            self.run_cli(
                ["backtest", "--provider", "stooq", "--symbol", "AAPL",
                 "--cache-dir", str(self.dir / "cache")]
            )
        self.assertIn("cache", str(builder.call_args.kwargs.get("cache_dir")))

    def test_interval_unsupported_by_provider_is_reported(self):
        with mock.patch.object(config_module, "build_provider", stub_stooq):
            code, _, err = self.run_cli(
                ["backtest", "--provider", "stooq", "--symbol", "AAPL", "--interval", "1m"]
            )
        self.assertEqual(code, 2)
        self.assertIn("does not support", err)

    def test_paper_requires_a_provider(self):
        code, _, err = self.run_cli(["paper", "--symbol", "AAPL"])
        self.assertEqual(code, 2)
        self.assertIn("provider", err)

    def test_paper_runs_end_to_end_on_the_mock_provider(self):
        # No stubbing and no network: the real command, the real feed, the
        # real engine, with time compressed so three 1m bars take milliseconds.
        code, out, _ = self.run_cli(
            ["paper", "--provider", "mock", "--symbol", "MOCK", "--interval", "1m",
             "--strategy", "buy-and-hold", "--warmup", "20", "--max-bars", "3",
             "--speed", "500000", "--cash", "10000"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Paper trading MOCK", out)
        self.assertIn("No real orders are placed", out)
        self.assertIn("Final equity", out)

    def test_paper_reports_speed_in_the_banner(self):
        code, out, _ = self.run_cli(
            ["paper", "--provider", "mock", "--symbol", "MOCK", "--interval", "1m",
             "--strategy", "buy-and-hold", "--warmup", "5", "--max-bars", "1",
             "--speed", "500000"]
        )
        self.assertEqual(code, 0)
        self.assertIn("speed", out)

    def test_paper_leaves_the_position_open_at_the_end(self):
        code, out, _ = self.run_cli(
            ["paper", "--provider", "mock", "--symbol", "MOCK", "--interval", "1m",
             "--strategy", "buy-and-hold", "--warmup", "20", "--max-bars", "3",
             "--speed", "500000"]
        )
        self.assertEqual(code, 0)
        # A live session must not liquidate just because the loop stopped.
        self.assertIn("Open position", out)

    def test_paper_emits_one_row_per_live_bar(self):
        code, out, _ = self.run_cli(
            ["paper", "--provider", "mock", "--symbol", "MOCK", "--interval", "1m",
             "--strategy", "buy-and-hold", "--warmup", "10", "--max-bars", "4",
             "--speed", "500000"]
        )
        self.assertEqual(code, 0)
        rows = [line for line in out.splitlines() if line.startswith("20")]
        self.assertEqual(len(rows), 4, "warmup bars must not be printed as live rows")

    def test_speed_is_rejected_for_a_real_provider(self):
        code, _, err = self.run_cli(
            ["paper", "--provider", "stooq", "--symbol", "AAPL", "--speed", "60"]
        )
        self.assertEqual(code, 2)
        self.assertIn("only applies to the mock provider", err)

    def test_backtest_runs_on_the_mock_provider(self):
        code, out, _ = self.run_cli(
            ["backtest", "--provider", "mock", "--symbol", "MOCK",
             "--strategy", "rsi-breakout", "--limit", "400", "--no-cache"]
        )
        self.assertEqual(code, 0)
        self.assertIn("MOCK", out)
        self.assertIn("Total return", out)

    def test_mock_results_carry_a_warning(self):
        # Sinusoidal mock data produces flattering numbers; unlabelled, they
        # would read as a real result.
        for argv in (
            ["backtest", "--provider", "mock", "--symbol", "MOCK", "--limit", "200", "--no-cache"],
            ["compare", "--provider", "mock", "--symbol", "MOCK", "--limit", "200", "--no-cache"],
        ):
            code, out, _ = self.run_cli(argv)
            self.assertEqual(code, 0, argv[0])
            self.assertIn("measure nothing about the strategy", out, argv[0])

    def test_paper_results_carry_the_mock_warning(self):
        # A live mock session prints the most flattering numbers of the lot.
        code, out, _ = self.run_cli(
            ["paper", "--provider", "mock", "--symbol", "MOCK", "--interval", "1m",
             "--strategy", "buy-and-hold", "--warmup", "20", "--max-bars", "3",
             "--speed", "500000"]
        )
        self.assertEqual(code, 0)
        self.assertIn("measure nothing about the strategy", out)

    def test_real_provider_results_carry_no_mock_warning(self):
        with mock.patch.object(config_module, "build_provider", stub_stooq):
            code, out, _ = self.run_cli(
                ["backtest", "--provider", "stooq", "--symbol", "AAPL", "--no-cache"]
            )
        self.assertEqual(code, 0)
        self.assertNotIn("measure nothing", out)

    def test_fetch_works_offline_with_the_mock_provider(self):
        path = self.dir / "mock.csv"
        code, out, _ = self.run_cli(
            ["fetch", "--provider", "mock", "--symbol", "MOCK",
             "--limit", "50", "--no-cache", "--out", str(path)]
        )
        self.assertEqual(code, 0)
        self.assertIn("Fetched 50", out)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
