"""Provider tests.

Every response body here mirrors the real API's documented shape, including
the awkward parts: Yahoo's null padding, Coinbase's low/high-before-open column
order and descending rows, Binance's still-forming final kline, and Stooq's
habit of reporting errors as HTTP 200 with a plain-text body.
"""

import json
import logging
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.providers import (
    PROVIDERS,
    BinanceProvider,
    CachingTransport,
    CoinbaseProvider,
    DataFeedError,
    LiveFeed,
    MarketDataFeed,
    Provider,
    RateLimited,
    StooqProvider,
    SymbolNotFound,
    Transport,
    TransportError,
    UrllibTransport,
    YahooProvider,
    build_provider,
)

HOUR = 3_600


def setUpModule():
    # Retry and recovery paths log warnings by design; they are the assertion,
    # not a problem, so keep them out of the test output.
    logging.getLogger("trading_bot.providers").setLevel(logging.ERROR)


class FakeTransport(Transport):
    """Returns canned bodies and records the URLs it was asked for."""

    def __init__(self, body=b"", bodies=None, error=None):
        self.body = body
        self.bodies = list(bodies) if bodies else None
        self.error = error
        self.urls: list[str] = []
        self.headers: list[dict] = []

    def get(self, url, headers=None):
        self.urls.append(url)
        self.headers.append(headers or {})
        if self.error is not None:
            raise self.error
        if self.bodies is not None:
            return self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return self.body


def encode(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

STOOQ_CSV = b"""Date,Open,High,Low,Close,Volume
2024-01-02,187.15,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
2024-01-04,182.15,183.09,180.88,181.91,71983600
"""


def yahoo_payload(timestamps, opens, highs, lows, closes, volumes):
    return {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD", "symbol": "AAPL", "exchangeTimezoneName": "America/New_York"},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def binance_klines(rows):
    """rows: (open_time_ms, o, h, l, c, v, close_time_ms)"""
    return [
        [r[0], str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), r[6],
         "0", 100, "0", "0", "0"]
        for r in rows
    ]


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------


class TestUrllibTransport(unittest.TestCase):
    def test_retries_are_bounded_and_backoff_grows(self):
        delays = []
        transport = UrllibTransport(retries=3, backoff=1.0, sleeper=delays.append)
        transport.get = UrllibTransport.get.__get__(transport)

        import urllib.error

        attempts = []

        def failing_urlopen(request, timeout=None):
            attempts.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

        import urllib.request as urlreq

        original = urlreq.urlopen
        urlreq.urlopen = failing_urlopen
        try:
            with self.assertRaises(TransportError):
                transport.get("https://example.test/x")
        finally:
            urlreq.urlopen = original

        self.assertEqual(len(attempts), 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_404_is_not_retried(self):
        import urllib.error
        import urllib.request as urlreq

        attempts = []

        def failing_urlopen(request, timeout=None):
            attempts.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 404, "nope", {}, None)

        transport = UrllibTransport(retries=3, sleeper=lambda _: None)
        original = urlreq.urlopen
        urlreq.urlopen = failing_urlopen
        try:
            with self.assertRaises(SymbolNotFound):
                transport.get("https://example.test/x")
        finally:
            urlreq.urlopen = original
        self.assertEqual(len(attempts), 1, "a missing symbol will still be missing on a retry")

    def test_sends_a_user_agent(self):
        import urllib.request as urlreq

        seen = {}

        class Response:
            def read(self):
                return b"ok"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(request, timeout=None):
            seen.update(request.headers)
            return Response()

        original = urlreq.urlopen
        urlreq.urlopen = urlopen
        try:
            UrllibTransport().get("https://example.test/x")
        finally:
            urlreq.urlopen = original
        self.assertTrue(any("user-agent" in key.lower() for key in seen))


class TestCachingTransport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_request_is_served_from_disk(self):
        inner = FakeTransport(body=b"payload")
        cache = CachingTransport(inner, self.dir)
        self.assertEqual(cache.get("https://x.test/a"), b"payload")
        self.assertEqual(cache.get("https://x.test/a"), b"payload")
        self.assertEqual(len(inner.urls), 1, "the second call must not hit the network")
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_different_urls_are_cached_separately(self):
        inner = FakeTransport(body=b"payload")
        cache = CachingTransport(inner, self.dir)
        cache.get("https://x.test/a")
        cache.get("https://x.test/b")
        self.assertEqual(len(inner.urls), 2)

    def test_expired_entries_are_refetched(self):
        inner = FakeTransport(body=b"payload")
        cache = CachingTransport(inner, self.dir, ttl=timedelta(seconds=1))
        cache.get("https://x.test/a")
        path = cache._path("https://x.test/a")
        import os

        stale = time.time() - 10
        os.utime(path, (stale, stale))
        cache.get("https://x.test/a")
        self.assertEqual(len(inner.urls), 2)

    def test_zero_ttl_never_expires(self):
        inner = FakeTransport(body=b"payload")
        cache = CachingTransport(inner, self.dir, ttl=timedelta(0))
        cache.get("https://x.test/a")
        cache.get("https://x.test/a")
        self.assertEqual(len(inner.urls), 1)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class TestStooqProvider(unittest.TestCase):
    def test_parses_bars(self):
        provider = StooqProvider(FakeTransport(STOOQ_CSV))
        candles = provider.fetch("AAPL")
        self.assertEqual(len(candles), 3)
        self.assertAlmostEqual(candles[0].close, 185.64)
        self.assertAlmostEqual(candles[-1].close, 181.91)
        self.assertEqual(candles[0].symbol, "AAPL")

    def test_timestamps_are_utc_aware_and_ascending(self):
        candles = StooqProvider(FakeTransport(STOOQ_CSV)).fetch("AAPL")
        for candle in candles:
            self.assertIsNotNone(candle.timestamp.tzinfo)
        for before, after in zip(candles, candles[1:]):
            self.assertLess(before.timestamp, after.timestamp)

    def test_bare_symbol_gains_the_us_suffix(self):
        transport = FakeTransport(STOOQ_CSV)
        StooqProvider(transport).fetch("AAPL")
        self.assertIn("aapl.us", transport.urls[0])

    def test_explicit_market_suffix_is_preserved(self):
        transport = FakeTransport(STOOQ_CSV)
        StooqProvider(transport).fetch("VOD.UK")
        self.assertIn("vod.uk", transport.urls[0])

    def test_index_symbol_is_preserved(self):
        transport = FakeTransport(STOOQ_CSV)
        StooqProvider(transport).fetch("^SPX")
        self.assertIn("%5Espx", transport.urls[0])

    def test_rate_limit_body_is_detected(self):
        transport = FakeTransport(b"Exceeded the daily hits limit")
        with self.assertRaises(RateLimited):
            StooqProvider(transport).fetch("AAPL")

    def test_no_data_body_is_detected(self):
        with self.assertRaises(SymbolNotFound):
            StooqProvider(FakeTransport(b"No data")).fetch("NOPE")

    def test_empty_body_is_rejected(self):
        with self.assertRaises(SymbolNotFound):
            StooqProvider(FakeTransport(b"")).fetch("AAPL")

    def test_unsupported_interval_is_rejected(self):
        with self.assertRaises(DataFeedError):
            StooqProvider(FakeTransport(STOOQ_CSV)).fetch("AAPL", interval="1m")

    def test_limit_keeps_the_most_recent_bars(self):
        candles = StooqProvider(FakeTransport(STOOQ_CSV)).fetch("AAPL", limit=2)
        self.assertEqual(len(candles), 2)
        self.assertAlmostEqual(candles[-1].close, 181.91)


class TestYahooProvider(unittest.TestCase):
    def test_parses_bars(self):
        payload = yahoo_payload(
            [1704207600, 1704294000],
            [187.15, 184.22], [188.44, 185.88], [183.89, 183.43],
            [185.64, 184.25], [82488700, 58414500],
        )
        candles = YahooProvider(FakeTransport(encode(payload))).fetch("AAPL")
        self.assertEqual(len(candles), 2)
        self.assertAlmostEqual(candles[0].close, 185.64)
        self.assertAlmostEqual(candles[1].volume, 58414500)

    def test_null_padded_bars_are_skipped(self):
        # Yahoo pads holidays and halts with nulls rather than omitting them.
        payload = yahoo_payload(
            [1704207600, 1704294000, 1704380400],
            [187.15, None, 182.15], [188.44, None, 183.09], [183.89, None, 180.88],
            [185.64, None, 181.91], [82488700, None, 71983600],
        )
        candles = YahooProvider(FakeTransport(encode(payload))).fetch("AAPL")
        self.assertEqual(len(candles), 2, "a bar with no close is not a bar")

    def test_partial_nulls_fall_back_to_the_close(self):
        payload = yahoo_payload(
            [1704207600], [None], [None], [None], [185.64], [None],
        )
        candles = YahooProvider(FakeTransport(encode(payload))).fetch("AAPL")
        self.assertEqual(len(candles), 1)
        self.assertAlmostEqual(candles[0].open, 185.64)
        self.assertAlmostEqual(candles[0].volume, 0.0)

    def test_error_payload_raises(self):
        payload = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}
        with self.assertRaises(SymbolNotFound):
            YahooProvider(FakeTransport(encode(payload))).fetch("NOPE")

    def test_empty_result_raises(self):
        with self.assertRaises(SymbolNotFound):
            YahooProvider(FakeTransport(encode({"chart": {"result": [], "error": None}}))).fetch("X")

    def test_interval_is_mapped_to_the_yahoo_code(self):
        payload = yahoo_payload([1704207600], [1], [1], [1], [1], [1])
        transport = FakeTransport(encode(payload))
        YahooProvider(transport).fetch("AAPL", interval="1h")
        self.assertIn("interval=60m", transport.urls[0])

    def test_weekly_interval_is_mapped(self):
        payload = yahoo_payload([1704207600], [1], [1], [1], [1], [1])
        transport = FakeTransport(encode(payload))
        YahooProvider(transport).fetch("AAPL", interval="1w")
        self.assertIn("interval=1wk", transport.urls[0])

    def test_range_widens_with_the_requested_bar_count(self):
        payload = yahoo_payload([1704207600], [1], [1], [1], [1], [1])
        short = FakeTransport(encode(payload))
        YahooProvider(short).fetch("AAPL", interval="1d", limit=20)
        long = FakeTransport(encode(payload))
        YahooProvider(long).fetch("AAPL", interval="1d", limit=6000)
        self.assertNotEqual(short.urls[0], long.urls[0])
        self.assertIn("range=max", long.urls[0])

    def test_unsupported_interval_is_rejected(self):
        with self.assertRaises(DataFeedError):
            YahooProvider(FakeTransport(b"{}")).fetch("AAPL", interval="4h")


class TestBinanceProvider(unittest.TestCase):
    def test_parses_bars(self):
        past = int((time.time() - 10 * 86_400) * 1000)
        rows = binance_klines([
            (past, 42000.0, 42500.0, 41800.0, 42300.0, 1234.5, past + 86_400_000),
            (past + 86_400_000, 42300.0, 43100.0, 42200.0, 43000.0, 987.6, past + 2 * 86_400_000),
        ])
        candles = BinanceProvider(FakeTransport(encode(rows))).fetch("BTCUSDT")
        self.assertEqual(len(candles), 2)
        self.assertAlmostEqual(candles[0].close, 42300.0)
        self.assertAlmostEqual(candles[1].volume, 987.6)

    def test_unclosed_final_kline_is_dropped(self):
        now_ms = time.time() * 1000
        past = int(now_ms - 3 * 86_400_000)
        rows = binance_klines([
            (past, 1.0, 2.0, 0.5, 1.5, 10.0, past + 86_400_000),
            # This bar has not closed yet; its price can still move.
            (int(now_ms - 1000), 1.5, 1.6, 1.4, 1.55, 3.0, int(now_ms + 86_400_000)),
        ])
        candles = BinanceProvider(FakeTransport(encode(rows))).fetch("BTCUSDT")
        self.assertEqual(len(candles), 1)
        self.assertAlmostEqual(candles[0].close, 1.5)

    def test_error_object_raises(self):
        body = encode({"code": -1121, "msg": "Invalid symbol."})
        with self.assertRaises(SymbolNotFound):
            BinanceProvider(FakeTransport(body)).fetch("NOTACOIN")

    def test_empty_array_raises(self):
        with self.assertRaises(SymbolNotFound):
            BinanceProvider(FakeTransport(encode([]))).fetch("BTCUSDT")

    def test_only_unclosed_bars_raises(self):
        now_ms = time.time() * 1000
        rows = binance_klines([(int(now_ms), 1.0, 2.0, 0.5, 1.5, 10.0, int(now_ms + 86_400_000))])
        with self.assertRaises(SymbolNotFound):
            BinanceProvider(FakeTransport(encode(rows))).fetch("BTCUSDT")

    def test_limit_is_clamped_to_the_api_maximum(self):
        past = int((time.time() - 10 * 86_400) * 1000)
        rows = binance_klines([(past, 1.0, 2.0, 0.5, 1.5, 10.0, past + 86_400_000)])
        transport = FakeTransport(encode(rows))
        BinanceProvider(transport).fetch("BTCUSDT", limit=99_999)
        self.assertIn("limit=1000", transport.urls[0])

    def test_supports_four_hour_bars(self):
        past = int((time.time() - 10 * 86_400) * 1000)
        rows = binance_klines([(past, 1.0, 2.0, 0.5, 1.5, 10.0, past + 4 * HOUR * 1000)])
        transport = FakeTransport(encode(rows))
        BinanceProvider(transport).fetch("BTCUSDT", interval="4h")
        self.assertIn("interval=4h", transport.urls[0])


class TestCoinbaseProvider(unittest.TestCase):
    def test_column_order_is_time_low_high_open_close(self):
        # Coinbase puts low and high *before* open and close, unlike everyone else.
        rows = [[1704240000, 41800.0, 42500.0, 42000.0, 42300.0, 1234.5]]
        candles = CoinbaseProvider(FakeTransport(encode(rows))).fetch("BTC-USD")
        self.assertAlmostEqual(candles[0].open, 42000.0)
        self.assertAlmostEqual(candles[0].close, 42300.0)
        self.assertAlmostEqual(candles[0].high, 42500.0)
        self.assertAlmostEqual(candles[0].low, 41800.0)

    def test_descending_rows_are_reordered(self):
        rows = [
            [1704326400, 42200.0, 43100.0, 42300.0, 43000.0, 987.6],
            [1704240000, 41800.0, 42500.0, 42000.0, 42300.0, 1234.5],
        ]
        candles = CoinbaseProvider(FakeTransport(encode(rows))).fetch("BTC-USD")
        self.assertLess(candles[0].timestamp, candles[1].timestamp)
        self.assertAlmostEqual(candles[0].close, 42300.0)

    def test_granularity_is_sent_in_seconds(self):
        rows = [[1704240000, 1.0, 2.0, 1.5, 1.8, 1.0]]
        transport = FakeTransport(encode(rows))
        CoinbaseProvider(transport).fetch("BTC-USD", interval="15m")
        self.assertIn("granularity=900", transport.urls[0])

    def test_error_object_raises(self):
        with self.assertRaises(SymbolNotFound):
            CoinbaseProvider(FakeTransport(encode({"message": "NotFound"}))).fetch("NO-PAIR")

    def test_unsupported_interval_is_rejected(self):
        with self.assertRaises(DataFeedError):
            CoinbaseProvider(FakeTransport(b"[]")).fetch("BTC-USD", interval="4h")


class TestSharedProviderBehaviour(unittest.TestCase):
    def test_wicks_are_clamped_not_rejected(self):
        # A high rounded below the close must not fail Candle validation.
        rows = [[1704240000, 100.0, 104.0, 100.5, 104.4, 1.0]]
        candles = CoinbaseProvider(FakeTransport(encode(rows))).fetch("BTC-USD")
        self.assertGreaterEqual(candles[0].high, candles[0].close)
        self.assertLessEqual(candles[0].low, candles[0].open)

    def test_duplicate_timestamps_are_collapsed(self):
        rows = [
            [1704240000, 1.0, 2.0, 1.5, 1.8, 1.0],
            [1704240000, 1.0, 2.0, 1.5, 1.9, 1.0],
        ]
        candles = CoinbaseProvider(FakeTransport(encode(rows))).fetch("BTC-USD")
        self.assertEqual(len(candles), 1)

    def test_every_provider_declares_intervals_and_a_name(self):
        for name, factory in PROVIDERS.items():
            provider = factory(FakeTransport(b"[]"))
            self.assertEqual(provider.name, name)
            self.assertTrue(provider.intervals)
            for interval in provider.intervals:
                self.assertIn(interval, __import__(
                    "trading_bot.providers", fromlist=["INTERVAL_SECONDS"]
                ).INTERVAL_SECONDS)

    def test_throttle_spaces_requests_out(self):
        class Slow(StooqProvider):
            min_request_interval = 2.0

        slept = []
        now = [100.0]
        provider = Slow(FakeTransport(STOOQ_CSV), clock=lambda: now[0], sleeper=slept.append)
        provider.fetch("AAPL")
        now[0] = 100.5
        provider.fetch("AAPL")
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(slept[0], 1.5)

    def test_no_throttle_by_default(self):
        slept = []
        provider = StooqProvider(FakeTransport(STOOQ_CSV), sleeper=slept.append)
        provider.fetch("AAPL")
        provider.fetch("AAPL")
        self.assertEqual(slept, [])


class TestBuildProvider(unittest.TestCase):
    def test_builds_each_registered_provider(self):
        for name in PROVIDERS:
            self.assertIsInstance(build_provider(name, FakeTransport(b"[]")), Provider)

    def test_unknown_provider_lists_alternatives(self):
        with self.assertRaises(KeyError) as ctx:
            build_provider("bloomberg")
        self.assertIn("stooq", str(ctx.exception))

    def test_cache_dir_wraps_the_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = build_provider("stooq", FakeTransport(STOOQ_CSV), cache_dir=tmp)
            self.assertIsInstance(provider.transport, CachingTransport)


# ---------------------------------------------------------------------------
# feeds
# ---------------------------------------------------------------------------


class TestMarketDataFeed(unittest.TestCase):
    def test_iterates_real_bars(self):
        feed = MarketDataFeed(StooqProvider(FakeTransport(STOOQ_CSV)), "AAPL")
        candles = list(feed)
        self.assertEqual(len(candles), 3)
        self.assertEqual(len(feed), 3)

    def test_fetches_once_and_reuses(self):
        transport = FakeTransport(STOOQ_CSV)
        feed = MarketDataFeed(StooqProvider(transport), "AAPL")
        list(feed)
        list(feed)
        len(feed)
        self.assertEqual(len(transport.urls), 1, "a grid search must not refetch per run")

    def test_drives_a_backtest(self):
        from trading_bot import PaperBroker, TradingEngine
        from trading_bot.risk import FixedFractionSizer, RiskLimits, RiskManager
        from trading_bot.strategy import BuyAndHold

        feed = MarketDataFeed(StooqProvider(FakeTransport(STOOQ_CSV)), "AAPL")
        engine = TradingEngine(
            BuyAndHold(),
            PaperBroker(starting_cash=10_000),
            RiskManager(RiskLimits(max_drawdown=None), FixedFractionSizer(0.95)),
        )
        result = engine.run(feed)
        self.assertEqual(result.bars, 3)
        self.assertEqual(result.symbol, "AAPL")


class TestLiveFeed(unittest.TestCase):
    def make_provider(self, batches):
        """A provider whose successive fetches return successive batches."""

        class Scripted(Provider):
            name = "scripted"
            intervals = ("1m", "1d")

            def __init__(self):
                super().__init__(FakeTransport(b""))
                self.calls = 0

            def fetch(self, symbol, interval="1d", limit=500):
                batch = batches[min(self.calls, len(batches) - 1)]
                self.calls += 1
                if isinstance(batch, Exception):
                    raise batch
                return list(batch)

        return Scripted()

    def bars(self, count, start_minute=0, base=100.0):
        start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        return [
            Candle_ := __import__("trading_bot.models", fromlist=["Candle"]).Candle(
                timestamp=start + timedelta(minutes=start_minute + i),
                open=base + i, high=base + i + 1, low=base + i - 1, close=base + i + 0.5,
                volume=10.0, symbol="TEST",
            )
            for i in range(count)
        ]

    def test_warmup_history_is_yielded_first(self):
        history = self.bars(5)
        feed = LiveFeed(
            self.make_provider([history, history]),
            "TEST", interval="1m", warmup=5, max_bars=0,
            sleeper=lambda _: None, clock=lambda: datetime(2024, 1, 1, 12, 10, tzinfo=timezone.utc),
        )
        candles = list(feed)
        self.assertEqual(len(candles), 5)
        self.assertEqual(feed.live_bars, 0)

    def test_new_bars_are_emitted_once(self):
        history = self.bars(3)
        later = self.bars(5)  # same first three, plus two new
        feed = LiveFeed(
            self.make_provider([history, later, later]),
            "TEST", interval="1m", warmup=3, max_bars=2,
            sleeper=lambda _: None, clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        candles = list(feed)
        self.assertEqual(len(candles), 5, "3 warmup bars plus 2 new ones")
        self.assertEqual(feed.live_bars, 2)
        timestamps = [c.timestamp for c in candles]
        self.assertEqual(len(timestamps), len(set(timestamps)), "no bar may be emitted twice")

    def test_repeated_identical_polls_emit_nothing(self):
        history = self.bars(3)
        feed = LiveFeed(
            self.make_provider([history]),
            "TEST", interval="1m", warmup=3, max_bars=1,
            sleeper=lambda _: None,
            clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        # max_bars counts live bars; the provider never produces a new one, so
        # this would spin forever if the loop had no other exit. Bound it.
        feed.until = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
        candles = list(feed)
        self.assertEqual(len(candles), 3)
        self.assertEqual(feed.live_bars, 0)

    def test_warmup_zero_still_suppresses_history(self):
        history = self.bars(3)
        feed = LiveFeed(
            self.make_provider([history, history]),
            "TEST", interval="1m", warmup=0, max_bars=1,
            sleeper=lambda _: None,
            clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        feed.until = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
        self.assertEqual(list(feed), [], "history must not be replayed as live bars")

    def test_transient_errors_are_survived(self):
        history = self.bars(3)
        later = self.bars(4)
        provider = self.make_provider([history, TransportError("blip"), later, later])
        feed = LiveFeed(
            provider, "TEST", interval="1m", warmup=0, max_bars=1,
            sleeper=lambda _: None,
            clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        candles = list(feed)
        self.assertEqual(len(candles), 1, "the feed recovered after the failed poll")

    def test_persistent_errors_stop_the_feed(self):
        history = self.bars(3)
        provider = self.make_provider([history, TransportError("down")])
        feed = LiveFeed(
            provider, "TEST", interval="1m", warmup=0, max_bars=5,
            max_consecutive_errors=3, sleeper=lambda _: None,
            clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        with self.assertRaises(DataFeedError) as ctx:
            list(feed)
        self.assertIn("3 times in a row", str(ctx.exception))

    def test_sleep_waits_for_the_next_bar_close(self):
        delays = []
        feed = LiveFeed(
            self.make_provider([self.bars(1), self.bars(2)]),
            "TEST", interval="1m", warmup=0, max_bars=1,
            poll_lag=5.0, sleeper=delays.append,
            clock=lambda: datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        list(feed)
        # The known bar opened at 12:00 and closes at 12:01, so the next one
        # closes at 12:02 — 120 seconds out, plus the 5 second lag buffer.
        self.assertEqual(len(delays), 1)
        self.assertAlmostEqual(delays[0], 125.0)

    def test_sleep_never_goes_negative(self):
        delays = []
        feed = LiveFeed(
            self.make_provider([self.bars(1), self.bars(2)]),
            "TEST", interval="1m", warmup=0, max_bars=1,
            sleeper=delays.append,
            # A clock far ahead of the data would otherwise busy-spin.
            clock=lambda: datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        list(feed)
        self.assertTrue(delays)
        self.assertTrue(all(d >= 1.0 for d in delays))

    def test_unsupported_interval_is_rejected_at_construction(self):
        with self.assertRaises(DataFeedError):
            LiveFeed(self.make_provider([[]]), "TEST", interval="4h")

    def test_negative_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            LiveFeed(self.make_provider([[]]), "TEST", interval="1m", warmup=-1)


if __name__ == "__main__":
    unittest.main()
