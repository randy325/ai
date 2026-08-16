import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.data import CSVFeed, CandleFeed, SyntheticFeed, parse_timestamp, write_csv


class TestParseTimestamp(unittest.TestCase):
    def test_iso_date(self):
        self.assertEqual(parse_timestamp("2020-03-15"), datetime(2020, 3, 15))

    def test_iso_datetime(self):
        self.assertEqual(parse_timestamp("2020-03-15 14:30:00"), datetime(2020, 3, 15, 14, 30))

    def test_iso_with_t_separator(self):
        self.assertEqual(parse_timestamp("2020-03-15T14:30:00"), datetime(2020, 3, 15, 14, 30))

    def test_epoch_seconds(self):
        self.assertEqual(
            parse_timestamp("1584280800"),
            datetime.fromtimestamp(1584280800, tz=timezone.utc),
        )

    def test_epoch_milliseconds(self):
        self.assertEqual(
            parse_timestamp("1584280800000"),
            datetime.fromtimestamp(1584280800, tz=timezone.utc),
        )

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_timestamp("not a date")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_timestamp("   ")


class TestCSVFeed(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text, name="prices.csv"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_standard_columns(self):
        path = self.write(
            "timestamp,open,high,low,close,volume\n"
            "2020-01-01,100,105,99,104,1000\n"
            "2020-01-02,104,108,103,107,1200\n"
        )
        candles = list(CSVFeed(path, symbol="TEST"))
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].symbol, "TEST")
        self.assertAlmostEqual(candles[0].close, 104)
        self.assertAlmostEqual(candles[1].volume, 1200)

    def test_column_names_are_case_and_alias_insensitive(self):
        path = self.write(
            "Date,Open,High,Low,Adj Close,Volume\n2020-01-01,100,105,99,104,1000\n"
        )
        candles = list(CSVFeed(path))
        self.assertAlmostEqual(candles[0].close, 104)

    def test_close_only_file_synthesises_ohlc(self):
        path = self.write("date,close\n2020-01-01,100\n2020-01-02,110\n")
        candles = list(CSVFeed(path))
        self.assertAlmostEqual(candles[0].open, 100)
        self.assertAlmostEqual(candles[0].high, 100)

    def test_missing_close_column_is_rejected(self):
        path = self.write("date,open\n2020-01-01,100\n")
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(path))
        self.assertIn("close", str(ctx.exception))

    def test_out_of_order_timestamps_are_rejected(self):
        path = self.write("date,close\n2020-01-02,100\n2020-01-01,110\n")
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(path))
        self.assertIn("chronological", str(ctx.exception))

    def test_bad_number_reports_the_line(self):
        path = self.write("date,close\n2020-01-01,abc\n")
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(path))
        self.assertIn(":2", str(ctx.exception))

    def test_blank_rows_are_skipped(self):
        path = self.write("date,close\n2020-01-01,100\n,\n2020-01-02,110\n")
        self.assertEqual(len(list(CSVFeed(path))), 2)

    def test_rounded_ohlc_is_clamped_rather_than_rejected(self):
        # A high below the close would fail Candle validation if not clamped.
        path = self.write("date,open,high,low,close\n2020-01-01,100,104,99,104.4\n")
        candles = list(CSVFeed(path))
        self.assertGreaterEqual(candles[0].high, candles[0].close)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            CSVFeed(self.dir / "nope.csv")

    def test_symbol_defaults_to_the_file_stem(self):
        path = self.write("date,close\n2020-01-01,100\n", name="aapl.csv")
        self.assertEqual(CSVFeed(path).symbol, "AAPL")

    def test_round_trips_through_write_csv(self):
        original = list(SyntheticFeed(symbol="RT", bars=25, seed=3))
        path = write_csv(original, self.dir / "rt.csv")
        restored = list(CSVFeed(path, symbol="RT"))
        self.assertEqual(len(original), len(restored))
        for before, after in zip(original, restored):
            self.assertAlmostEqual(before.close, after.close, places=4)
            self.assertEqual(before.timestamp, after.timestamp)


class TestSyntheticFeed(unittest.TestCase):
    def test_produces_the_requested_number_of_bars(self):
        self.assertEqual(len(list(SyntheticFeed(bars=100))), 100)

    def test_is_reproducible_for_a_given_seed(self):
        first = [c.close for c in SyntheticFeed(bars=50, seed=42)]
        second = [c.close for c in SyntheticFeed(bars=50, seed=42)]
        self.assertEqual(first, second)

    def test_different_seeds_diverge(self):
        first = [c.close for c in SyntheticFeed(bars=50, seed=1)]
        second = [c.close for c in SyntheticFeed(bars=50, seed=2)]
        self.assertNotEqual(first, second)

    def test_every_bar_satisfies_ohlc_invariants(self):
        for candle in SyntheticFeed(bars=300, volatility=0.9, seed=11):
            self.assertLessEqual(candle.low, candle.open)
            self.assertLessEqual(candle.low, candle.close)
            self.assertGreaterEqual(candle.high, candle.open)
            self.assertGreaterEqual(candle.high, candle.close)
            self.assertGreater(candle.low, 0)

    def test_timestamps_are_strictly_increasing(self):
        candles = list(SyntheticFeed(bars=50))
        for before, after in zip(candles, candles[1:]):
            self.assertLess(before.timestamp, after.timestamp)

    def test_strong_drift_trends_upward(self):
        candles = list(SyntheticFeed(bars=500, drift=0.5, volatility=0.05, seed=5))
        self.assertGreater(candles[-1].close, candles[0].close)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            SyntheticFeed(bars=0)
        with self.assertRaises(ValueError):
            SyntheticFeed(start_price=0)


class TestCandleFeed(unittest.TestCase):
    def test_wraps_a_sequence(self):
        candles = list(SyntheticFeed(bars=10, symbol="Z"))
        feed = CandleFeed(candles)
        self.assertEqual(len(feed), 10)
        self.assertEqual(feed.symbol, "Z")
        self.assertEqual(list(feed), candles)

    def test_empty_sequence_is_allowed(self):
        self.assertEqual(len(CandleFeed([])), 0)


if __name__ == "__main__":
    unittest.main()
