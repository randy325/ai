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


class TestErrorBodyDiagnosis(unittest.TestCase):
    """A refused download saves as .csv and must be named as such.

    Providers answer a rejected request with a short message or an HTML error
    page. Both save happily as ``spy.csv``, and the column-detection failure
    they used to produce sent people hunting for a formatting problem that did
    not exist.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text, name="spy.csv"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_stooq_access_denied_is_named(self):
        from trading_bot.data import CSVFeed
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(self.write("Access denied")))
        message = str(ctx.exception)
        self.assertIn("not market data", message)
        self.assertIn("refused", message)
        self.assertNotIn("missing required column", message)

    def test_rate_limit_body_is_named(self):
        from trading_bot.data import CSVFeed
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(self.write("Exceeded the daily hits limit")))
        self.assertIn("rate limit", str(ctx.exception))

    def test_html_error_page_is_named(self):
        from trading_bot.data import CSVFeed
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(self.write("<!DOCTYPE html><html><body>403</body></html>")))
        self.assertIn("HTML", str(ctx.exception))

    def test_empty_file_is_named(self):
        from trading_bot.data import CSVFeed
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(self.write("   \n")))
        self.assertIn("empty", str(ctx.exception))

    def test_real_data_is_untouched(self):
        from trading_bot.data import CSVFeed
        path = self.write("Date,Open,High,Low,Close,Volume\n2024-01-02,1,2,0.5,1.5,100\n")
        self.assertEqual(len(list(CSVFeed(path))), 1)

    def test_a_symbol_containing_a_marker_word_still_parses(self):
        # "No data" appears inside a legitimate header comment position only as
        # a body; a real CSV whose first bytes are column names must survive.
        from trading_bot.data import CSVFeed, diagnose_error_body
        self.assertIsNone(diagnose_error_body("Date,Open,High,Low,Close\n2024-01-02,1,2,0.5,1.5\n"))

    def test_diagnosis_returns_none_for_plausible_data(self):
        from trading_bot.data import diagnose_error_body
        self.assertIsNone(diagnose_error_body("timestamp,close\n2020-01-01,100\n"))


class TestSpreadsheetExportFormats(unittest.TestCase):
    """A GOOGLEFINANCE export must parse without hand-editing.

    ``=GOOGLEFINANCE("SPY","all",...)`` writes its dates as "1/2/2024 16:00:00",
    which matched none of the accepted formats — the export would have been
    rejected wholesale the first time anyone pasted one in.
    """

    def test_googlefinance_datetime_parses(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(parse_timestamp("1/2/2024 16:00:00"), datetime(2024, 1, 2, 16, 0))
        self.assertEqual(parse_timestamp("01/02/2024 16:00:00"), datetime(2024, 1, 2, 16, 0))

    def test_minute_precision_variant_parses(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(parse_timestamp("3/15/2024 09:30"), datetime(2024, 3, 15, 9, 30))

    def test_month_first_still_wins_for_ambiguous_slash_dates(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(parse_timestamp("3/4/2024 16:00:00"), datetime(2024, 3, 4, 16, 0))

    def test_a_day_first_datetime_needs_the_day_first_order(self):
        from trading_bot.data import parse_timestamp
        self.assertEqual(
            parse_timestamp("25/12/2024 16:00:00", date_order="day-first"),
            datetime(2024, 12, 25, 16, 0),
        )

    def test_a_full_googlefinance_export_loads(self):
        import tempfile
        from trading_bot.data import CSVFeed
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spy.csv"
            path.write_text(
                # A real export spans months, so some row disambiguates the
                # ordering. Three rows all <= 12 would be genuinely ambiguous
                # and are now rejected rather than guessed.
                "Date,Open,High,Low,Close,Volume\n"
                "1/2/2024 16:00:00,472.16,473.67,470.49,472.65,123963000\n"
                "1/3/2024 16:00:00,470.43,471.19,468.17,468.79,103800800\n"
                "1/16/2024 16:00:00,468.3,470.96,467.05,467.28,84232200\n",
                encoding="utf-8",
            )
            candles = list(CSVFeed(path))
        self.assertEqual(len(candles), 3)
        self.assertAlmostEqual(candles[0].close, 472.65)
        self.assertAlmostEqual(candles[-1].volume, 84232200)


class TestSlashDateOrderIsResolvedPerFile(unittest.TestCase):
    """Slash-date ordering is a property of the file, not of a row.

    With both orderings in one per-row fallback list, month-first did not
    "fail consistently": it succeeded WRONGLY on days 1-12 and fell through
    correctly on 13-31, interleaving two calendars in one column.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rows, name="p.csv"):
        path = self.dir / name
        path.write_text("Date,Close\n" + "".join(f"{d},{c}\n" for d, c in rows), encoding="utf-8")
        return path

    def test_a_european_daily_file_is_not_interleaved(self):
        from trading_bot.data import CSVFeed
        path = self.write([("01/04/2024", 100), ("02/04/2024", 101), ("15/04/2024", 102)])
        dates = [c.timestamp.date() for c in CSVFeed(path)]
        self.assertEqual([d.month for d in dates], [4, 4, 4],
                         "one row above 12 settles the whole file as day-first")
        self.assertEqual([d.day for d in dates], [1, 2, 15])

    def test_a_us_daily_file_is_read_month_first(self):
        from trading_bot.data import CSVFeed
        path = self.write([("04/01/2024", 100), ("04/02/2024", 101), ("04/15/2024", 102)])
        dates = [c.timestamp.date() for c in CSVFeed(path)]
        self.assertEqual([d.month for d in dates], [4, 4, 4])
        self.assertEqual([d.day for d in dates], [1, 2, 15])

    def test_a_european_monthly_series_raises_instead_of_corrupting(self):
        # The silent case: twelve bars dated the 1st parse as twelve
        # consecutive days of January, ascending, so no ordering check fires.
        from trading_bot.data import AmbiguousDateOrder, CSVFeed
        path = self.write([(f"01/{m:02d}/2024", 100 + m) for m in range(1, 13)])
        with self.assertRaises(AmbiguousDateOrder) as ctx:
            list(CSVFeed(path))
        self.assertIn("date_order", str(ctx.exception))

    def test_an_explicit_order_resolves_the_ambiguous_case(self):
        from trading_bot.data import CSVFeed
        path = self.write([(f"01/{m:02d}/2024", 100 + m) for m in range(1, 13)])
        dates = [c.timestamp.date() for c in CSVFeed(path, date_order="day-first")]
        self.assertEqual([d.month for d in dates], list(range(1, 13)))
        self.assertTrue(all(d.day == 1 for d in dates))

    def test_the_wrong_explicit_order_is_honoured_not_second_guessed(self):
        from trading_bot.data import CSVFeed
        path = self.write([(f"01/{m:02d}/2024", 100 + m) for m in range(1, 13)])
        dates = [c.timestamp.date() for c in CSVFeed(path, date_order="month-first")]
        self.assertTrue(all(d.month == 1 for d in dates))

    def test_a_file_mixing_both_orderings_is_rejected(self):
        from trading_bot.data import CSVFeed
        path = self.write([("15/04/2024", 100), ("04/15/2024", 101)])
        with self.assertRaises(ValueError) as ctx:
            list(CSVFeed(path))
        self.assertIn("not one consistent format", str(ctx.exception))

    def test_iso_files_are_unaffected(self):
        from trading_bot.data import CSVFeed
        path = self.write([("2024-01-02", 100), ("2024-02-02", 101)])
        self.assertEqual(len(list(CSVFeed(path))), 2)

    def test_googlefinance_export_still_loads(self):
        from trading_bot.data import CSVFeed
        path = self.write([("1/2/2024 16:00:00", 100), ("1/15/2024 16:00:00", 101)])
        dates = [c.timestamp.date() for c in CSVFeed(path)]
        self.assertEqual([d.month for d in dates], [1, 1])
        self.assertEqual([d.day for d in dates], [2, 15])

    def test_detect_rejects_an_invalid_override(self):
        from trading_bot.data import CSVFeed
        with self.assertRaises(ValueError):
            CSVFeed(self.write([("2024-01-02", 100)]), date_order="whenever")

    def test_detection_reports_the_file_in_its_message(self):
        from trading_bot.data import AmbiguousDateOrder, CSVFeed
        path = self.write([("01/02/2024", 100), ("01/03/2024", 101)], name="eu.csv")
        with self.assertRaises(AmbiguousDateOrder) as ctx:
            list(CSVFeed(path))
        self.assertIn("eu.csv", str(ctx.exception))
