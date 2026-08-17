"""The offline evaluation path and the robustness sweep's detectors.

The sweep's job is to spot a strategy fitted to one parameter value. A detector
that has never been shown to fire is not evidence of anything, so the roughness
and peak metrics are tested against sequences whose shape is known by
construction.
"""

import datetime
import importlib.util
import io
import random
import statistics
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from trading_bot.data import SyntheticFeed, write_csv  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load("robustness_sweep")
evaluate = load("evaluate_real")


class TestRoughness(unittest.TestCase):
    def test_a_monotonic_ramp_is_smooth(self):
        ramp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertAlmostEqual(sweep.roughness(ramp), sweep.smooth_reference(len(ramp)), places=6)

    def test_an_alternating_surface_is_maximally_rough(self):
        spiky = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(sweep.roughness(spiky), 1.0)

    def test_roughness_alone_does_not_catch_a_lone_spike(self):
        # A single spike has only two non-zero steps out of six, so it scores
        # low on roughness. Roughness measures jaggedness, not isolation —
        # is_peaked is the detector for this shape, and the two are used
        # together for exactly this reason.
        spike = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.assertLess(sweep.roughness(spike), 0.5)
        values = [10, 15, 20, 25, 30, 35, 40]
        peaked, _ = sweep.is_peaked(values, spike, 25)
        self.assertTrue(peaked, "the peak detector must catch what roughness misses")

    def test_a_gentle_hill_stays_below_the_threshold(self):
        hill = [0.0, 0.2, 0.4, 0.5, 0.4, 0.2, 0.0]
        self.assertLess(sweep.roughness(hill), 0.5)

    def test_a_flat_surface_is_zero(self):
        self.assertEqual(sweep.roughness([0.3] * 5), 0.0)

    def test_degenerate_inputs_are_safe(self):
        self.assertEqual(sweep.roughness([]), 0.0)
        self.assertEqual(sweep.roughness([0.5]), 0.0)

    def test_smooth_reference_shrinks_with_more_points(self):
        self.assertGreater(sweep.smooth_reference(3), sweep.smooth_reference(11))


class TestNullCalibration(unittest.TestCase):
    """The threshold must come from the noise distribution, not a constant.

    Simulating roughness under iid random values shows the statistic's null is
    strongly dependent on the point count: mean about 0.67 at 3 points falling
    to 0.28 at 31. A fixed 0.5 cutoff therefore sat on the noise *mean* for a
    5-point sweep, flagging pure noise as SPIKY roughly half the time.
    """

    def test_the_null_mean_falls_as_points_increase(self):
        means = [statistics.mean(sweep.null_roughness(n, trials=4000)) for n in (5, 11, 21)]
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])

    def test_the_five_point_null_matches_the_reported_simulation(self):
        values = sweep.null_roughness(5, trials=20_000)
        self.assertAlmostEqual(statistics.mean(values), 0.49, delta=0.03)
        self.assertAlmostEqual(sweep.null_percentile(5, 0.05), 0.30, delta=0.03)
        self.assertAlmostEqual(sweep.null_percentile(5, 0.95), 0.70, delta=0.03)

    def test_the_old_constant_threshold_was_a_coin_flip_at_five_points(self):
        values = sweep.null_roughness(5, trials=20_000)
        false_positive = sum(1 for v in values if v > 0.5) / len(values)
        self.assertGreater(false_positive, 0.30,
                           "a fixed 0.5 cutoff fires on pure noise far too often")

    def test_the_calibrated_threshold_holds_its_error_rate(self):
        # By construction the p95 cutoff should fire on ~5% of pure noise, at
        # every point count — which the constant threshold did not.
        for count in (5, 7, 11, 15, 21):
            values = sweep.null_roughness(count, trials=20_000)
            cutoff = sweep.null_percentile(count, 0.95)
            rate = sum(1 for v in values if v > cutoff) / len(values)
            self.assertAlmostEqual(rate, 0.05, delta=0.01, msg=f"count={count}")

    def test_the_null_is_deterministic(self):
        self.assertEqual(sweep.null_roughness(7, trials=500),
                         sweep.null_roughness(7, trials=500))

    def test_classification_is_three_way(self):
        # Above the upper tail, below the lower tail, and the large middle
        # where the statistic simply cannot tell.
        self.assertEqual(sweep.classify_roughness(0.99, 11), "jagged")
        self.assertEqual(sweep.classify_roughness(0.05, 11), "smooth")
        self.assertEqual(
            sweep.classify_roughness(statistics.median(sweep.null_roughness(11)), 11),
            "noise",
        )

    def test_the_previously_flagged_oversold_value_is_only_noise(self):
        # rsi-mean-reversion's oversold measured 0.63 over 5 settings and was
        # labelled SPIKY. The 5-point null's 95th percentile is about 0.70.
        self.assertEqual(sweep.classify_roughness(0.63, 5), "noise")


class TestResolvingPower(unittest.TestCase):
    """Between-point signal must be separated from within-point sampling noise."""

    def test_identical_noisy_points_have_no_resolving_power(self):
        rng = random.Random(1)
        per_seed = [[rng.gauss(0.0, 0.10) for _ in range(20)] for _ in range(9)]
        _, _, ratio, _ = sweep.resolving_power(per_seed)
        self.assertLess(ratio, 2.0, "a flat surface must not appear resolvable")

    def test_a_strong_trend_resolves(self):
        rng = random.Random(2)
        per_seed = [
            [i * 0.10 + rng.gauss(0.0, 0.01) for _ in range(20)] for i in range(9)
        ]
        _, _, ratio, _ = sweep.resolving_power(per_seed)
        self.assertGreater(ratio, 2.0)

    def test_more_seeds_increase_resolving_power(self):
        def ratio_for(seed_count):
            rng = random.Random(3)
            per_seed = [
                [i * 0.02 + rng.gauss(0.0, 0.10) for _ in range(seed_count)]
                for i in range(9)
            ]
            return sweep.resolving_power(per_seed)[2]

        self.assertGreater(ratio_for(100), ratio_for(4))

    def test_degenerate_input_is_safe(self):
        self.assertEqual(sweep.resolving_power([]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(sweep.resolving_power([[0.1]]), (0.0, 0.0, 0.0, 0.0))

    def test_a_flat_surface_has_zero_noise_corrected_signal(self):
        # Every point shares one true value; the medians still scatter, and the
        # correction must recognise that scatter as noise rather than shape.
        rng = random.Random(9)
        per_seed = [[rng.gauss(0.0, 0.10) for _ in range(30)] for _ in range(11)]
        *_, true_between = sweep.resolving_power(per_seed)
        self.assertLess(true_between, 0.01)

    def test_a_real_surface_survives_the_correction(self):
        rng = random.Random(10)
        per_seed = [[i * 0.10 + rng.gauss(0.0, 0.02) for _ in range(30)] for i in range(11)]
        *_, true_between = sweep.resolving_power(per_seed)
        self.assertGreater(true_between, 0.20)

    def test_more_seeds_do_not_manufacture_signal_from_a_flat_surface(self):
        # The decisive property: on a flat surface the ratio does not improve
        # with seed count, because between and within shrink together.
        def ratio_for(k):
            rng = random.Random(11)
            return sweep.resolving_power(
                [[rng.gauss(0.0, 0.10) for _ in range(k)] for _ in range(11)]
            )[2]
        self.assertLess(ratio_for(400), 2.0)
        self.assertLess(ratio_for(25), 2.0)

    def test_within_point_error_shrinks_with_seed_count(self):
        rng = random.Random(4)
        few = [[rng.gauss(0.0, 0.1) for _ in range(4)] for _ in range(6)]
        many = [[rng.gauss(0.0, 0.1) for _ in range(64)] for _ in range(6)]
        self.assertGreater(sweep.resolving_power(few)[1], sweep.resolving_power(many)[1])


class TestPeakDetection(unittest.TestCase):
    def test_an_isolated_best_is_peaked(self):
        values = [10, 15, 20, 25, 30]
        returns = [0.0, 0.0, 0.5, 0.0, 0.0]
        peaked, margin = sweep.is_peaked(values, returns, 20)
        self.assertTrue(peaked)
        self.assertAlmostEqual(margin, 0.5)

    def test_a_broad_plateau_is_not_peaked(self):
        values = [10, 15, 20, 25, 30]
        returns = [0.30, 0.32, 0.34, 0.33, 0.31]
        peaked, _ = sweep.is_peaked(values, returns, 20)
        self.assertFalse(peaked, "a value barely ahead of its neighbours is not a peak")

    def test_a_configured_value_that_is_not_best_is_not_peaked(self):
        values = [10, 15, 20]
        returns = [0.9, 0.1, 0.1]
        peaked, _ = sweep.is_peaked(values, returns, 20)
        self.assertFalse(peaked)

    def test_missing_default_is_handled(self):
        peaked, margin = sweep.is_peaked([1, 2], [0.1, 0.2], 99)
        self.assertFalse(peaked)
        self.assertEqual(margin, 0.0)


class TestParameterVariation(unittest.TestCase):
    def test_integers_stay_integral(self):
        for value in sweep.vary(20, 0.3, 7):
            self.assertIsInstance(value, int)

    def test_the_current_value_is_included(self):
        self.assertIn(20, sweep.vary(20, 0.3, 7))

    def test_the_span_matches_the_requested_spread(self):
        values = sweep.vary(100, 0.3, 7)
        self.assertEqual(min(values), 70)
        self.assertEqual(max(values), 130)

    def test_floats_keep_precision(self):
        values = sweep.vary(2.5, 0.2, 5)
        self.assertAlmostEqual(min(values), 2.0)
        self.assertAlmostEqual(max(values), 3.0)

    def test_values_never_fall_below_one(self):
        self.assertTrue(all(v >= 1 for v in sweep.vary(1, 0.9, 5)))

    def test_duplicates_are_collapsed(self):
        values = sweep.vary(3, 0.1, 9)
        self.assertEqual(len(values), len(set(values)))


class TestStrategyParameterDiscovery(unittest.TestCase):
    def test_numeric_defaults_are_found(self):
        params = sweep.current_parameters("rsi-breakout")
        self.assertIn("lookback", params)
        self.assertIn("max_entry_rsi", params)

    def test_non_numeric_and_flag_parameters_are_skipped(self):
        params = sweep.current_parameters("sma-crossover")
        self.assertNotIn("allow_short", params, "booleans are not dials to sweep")

    def test_ensemble_skips_its_member_list(self):
        params = sweep.current_parameters("ensemble")
        self.assertNotIn("members", params)
        self.assertNotIn("mode", params)


class TestDetectionFloor(unittest.TestCase):
    """A clamped estimate must never be reported as a measurement of zero."""

    def test_a_clamped_signal_is_labelled_not_zeroed(self):
        text = sweep.describe_signal(between=0.01, within=0.03, true_between=0.0)
        self.assertIn("below detection floor", text)
        self.assertNotIn("0.00%", text)

    def test_a_resolved_signal_reports_its_value_and_floor(self):
        text = sweep.describe_signal(between=0.10, within=0.02, true_between=0.098)
        self.assertIn("9.8", text)
        self.assertIn("floor", text)

    def test_the_floor_scales_with_the_within_point_error(self):
        self.assertGreater(sweep.detection_floor(0.05), sweep.detection_floor(0.01))

    def test_repeated_clamps_are_not_independent_measurements(self):
        # Three seed counts all bottoming out is one floor hit three times.
        # Each must describe itself as bounded, never as a measured zero.
        for within in (0.07, 0.04, 0.02):
            text = sweep.describe_signal(between=within * 0.5, within=within,
                                         true_between=0.0)
            self.assertIn("below detection floor", text)


class TestMonotonicTrend(unittest.TestCase):
    """Ordering is exactly what roughness discards, so it needs its own test."""

    def test_spearman_handles_perfect_orderings(self):
        self.assertAlmostEqual(sweep.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(sweep.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_spearman_is_safe_on_ties(self):
        self.assertAlmostEqual(sweep.spearman([1, 1, 1], [1, 2, 3]), 0.0)

    def test_a_monotone_ramp_is_the_smoothest_possible_sequence(self):
        # The reason a real trend is invisible to roughness: it scores the
        # minimum, so it looks maximally unremarkable.
        ramp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertAlmostEqual(sweep.roughness(ramp), sweep.smooth_reference(7), places=6)
        self.assertEqual(sweep.classify_roughness(sweep.roughness(ramp), 7), "smooth")

    def test_a_trend_invisible_to_roughness_is_caught_by_the_trend_test(self):
        rng = random.Random(21)
        values = list(range(9))
        # Effect far smaller than the per-seed noise, but consistent in sign.
        per_seed = [
            [i * -0.004 + rng.gauss(0.0, 0.05) for _ in range(60)] for i in range(9)
        ]
        _, _, ratio, _ = sweep.resolving_power(per_seed)
        trend = sweep.monotonic_trend(values, per_seed)
        self.assertLess(ratio, 2.0, "the variance ratio should miss this")
        self.assertLess(trend["p"], 0.05, "the trend test should catch it")
        self.assertLess(trend["rho"], 0)

    def test_no_trend_on_pure_noise(self):
        rng = random.Random(22)
        per_seed = [[rng.gauss(0.0, 0.05) for _ in range(60)] for _ in range(9)]
        self.assertGreater(sweep.monotonic_trend(list(range(9)), per_seed)["p"], 0.05)

    def test_degenerate_input_is_safe(self):
        self.assertEqual(sweep.monotonic_trend([1, 2], [[0.1], [0.2]])["p"], 1.0)


class TestPositiveControl(unittest.TestCase):
    """The null test alone is unfalsifiable: a broken detector also passes it."""

    def test_zero_effect_fires_at_about_the_alpha_rate(self):
        rate = sweep.positive_control(effect=0.0, points=9, seeds=25, trials=200)
        self.assertLess(rate, 0.12, "a detector that fires on nothing is broken")

    def test_a_large_effect_is_detected_essentially_always(self):
        rate = sweep.positive_control(effect=0.20, points=9, seeds=25, trials=100)
        self.assertGreater(rate, 0.90,
                           "a detector that never fires would pass the null test too")

    def test_detection_rate_rises_with_effect_size(self):
        small = sweep.positive_control(effect=0.01, points=9, seeds=25, trials=150)
        large = sweep.positive_control(effect=0.10, points=9, seeds=25, trials=150)
        self.assertGreater(large, small)

    def test_more_seeds_lower_the_minimum_detectable_effect(self):
        few = sweep.minimum_detectable_effect(9, 10, trials=60)
        many = sweep.minimum_detectable_effect(9, 200, trials=60)
        self.assertGreaterEqual(few, many)

    def test_the_minimum_detectable_effect_is_finite_at_the_defaults(self):
        mde = sweep.minimum_detectable_effect(15, 25, trials=60)
        self.assertLess(mde, 0.10, "the sweep must be able to see something")


class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_csvs(self, count=2, bars=400):
        for index in range(count):
            write_csv(
                SyntheticFeed(symbol=f"SYN{index}", bars=bars, seed=index + 1),
                self.dir / f"syn{index}.csv",
            )

    def run_script(self, argv):
        out, err = io.StringIO(), io.StringIO()
        old = sys.argv
        sys.argv = ["evaluate_real.py", *argv]
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = evaluate.main()
        finally:
            sys.argv = old
        return code, out.getvalue(), err.getvalue()

    def test_csv_dir_runs_without_any_network(self):
        self.make_csvs()
        code, out, _ = self.run_script(["--csv-dir", str(self.dir), "--strategy", "breakout"])
        self.assertEqual(code, 0)
        self.assertIn("local CSV files", out)
        self.assertIn("SYN0", out)
        self.assertIn("SYN1", out)

    def test_each_instrument_is_reported_separately(self):
        self.make_csvs(count=3)
        code, out, _ = self.run_script(["--csv-dir", str(self.dir)])
        self.assertEqual(code, 0)
        for name in ("SYN0", "SYN1", "SYN2"):
            self.assertIn(name, out)

    def test_filename_becomes_the_instrument_name(self):
        write_csv(SyntheticFeed(symbol="X", bars=300, seed=4), self.dir / "spy.csv")
        code, out, _ = self.run_script(["--csv-dir", str(self.dir)])
        self.assertEqual(code, 0)
        self.assertIn("SPY", out)

    def test_cost_comparison_is_present(self):
        self.make_csvs()
        code, out, _ = self.run_script(["--csv-dir", str(self.dir)])
        self.assertIn("with costs", out)
        self.assertIn("no costs", out)

    def test_a_missing_directory_is_reported(self):
        code, _, err = self.run_script(["--csv-dir", str(self.dir / "nope")])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)

    def test_an_empty_directory_explains_the_format(self):
        code, _, err = self.run_script(["--csv-dir", str(self.dir)])
        self.assertEqual(code, 2)
        self.assertIn("no .csv files", err)
        self.assertIn("one file per instrument", err)

    def test_an_unreadable_file_is_skipped_not_fatal(self):
        self.make_csvs(count=1)
        (self.dir / "broken.csv").write_text("date,close\nnot-a-date,abc\n", encoding="utf-8")
        code, out, err = self.run_script(["--csv-dir", str(self.dir)])
        self.assertEqual(code, 0, "one bad file must not sink the whole evaluation")
        self.assertIn("BROKEN", err)
        self.assertIn("SYN0", out)

    def test_csv_format_help_is_printed_and_exits(self):
        code, out, _ = self.run_script(["--csv-format"])
        self.assertEqual(code, 0)
        self.assertIn("one file per instrument", out)
        self.assertIn("chronological", out)


class TestBenchmarkIsUnmanaged(unittest.TestCase):
    """The buy-and-hold baseline must not be run through the risk overlay.

    Found on real data: over 1999-2018 the S&P rose 104%, but the evaluation
    reported buy-and-hold at -8.5%, because the default 25% drawdown limit
    flattened the hold position in the 2001 crash and never re-entered. Every
    "edge vs B&H" number in the table was then measuring the circuit breaker.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "crash.csv"
        # Up 50%, down 40% (breaching any 25% drawdown limit), then up to a
        # new high. A real hold ends well ahead; a halted one ends at the
        # bottom, in cash.
        legs = [(0, 60, 100.0, 150.0), (60, 140, 150.0, 90.0), (140, 260, 90.0, 240.0)]
        rows = []
        for start, end, first, last in legs:
            span = end - start
            for step in range(span):
                rows.append(first + (last - first) * step / span)
        rows.append(240.0)
        lines = ["date,open,high,low,close,volume"]
        start_day = datetime.date(2000, 1, 3)
        for index, price in enumerate(rows):
            day = start_day + datetime.timedelta(days=index)
            lines.append(f"{day.isoformat()},{price:.4f},{price * 1.005:.4f},"
                         f"{price * 0.995:.4f},{price:.4f},1000000")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.total = rows[-1] / rows[0] - 1.0

    def tearDown(self):
        self._tmp.cleanup()

    def test_hold_through_a_deep_drawdown_tracks_the_instrument(self):
        result = evaluate.benchmark(self.path, "CRASH", 10_000.0)
        self.assertFalse(result.halted, f"benchmark was halted: {result.halt_reason}")
        self.assertGreater(
            result.metrics.total_return, 0.9 * self.total,
            "buy-and-hold must track the instrument, not be flattened by the "
            "drawdown limit partway through",
        )

    def test_the_managed_run_is_what_the_benchmark_must_not_be(self):
        # Guards the premise: with the overlay on, the same hold really does
        # halt and end far below the instrument. If this ever stops halting the
        # test above proves nothing.
        managed = evaluate.run("buy-and-hold", self.path, "CRASH", 10_000.0)
        self.assertTrue(managed.halted)
        self.assertLess(managed.metrics.total_return, 0.5 * self.total)


if __name__ == "__main__":
    unittest.main()
