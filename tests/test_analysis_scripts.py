"""The offline evaluation path and the robustness sweep's detectors.

The sweep's job is to spot a strategy fitted to one parameter value. A detector
that has never been shown to fire is not evidence of anything, so the roughness
and peak metrics are tested against sequences whose shape is known by
construction.
"""

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


if __name__ == "__main__":
    unittest.main()
