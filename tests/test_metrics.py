import math
import unittest
from datetime import datetime, timedelta

from trading_bot.metrics import compute_metrics, max_drawdown
from trading_bot.models import EquityPoint, Side, Trade


def curve(values, start=None, step=timedelta(days=1), exposure=1.0):
    start = start or datetime(2020, 1, 1)
    return [
        EquityPoint(start + step * i, cash=0.0, equity=value, exposure=exposure)
        for i, value in enumerate(values)
    ]


def trade(pnl, day=1):
    return Trade(
        symbol="X",
        side=Side.BUY,
        quantity=10,
        entry_time=datetime(2020, 1, day),
        entry_price=100.0,
        exit_time=datetime(2020, 1, day + 1),
        exit_price=100.0 + pnl / 10,
        pnl=pnl,
        commission=0.0,
    )


class TestMaxDrawdown(unittest.TestCase):
    def test_monotonic_rise_has_no_drawdown(self):
        worst, duration = max_drawdown(curve([100, 110, 120, 130]))
        self.assertAlmostEqual(worst, 0.0)
        self.assertEqual(duration, 0)

    def test_measures_peak_to_trough(self):
        worst, _ = max_drawdown(curve([100, 200, 100, 150]))
        self.assertAlmostEqual(worst, 0.5)

    def test_duration_counts_bars_below_the_peak(self):
        _, duration = max_drawdown(curve([100, 200, 150, 150, 150]))
        self.assertEqual(duration, 3)

    def test_empty_curve_is_safe(self):
        self.assertEqual(max_drawdown([]), (0.0, 0))


class TestComputeMetrics(unittest.TestCase):
    def test_empty_inputs_produce_zeroed_metrics(self):
        metrics = compute_metrics([], [])
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.num_trades, 0)
        self.assertIsNone(metrics.start)

    def test_total_return(self):
        metrics = compute_metrics(curve([100, 110, 120]), [])
        self.assertAlmostEqual(metrics.total_return, 0.2)
        self.assertAlmostEqual(metrics.starting_equity, 100)
        self.assertAlmostEqual(metrics.ending_equity, 120)

    def test_cagr_over_one_year(self):
        points = [
            EquityPoint(datetime(2020, 1, 1), 0, 100.0),
            EquityPoint(datetime(2020, 7, 1), 0, 120.0),
            EquityPoint(datetime(2020, 12, 31), 0, 150.0),
        ]
        metrics = compute_metrics(points, [])
        self.assertAlmostEqual(metrics.cagr, 0.5, places=1)

    def test_flat_curve_has_zero_volatility_and_sharpe(self):
        metrics = compute_metrics(curve([100] * 20), [])
        self.assertAlmostEqual(metrics.annual_volatility, 0.0)
        self.assertAlmostEqual(metrics.sharpe_ratio, 0.0)

    def test_steady_gains_produce_a_high_sharpe(self):
        metrics = compute_metrics(curve([100 * 1.001**i for i in range(200)]), [])
        self.assertGreater(metrics.sharpe_ratio, 10)

    def test_sortino_ignores_upside_volatility(self):
        # Alternating flat and up moves: no downside deviation at all.
        values = [100.0]
        for i in range(40):
            values.append(values[-1] * (1.02 if i % 2 else 1.0))
        metrics = compute_metrics(curve(values), [])
        self.assertEqual(metrics.sortino_ratio, math.inf)

    def test_risk_free_rate_lowers_the_sharpe_ratio(self):
        points = curve([100 * 1.0005**i for i in range(200)])
        base = compute_metrics(points, []).sharpe_ratio
        adjusted = compute_metrics(points, [], risk_free_rate=0.05).sharpe_ratio
        self.assertLess(adjusted, base)

    def test_trade_statistics(self):
        trades = [trade(100, 1), trade(50, 3), trade(-60, 5), trade(-40, 7)]
        metrics = compute_metrics(curve([100, 105]), trades)
        self.assertEqual(metrics.num_trades, 4)
        self.assertAlmostEqual(metrics.win_rate, 0.5)
        self.assertAlmostEqual(metrics.average_win, 75.0)
        self.assertAlmostEqual(metrics.average_loss, -50.0)
        self.assertAlmostEqual(metrics.largest_win, 100.0)
        self.assertAlmostEqual(metrics.largest_loss, -60.0)
        self.assertAlmostEqual(metrics.profit_factor, 150 / 100)
        self.assertAlmostEqual(metrics.expectancy, 12.5)

    def test_profit_factor_is_infinite_without_losses(self):
        metrics = compute_metrics(curve([100, 120]), [trade(50), trade(70)])
        self.assertEqual(metrics.profit_factor, math.inf)

    def test_exposure_counts_invested_bars(self):
        points = curve([100] * 4)
        points[0].exposure = 0.0
        points[1].exposure = 0.0
        metrics = compute_metrics(points, [])
        self.assertAlmostEqual(metrics.exposure, 0.5)

    def test_calmar_relates_cagr_to_drawdown(self):
        metrics = compute_metrics(curve([100, 200, 100, 150]), [])
        self.assertAlmostEqual(metrics.calmar_ratio, metrics.cagr / metrics.max_drawdown)

    def test_report_renders_every_line(self):
        metrics = compute_metrics(curve([100, 110, 120]), [trade(10)], total_commission=1.0)
        report = metrics.format_report("Test run")
        self.assertIn("Test run", report)
        self.assertIn("Total return", report)
        self.assertIn("Sharpe ratio", report)
        self.assertIn("Commission paid", report)

    def test_report_handles_infinite_profit_factor(self):
        metrics = compute_metrics(curve([100, 120]), [trade(50)])
        self.assertIn("inf", metrics.format_report())

    def test_as_dict_is_json_friendly(self):
        metrics = compute_metrics(curve([100, 110]), [trade(10)])
        data = metrics.as_dict()
        self.assertIsInstance(data["start"], str)
        self.assertIn("total_return", data)

    def test_extras_appear_in_dict_and_report(self):
        metrics = compute_metrics(curve([100, 110]), [])
        metrics.extras["rejected_orders"] = 3
        self.assertEqual(metrics.as_dict()["rejected_orders"], 3)
        self.assertIn("Rejected orders", metrics.format_report())


if __name__ == "__main__":
    unittest.main()
