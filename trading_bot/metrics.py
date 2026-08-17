"""Performance statistics computed from an equity curve and a trade list."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .models import EquityPoint, Trade

TRADING_DAYS_PER_YEAR = 252

MIN_YEARS_TO_ANNUALISE = 7 / 365.25
"""Below one week, annualising is meaningless and numerically explosive: a 5
minute session raises the return ratio to the ~100,000th power, which overflows
long before it says anything about a year. Short runs report the period return
instead, and ``Metrics.annualised`` records which one you are looking at."""


@dataclass
class Metrics:
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    exposure: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    total_commission: float = 0.0
    #: False when the run was too short to annualise; ``cagr`` is then the
    #: plain period return, and Calmar is derived from it.
    annualised: bool = False
    start: datetime | None = None
    end: datetime | None = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"start", "end", "extras"}
        }
        data["start"] = self.start.isoformat() if self.start else None
        data["end"] = self.end.isoformat() if self.end else None
        data.update(self.extras)
        return data

    def format_report(self, title: str = "Backtest results") -> str:
        """Render the metrics as an aligned plain-text report."""
        period = ""
        if self.start and self.end:
            period = f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"

        rows: list[tuple[str, str]] = [
            ("Period", period),
            ("Starting equity", f"${self.starting_equity:,.2f}"),
            ("Ending equity", f"${self.ending_equity:,.2f}"),
            ("Total return", f"{self.total_return:+.2%}"),
            ("CAGR" if self.annualised else "Period return", f"{self.cagr:+.2%}"),
            ("Annual volatility", f"{self.annual_volatility:.2%}"),
            ("Sharpe ratio", f"{self.sharpe_ratio:.2f}"),
            ("Sortino ratio", f"{self.sortino_ratio:.2f}"),
            ("Calmar ratio", f"{self.calmar_ratio:.2f}"),
            ("Max drawdown", f"{self.max_drawdown:.2%}"),
            ("Max DD duration", f"{self.max_drawdown_duration} bars"),
            ("Time in market", f"{self.exposure:.1%}"),
            ("Trades", f"{self.num_trades}"),
            ("Win rate", f"{self.win_rate:.1%}"),
            ("Profit factor", f"{self.profit_factor:.2f}" if self.profit_factor != math.inf else "inf"),
            ("Average win", f"${self.average_win:,.2f}"),
            ("Average loss", f"${self.average_loss:,.2f}"),
            ("Largest win", f"${self.largest_win:,.2f}"),
            ("Largest loss", f"${self.largest_loss:,.2f}"),
            ("Expectancy/trade", f"${self.expectancy:,.2f}"),
            ("Commission paid", f"${self.total_commission:,.2f}"),
        ]
        for key, value in self.extras.items():
            rows.append((key.replace("_", " ").capitalize(), str(value)))

        width = max(len(label) for label, _ in rows)
        lines = [title, "=" * max(len(title), width + 22)]
        lines += [f"{label:<{width}}  {value:>18}" for label, value in rows if value != ""]
        return "\n".join(lines)


def _returns(curve: list[EquityPoint]) -> list[float]:
    out = []
    for previous, current in zip(curve, curve[1:]):
        if previous.equity > 0:
            out.append(current.equity / previous.equity - 1.0)
    return out


SECONDS_PER_YEAR = 365.25 * 86_400

#: Below this much elapsed history, bar density is too noisy to trust and the
#: spacing heuristic is used instead.
MIN_SECONDS_FOR_DENSITY = 30 * 86_400


def _bars_per_year(curve: list[EquityPoint]) -> float:
    """How many bars this series actually contains per year.

    Measured, not assumed. The previous version inferred a market from the
    median gap, which forced every daily series to 252 bars a year and every
    intraday one to a 6.5 hour session — understating a 24/7 crypto Sharpe by
    about 17%, and a weekly series by treating 52 bars as 36.

    Counting bars over elapsed time needs no such guess: a weekday-only series
    lands near 252 on its own because the weekend gaps are already in the
    elapsed time, and a continuous series lands near 365 for the same reason.
    """
    if len(curve) < 3:
        return TRADING_DAYS_PER_YEAR

    elapsed = (curve[-1].timestamp - curve[0].timestamp).total_seconds()
    if elapsed >= MIN_SECONDS_FOR_DENSITY:
        return (len(curve) - 1) * SECONDS_PER_YEAR / elapsed

    # Too short to measure: fall back to spacing, but annualise on the calendar
    # rather than assuming an equity session.
    gaps = sorted(
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in zip(curve, curve[1:])
        if (b.timestamp - a.timestamp).total_seconds() > 0
    )
    if not gaps:
        return TRADING_DAYS_PER_YEAR
    median = gaps[len(gaps) // 2]
    return SECONDS_PER_YEAR / median


def max_drawdown(curve: list[EquityPoint]) -> tuple[float, int]:
    """Return the worst peak-to-trough decline and its longest duration in bars."""
    peak = -math.inf
    worst = 0.0
    peak_index = 0
    longest = 0
    for index, point in enumerate(curve):
        if point.equity > peak:
            peak = point.equity
            peak_index = index
        elif peak > 0:
            worst = max(worst, (peak - point.equity) / peak)
            longest = max(longest, index - peak_index)
    return worst, longest


def compute_metrics(
    curve: list[EquityPoint],
    trades: list[Trade],
    total_commission: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Metrics:
    """Summarise an equity curve and its trades.

    ``risk_free_rate`` is annualised and subtracted from returns before the
    Sharpe and Sortino ratios are computed.
    """
    metrics = Metrics(total_commission=total_commission, num_trades=len(trades))
    if not curve:
        return metrics

    metrics.start = curve[0].timestamp
    metrics.end = curve[-1].timestamp
    metrics.starting_equity = curve[0].equity
    metrics.ending_equity = curve[-1].equity
    if metrics.starting_equity > 0:
        metrics.total_return = metrics.ending_equity / metrics.starting_equity - 1.0

    returns = _returns(curve)
    periods_per_year = _bars_per_year(curve)

    elapsed_days = max((metrics.end - metrics.start).total_seconds() / 86_400, 0.0)
    years = elapsed_days / 365.25
    if (
        years >= MIN_YEARS_TO_ANNUALISE
        and metrics.starting_equity > 0
        and metrics.ending_equity > 0
    ):
        try:
            metrics.cagr = (metrics.ending_equity / metrics.starting_equity) ** (1 / years) - 1.0
            metrics.annualised = True
        except OverflowError:
            metrics.cagr = math.inf if metrics.ending_equity > metrics.starting_equity else -1.0
            metrics.annualised = True
    else:
        metrics.cagr = metrics.total_return

    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        metrics.annual_volatility = std * math.sqrt(periods_per_year)

        periodic_rf = risk_free_rate / periods_per_year
        excess = mean - periodic_rf
        if std > 0:
            metrics.sharpe_ratio = (excess / std) * math.sqrt(periods_per_year)

        downside = [r - periodic_rf for r in returns if r < periodic_rf]
        if downside:
            downside_std = math.sqrt(sum(d**2 for d in downside) / len(downside))
            if downside_std > 0:
                metrics.sortino_ratio = (excess / downside_std) * math.sqrt(periods_per_year)
        elif excess > 0:
            metrics.sortino_ratio = math.inf

    metrics.max_drawdown, metrics.max_drawdown_duration = max_drawdown(curve)
    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown

    invested = sum(1 for point in curve if abs(point.exposure) > 1e-9)
    metrics.exposure = invested / len(curve)

    if trades:
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        metrics.win_rate = len(wins) / len(trades)
        metrics.average_win = sum(wins) / len(wins) if wins else 0.0
        metrics.average_loss = sum(losses) / len(losses) if losses else 0.0
        metrics.largest_win = max(wins) if wins else 0.0
        metrics.largest_loss = min(losses) if losses else 0.0
        metrics.expectancy = sum(t.pnl for t in trades) / len(trades)
        gross_loss = abs(sum(losses))
        gross_profit = sum(wins)
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics.profit_factor = math.inf

    return metrics
