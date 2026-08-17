"""Evaluate a strategy against real market data across several regimes.

    python scripts/evaluate_real.py                       # default basket
    python scripts/evaluate_real.py --strategy breakout
    python scripts/evaluate_real.py --instruments SPY:yahoo QQQ:yahoo

Runs each instrument separately — never pooled — because a single blended
number hides the case where one instrument carries everything. For each it
reports the strategy against buy-and-hold over the identical period, then
repeats the run with costs stripped out, then flags the statistical problems
that make a good-looking backtest untradeable.

Bars are fetched once per instrument and frozen to CSV, so every variant runs
on byte-identical data.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import RunConfig  # noqa: E402
from trading_bot.data import write_csv  # noqa: E402
from trading_bot.providers import DataFeedError, build_provider  # noqa: E402

# A deliberate spread of regimes. Trend, chop and drawdown are different
# problems, and a strategy that only survives one of them has not been tested.
DEFAULT_BASKET = [
    ("SPY", "yahoo", "broad US equity: long uptrend with sharp drawdowns"),
    ("QQQ", "yahoo", "tech: strong trend then a deep 2022 bear market"),
    ("XLU", "yahoo", "utilities: low-beta, largely sideways"),
    ("BTCUSDT", "binance", "crypto: violent trends and 70%+ drawdowns"),
    ("TLT", "yahoo", "long bonds: sustained multi-year downtrend"),
]

#: Below this, per-trade statistics are noise rather than evidence.
MIN_TRADES_FOR_SIGNIFICANCE = 30


def fetch(symbol: str, provider_name: str, interval: str, limit: int, out_dir: Path) -> Path:
    provider = build_provider(provider_name, cache_dir=".cache/market-data")
    candles = provider.fetch(symbol, interval, limit)
    destination = out_dir / f"{symbol.lower()}-{interval}.csv"
    write_csv(candles, destination)
    return destination


def run(strategy: str, path: Path, symbol: str, cash: float, **overrides):
    config = RunConfig(
        strategy=strategy, symbol=symbol, data_file=str(path),
        starting_cash=cash, cache_dir=None, **overrides,
    )
    return config.build_engine().run(config.build_feed())


def hold_durations(result) -> float:
    if not result.trades:
        return 0.0
    days = [(t.exit_time - t.entry_time).total_seconds() / 86_400 for t in result.trades]
    return statistics.mean(days)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="rsi-breakout")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=750, help="bars per instrument (~3y daily)")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--out-dir", default="data/evaluation")
    parser.add_argument("--instruments", nargs="*",
                        help="SYMBOL:PROVIDER pairs, overriding the default basket")
    args = parser.parse_args()

    if args.instruments:
        basket = []
        for item in args.instruments:
            symbol, _, provider = item.partition(":")
            basket.append((symbol, provider or "yahoo", ""))
    else:
        basket = DEFAULT_BASKET

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for symbol, provider_name, description in basket:
        try:
            path = fetch(symbol, provider_name, args.interval, args.limit, out_dir)
        except DataFeedError as exc:
            failures.append((symbol, str(exc)))
            continue

        strat = run(args.strategy, path, symbol, args.cash)
        hold = run("buy-and-hold", path, symbol, args.cash)
        free = run(args.strategy, path, symbol, args.cash,
                   commission_percent=0.0, slippage_percent=0.0, spread_fraction=0.0)
        rows.append((symbol, description, strat, hold, free))

    if failures:
        print("Could not fetch:", file=sys.stderr)
        for symbol, error in failures:
            print(f"  {symbol}: {error}", file=sys.stderr)
        if not rows:
            print(
                "\nNo data retrieved, so there is nothing to evaluate. If these are "
                "403s from a proxy, the hosts are blocked by an egress policy and an "
                "administrator has to allow them; the strategy is untested until then.",
                file=sys.stderr,
            )
            return 2

    # -- per instrument, never pooled ---------------------------------------
    header = (f"{'instrument':<11}{'period':<20}{'strat':>8}{'B&H':>8}{'edge':>8}"
              f"{'sharpe':>8}{'maxDD':>8}{'trades':>8}{'win%':>7}{'avg hold':>10}")
    print(header)
    print("-" * len(header))
    for symbol, _, strat, hold, _free in rows:
        m, b = strat.metrics, hold.metrics
        period = f"{m.start:%Y-%m}..{m.end:%Y-%m}" if m.start else "?"
        print(
            f"{symbol:<11}{period:<20}{m.total_return:>7.1%}{b.total_return:>8.1%}"
            f"{m.total_return - b.total_return:>+8.1%}{m.sharpe_ratio:>8.2f}"
            f"{m.max_drawdown:>8.1%}{m.num_trades:>8}{m.win_rate:>7.0%}"
            f"{hold_durations(strat):>9.0f}d"
        )

    # -- cost sensitivity on the same real data -----------------------------
    print(f"\n{'instrument':<11}{'with costs':>12}{'no costs':>12}{'cost drag':>12}"
          f"{'fees paid':>12}{'beats B&H?':>12}")
    print("-" * 71)
    for symbol, _, strat, hold, free in rows:
        m, b, f = strat.metrics, hold.metrics, free.metrics
        beats = "yes" if m.total_return > b.total_return else "NO"
        print(
            f"{symbol:<11}{m.total_return:>11.1%}{f.total_return:>12.1%}"
            f"{f.total_return - m.total_return:>+12.1%}"
            f"{m.total_commission:>11,.0f}{beats:>12}"
        )

    # -- the things that make a good backtest untradeable -------------------
    print("\nWarnings")
    print("-" * 71)
    warnings = []
    for symbol, _, strat, hold, free in rows:
        m, b, f = strat.metrics, hold.metrics, free.metrics
        if m.num_trades < MIN_TRADES_FOR_SIGNIFICANCE:
            warnings.append(
                f"{symbol}: only {m.num_trades} trades — below {MIN_TRADES_FOR_SIGNIFICANCE}, "
                "so win rate and expectancy are noise, not evidence"
            )
        if f.total_return > b.total_return >= m.total_return:
            warnings.append(
                f"{symbol}: beats buy-and-hold before costs ({f.total_return:.1%}) but not "
                f"after ({m.total_return:.1%}) — the edge is smaller than the friction"
            )
        if m.num_trades and m.largest_win > 0 and m.total_return > 0:
            gross = m.average_win * m.win_rate * m.num_trades
            if gross > 0 and m.largest_win / gross > 0.5:
                warnings.append(
                    f"{symbol}: one trade is {m.largest_win / gross:.0%} of gross profit — "
                    "the result rests on a single lucky period"
                )
        if m.exposure < 0.05 and m.num_trades:
            warnings.append(f"{symbol}: in the market only {m.exposure:.1%} of the time")

    beat_count = sum(1 for _, _, s, h, _ in rows if s.metrics.total_return > h.metrics.total_return)
    if beat_count == 0:
        warnings.append("beats buy-and-hold on ZERO instruments after costs")
    elif beat_count < len(rows) / 2:
        warnings.append(
            f"beats buy-and-hold on only {beat_count} of {len(rows)} instruments — "
            "consistent with noise rather than an edge"
        )

    for warning in warnings or ["none"]:
        print(f"  - {warning}")

    # -- verdict ------------------------------------------------------------
    total_trades = sum(s.metrics.num_trades for _, _, s, _, _ in rows)
    median_edge = statistics.median(
        s.metrics.total_return - h.metrics.total_return for _, _, s, h, _ in rows
    )
    print("\nVerdict")
    print("-" * 71)
    print(f"  instruments tested     : {len(rows)}")
    print(f"  beats buy-and-hold     : {beat_count}/{len(rows)} after costs")
    print(f"  median edge vs B&H     : {median_edge:+.1%}")
    print(f"  total trades           : {total_trades}")
    if beat_count == 0 or median_edge <= 0:
        print("\n  NO EDGE. The strategy does not beat buy-and-hold after realistic")
        print("  costs on this basket. Do not deploy it.")
    elif total_trades < MIN_TRADES_FOR_SIGNIFICANCE * len(rows):
        print("\n  UNPROVEN. There is a positive median edge, but too few trades to")
        print("  distinguish it from luck. Widen the sample before believing it.")
    else:
        print("\n  POSITIVE but unconfirmed. Test out-of-sample on periods and")
        print("  instruments not used here before treating this as real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
