"""Compare strategies on real market data, on a frozen window.

    python scripts/compare_real.py                       # yahoo AAPL, daily
    python scripts/compare_real.py --provider binance --symbol BTCUSDT --interval 4h
    python scripts/compare_real.py --strategies breakout rsi-breakout

Bars are fetched once to a CSV and every strategy is run against that file, so
the comparison is over identical data. Fetching per strategy would give each a
slightly different window — the newest bar closes while the sweep runs — and
differences of a few percent would then be an artefact of timing rather than a
result.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import RunConfig  # noqa: E402
from trading_bot.data import write_csv  # noqa: E402
from trading_bot.providers import PROVIDERS, DataFeedError, build_provider  # noqa: E402

DEFAULT_STRATEGIES = ("buy-and-hold", "breakout", "rsi-breakout", "sma-crossover")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="yahoo", choices=sorted(PROVIDERS))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--strategies", nargs="*", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--out", default=None, help="where to freeze the bars")
    args = parser.parse_args()

    destination = Path(args.out or f"data/{args.symbol.replace('/', '-').lower()}-{args.interval}.csv")

    try:
        provider = build_provider(args.provider, cache_dir=".cache/market-data")
        candles = provider.fetch(args.symbol, args.interval, args.limit)
    except DataFeedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nIf this is a 403 from a proxy, the host is blocked by an egress "
            "policy rather than being down; an administrator has to allow it.",
            file=sys.stderr,
        )
        return 2

    write_csv(candles, destination)
    print(
        f"{len(candles)} {args.interval} bars of {args.symbol.upper()} from {args.provider}: "
        f"{candles[0].timestamp:%Y-%m-%d} to {candles[-1].timestamp:%Y-%m-%d}, "
        f"last close {candles[-1].close:,.4f}\nFrozen at {destination}\n"
    )

    rows = []
    for name in args.strategies:
        config = RunConfig(
            strategy=name,
            symbol=args.symbol.upper(),
            data_file=str(destination),
            starting_cash=args.cash,
        )
        try:
            result = config.build_engine().run(config.build_feed())
        except (ValueError, KeyError) as exc:
            print(f"  {name}: skipped ({exc})", file=sys.stderr)
            continue
        rows.append((name, result))

    if not rows:
        print("No strategies ran.", file=sys.stderr)
        return 1

    header = (f"{'strategy':<18}{'return':>9}{'CAGR':>8}{'sharpe':>8}{'maxDD':>8}"
              f"{'trades':>8}{'win%':>7}{'profitF':>9}{'in-mkt':>8}")
    print(header)
    print("-" * len(header))
    for name, result in sorted(rows, key=lambda r: r[1].metrics.sharpe_ratio, reverse=True):
        m = result.metrics
        factor = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        print(
            f"{name:<18}{m.total_return:>8.1%}{m.cagr:>8.1%}{m.sharpe_ratio:>8.2f}"
            f"{m.max_drawdown:>8.1%}{m.num_trades:>8}{m.win_rate:>7.0%}{factor:>9}"
            f"{m.exposure:>8.1%}"
        )

    benchmark = next((r for n, r in rows if n == "buy-and-hold"), None)
    if benchmark is not None:
        print(f"\nBenchmark (buy-and-hold): {benchmark.metrics.total_return:+.1%}")
        beat = [n for n, r in rows
                if n != "buy-and-hold"
                and r.metrics.total_return > benchmark.metrics.total_return]
        print(f"Beat it: {', '.join(beat) if beat else 'none'}")

    print(
        "\nOne symbol over one window is one sample. Before believing any of "
        "this, re-run it on other symbols, other periods, and neighbouring "
        "parameters — and remember the fills are simulated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
