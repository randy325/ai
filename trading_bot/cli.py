"""Command line interface.

    python -m trading_bot backtest --provider yahoo --symbol AAPL
    python -m trading_bot backtest --strategy sma-crossover --data prices.csv
    python -m trading_bot compare --bars 1000
    python -m trading_bot optimize --strategy sma-crossover
    python -m trading_bot fetch --provider binance --symbol BTCUSDT --limit 1000
    python -m trading_bot paper --provider binance --symbol BTCUSDT --interval 1m
    python -m trading_bot demo-data --out data/demo.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

from .config import RunConfig
from .data import SyntheticFeed, write_csv
from .engine import BacktestResult
from .providers import INTERVAL_SECONDS, PROVIDERS, DataFeedError, build_provider
from .strategy import STRATEGIES


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("data")
    data.add_argument("--data", dest="data_file", help="CSV file of OHLCV bars")
    data.add_argument("--provider", choices=sorted(PROVIDERS),
                      help="fetch real market prices from this provider")
    data.add_argument("--interval", default="1d", choices=sorted(INTERVAL_SECONDS),
                      help="bar size for --provider (default: 1d)")
    data.add_argument("--limit", type=int, default=500,
                      help="bars to fetch from --provider (default: 500)")
    data.add_argument("--cache-dir", default=".cache/market-data",
                      help="where to cache provider responses")
    data.add_argument("--no-cache", action="store_true",
                      help="bypass the response cache and always refetch")
    data.add_argument("--cache-hours", type=float, default=12.0,
                      help="how long a cached response stays fresh (default: 12)")
    data.add_argument("--symbol", default="SYNTH", help="symbol name (default: SYNTH)")
    data.add_argument("--bars", type=int, default=750, help="synthetic bars when no --data")
    data.add_argument("--seed", type=int, default=7, help="synthetic data seed")
    data.add_argument("--volatility", type=float, default=0.25, help="synthetic annual volatility")
    data.add_argument("--drift", type=float, default=0.08, help="synthetic annual drift")

    account = parser.add_argument_group("account")
    account.add_argument("--cash", type=float, default=100_000.0, help="starting cash")
    account.add_argument("--commission", type=float, default=0.001, help="commission as a fraction of notional")
    account.add_argument("--slippage", type=float, default=0.0005, help="slippage as a fraction of price")
    account.add_argument("--leverage", type=float, default=1.0, help="maximum gross leverage")

    risk = parser.add_argument_group("risk")
    risk.add_argument("--sizer", choices=["fixed", "volatility"], default="fixed")
    risk.add_argument("--position-fraction", type=float, default=0.95, help="max equity per position")
    risk.add_argument("--risk-per-trade", type=float, default=0.02, help="equity risked per trade (volatility sizer)")
    risk.add_argument("--atr-multiple", type=float, default=2.0, help="ATR stop distance (volatility sizer)")
    risk.add_argument("--max-drawdown", type=float, default=0.25, help="halt trading past this drawdown; 0 disables")
    risk.add_argument("--daily-loss-limit", type=float, default=None, help="pause for the day past this loss")
    risk.add_argument("--max-trades-per-day", type=int, default=None)

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--allow-short", action="store_true", help="permit short positions")
    behaviour.add_argument("--trend-filter", type=int, default=None, help="suppress trades against an N-period EMA")
    behaviour.add_argument(
        "--execute-on",
        choices=["next_open", "close"],
        default="next_open",
        help="next_open avoids lookahead bias (default); close is optimistic",
    )
    behaviour.add_argument("--config", help="JSON config file; CLI flags override its values")
    behaviour.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                           help="strategy parameter, repeatable (e.g. --param fast=10)")


def _coerce(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        params[key.strip()] = _coerce(value)
    return params


def _provided_flags(argv: list[str]) -> set[str]:
    """Names of the long options that appear literally in ``argv``.

    Only these override a config file, so a flag left at its argparse default
    doesn't silently replace a value the config file set.
    """
    names = set()
    for token in argv:
        if token.startswith("--"):
            name = token[2:].partition("=")[0]
            if name:
                names.add(name.replace("-", "_"))
    return names


def _config_from_args(args: argparse.Namespace, strategy: str | None = None) -> RunConfig:
    config = RunConfig.from_file(args.config) if getattr(args, "config", None) else RunConfig()
    provided = _provided_flags(getattr(args, "_argv", sys.argv[1:]))

    def take(name: str, value, config_field: str | None = None) -> None:
        """Apply a CLI value unless it is a default and the config file set one."""
        field = config_field or name
        if name in provided or not getattr(args, "config", None):
            setattr(config, field, value)

    if strategy is not None:
        config.strategy = strategy
    elif getattr(args, "strategy", None):
        take("strategy", args.strategy)

    take("data", args.data_file, "data_file")
    take("provider", args.provider)
    take("interval", args.interval)
    take("limit", args.limit)
    take("cache_hours", args.cache_hours)
    take("symbol", args.symbol)
    if args.no_cache:
        config.cache_dir = None
    else:
        take("cache_dir", args.cache_dir)
    take("bars", args.bars)
    take("seed", args.seed)
    take("volatility", args.volatility)
    take("drift", args.drift)
    take("cash", args.cash, "starting_cash")
    take("commission", args.commission, "commission_percent")
    take("slippage", args.slippage, "slippage_percent")
    take("leverage", args.leverage, "max_leverage")
    take("sizer", args.sizer)
    take("position_fraction", args.position_fraction)
    take("risk_per_trade", args.risk_per_trade)
    take("atr_multiple", args.atr_multiple)
    take("daily_loss_limit", args.daily_loss_limit)
    take("max_trades_per_day", args.max_trades_per_day)
    take("allow_short", args.allow_short)
    take("trend_filter", args.trend_filter)
    take("execute_on", args.execute_on)

    if "max_drawdown" in provided or not getattr(args, "config", None):
        config.max_drawdown = args.max_drawdown if args.max_drawdown else None

    params = _parse_params(args.param)
    if params:
        config.strategy_params = {**config.strategy_params, **params}

    return config


def _run(config: RunConfig) -> BacktestResult:
    engine = config.build_engine()
    return engine.run(config.build_feed())


def _export(result: BacktestResult, equity_path: str | None, trades_path: str | None) -> None:
    if equity_path:
        destination = Path(equity_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "cash", "equity", "exposure"])
            for point in result.equity_curve:
                writer.writerow(
                    [point.timestamp.isoformat(), f"{point.cash:.2f}",
                     f"{point.equity:.2f}", f"{point.exposure:.4f}"]
                )
        print(f"\nEquity curve written to {destination}")

    if trades_path:
        destination = Path(trades_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["symbol", "side", "quantity", "entry_time", "entry_price",
                 "exit_time", "exit_price", "pnl", "return_pct", "commission", "reason"]
            )
            for trade in result.trades:
                writer.writerow(
                    [trade.symbol, trade.side.value, f"{trade.quantity:.6f}",
                     trade.entry_time.isoformat(), f"{trade.entry_price:.4f}",
                     trade.exit_time.isoformat(), f"{trade.exit_price:.4f}",
                     f"{trade.pnl:.2f}", f"{trade.return_pct:.4f}",
                     f"{trade.commission:.2f}", trade.reason]
                )
        print(f"Trade log written to {destination}")


def cmd_backtest(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    result = _run(config)

    if args.json:
        payload = {
            "strategy": result.strategy,
            "symbol": result.symbol,
            "bars": result.bars,
            "halted": result.halted,
            "halt_reason": result.halt_reason,
            "metrics": result.metrics.as_dict(),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(result.summary())
        if args.show_trades and result.trades:
            print("\nTrades")
            print("-" * 78)
            print(f"{'entry':<12}{'exit':<12}{'side':<6}{'qty':>10}{'in':>10}{'out':>10}{'pnl':>14}")
            for trade in result.trades[: args.show_trades]:
                print(
                    f"{trade.entry_time:%Y-%m-%d}  {trade.exit_time:%Y-%m-%d}  "
                    f"{trade.side.value:<6}{trade.quantity:>10.2f}"
                    f"{trade.entry_price:>10.2f}{trade.exit_price:>10.2f}{trade.pnl:>14,.2f}"
                )
            if len(result.trades) > args.show_trades:
                print(f"... and {len(result.trades) - args.show_trades} more")

    _export(result, args.export_equity, args.export_trades)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    names = args.strategies or sorted(STRATEGIES)
    rows = []
    for name in names:
        config = _config_from_args(args, strategy=name)
        config.strategy_params = {}
        try:
            result = _run(config)
        except Exception as exc:  # a bad parameter set shouldn't kill the sweep
            print(f"  {name}: failed ({exc})", file=sys.stderr)
            continue
        rows.append((name, result))

    if not rows:
        print("No strategies ran successfully.", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: row[1].metrics.sharpe_ratio, reverse=True)
    header = f"{'strategy':<22}{'return':>10}{'CAGR':>9}{'sharpe':>9}{'maxDD':>9}{'trades':>8}{'win%':>8}"
    print(header)
    print("-" * len(header))
    for name, result in rows:
        m = result.metrics
        print(
            f"{name:<22}{m.total_return:>9.1%}{m.cagr:>9.1%}{m.sharpe_ratio:>9.2f}"
            f"{m.max_drawdown:>9.1%}{m.num_trades:>8}{m.win_rate:>8.0%}"
        )
    print(f"\nRanked by Sharpe ratio. Best: {rows[0][0]}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    grid = {}
    for spec in args.grid:
        if "=" not in spec:
            raise SystemExit(f"--grid expects KEY=V1,V2,..., got {spec!r}")
        key, _, values = spec.partition("=")
        grid[key.strip()] = [_coerce(v) for v in values.split(",")]

    if not grid:
        grid = {"fast": [5, 10, 20, 30], "slow": [40, 60, 100, 150]}

    keys = list(grid)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]
    print(f"Testing {len(combinations)} parameter combinations of {args.strategy}...\n")

    results = []
    for params in combinations:
        config = _config_from_args(args)
        config.strategy_params = {**config.strategy_params, **params}
        try:
            result = _run(config)
        except (ValueError, KeyError, TypeError):
            continue  # invalid combination, e.g. fast >= slow
        results.append((params, result.metrics))

    if not results:
        print("No valid parameter combinations.", file=sys.stderr)
        return 1

    results.sort(key=lambda row: row[1].sharpe_ratio, reverse=True)
    label_width = max(len(str(params)) for params, _ in results[: args.top])
    header = f"{'params':<{label_width}}{'return':>10}{'sharpe':>9}{'maxDD':>9}{'trades':>8}"
    print(header)
    print("-" * len(header))
    for params, metrics in results[: args.top]:
        print(
            f"{str(params):<{label_width}}{metrics.total_return:>9.1%}"
            f"{metrics.sharpe_ratio:>9.2f}{metrics.max_drawdown:>9.1%}{metrics.num_trades:>8}"
        )

    print(
        "\nNote: the best row here is the best fit to this one price series. "
        "Expect materially worse results out of sample."
    )
    return 0


def cmd_demo_data(args: argparse.Namespace) -> int:
    feed = SyntheticFeed(
        symbol=args.symbol,
        bars=args.bars,
        seed=args.seed,
        volatility=args.volatility,
        drift=args.drift,
    )
    path = write_csv(feed, args.out)
    print(f"Wrote {args.bars} bars of synthetic {args.symbol} data to {path}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download real bars and save them as a CSV the other commands can read."""
    provider = build_provider(
        args.provider,
        cache_dir=None if args.no_cache else args.cache_dir,
        cache_ttl=timedelta(hours=args.cache_hours),
    )
    candles = provider.fetch(args.symbol, args.interval, args.limit)
    destination = Path(args.out or f"data/{args.symbol.replace('/', '-').lower()}.csv")
    write_csv(candles, destination)
    print(
        f"Fetched {len(candles)} {args.interval} bars of {args.symbol.upper()} "
        f"from {provider.name}"
    )
    print(f"  {candles[0].timestamp:%Y-%m-%d %H:%M} to {candles[-1].timestamp:%Y-%m-%d %H:%M} UTC")
    print(f"  last close {candles[-1].close:,.4f}")
    print(f"  written to {destination}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    print("Live data providers (no API key required)\n")
    for name, factory in sorted(PROVIDERS.items()):
        doc = (factory.__doc__ or "").strip().splitlines()
        print(f"  {name:<10} {doc[0] if doc else ''}")
        print(f"  {'':<10} intervals: {', '.join(factory.intervals)}")
    print("\nSymbol formats: stooq aapl.us / ^spx, yahoo AAPL, "
          "binance BTCUSDT, coinbase BTC-USD")
    print("Example: python -m trading_bot backtest --provider yahoo --symbol AAPL")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    """Paper-trade live bars as they close. Places no real orders."""
    config = _config_from_args(args)
    if not config.provider:
        print("error: paper trading needs --provider", file=sys.stderr)
        return 2

    engine = config.build_engine()
    engine.close_at_end = False
    feed = config.build_live_feed(warmup=args.warmup, max_bars=args.max_bars)

    print(
        f"Paper trading {config.symbol} {config.interval} bars from {config.provider}.\n"
        f"Strategy: {engine.strategy.describe()}\n"
        f"Starting cash: ${config.starting_cash:,.2f}   "
        f"{'Bars: ' + str(args.max_bars) if args.max_bars else 'Running until interrupted'}\n"
        "No real orders are placed. Press Ctrl-C to stop.\n"
    )

    print(f"Warming up on {args.warmup} historical bars, then waiting for the next close.\n")
    print(f"{'time':<17}{'close':>12}{'equity':>14}{'target':>8}  signal")
    print("-" * 78)

    def on_bar(candle, signal, equity):
        # Warmup bars are history; only narrate what is happening now.
        if feed.live_bars == 0:
            return
        print(
            f"{candle.timestamp:%Y-%m-%d %H:%M}{candle.close:>12,.4f}"
            f"{equity:>14,.2f}{signal.target:>8.1f}  {signal.reason}"
        )

    engine.on_bar = on_bar
    engine.on_fill = lambda fill: print(
        f"  -> {fill.side.value.upper()} {fill.quantity:,.6f} @ {fill.price:,.4f} "
        f"(fee {fill.commission:,.2f})"
    )

    try:
        result = engine.run(feed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        result = None
    except DataFeedError as exc:
        print(f"\nfeed stopped: {exc}", file=sys.stderr)
        result = None

    broker = engine.broker
    print(f"\nFinal equity ${broker.equity():,.2f} "
          f"({broker.equity() / config.starting_cash - 1:+.2%})")
    position = broker.position(config.symbol)
    if not position.is_flat:
        print(f"Open position: {position.quantity:+,.6f} @ {position.average_price:,.4f}")
    if result is not None and result.trades:
        print(result.metrics.format_report(f"Live paper session: {config.symbol}"))
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    print("Available strategies\n")
    for name, factory in sorted(STRATEGIES.items()):
        doc = (factory.__doc__ or "").strip().splitlines()
        print(f"  {name:<22} {doc[0] if doc else ''}")
    print("\nPass parameters with --param, e.g. --param fast=10 --param slow=40")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading_bot",
        description="Backtesting and paper-trading bot. Simulation only — it places no real orders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every fill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="run one strategy over one series")
    backtest.add_argument("--strategy", default="sma-crossover", choices=sorted(STRATEGIES))
    backtest.add_argument("--json", action="store_true", help="emit metrics as JSON")
    backtest.add_argument("--show-trades", type=int, default=0, metavar="N", help="print the first N trades")
    backtest.add_argument("--export-equity", metavar="PATH", help="write the equity curve to CSV")
    backtest.add_argument("--export-trades", metavar="PATH", help="write the trade log to CSV")
    _add_common_arguments(backtest)
    backtest.set_defaults(func=cmd_backtest)

    compare = subparsers.add_parser("compare", help="run every strategy over the same series")
    compare.add_argument("--strategies", nargs="*", choices=sorted(STRATEGIES))
    _add_common_arguments(compare)
    compare.set_defaults(func=cmd_compare)

    optimize = subparsers.add_parser("optimize", help="grid-search a strategy's parameters")
    optimize.add_argument("--strategy", default="sma-crossover", choices=sorted(STRATEGIES))
    optimize.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2",
                          help="parameter values to sweep, repeatable")
    optimize.add_argument("--top", type=int, default=10, help="rows to display")
    _add_common_arguments(optimize)
    optimize.set_defaults(func=cmd_optimize)

    demo = subparsers.add_parser("demo-data", help="generate a synthetic OHLCV CSV file")
    demo.add_argument("--out", default="data/demo.csv")
    demo.add_argument("--symbol", default="DEMO")
    demo.add_argument("--bars", type=int, default=1000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--volatility", type=float, default=0.25)
    demo.add_argument("--drift", type=float, default=0.08)
    demo.set_defaults(func=cmd_demo_data)

    strategies = subparsers.add_parser("strategies", help="list available strategies")
    strategies.set_defaults(func=cmd_strategies)

    fetch = subparsers.add_parser("fetch", help="download real market bars to a CSV file")
    fetch.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    fetch.add_argument("--symbol", required=True, help="e.g. AAPL, BTCUSDT, BTC-USD")
    fetch.add_argument("--interval", default="1d", choices=sorted(INTERVAL_SECONDS))
    fetch.add_argument("--limit", type=int, default=500, help="how many bars to fetch")
    fetch.add_argument("--out", help="output path (default: data/<symbol>.csv)")
    fetch.add_argument("--cache-dir", default=".cache/market-data")
    fetch.add_argument("--cache-hours", type=float, default=12.0)
    fetch.add_argument("--no-cache", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    providers = subparsers.add_parser("providers", help="list live data providers")
    providers.set_defaults(func=cmd_providers)

    paper = subparsers.add_parser(
        "paper", help="paper-trade live bars as they close (places no real orders)"
    )
    paper.add_argument("--strategy", default="sma-crossover", choices=sorted(STRATEGIES))
    paper.add_argument("--warmup", type=int, default=200,
                       help="historical bars to prime indicators (default: 200)")
    paper.add_argument("--max-bars", type=int, default=None,
                       help="stop after this many live bars")
    _add_common_arguments(paper)
    paper.set_defaults(func=cmd_paper)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(tokens)
    args._argv = tokens
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, DataFeedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
