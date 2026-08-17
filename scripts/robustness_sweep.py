"""Parameter robustness sweep on SYNTHETIC data.

    python scripts/robustness_sweep.py --strategy rsi-breakout
    python scripts/robustness_sweep.py --strategy breakout --spread 0.3 --seeds 8

Varies each numeric parameter around its current value and reports how much the
result moves. A strategy whose performance collapses one step either side of
its configured setting is fitted to that setting, not to the market: real edges
are broad, because the effect they exploit does not know your parameter value.

Every number this prints comes from a generated random walk. It says nothing
about whether the strategy makes money — only whether it depends on precise
parameter values, which is a question synthetic data can answer honestly
because there is no market structure to overfit *to*. Treat a smooth surface as
a necessary condition, never a sufficient one.

Each setting is run across several seeds and the median taken, so one lucky
path cannot make a parameter look good.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import RunConfig  # noqa: E402
from trading_bot.strategy import STRATEGIES, build_strategy  # noqa: E402

BANNER = "*** SYNTHETIC DATA — measures parameter sensitivity, NOT profitability ***"

#: Parameters that are not sweepable dial settings.
SKIP = {"allow_short", "members", "mode", "member_params"}


def current_parameters(strategy: str) -> dict[str, float]:
    """The strategy's own defaults, read from its constructor signature."""
    import inspect

    signature = inspect.signature(STRATEGIES[strategy].__init__)
    params = {}
    for name, parameter in signature.parameters.items():
        if name in ("self", *SKIP) or parameter.default is inspect.Parameter.empty:
            continue
        if isinstance(parameter.default, bool) or not isinstance(parameter.default, (int, float)):
            continue
        params[name] = parameter.default
    return params


def vary(value: float, spread: float, steps: int) -> list[float]:
    """Values from -spread to +spread around ``value``, keeping integers integral."""
    out = []
    for i in range(steps):
        factor = 1.0 - spread + (2 * spread * i / (steps - 1))
        scaled = value * factor
        candidate = max(int(round(scaled)), 1) if isinstance(value, int) else round(scaled, 4)
        if candidate not in out:
            out.append(candidate)
    return out


def roughness(returns: list[float]) -> float:
    """How jagged a sweep is: mean step between neighbours, over the total range.

    A smooth curve over n points steps about ``1/(n-1)`` of its range each time.
    A surface that alternates high and low traverses the whole range every step
    and approaches 1.0. Above roughly 0.5 the neighbours of a setting tell you
    nothing about it, which means the setting itself is a draw from noise.
    """
    if len(returns) < 2:
        return 0.0
    spread = max(returns) - min(returns)
    if spread <= 1e-9:
        return 0.0
    steps = [abs(b - a) for a, b in zip(returns, returns[1:])]
    return statistics.mean(steps) / spread


def smooth_reference(count: int) -> float:
    """The roughness a perfectly monotonic sweep of ``count`` points would show."""
    return 1.0 / (count - 1) if count > 1 else 0.0


def is_peaked(values: list, returns: list[float], default) -> tuple[bool, float]:
    """Whether the configured value is an isolated best in its neighbourhood."""
    baseline = next((r for v, r in zip(values, returns) if v == default), None)
    others = [r for v, r in zip(values, returns) if v != default]
    if baseline is None or not others:
        return False, 0.0
    margin = baseline - statistics.median(others)
    spread = max(returns) - min(returns)
    return (baseline == max(returns) and margin > max(0.05, spread * 0.5)), margin


def score(strategy: str, params: dict, seeds: list[int], bars: int, cash: float) -> dict | None:
    """Median return and trade count across seeds, or None if the combination is invalid."""
    returns, trades, sharpes = [], [], []
    for seed in seeds:
        config = RunConfig(
            strategy=strategy, strategy_params=params, bars=bars, seed=seed,
            starting_cash=cash, cache_dir=None,
        )
        try:
            result = config.build_engine().run(config.build_feed())
        except (ValueError, KeyError, TypeError):
            return None
        returns.append(result.metrics.total_return)
        trades.append(result.metrics.num_trades)
        sharpes.append(result.metrics.sharpe_ratio)
    return {
        "return": statistics.median(returns),
        "trades": statistics.median(trades),
        "sharpe": statistics.median(sharpes),
        "worst": min(returns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="rsi-breakout", choices=sorted(STRATEGIES))
    parser.add_argument("--spread", type=float, default=0.30, help="vary by +/- this fraction")
    parser.add_argument("--steps", type=int, default=7, help="settings per parameter")
    parser.add_argument("--seeds", type=int, default=6, help="synthetic paths per setting")
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--cash", type=float, default=10_000.0)
    args = parser.parse_args()

    if args.steps < 3:
        print("error: --steps must be at least 3", file=sys.stderr)
        return 2

    seeds = list(range(1, args.seeds + 1))
    baseline_params = current_parameters(args.strategy)
    if not baseline_params:
        print(f"{args.strategy} has no numeric parameters to sweep", file=sys.stderr)
        return 1

    print(BANNER)
    print(f"\nStrategy   : {build_strategy(args.strategy).describe()}")
    print(f"Sweep      : +/-{args.spread:.0%} in {args.steps} steps, "
          f"median of {len(seeds)} synthetic paths, {args.bars} bars each")

    baseline = score(args.strategy, {}, seeds, args.bars, args.cash)
    if baseline is None:
        print("baseline configuration failed to run", file=sys.stderr)
        return 1
    print(f"Baseline   : return {baseline['return']:+.1%}  "
          f"sharpe {baseline['sharpe']:.2f}  trades {baseline['trades']:.0f}\n")

    header = (f"{'parameter':<16}{'value':>9}{'return':>10}{'sharpe':>8}"
              f"{'trades':>8}{'worst seed':>12}")
    print(header)
    print("-" * len(header))

    findings = []
    for name, default in sorted(baseline_params.items()):
        results = []
        for value in vary(default, args.spread, args.steps):
            outcome = score(args.strategy, {name: value}, seeds, args.bars, args.cash)
            if outcome is None:
                continue
            results.append((value, outcome))
            marker = "  <- current" if value == default else ""
            print(
                f"{name if value == results[0][0] else '':<16}{value:>9}"
                f"{outcome['return']:>10.1%}{outcome['sharpe']:>8.2f}"
                f"{outcome['trades']:>8.0f}{outcome['worst']:>11.1%}{marker}"
            )
        if len(results) >= 3:
            findings.append((name, default, results))
        print()

    # -- shape of the surface ------------------------------------------------
    # The question here is SHAPE, not profit. On a random walk with costs the
    # expected return of any strategy is negative, so "unprofitable" is the
    # null result and says nothing. What synthetic data *can* answer is whether
    # performance varies smoothly with a parameter or jumps around — and a
    # parameter whose neighbours look nothing like it was fitted, not found.
    print("Sensitivity")
    print("-" * len(header))
    flags = []
    roughness_by_param = {}

    for name, default, results in findings:
        returns = [outcome["return"] for _, outcome in results]
        spread = max(returns) - min(returns)
        median = statistics.median(returns)

        # Mean step between adjacent settings, relative to the total range.
        # A smooth curve over n points steps about 1/(n-1) of the range each
        # time; an alternating surface steps the whole range every time.
        jaggedness = roughness(returns)
        reference = smooth_reference(len(returns))
        roughness_by_param[name] = jaggedness

        values = [v for v, _ in results]
        stands_alone, margin = is_peaked(values, returns, default)

        verdict = "smooth"
        if spread < 1e-9:
            # Changing the value moved nothing, so the parameter is not wired
            # into this configuration — long-only strategies never read their
            # short-side thresholds, for instance.
            verdict = "no effect"
            flags.append(
                f"{name}: varying it changes nothing — inert in this configuration, "
                "so it is not a tuned parameter but it is also not doing anything"
            )
        elif jaggedness > 0.5:
            verdict = "SPIKY"
            flags.append(
                f"{name}: adjacent settings differ by {jaggedness:.0%} of the total range "
                f"(a smooth curve would be about {reference:.0%}) — the surface is "
                "jagged, so the configured value is not a stable choice"
            )
        if stands_alone:
            verdict = "PEAKED"
            flags.append(
                f"{name}: the configured value is the single best setting and beats the "
                f"median of its neighbours by {margin:.1%} — the signature of a value "
                "that was tuned rather than found"
            )

        print(
            f"{name:<16}{'':>9}{'range':>10} {spread:>7.1%}"
            f"   roughness {jaggedness:>4.0%} (smooth ~{reference:.0%})   {verdict}"
        )

    all_returns = [o["return"] for _, _, rs in findings for _, o in rs]
    mean_roughness = statistics.mean(roughness_by_param.values())

    print("\nVerdict")
    print("-" * len(header))
    print(f"  settings tested        : {len(all_returns)}")
    print(f"  median across sweep    : {statistics.median(all_returns):+.1%}")
    print(f"  baseline               : {baseline['return']:+.1%}")
    print(f"  baseline percentile    : "
          f"{sum(1 for r in all_returns if r < baseline['return']) / len(all_returns):.0%}")
    print(f"  mean roughness         : {mean_roughness:.0%} "
          f"(smooth is near {1.0 / (args.steps - 1):.0%}, jagged approaches 100%)")

    if flags:
        print("\n  Flags:")
        for flag in flags:
            print(f"    - {flag}")

    if mean_roughness > 0.5:
        print("\n  JAGGED surface. Results jump between neighbouring settings, so the")
        print("  configured values are not meaningfully better than their neighbours —")
        print("  they are a draw from noise. Treat the parameters as unvalidated.")
    elif flags:
        print("\n  MOSTLY SMOOTH, with the flagged parameters looking tuned. Widen or")
        print("  remove those before trusting the configuration.")
    else:
        print("\n  SMOOTH surface. Performance changes gradually in every direction and")
        print("  the configured values are not isolated peaks. That is consistent with")
        print("  parameters that were chosen rather than fitted.")

    print("\n  Note: near-zero or negative returns above are the EXPECTED result on a")
    print("  random walk, where there is no structure to exploit and costs are real.")
    print("  They are not evidence against the strategy, and a profitable sweep here")
    print("  would be evidence of a bug rather than an edge.")

    print(f"\n{BANNER}")
    print("Synthetic data cannot tell you whether this strategy has an edge.")
    print("Run scripts/evaluate_real.py on real bars for that question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
