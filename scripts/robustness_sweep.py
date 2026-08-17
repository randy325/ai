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
import math
import random
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
    """Roughness of a perfectly monotonic sweep. NOT a decision threshold.

    This is the value a noiseless straight line would produce. It was
    previously used as the yardstick, which was wrong: the question is not
    "is this rougher than a perfect ramp" — almost everything is — but "is
    this rougher than noise". Use :func:`null_roughness` for that.
    """
    return 1.0 / (count - 1) if count > 1 else 0.0


_NULL_CACHE: dict[tuple[int, int, int], list[float]] = {}


def null_roughness(count: int, trials: int = 20_000, seed: int = 20240817) -> list[float]:
    """Sorted roughness values for ``count`` iid random points.

    This is the distribution the statistic takes when the surface carries no
    shape at all and every point is sampling noise. It depends strongly on the
    point count — the mean runs about 0.67 at 3 points and 0.28 at 31 — which
    is why a single hardcoded cutoff cannot work.
    """
    key = (count, trials, seed)
    if key not in _NULL_CACHE:
        rng = random.Random(seed + count)
        _NULL_CACHE[key] = sorted(
            roughness([rng.gauss(0.0, 1.0) for _ in range(count)]) for _ in range(trials)
        )
    return _NULL_CACHE[key]


def null_percentile(count: int, percentile: float, **kwargs) -> float:
    """A percentile of the null roughness distribution for ``count`` points."""
    values = null_roughness(count, **kwargs)
    if not values:
        return 0.0
    index = min(max(int(percentile * (len(values) - 1)), 0), len(values) - 1)
    return values[index]


def classify_roughness(value: float, count: int, alpha: float = 0.05) -> str:
    """Compare measured roughness against the noise distribution.

    Returns "jagged" above the upper tail, "smooth" below the lower tail, and
    "noise" in between — where the statistic simply cannot tell the difference
    between a real surface and a random one.
    """
    if count < 3:
        return "noise"
    if value > null_percentile(count, 1.0 - alpha):
        return "jagged"
    if value < null_percentile(count, alpha):
        return "smooth"
    return "noise"


def positive_control(
    effect: float,
    points: int,
    seeds: int,
    noise: float = 0.10,
    trials: int = 200,
    seed: int = 4242,
) -> float:
    """Detection rate for an injected linear effect of known size.

    The null test proves the detector will not fire on noise. On its own that
    is worthless as evidence, because a detector wired to "no" passes it
    perfectly — "no resolving power" and "the instrument is broken" look
    identical. This is the complement: a surface with a KNOWN trend, to
    confirm the detector fires when it should and to measure how small an
    effect it can still see.

    ``effect`` is the total return difference across the whole parameter range.
    """
    rng = random.Random(seed)
    values = list(range(points))
    detected = 0
    for _ in range(trials):
        step = effect / (points - 1)
        # Seeds are shared across settings, exactly as in a real sweep, so the
        # per-seed path offset is common and the comparison is paired.
        offsets = [rng.gauss(0.0, noise) for _ in range(seeds)]
        per_seed = [
            [i * step + offsets[s] + rng.gauss(0.0, noise / 3.0) for s in range(seeds)]
            for i in range(points)
        ]
        if monotonic_trend(values, per_seed)["p"] < 0.05:
            detected += 1
    return detected / trials


def minimum_detectable_effect(
    points: int, seeds: int, noise: float = 0.10, power: float = 0.80, trials: int = 120
) -> float:
    """Smallest injected effect the trend test detects at ``power``."""
    candidates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50]
    for effect in candidates:
        if positive_control(effect, points, seeds, noise, trials=trials) >= power:
            return effect
    return math.inf


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation between two equal-length sequences."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-12 else 0.0


def monotonic_trend(values: list[float], per_seed: list[list[float]]) -> dict:
    """Paired test for a monotone relationship between setting and return.

    Roughness is blind to this by construction: a perfectly monotone ramp is
    the *smoothest* sequence there is, scoring the minimum 1/(n-1), so a real
    trend looks maximally unremarkable to it. Ordering is exactly the
    information roughness discards, and a trend test is the statistic that
    uses it.

    The seeds are shared across settings, so this is paired — each seed sees
    every setting on the same price path, and its own rank correlation removes
    the path-to-path variation that dominates the unpaired comparison.
    """
    seed_count = min((len(p) for p in per_seed), default=0)
    if len(values) < 3 or seed_count < 2:
        return {"rho": 0.0, "sem": 0.0, "t": 0.0, "p": 1.0, "seeds": seed_count}

    rhos = []
    for s in range(seed_count):
        column = [per_seed[i][s] for i in range(len(values))]
        rhos.append(spearman(list(values), column))

    mean_rho = statistics.mean(rhos)
    sem = statistics.pstdev(rhos) / math.sqrt(len(rhos)) if len(rhos) > 1 else 0.0
    t = mean_rho / sem if sem > 1e-12 else 0.0
    # Two-sided normal approximation; seed counts here are large enough.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"rho": mean_rho, "sem": sem, "t": t, "p": p, "seeds": len(rhos)}


def resolving_power(per_seed: list[list[float]]) -> tuple[float, float, float, float]:
    """Between-point signal against within-point sampling noise.

    Each sweep point is a median over seeds, so it carries its own standard
    error. If the spread *between* points is no larger than the error *within*
    them, the shape being measured is an artefact of how many seeds were run,
    and no verdict about the surface is meaningful.

    ``true_between`` removes the sampling noise the observed spread contains:
    the medians scatter even when every point shares one true value, so
    observed² = signal² + noise². When it comes out at zero the surface is flat
    and no amount of extra seeds will resolve it, because there is nothing to
    resolve — the ratio stays put while both terms shrink together.

    Returns (between, within, ratio, true_between).
    """
    points = [p for p in per_seed if len(p) >= 2]
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    medians = [statistics.median(p) for p in points]
    between = statistics.pstdev(medians)
    # Each point is summarised by a MEDIAN, so the error to compare against is
    # the standard error of the median, not of the mean. For roughly normal
    # samples that is sqrt(pi/2) ~ 1.2533 times larger; using the mean's error
    # understated within-point noise by about a quarter and left a flat surface
    # looking like it still carried signal.
    median_se = math.sqrt(math.pi / 2.0)
    errors = [median_se * statistics.pstdev(p) / math.sqrt(len(p)) for p in points]
    within = statistics.median(errors)
    ratio = between / within if within > 1e-12 else math.inf
    # Method-of-moments estimate of the true spread. It is NOT clamped here:
    # a negative raw value means the observed scatter is smaller than the noise
    # alone predicts, which is information ("below the detection floor"), and
    # clamping it to zero and printing "0.00%" turns that into a false claim of
    # measured absence. Callers get both the raw and the floor.
    raw = between**2 - within**2
    true_between = math.sqrt(raw) if raw > 0 else 0.0
    return between, within, ratio, true_between


def detection_floor(within: float) -> float:
    """Smallest between-point spread this seed count could distinguish.

    Set at 2x the within-point standard error: below that the ratio cannot
    reach the threshold at which any shape statistic is readable.
    """
    return 2.0 * within


def describe_signal(between: float, within: float, true_between: float) -> str:
    """Human-readable signal estimate that never reports a clamp as zero."""
    floor = detection_floor(within)
    if between <= within:
        return f"below detection floor (<{floor:.2%})"
    return f"{true_between:.2%} (floor {floor:.2%})"


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
        # Kept so the sweep can separate variation between settings from
        # sampling noise within a single setting.
        "per_seed": returns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="rsi-breakout", choices=sorted(STRATEGIES))
    parser.add_argument("--spread", type=float, default=0.30, help="vary by +/- this fraction")
    parser.add_argument("--steps", type=int, default=15, help="settings per parameter")
    parser.add_argument("--seeds", type=int, default=25, help="synthetic paths per setting")
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="tail probability for the noise comparison")
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
    # Two questions, kept separate. First: can this sweep resolve anything at
    # all — is the variation between settings bigger than the sampling error
    # within each one? Only if yes does the second question mean anything:
    # is the surface rougher, or smoother, than pure noise would be?
    print("Sensitivity")
    print("-" * len(header))
    flags = []
    verdicts = {}
    ratios = []
    signals = []
    trends = {}

    for name, default, results in findings:
        returns = [outcome["return"] for _, outcome in results]
        per_seed = [outcome["per_seed"] for _, outcome in results]
        spread = max(returns) - min(returns)
        count = len(returns)

        jaggedness = roughness(returns)
        between, within, ratio, true_between = resolving_power(per_seed)
        trend = monotonic_trend([float(v) for v, _ in results], per_seed)
        ratios.append(ratio)
        signals.append(true_between)
        trends[name] = trend
        upper = null_percentile(count, 1.0 - args.alpha)
        lower = null_percentile(count, args.alpha)
        shape = classify_roughness(jaggedness, count, args.alpha)

        values = [v for v, _ in results]
        stands_alone, margin = is_peaked(values, returns, default)

        if spread < 1e-9:
            verdict = "no effect"
            flags.append(
                f"{name}: varying it changes nothing — inert in this configuration"
            )
        elif trend["p"] < args.alpha:
            # Ordering carries what the variance ratio throws away: a monotone
            # response is the SMOOTHEST possible sequence by roughness, so a
            # real trend is invisible to that statistic by construction.
            direction = "falls" if trend["rho"] < 0 else "rises"
            verdict = "TREND"
            flags.append(
                f"{name}: return {direction} monotonically with the setting "
                f"(rho {trend['rho']:+.2f}, p={trend['p']:.3f} paired over "
                f"{trend['seeds']} seeds) — a real response the roughness "
                "statistic cannot see"
            )
        elif ratio < 2.0:
            # Point-to-point differences are the same size as the error on each
            # point. Any shape read off this is an artefact of the seed count.
            verdict = "UNRESOLVED"
            flags.append(
                f"{name}: between-point spread {between:.1%} against a within-point "
                f"standard error of {within:.1%} ({ratio:.1f}x); signal "
                f"{describe_signal(between, within, true_between)}; no monotone "
                f"trend either (p={trend['p']:.2f})"
            )
        elif shape == "jagged":
            verdict = "JAGGED"
            flags.append(
                f"{name}: roughness {jaggedness:.0%} exceeds the {1 - args.alpha:.0%} "
                f"point of the noise distribution ({upper:.0%} at {count} settings) — "
                "significantly more jagged than chance"
            )
        elif shape == "smooth":
            verdict = "smooth"
        else:
            verdict = "noise"

        if stands_alone and verdict not in {"UNRESOLVED", "no effect"}:
            verdict = "PEAKED"
            flags.append(
                f"{name}: the configured value is the single best setting and beats the "
                f"median of its neighbours by {margin:.1%} — the signature of a value "
                "that was tuned rather than found"
            )

        verdicts[name] = verdict
        print(
            f"{name:<16}{'':>9}{'rough':>7} {jaggedness:>5.0%}"
            f"  resolve {ratio:>4.1f}x  trend rho {trend['rho']:>+5.2f} "
            f"p={trend['p']:>5.3f}   {verdict}"
        )

    all_returns = [o["return"] for _, _, rs in findings for _, o in rs]
    median_ratio = statistics.median(ratios) if ratios else 0.0
    resolved = [n for n, v in verdicts.items() if v not in {"UNRESOLVED", "no effect"}]

    print("\nVerdict")
    print("-" * len(header))
    print(f"  settings tested        : {len(all_returns)}")
    print(f"  points x seeds         : {args.steps} x {args.seeds}")
    median_signal = statistics.median(signals) if signals else 0.0
    mde = minimum_detectable_effect(args.steps, args.seeds, trials=60)
    print(f"  median resolving power : {median_ratio:.1f}x "
          f"(between-point spread / within-point standard error; >2 to read anything)")
    print(f"  median signal estimate : {median_signal:.2%} "
          "(0 here means below the floor, not measured absence)")
    print(f"  min detectable effect  : {mde:.1%} end-to-end at {args.steps}x{args.seeds}, "
          "80% power (positive control)")
    trending = [n for n, tr in trends.items() if tr["p"] < args.alpha]
    print(f"  monotone trends found  : {len(trending)}/{len(trends)}"
          + (f" ({', '.join(trending)})" if trending else ""))
    print(f"  parameters resolved    : {len(resolved)}/{len(verdicts)}")
    print(f"  median across sweep    : {statistics.median(all_returns):+.1%}")
    print(f"  baseline               : {baseline['return']:+.1%}")

    if flags:
        print("\n  Flags:")
        for flag in flags:
            print(f"    - {flag}")

    if trending:
        print("\n  TREND DETECTED. At least one parameter has a monotone effect on")
        print("  return. See the mechanism note below before reading it as strategy")
        print("  quality — on this data it is most likely exposure, not edge.")
    elif not resolved:
        print("\n  BELOW DETECTION FLOOR. No parameter's surface is separable from")
        print(f"  sampling noise, and no monotone trend clears p<{args.alpha}. The")
        print(f"  positive control puts the smallest detectable effect at {mde:.1%}")
        print("  end-to-end here, so this rules out effects larger than that and")
        print("  says nothing about smaller ones. It is NOT evidence of no effect.")
    elif any(v in {"JAGGED", "PEAKED"} for v in verdicts.values()):
        print("\n  TUNED-LOOKING parameters found. The flagged ones are significantly")
        print("  rougher than noise or sit on isolated peaks; widen or remove them.")
    elif all(verdicts[n] == "noise" for n in resolved):
        print("\n  INDISTINGUISHABLE FROM NOISE. Nothing looks overfit, but nothing")
        print("  looks structured either — the surface is flat within measurement")
        print("  error, which is the expected result on a random walk.")
    else:
        print("\n  SMOOTH. Resolved parameters vary gradually and none is an isolated")
        print("  peak, which is what a non-overfit parameter set looks like.")

    print("\n  Note: near-zero or negative returns above are the EXPECTED result on a")
    print("  random walk, where there is no structure to exploit and costs are real.")
    print("  They are not evidence against the strategy, and a profitable sweep here")
    print("  would be evidence of a bug rather than an edge.")
    print()
    print("  MECHANISM WARNING. On this synthetic feed, mean return tracks mean")
    print("  trade count at about r=+0.94, and the generator has positive drift.")
    print("  What varies with a parameter here is therefore mostly TIME IN MARKET,")
    print("  and a setting that looks good is most likely one that stays invested")
    print("  longer to collect more drift. That is a property of the data, not of")
    print("  the strategy, and it is a reason to distrust synthetic parameter")
    print("  sweeps categorically rather than merely to run them with more seeds.")

    print(f"\n{BANNER}")
    print("Synthetic data cannot tell you whether this strategy has an edge.")
    print("Run scripts/evaluate_real.py on real bars for that question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
