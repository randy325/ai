# trading-bot

A backtesting and paper-trading bot in pure Python — no third-party packages,
no build step, no API keys.

**This is a simulator.** Nothing in it connects to a broker or an exchange, and
no code path can place a real order. See [docs/going-live.md](docs/going-live.md)
for what live trading would actually require, and why the gap is wider than it
looks.

## Quick start

Python 3.10 or newer. No installation needed:

```bash
python -m trading_bot backtest --strategy breakout
```

That runs on a reproducible synthetic price series, so it works before you have
any data. To use your own:

```bash
python -m trading_bot backtest --strategy sma-crossover --data prices.csv --symbol AAPL
```

Any CSV with a date column and a close column works. Column names are matched
case-insensitively, and `open`/`high`/`low`/`volume` are optional:

```csv
timestamp,open,high,low,close,volume
2020-01-02,74.06,75.15,73.80,75.09,135480400
```

## Commands

| Command | What it does |
| --- | --- |
| `backtest` | Run one strategy over one series and print a performance report |
| `compare` | Run every strategy over the same series, ranked by Sharpe ratio |
| `optimize` | Grid-search a strategy's parameters |
| `demo-data` | Write a synthetic OHLCV file you can experiment with |
| `strategies` | List the available strategies |

```bash
python -m trading_bot compare --bars 1000
python -m trading_bot optimize --strategy sma-crossover --grid fast=5,10,20 --grid slow=50,100,200
python -m trading_bot backtest --strategy breakout --export-trades trades.csv
```

Run `python -m trading_bot <command> --help` for the full flag list.

## Strategies

| Name | Idea |
| --- | --- |
| `buy-and-hold` | Long from the first bar. The benchmark everything else has to beat |
| `sma-crossover` | Long while the fast moving average is above the slow one |
| `macd-trend` | Long while the MACD line is above zero |
| `breakout` | Donchian channel breakout with an ATR trailing stop |
| `rsi-mean-reversion` | Buy oversold, exit when the RSI reverts to the middle |
| `bollinger-reversion` | Fade moves outside the bands, exit at the middle band |

Add `--allow-short` to let a strategy take the other side, and `--trend-filter 200`
to suppress any trade fighting the 200-period EMA.

## Writing a strategy

Implement `on_candle`, return the exposure you want as a fraction of equity:
`1.0` fully long, `-1.0` fully short, `0.0` flat. Sizing, cash and orders are
somebody else's problem — which is what keeps a strategy testable.

```python
from trading_bot import Strategy, Signal
from trading_bot.indicators import SMA

class AboveAverage(Strategy):
    name = "above-average"

    def __init__(self, period: int = 50):
        self.sma = SMA(period)

    def on_candle(self, candle):
        self.sma.update(candle.close)
        if not self.sma.ready:
            return Signal(0.0)
        return Signal(1.0 if candle.close > self.sma.value else 0.0)
```

Register it to reach it from the CLI:

```python
from trading_bot.strategy import STRATEGIES
STRATEGIES["above-average"] = AboveAverage
```

Indicators are streaming — you feed them one value at a time and they tell you
when they have enough history (`ready`). A backtest and a live session therefore
run the exact same code, rather than one vectorised path and one incremental one
that quietly disagree.

## Using it as a library

```python
from trading_bot import RunConfig

config = RunConfig(strategy="breakout", bars=1000, starting_cash=50_000)
result = config.build_engine().run(config.build_feed())

print(result.summary())
print(result.metrics.sharpe_ratio, result.metrics.max_drawdown)
for trade in result.trades:
    print(trade.entry_time, trade.pnl)
```

Or assemble the pieces yourself:

```python
from trading_bot import PaperBroker, SyntheticFeed, TradingEngine, SMACrossover
from trading_bot.risk import RiskLimits, RiskManager, VolatilitySizer

engine = TradingEngine(
    strategy=SMACrossover(fast=10, slow=40),
    broker=PaperBroker(starting_cash=100_000),
    risk=RiskManager(RiskLimits(max_drawdown=0.20), VolatilitySizer(risk_per_trade=0.01)),
)
result = engine.run(SyntheticFeed(bars=500))
```

`RunConfig` round-trips to JSON, so a run stays reproducible:

```bash
python -m trading_bot backtest --config my-run.json
```

## Risk controls

Risk lives outside the strategy, so you can change how much you bet without
touching the logic that decides *whether* to bet.

- **Position sizing** — `--sizer fixed` allocates a constant fraction of equity.
  `--sizer volatility` sizes so that a move to the stop costs a fixed fraction of
  equity (`--risk-per-trade 0.02`), which means quiet markets get bigger positions
  and volatile ones get smaller.
- **`--max-drawdown 0.25`** — flatten and stop trading for good once equity falls
  25% below its high-water mark. This is the one control that matters most.
- **`--daily-loss-limit 0.03`** — pause until the next session after a 3% daily
  loss, measured from the prior close.
- **`--max-trades-per-day`** — throttle a signal that churns.
- **`--leverage`** — cap gross exposure. Orders are trimmed to fit; exits are
  never blocked.

## How a bar is processed

Order of operations is the whole ballgame in a backtester:

1. Fill the order decided on the *previous* bar, at this bar's open.
2. Mark the book to this bar's close and compute equity.
3. Update risk state (drawdown, daily limits).
4. Ask the strategy for a signal from this bar's close.
5. Convert the signal to a target position, and queue the order for the next bar.

Step 5 queuing rather than filling is deliberate. A signal derived from a bar's
close cannot be filled at that same close — you did not know the close until it
happened. `--execute-on close` will do it anyway if you want to see how much that
single assumption flatters a result; the difference is usually larger than the
strategy's entire edge. There is a test pinning this behaviour
(`test_signal_fills_on_the_next_bar_open_not_the_signal_bar_close`).

Costs are on by default: 10 bps commission and 5 bps slippage. A strategy that
only works with `--commission 0 --slippage 0` does not work.

## Reading the results

```
Total return                  +41.75%
CAGR                          +18.54%
Sharpe ratio                     0.91
Max drawdown                   12.30%
Trades                             11
Profit factor                    5.79
```

Look at max drawdown and trade count before return. A 40% return from 11 trades
is 11 data points — nowhere near enough to distinguish skill from luck. Profit
factor (gross profit over gross loss) below about 1.2 will not survive real
costs, and a Sharpe ratio computed over a few hundred bars has enormous error
bars around it.

`optimize` is the most dangerous command here. The best row in its grid is, by
construction, the parameter set that best fit that one price series, and its
reported return is an estimate of nothing. Treat it as a way to check whether a
strategy is robust across neighbouring parameters — a lone spike surrounded by
losers is noise — not as a way to pick the winner.

## Tests

```bash
python -m unittest discover -s tests
```

193 tests covering position accounting through to the CLI. The ones worth
knowing about assert properties that are easy to get quietly wrong: that a
signal cannot be filled on its own bar, that buy-and-hold produces exactly one
round trip (an early version churned 33 times as its fixed-fraction target
drifted), that exits are never blocked by buying power, and that a drawdown halt
is permanent.
