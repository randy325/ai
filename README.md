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
any data. To run on real market prices, name a provider — no API key needed:

```bash
python -m trading_bot backtest --provider yahoo --symbol AAPL --strategy breakout
python -m trading_bot backtest --provider binance --symbol BTCUSDT --interval 4h
```

Or use your own CSV. Any file with a date column and a close column works;
column names are matched case-insensitively, and `open`/`high`/`low`/`volume`
are optional:

```csv
timestamp,open,high,low,close,volume
2020-01-02,74.06,75.15,73.80,75.09,135480400
```

```bash
python -m trading_bot backtest --strategy sma-crossover --data prices.csv --symbol AAPL
```

## Commands

| Command | What it does |
| --- | --- |
| `backtest` | Run one strategy over one series and print a performance report |
| `compare` | Run every strategy over the same series, ranked by Sharpe ratio |
| `optimize` | Grid-search a strategy's parameters |
| `fetch` | Download real market bars to a CSV file |
| `paper` | Paper-trade live bars as they close |
| `providers` | List the live data providers |
| `demo-data` | Write a synthetic OHLCV file you can experiment with |
| `strategies` | List the available strategies |

```bash
python -m trading_bot compare --provider yahoo --symbol MSFT
python -m trading_bot optimize --strategy sma-crossover --grid fast=5,10,20 --grid slow=50,100,200
python -m trading_bot backtest --strategy breakout --export-trades trades.csv
```

Run `python -m trading_bot <command> --help` for the full flag list.

## Live market data

Four providers, none of which needs an API key:

| Provider | Covers | Intervals | Symbol format |
| --- | --- | --- | --- |
| `stooq` | stocks, ETFs, indices | `1d` `1w` | `aapl.us`, `^spx` (a bare `AAPL` gets `.us`) |
| `yahoo` | stocks, ETFs | `1m` … `1w` | `AAPL` |
| `binance` | crypto | `1m` … `1w` | `BTCUSDT` |
| `coinbase` | crypto | `1m` … `1d` | `BTC-USD` |

```bash
python -m trading_bot providers
python -m trading_bot fetch --provider stooq --symbol AAPL --limit 2000 --out data/aapl.csv
python -m trading_bot backtest --provider coinbase --symbol ETH-USD --interval 1h
```

Responses are cached under `.cache/market-data` for 12 hours. That matters more
than it sounds: `optimize` re-runs the same feed dozens of times, and a free
provider will throttle you long before a grid search finishes. Use
`--cache-hours` to change the window, or `--no-cache` to always refetch.

Providers have real limits. Yahoo caps intraday history at roughly 7 days at
`1m` and 60 days otherwise; Binance returns at most 1000 bars per request and
Coinbase 300; Stooq is daily-and-slower only, and rate-limits by IP. Ask for
more bars than a provider will give and you get what it has, not an error.

### Live paper trading

```bash
python -m trading_bot paper --provider binance --symbol BTCUSDT --interval 1m \
    --strategy breakout --warmup 200
```

This polls for new bars and runs the same engine on each one as it closes,
against the paper broker. It prints a row per bar and every simulated fill, and
runs until you interrupt it or `--max-bars` is reached.

Three things it does deliberately:

- **Only closed bars are traded.** A bar still forming has a price that can
  still move; Binance's most recent kline is dropped for exactly this reason.
- **`--warmup 200` replays history first**, so a 200-period average is ready
  before the first live bar. A freshly started bot is otherwise blind for as
  many bars as its slowest indicator needs.
- **It does not liquidate when the session ends.** Stopping the loop is not a
  decision to close the position, so the open position is reported instead.

It still places no real orders. It is a simulator being fed live prices.

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

Real prices, from Python:

```python
from trading_bot import MarketDataFeed, build_provider

provider = build_provider("yahoo", cache_dir=".cache/market-data")
feed = MarketDataFeed(provider, "AAPL", interval="1d", limit=1000)
for candle in feed:
    print(candle.timestamp, candle.close)
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

277 tests covering position accounting through to the CLI. The ones worth
knowing about assert properties that are easy to get quietly wrong: that a
signal cannot be filled on its own bar, that buy-and-hold produces exactly one
round trip (an early version churned 33 times as its fixed-fraction target
drifted), that exits are never blocked by buying power, and that a drawdown halt
is permanent.

The provider tests run against recorded response bodies rather than the
network, and they cover each API's awkward corners: Yahoo pads holidays with
nulls, Coinbase orders its columns `low, high, open, close` and returns rows
newest-first, Binance's last kline is usually still forming, and Stooq reports
errors as HTTP 200 with a plain-text body.

**The live providers have not been exercised against the real endpoints.** They
were built and tested in a sandbox whose egress policy blocks every market-data
host, so the URL construction and response parsing are verified only against
those recorded fixtures. The shapes match each API's documented format, but
treat the first real call as unverified. This checks all four in one go:

```bash
for p in stooq yahoo binance coinbase; do
  case $p in
    stooq|yahoo) s=AAPL ;; binance) s=BTCUSDT ;; coinbase) s=BTC-USD ;;
  esac
  echo "== $p $s"
  python -m trading_bot fetch --provider $p --symbol $s --limit 5 --no-cache \
      --out /tmp/$p.csv || echo "FAILED"
done
```
