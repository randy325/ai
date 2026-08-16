# Going live

This package is a simulator. It has no exchange adapter, no credential handling,
and no order-state machine, and adding one is not a small change. This document
exists so the gap is explicit rather than discovered with money on the line.

## What a backtest does not tell you

A backtest is a claim about what would have happened. Every one of the following
is assumed away by the simulation here:

**Your fills are not free.** `PaperBroker` fills every market order at the bar's
open plus a fixed slippage. In reality you get partial fills, you move the price
when your order is large relative to volume, and the spread widens exactly when
your strategy most wants to trade. Backtested slippage is a guess; measure real
slippage against your own fills and feed the number back in.

**Your data has survivorship bias.** If your instrument list is "what trades
today", you have excluded everything that went to zero. A strategy tested on
today's index members will look considerably better than it would have been.

**The bars lie about intrabar order.** A bar tells you the high and the low, not
which came first. Any logic reading both — a stop and a target in the same bar —
is guessing. This package sidesteps it by acting only on closes; a live system
with real stops does not have that luxury.

**One backtest is one sample.** A strategy tested on one series over one period
has produced a single number with a wide confidence interval. Run it across many
instruments, many periods, and neighbouring parameter values. If it only works
in one corner of that space, you found noise.

**You tested on the data you chose.** Every parameter you tuned while looking at
results, including the ones you tuned by discarding a strategy and trying
another, spent some of your data's information. Hold out a period you have never
looked at, and look at it once.

## What a live adapter would need

To place real orders, `PaperBroker` would need a sibling implementing the same
interface (`submit`, `position`, `equity`, `mark`) against an exchange API. The
engine would not have to change. Everything below would:

**Order state is not instantaneous.** `submit` currently returns a `Fill` or
`None`. A real order is acknowledged, then partially filled, then filled or
cancelled or rejected — across seconds or minutes, possibly spanning bars. That
is a state machine, and it needs to survive your process restarting.

**Reconciliation, not assumption.** This broker believes its own arithmetic about
what it holds. A live one must treat the exchange as the source of truth, query
actual positions and balances on every startup, and refuse to trade when its
view and the exchange's disagree. Assuming your local position is correct is how
a bug becomes an unhedged position.

**Idempotency.** Send a client-assigned order ID with every order so a retry
after a timeout cannot become a second position. Network timeouts do not tell
you whether the order arrived.

**Credentials.** API keys belong in environment variables or a secrets manager,
never in a config file next to a strategy, never in git. Use the most restricted
key the exchange offers — trading permissions without withdrawal permissions —
and IP-allowlist it.

**A kill switch that works when your code does not.** `RiskLimits.max_drawdown`
only fires if the loop is running. Real protection is outside the process:
exchange-side stop orders, position limits set at the account level, and a
manual way to flatten everything that does not depend on your bot being healthy.

**Rate limits and reconnection.** Exchanges throttle, disconnect, and go into
maintenance. A feed that silently stops delivering bars leaves your bot holding
a position it thinks it is managing. Treat stale data as an emergency, not a
gap — if the last bar is older than expected, stop trading.

**Time.** Backtests have one clock. Live systems have exchange time, your
server's clock, and bar-close time, and they disagree. Act on completed bars
only, and know your data provider's timestamp convention (bar open or bar
close — this package uses close).

## Before any of that

Paper trade against live data first, for longer than feels necessary, and
compare the paper results against what the backtest predicted for the same
period. If they diverge, your model of costs is wrong, and that is far better to
learn from a spreadsheet than from a statement.

Then size down. The first live capital should be an amount whose complete loss
would be annoying rather than significant, because the purpose of the first
deployment is to discover what you got wrong about execution, not to make money.

## A note on expectations

Most retail strategies do not beat buy-and-hold after costs and taxes. `compare`
includes `buy-and-hold` for exactly this reason: it is the benchmark, it is free
to implement, and beating it consistently is genuinely difficult. Nothing in
this repository changes that, and a good backtest result is not evidence to the
contrary — it is the thing you should be most suspicious of.
