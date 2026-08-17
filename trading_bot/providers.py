"""Live market data from public HTTP APIs.

Four live providers, none of which needs an API key, plus a mock:

    stooq      daily/weekly stock, ETF and index bars (CSV)
    yahoo      stock and ETF bars, intraday to monthly (JSON)
    binance    crypto bars, 1m to 1w (JSON)
    coinbase   crypto bars, 1m to 1d (JSON)
    mock       synthetic bars generated locally, for testing without a network

Two ways to consume them. :class:`MarketDataFeed` fetches a block of history
and replays it, so a backtest runs on real prices. :class:`LiveFeed` polls and
yields each bar as it closes, so the same engine can paper-trade in real time.

Network access goes through a :class:`Transport`, which is injected. That keeps
the parsing logic testable without a network, and it is how the test suite
covers every provider against recorded responses.

Nothing here places orders. A live feed drives the paper broker only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .data import DataFeed
from .models import Candle

logger = logging.getLogger("trading_bot.providers")

USER_AGENT = "trading-bot/0.1 (+https://github.com/randy325/ai)"

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
    "1w": 604_800,
}


class DataFeedError(Exception):
    """Base class for provider failures."""


class SymbolNotFound(DataFeedError):
    """The provider has no data for the requested symbol."""


class RateLimited(DataFeedError):
    """The provider is throttling us."""


class TransportError(DataFeedError):
    """The request did not complete."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(timestamp: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    Providers disagree — epochs are UTC, Stooq's dates carry no zone at all.
    Mixing naive and aware datetimes raises on comparison, so everything that
    leaves this module is aware.
    """
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _candle(
    timestamp: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    symbol: str,
) -> Candle:
    """Build a candle, clamping the wicks.

    Providers round OHLC independently, which regularly yields a high a hair
    below the close. That is a rounding artefact, not bad data, so clamp rather
    than reject the bar.
    """
    return Candle(
        timestamp=_utc(timestamp),
        open=open_,
        high=max(high, open_, close),
        low=min(low, open_, close),
        close=close,
        volume=volume,
        symbol=symbol,
    )


def _clean(candles: list[Candle]) -> list[Candle]:
    """Sort ascending and drop duplicate timestamps, keeping the last."""
    by_timestamp: dict[datetime, Candle] = {}
    for candle in candles:
        by_timestamp[candle.timestamp] = candle
    return [by_timestamp[key] for key in sorted(by_timestamp)]


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


class Transport:
    """Fetches a URL and returns the raw body."""

    def get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        raise NotImplementedError  # pragma: no cover - abstract


class UrllibTransport(Transport):
    """Standard-library HTTP client, with retries on transient failures.

    Retries 429 and 5xx with exponential backoff. A 404 is not retried — the
    symbol will not exist on the second attempt either.
    """

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleeper = sleeper

    def get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise SymbolNotFound(f"{url} returned 404") from exc
                if exc.code == 429:
                    last_error = RateLimited(f"{url} returned 429")
                elif 500 <= exc.code < 600:
                    last_error = TransportError(f"{url} returned {exc.code}")
                else:
                    raise TransportError(f"{url} returned {exc.code}: {exc.reason}") from exc
            except urllib.error.URLError as exc:
                last_error = TransportError(f"{url} unreachable: {exc.reason}")
            except TimeoutError as exc:
                last_error = TransportError(f"{url} timed out")

            if attempt < self.retries - 1:
                delay = self.backoff * (2**attempt)
                logger.warning("%s; retrying in %.1fs", last_error, delay)
                self.sleeper(delay)

        raise last_error or TransportError(f"{url} failed")


class CachingTransport(Transport):
    """Caches response bodies on disk so repeated backtests don't re-fetch.

    Entries older than ``ttl`` are refetched. A cache is close to mandatory
    when tuning parameters: a grid search re-runs the same feed dozens of
    times, and free providers will throttle long before the sweep finishes.
    """

    def __init__(self, inner: Transport, directory: str | Path, ttl: timedelta | None = None) -> None:
        self.inner = inner
        self.directory = Path(directory)
        self.ttl = ttl if ttl is not None else timedelta(hours=12)
        self.hits = 0
        self.misses = 0

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.body"

    def get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        path = self._path(url)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if self.ttl.total_seconds() <= 0 or age < self.ttl.total_seconds():
                self.hits += 1
                return path.read_bytes()

        body = self.inner.get(url, headers)
        self.misses += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class Provider:
    """Fetches historical bars for one symbol."""

    name: str = "provider"
    intervals: tuple[str, ...] = ()
    #: Minimum seconds between requests, to stay inside free-tier limits.
    min_request_interval: float = 0.0

    def __init__(
        self,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport or UrllibTransport()
        self.clock = clock
        self.sleeper = sleeper
        self._last_request: float = 0.0

    def _throttle(self) -> None:
        """Space requests out so a grid search doesn't trip a free-tier limit."""
        if self.min_request_interval <= 0:
            return
        now = self.clock()
        if self._last_request:
            elapsed = now - self._last_request
            if elapsed < self.min_request_interval:
                self.sleeper(self.min_request_interval - elapsed)
        self._last_request = self.clock()

    def check_interval(self, interval: str) -> None:
        if interval not in self.intervals:
            raise DataFeedError(
                f"{self.name} does not support interval {interval!r}; "
                f"supported: {', '.join(self.intervals)}"
            )

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        raise NotImplementedError  # pragma: no cover - abstract


class StooqProvider(Provider):
    """Free daily/weekly/monthly bars for stocks, ETFs and indices.

    Symbols carry a market suffix: ``aapl.us``, ``vod.uk``. A bare ``aapl`` is
    rewritten to ``aapl.us``. Indices use a caret, e.g. ``^spx``.
    """

    name = "stooq"
    intervals = ("1d", "1w")
    BASE = "https://stooq.com/q/d/l/"
    _CODES = {"1d": "d", "1w": "w"}

    def normalise_symbol(self, symbol: str) -> str:
        lowered = symbol.strip().lower()
        if lowered.startswith("^") or "." in lowered:
            return lowered
        return f"{lowered}.us"

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        self.check_interval(interval)
        self._throttle()
        query = urllib.parse.urlencode(
            {"s": self.normalise_symbol(symbol), "i": self._CODES[interval]}
        )
        body = self.transport.get(f"{self.BASE}?{query}").decode("utf-8", "replace")

        stripped = body.strip()
        # Stooq signals failure with a plain-text body, not an HTTP status.
        if stripped.lower().startswith("exceeded the daily hits limit"):
            raise RateLimited("stooq daily request limit exceeded")
        if not stripped or stripped.lower().startswith("no data"):
            raise SymbolNotFound(f"stooq has no data for {symbol!r}")

        reader = csv.DictReader(io.StringIO(stripped))
        if not reader.fieldnames or "Close" not in reader.fieldnames:
            raise DataFeedError(f"unexpected stooq response for {symbol!r}: {stripped[:120]!r}")

        candles: list[Candle] = []
        for row in reader:
            if not row.get("Date"):
                continue
            try:
                candles.append(
                    _candle(
                        datetime.strptime(row["Date"], "%Y-%m-%d"),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        float(row.get("Volume") or 0.0),
                        symbol.upper(),
                    )
                )
            except (TypeError, ValueError):
                continue  # holidays and halts appear as blank rows

        if not candles:
            raise SymbolNotFound(f"stooq returned no usable rows for {symbol!r}")
        return _clean(candles)[-limit:]


class YahooProvider(Provider):
    """Free stock and ETF bars from the Yahoo Finance chart endpoint.

    Intraday history is limited by Yahoo: roughly 7 days at 1m and 60 days at
    other intraday intervals, regardless of the range requested.
    """

    name = "yahoo"
    intervals = ("1m", "5m", "15m", "30m", "1h", "1d", "1w")
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
    _CODES = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d", "1w": "1wk"}

    def _range_for(self, interval: str, limit: int) -> str:
        span = INTERVAL_SECONDS[interval] * limit
        # Ask for enough calendar time to cover `limit` trading bars, allowing
        # for weekends and holidays, then trim after parsing.
        for threshold, label in (
            (5 * 86_400, "5d"),
            (28 * 86_400, "1mo"),
            (85 * 86_400, "3mo"),
            (170 * 86_400, "6mo"),
            (350 * 86_400, "1y"),
            (700 * 86_400, "2y"),
            (1_800 * 86_400, "5y"),
            (3_600 * 86_400, "10y"),
        ):
            if span <= threshold:
                return label
        return "max"

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        self.check_interval(interval)
        self._throttle()
        query = urllib.parse.urlencode(
            {"range": self._range_for(interval, limit), "interval": self._CODES[interval]}
        )
        url = f"{self.BASE}{urllib.parse.quote(symbol.strip().upper())}?{query}"
        payload = json.loads(self.transport.get(url).decode("utf-8", "replace"))

        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            description = error.get("description") if isinstance(error, dict) else str(error)
            raise SymbolNotFound(f"yahoo rejected {symbol!r}: {description}")

        results = chart.get("result") or []
        if not results:
            raise SymbolNotFound(f"yahoo returned no result for {symbol!r}")

        result = results[0]
        timestamps = result.get("timestamp") or []
        quotes = (result.get("indicators") or {}).get("quote") or [{}]
        quote = quotes[0]
        opens, highs = quote.get("open") or [], quote.get("high") or []
        lows, closes = quote.get("low") or [], quote.get("close") or []
        volumes = quote.get("volume") or []

        candles: list[Candle] = []
        for index, epoch in enumerate(timestamps):
            def at(values: Sequence, default=None):
                value = values[index] if index < len(values) else None
                return default if value is None else value

            close = at(closes)
            # Yahoo pads its arrays with nulls for halts and holidays. A bar
            # with no close is not a bar.
            if close is None or epoch is None:
                continue
            open_ = at(opens, close)
            candles.append(
                _candle(
                    datetime.fromtimestamp(int(epoch), tz=timezone.utc),
                    float(open_),
                    float(at(highs, max(open_, close))),
                    float(at(lows, min(open_, close))),
                    float(close),
                    float(at(volumes, 0.0)),
                    symbol.upper(),
                )
            )

        if not candles:
            raise SymbolNotFound(f"yahoo returned no usable bars for {symbol!r}")
        return _clean(candles)[-limit:]


class BinanceProvider(Provider):
    """Free crypto bars from Binance. Symbols look like ``BTCUSDT``."""

    name = "binance"
    intervals = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")
    BASE = "https://api.binance.com/api/v3/klines"
    MAX_LIMIT = 1_000

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        self.check_interval(interval)
        self._throttle()
        query = urllib.parse.urlencode(
            {
                "symbol": symbol.strip().upper(),
                "interval": interval,
                "limit": min(max(limit, 1), self.MAX_LIMIT),
            }
        )
        payload = json.loads(self.transport.get(f"{self.BASE}?{query}").decode("utf-8", "replace"))

        if isinstance(payload, dict):
            # Binance reports symbol errors as a JSON object with a code.
            raise SymbolNotFound(f"binance rejected {symbol!r}: {payload.get('msg', payload)}")
        if not payload:
            raise SymbolNotFound(f"binance returned no bars for {symbol!r}")

        now_ms = time.time() * 1000
        candles: list[Candle] = []
        for row in payload:
            open_time, open_, high, low, close, volume = row[0], row[1], row[2], row[3], row[4], row[5]
            close_time = row[6]
            # The final kline is usually still forming. Acting on a bar that
            # has not closed means acting on a price that can still move.
            if close_time >= now_ms:
                continue
            candles.append(
                _candle(
                    datetime.fromtimestamp(int(open_time) / 1000, tz=timezone.utc),
                    float(open_),
                    float(high),
                    float(low),
                    float(close),
                    float(volume),
                    symbol.upper(),
                )
            )

        if not candles:
            raise SymbolNotFound(f"binance returned no closed bars for {symbol!r}")
        return _clean(candles)[-limit:]


class CoinbaseProvider(Provider):
    """Free crypto bars from Coinbase Exchange. Symbols look like ``BTC-USD``.

    The endpoint returns at most 300 candles per request, newest first, and its
    rows are ordered ``[time, low, high, open, close, volume]`` — note that low
    and high precede open, unlike every other provider here.
    """

    name = "coinbase"
    intervals = ("1m", "5m", "15m", "1h", "1d")
    BASE = "https://api.exchange.coinbase.com/products"
    MAX_LIMIT = 300
    _GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3_600, "1d": 86_400}

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 300) -> list[Candle]:
        self.check_interval(interval)
        self._throttle()
        product = symbol.strip().upper()
        query = urllib.parse.urlencode({"granularity": self._GRANULARITY[interval]})
        url = f"{self.BASE}/{urllib.parse.quote(product)}/candles?{query}"
        payload = json.loads(self.transport.get(url).decode("utf-8", "replace"))

        if isinstance(payload, dict):
            raise SymbolNotFound(f"coinbase rejected {symbol!r}: {payload.get('message', payload)}")
        if not payload:
            raise SymbolNotFound(f"coinbase returned no bars for {symbol!r}")

        candles = [
            _candle(
                datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
                float(row[3]),
                float(row[2]),
                float(row[1]),
                float(row[4]),
                float(row[5]),
                product,
            )
            for row in payload
        ]
        return _clean(candles)[-limit:]


class SimulatedClock:
    """A wall clock that can run faster than real time.

    ``speed`` of 60 makes a virtual minute pass every real second, so a 1m
    ``paper`` session produces a bar per second instead of per minute. Sleeps
    are divided by the same factor, which keeps :class:`LiveFeed`'s timing
    arithmetic — expressed in virtual seconds — correct at any speed.
    """

    def __init__(self, speed: float = 1.0, sleeper: Callable[[float], None] = time.sleep) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed
        self._sleeper = sleeper
        self._real_origin = time.time()
        self._origin = self._real_origin

    def now(self) -> datetime:
        elapsed = (time.time() - self._real_origin) * self.speed
        return datetime.fromtimestamp(self._origin + elapsed, tz=timezone.utc)

    def sleep(self, virtual_seconds: float) -> None:
        self._sleeper(max(virtual_seconds, 0.0) / self.speed)


class MockProvider(Provider):
    """Generates synthetic bars anchored to the clock. No network involved.

    Bars are a deterministic function of their absolute index, so the same
    timestamp always carries the same price no matter when it is fetched, and
    new bars appear as time passes — which is what :class:`LiveFeed` needs to
    behave exactly as it would against a real provider.

    The series is built from layered sine waves plus hashed noise rather than a
    random walk. That is a deliberate trade: it is O(1) per bar and produces
    trends, breakouts and reversals on demand, which makes it good for
    exercising a strategy's plumbing. It is not a realistic market, and a
    backtest against it says nothing about whether a strategy works.
    """

    name = "mock"
    intervals = tuple(INTERVAL_SECONDS)

    def __init__(
        self,
        transport: Transport | None = None,
        seed: int = 7,
        start_price: float = 100.0,
        amplitude: float = 0.09,
        noise: float = 0.004,
        time_source: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(transport or UrllibTransport())
        self.seed = seed
        self.start_price = start_price
        self.amplitude = amplitude
        self.noise = noise
        self.time_source = time_source or _utc_now

    def _jitter(self, index: int, salt: str) -> float:
        """A stable pseudo-random value in [-1, 1] for this bar and purpose."""
        digest = hashlib.sha256(f"{self.seed}:{salt}:{index}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 0x7FFFFFFF - 1.0

    def _price(self, index: int) -> float:
        """Price for an absolute bar index.

        Every term is bounded. Indices here are epoch-derived, so a 1m bar
        today has an index around 29 million — any term growing with the index
        (a linear drift, say) overflows to exp(580) rather than trending.
        The slowest wave supplies the trend instead.
        """
        wave = (
            math.sin(index / 2003.0 + self.seed) * 1.2
            + math.sin(index / 61.0 + self.seed * 2) * 1.0
            + math.sin(index / 17.0 + self.seed * 3) * 0.45
            + math.sin(index / 7.0 + self.seed * 4) * 0.2
        )
        exponent = self.amplitude * wave + self.noise * self._jitter(index, "n")
        return self.start_price * math.exp(exponent)

    def _bar(self, index: int, interval: str, symbol: str) -> Candle:
        seconds = INTERVAL_SECONDS[interval]
        open_ = self._price(index)
        close = self._price(index + 1)
        span = abs(close - open_) + self.start_price * self.noise
        high = max(open_, close) + span * abs(self._jitter(index, "h"))
        low = min(open_, close) - span * abs(self._jitter(index, "l"))
        return _candle(
            datetime.fromtimestamp(index * seconds, tz=timezone.utc),
            round(open_, 4),
            round(high, 4),
            round(max(low, 0.01), 4),
            round(close, 4),
            round(1_000 + 9_000 * abs(self._jitter(index, "v")), 2),
            symbol.upper(),
        )

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        self.check_interval(interval)
        seconds = INTERVAL_SECONDS[interval]
        # The bar containing "now" has not closed yet, so the newest closed bar
        # is the one before it — the same rule the real providers follow.
        current = int(self.time_source().timestamp()) // seconds
        newest = current - 1
        count = max(min(limit, newest + 1), 0)
        if count <= 0:
            raise SymbolNotFound(f"mock has no closed {interval} bars yet for {symbol!r}")
        return [self._bar(i, interval, symbol) for i in range(newest - count + 1, newest + 1)]


class FallbackProvider(Provider):
    """Tries several providers in order until one returns bars.

    Free endpoints are unofficial and go down; a chain means a session survives
    one of them failing. It is a redundancy mechanism, not a consensus one —
    the first provider that answers wins, and its prices are used as-is.

    Prices are *not* blended across providers. Two feeds for the same
    instrument disagree on venue, timestamp convention and adjustment, so
    averaging them produces bars that never traded anywhere.
    """

    name = "fallback"

    def __init__(self, providers: Sequence[Provider], symbols: dict[str, str] | None = None) -> None:
        if not providers:
            raise ValueError("a fallback chain needs at least one provider")
        super().__init__()
        self.providers = list(providers)
        #: Per-provider symbol overrides, since tickers differ by venue
        #: (AAPL vs aapl.us, BTCUSDT vs BTC-USD).
        self.symbols = symbols or {}
        self.used: str | None = None

    @property
    def intervals(self) -> tuple[str, ...]:
        """Intervals every member supports — the chain is only as broad."""
        common = set(self.providers[0].intervals)
        for provider in self.providers[1:]:
            common &= set(provider.intervals)
        return tuple(i for i in INTERVAL_SECONDS if i in common)

    def describe(self) -> str:
        return " -> ".join(p.name for p in self.providers)

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[Candle]:
        errors: list[str] = []
        for provider in self.providers:
            if interval not in provider.intervals:
                errors.append(f"{provider.name}: no {interval} bars")
                continue
            resolved = self.symbols.get(provider.name, symbol)
            try:
                candles = provider.fetch(resolved, interval, limit)
            except DataFeedError as exc:
                errors.append(f"{provider.name}: {exc}")
                logger.warning("%s failed, trying next provider: %s", provider.name, exc)
                continue
            self.used = provider.name
            return candles

        raise DataFeedError(
            f"every provider in the chain failed for {symbol!r} at {interval}:\n  "
            + "\n  ".join(errors)
        )


PROVIDERS: dict[str, type[Provider]] = {
    MockProvider.name: MockProvider,
    StooqProvider.name: StooqProvider,
    YahooProvider.name: YahooProvider,
    BinanceProvider.name: BinanceProvider,
    CoinbaseProvider.name: CoinbaseProvider,
}


def build_provider(
    name: str,
    transport: Transport | None = None,
    cache_dir: str | Path | None = None,
    cache_ttl: timedelta | None = None,
) -> Provider:
    """Instantiate a provider by name, optionally wrapping it in a disk cache."""
    try:
        factory = PROVIDERS[name]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise KeyError(f"unknown provider {name!r}; available: {available}") from None

    resolved = transport or UrllibTransport()
    if cache_dir is not None:
        resolved = CachingTransport(resolved, cache_dir, cache_ttl)
    return factory(resolved)


# ---------------------------------------------------------------------------
# feeds
# ---------------------------------------------------------------------------


class MarketDataFeed(DataFeed):
    """A block of real historical bars, replayed like any other feed.

    Bars are fetched once on first iteration and reused, so a grid search over
    this feed makes one request rather than one per run.
    """

    def __init__(
        self,
        provider: Provider,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> None:
        self.provider = provider
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self._candles: list[Candle] | None = None

    def load(self) -> list[Candle]:
        if self._candles is None:
            self._candles = self.provider.fetch(self.symbol, self.interval, self.limit)
            logger.info(
                "fetched %d %s bars of %s from %s",
                len(self._candles), self.interval, self.symbol, self.provider.name,
            )
        return self._candles

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.load())


@dataclass
class LiveFeed(DataFeed):
    """Polls a provider and yields each bar as it closes.

    Only closed bars are emitted, and each timestamp is emitted once. The feed
    replays ``warmup`` historical bars first so a strategy's indicators are
    ready before the first live bar arrives — a freshly started bot with a
    200-period average is otherwise blind for 200 bars.

    Iteration blocks between bars. ``max_bars`` and ``until`` bound the run;
    without either, it runs until interrupted.
    """

    provider: Provider
    symbol: str
    interval: str = "1m"
    warmup: int = 200
    max_bars: int | None = None
    until: datetime | None = None
    poll_lag: float = 5.0
    max_consecutive_errors: int = 5
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], datetime] = _utc_now

    def __post_init__(self) -> None:
        self.provider.check_interval(self.interval)
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.symbol = self.symbol.upper()
        self._seen: set[datetime] = set()
        self.live_bars = 0

    @property
    def interval_seconds(self) -> int:
        return INTERVAL_SECONDS[self.interval]

    def _finished(self, emitted: int) -> bool:
        if self.max_bars is not None and emitted >= self.max_bars:
            return True
        if self.until is not None and self.clock() >= self.until:
            return True
        return False

    def _sleep_until_next_bar(self, last: datetime) -> None:
        """Wait until the bar after ``last`` should have closed.

        Bars are labelled by opening time, so the bar opening at ``last``
        closes one interval later and the next one closes two intervals later.
        """
        target = last + timedelta(seconds=2 * self.interval_seconds)
        delay = (target - self.clock()).total_seconds() + self.poll_lag
        # A provider lagging behind real time would otherwise spin.
        self.sleeper(max(delay, 1.0))

    def __iter__(self) -> Iterator[Candle]:
        emitted = 0
        history = self.provider.fetch(self.symbol, self.interval, max(self.warmup, 1))

        if self.warmup:
            for candle in history:
                self._seen.add(candle.timestamp)
                yield candle
        elif history:
            # Still mark history as seen, or the first poll replays it as live.
            self._seen.update(candle.timestamp for candle in history)

        last_timestamp = history[-1].timestamp if history else self.clock()
        errors = 0

        while not self._finished(emitted):
            self._sleep_until_next_bar(last_timestamp)
            if self._finished(emitted):
                break

            try:
                batch = self.provider.fetch(self.symbol, self.interval, 10)
                errors = 0
            except DataFeedError as exc:
                errors += 1
                # A transient outage must not kill a running bot, but a
                # persistent one must not be mistaken for a quiet market.
                logger.warning("live fetch failed (%d/%d): %s", errors, self.max_consecutive_errors, exc)
                if errors >= self.max_consecutive_errors:
                    raise DataFeedError(
                        f"{self.provider.name} failed {errors} times in a row; stopping"
                    ) from exc
                continue

            for candle in batch:
                if candle.timestamp in self._seen:
                    continue
                self._seen.add(candle.timestamp)
                last_timestamp = max(last_timestamp, candle.timestamp)
                self.live_bars += 1
                emitted += 1
                yield candle
                if self._finished(emitted):
                    return
