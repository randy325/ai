"""Market data feeds.

A feed is any iterable of :class:`~trading_bot.models.Candle` in chronological
order. That keeps the engine indifferent to whether bars come from a CSV file,
a generator, or a live socket.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .models import Candle

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    # Slash dates, month-first before day-first. Both orderings are guesses for
    # a date like 03/04/2020, but day-first silently produced an interleaved
    # sequence on US exports: day-first for days <= 12, month-first for >= 13.
    # Month-first at least fails consistently, and matches what Stooq, Yahoo
    # and Google Sheets emit.
    #
    # The time-bearing variants are what GOOGLEFINANCE("TICKER","all",...)
    # produces — "1/2/2024 16:00:00" — which parsed as nothing at all until a
    # real export was about to be pasted in.
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
)


def parse_timestamp(raw: str) -> datetime:
    """Parse the timestamp formats commonly found in exported market data."""
    text = raw.strip()
    if not text:
        raise ValueError("empty timestamp")

    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        epoch = int(text)
        # Heuristic: values this large are milliseconds, not seconds.
        if abs(epoch) > 10_000_000_000:
            epoch //= 1000
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unrecognized timestamp format: {raw!r}") from exc


#: Bodies a data provider returns instead of a CSV when it refuses the request.
#: Saved to disk with a .csv extension they look like data files, and the
#: column-detection error they used to produce ("missing required column(s)")
#: sent people looking for a formatting problem that was not there.
_ERROR_BODIES = (
    ("access denied", "the provider refused the request"),
    ("exceeded the daily hits limit", "the provider's per-IP rate limit was hit"),
    ("no data", "the provider has no data for that symbol"),
    ("<!doctype", "an HTML page was saved instead of a CSV"),
    ("<html", "an HTML page was saved instead of a CSV"),
    ("403 forbidden", "the request was rejected with HTTP 403"),
    ("404 not found", "the request was rejected with HTTP 404"),
)


def diagnose_error_body(text: str) -> str | None:
    """Explain a non-CSV body, or return None if it might be real data.

    Providers answer a refused download with a short message or an HTML error
    page, both of which happily save as ``something.csv``. Recognising them is
    the difference between "your file is an error page, re-download it" and a
    confusing complaint about missing columns.
    """
    head = text.strip()[:400].lower()
    if not head:
        return "the file is empty"
    for marker, explanation in _ERROR_BODIES:
        if head.startswith(marker) or marker in head[:80]:
            return explanation
    return None


class DataFeed:
    """Base feed. Subclasses implement ``__iter__``."""

    symbol: str = ""

    def __iter__(self) -> Iterator[Candle]:  # pragma: no cover - abstract
        raise NotImplementedError


class CandleFeed(DataFeed):
    """Wraps an in-memory sequence of candles."""

    def __init__(self, candles: Sequence[Candle], symbol: str = "") -> None:
        self.candles = list(candles)
        self.symbol = symbol or (self.candles[0].symbol if self.candles else "")

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.candles)


class CSVFeed(DataFeed):
    """Reads OHLCV bars from a CSV file.

    Column names are matched case-insensitively and tolerate the common
    variants (``date``/``time``/``timestamp``, ``adj close``, and so on).
    """

    _ALIASES = {
        "timestamp": ("timestamp", "time", "date", "datetime", "open_time"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c", "adj close", "adj_close", "close_price"),
        "volume": ("volume", "v", "vol", "quantity"),
    }

    def __init__(self, path: str | Path, symbol: str = "") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such data file: {self.path}")
        self.symbol = symbol or self.path.stem.upper()

    def _resolve_columns(self, fieldnames: Sequence[str]) -> dict[str, str]:
        lookup = {name.strip().lower(): name for name in fieldnames}
        resolved: dict[str, str] = {}
        for field, aliases in self._ALIASES.items():
            for alias in aliases:
                if alias in lookup:
                    resolved[field] = lookup[alias]
                    break
        missing = {"timestamp", "close"} - resolved.keys()
        if missing:
            raise ValueError(
                f"{self.path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        return resolved

    def __iter__(self) -> Iterator[Candle]:
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            preview = handle.read(512)
            problem = diagnose_error_body(preview)
            if problem is not None:
                raise ValueError(
                    f"{self.path} is not market data: {problem}. "
                    f"It begins {preview.strip()[:60]!r}. Re-download it — a browser "
                    "session usually succeeds where a direct fetch is refused."
                )
            handle.seek(0)

            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return
            columns = self._resolve_columns(reader.fieldnames)

            previous: datetime | None = None
            for line_number, row in enumerate(reader, start=2):
                raw_timestamp = row[columns["timestamp"]]
                if raw_timestamp is None or not raw_timestamp.strip():
                    continue
                try:
                    timestamp = parse_timestamp(raw_timestamp)
                    close = float(row[columns["close"]])
                    open_ = float(row[columns["open"]]) if "open" in columns else close
                    high = float(row[columns["high"]]) if "high" in columns else max(open_, close)
                    low = float(row[columns["low"]]) if "low" in columns else min(open_, close)
                    volume = float(row[columns["volume"]]) if "volume" in columns else 0.0
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc

                if previous is not None:
                    try:
                        out_of_order = timestamp < previous
                        duplicated = timestamp == previous
                    except TypeError as exc:
                        # One row carried a zone and another did not; comparing
                        # them raises a bare TypeError deep in the loop.
                        raise ValueError(
                            f"{self.path}:{line_number}: mixed timezone awareness — "
                            f"{timestamp!r} and {previous!r} cannot be compared. "
                            "Use one timestamp format throughout the file."
                        ) from exc
                    if duplicated:
                        raise ValueError(
                            f"{self.path}:{line_number}: duplicate timestamp {timestamp}; "
                            "repeated bars inflate the bar count and the return series"
                        )
                    if out_of_order:
                        raise ValueError(
                            f"{self.path}:{line_number}: timestamps must be chronological "
                            f"({timestamp} follows {previous})"
                        )
                previous = timestamp

                # Clamp so a rounded high/low in the source can't fail validation.
                high = max(high, open_, close)
                low = min(low, open_, close)

                yield Candle(
                    timestamp=timestamp,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    symbol=self.symbol,
                )


class SyntheticFeed(DataFeed):
    """Generates a geometric-Brownian-motion price series.

    Useful for smoke-testing a strategy end to end without a data file. A fixed
    ``seed`` makes the series reproducible.
    """

    def __init__(
        self,
        symbol: str = "SYNTH",
        bars: int = 500,
        start_price: float = 100.0,
        drift: float = 0.05,
        volatility: float = 0.20,
        bars_per_year: int = 252,
        start: datetime | None = None,
        interval: timedelta = timedelta(days=1),
        seed: int | None = 7,
    ) -> None:
        if bars < 1:
            raise ValueError("bars must be >= 1")
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        self.symbol = symbol
        self.bars = bars
        self.start_price = start_price
        self.drift = drift
        self.volatility = volatility
        self.bars_per_year = bars_per_year
        self.start = start or datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.interval = interval
        self.seed = seed

    def __iter__(self) -> Iterator[Candle]:
        rng = random.Random(self.seed)
        dt = 1.0 / self.bars_per_year
        mu = (self.drift - 0.5 * self.volatility**2) * dt
        sigma = self.volatility * math.sqrt(dt)

        price = self.start_price
        timestamp = self.start
        for _ in range(self.bars):
            open_ = price
            price = open_ * math.exp(mu + sigma * rng.gauss(0.0, 1.0))
            close = price
            wick = abs(sigma) * open_ * abs(rng.gauss(0.0, 0.6))
            high = max(open_, close) + wick
            low = max(min(open_, close) - wick, 0.01)
            volume = round(rng.uniform(5_000, 50_000), 2)

            yield Candle(
                timestamp=timestamp,
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=volume,
                symbol=self.symbol,
            )
            timestamp += self.interval


def write_csv(candles: Iterable[Candle], path: str | Path) -> Path:
    """Write candles to CSV in the format :class:`CSVFeed` reads back."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for candle in candles:
            writer.writerow(
                [
                    candle.timestamp.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ]
            )
    return destination
