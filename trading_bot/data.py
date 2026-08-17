"""Market data feeds.

A feed is any iterable of :class:`~trading_bot.models.Candle` in chronological
order. That keeps the engine indifferent to whether bars come from a CSV file,
a generator, or a live socket.
"""

from __future__ import annotations

import csv
import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .models import Candle

#: Formats that mean exactly one thing whatever the file's origin.
_UNAMBIGUOUS_FORMATS = (
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)

#: Slash dates, which do NOT mean one thing. 03/04/2024 is 3 April in most of
#: the world and 4 March in the US, and no amount of per-row cleverness can tell
#: them apart — the ordering is a property of the FILE. Resolve it once with
#: :func:`detect_date_order` and pass the answer in.
_MONTH_FIRST_FORMATS = (
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)
_DAY_FIRST_FORMATS = (
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)

_SLASH_DATE = re.compile(
    r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*$"
)


class AmbiguousDateOrder(ValueError):
    """A file's slash dates could be read either way and nothing settles it."""


def slash_fields(text: str) -> tuple[int, int] | None:
    """The first two numbers of a slash date, or None if it is not one."""
    match = _SLASH_DATE.match(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def detect_date_order(samples) -> str:
    """Resolve slash-date ordering for a whole file from its own contents.

    A value above 12 in the first field can only be a day; above 12 in the
    second can only be a month. One such row settles the entire file.

    Deciding per row instead is the trap this replaces: with both orderings in
    one fallback list, a European file parses days 1-12 as months and days
    13-31 correctly, interleaving two calendars. Worse, a monthly series dated
    the 1st reads as consecutive days of January — ascending, so the
    chronological check never fires and the corruption is silent.
    """
    first_over_12 = second_over_12 = False
    saw_slash = False
    for sample in samples:
        fields = slash_fields(sample) if sample else None
        if fields is None:
            continue
        saw_slash = True
        first, second = fields
        first_over_12 |= first > 12
        second_over_12 |= second > 12

    if not saw_slash:
        return "month-first"  # no slash dates; the answer is irrelevant
    if first_over_12 and second_over_12:
        raise ValueError(
            "date column contains both DD/MM and MM/DD rows — it is not one "
            "consistent format, so it cannot be read safely"
        )
    if first_over_12:
        return "day-first"
    if second_over_12:
        return "month-first"
    raise AmbiguousDateOrder(
        "every slash date in this file has both fields <= 12, so day-first and "
        "month-first are equally consistent with it. Pass date_order='day-first' "
        "or 'month-first' explicitly — guessing would silently shift every bar"
    )



def parse_timestamp(raw: str, date_order: str = "month-first") -> datetime:
    """Parse a timestamp from exported market data.

    ``date_order`` settles slash dates and must come from
    :func:`detect_date_order` when reading a file. The default here serves
    single-value use only; :class:`CSVFeed` never relies on it, because a lone
    date genuinely cannot be disambiguated and a file usually can.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty timestamp")

    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        epoch = int(text)
        # Heuristic: values this large are milliseconds, not seconds.
        if abs(epoch) > 10_000_000_000:
            epoch //= 1000
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    if date_order not in ("month-first", "day-first"):
        raise ValueError(f"date_order must be 'month-first' or 'day-first', got {date_order!r}")
    slash = _DAY_FIRST_FORMATS if date_order == "day-first" else _MONTH_FIRST_FORMATS
    for fmt in (*_UNAMBIGUOUS_FORMATS, *slash):
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

    def __init__(self, path: str | Path, symbol: str = "", date_order: str = "auto") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such data file: {self.path}")
        if date_order not in ("auto", "month-first", "day-first"):
            raise ValueError(
                f"date_order must be 'auto', 'month-first' or 'day-first', got {date_order!r}"
            )
        self.symbol = symbol or self.path.stem.upper()
        #: "auto" resolves slash-date ordering from the file's own contents and
        #: raises if nothing in it settles the question.
        self.date_order = date_order

    def _scan_date_order(self, column: str) -> str:
        """Resolve the file's slash-date ordering in one pass, before parsing.

        Ordering is a property of the file, not of a row, so it is decided once
        and applied to every row. Deciding per row is what let a European file
        parse days 1-12 as months while reading 13-31 correctly.
        """
        if self.date_order != "auto":
            return self.date_order
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            samples = []
            for row in reader:
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                if not samples and slash_fields(raw) is None:
                    # First real value is not a slash date, so the question does
                    # not arise. Avoids reading the whole file for nothing.
                    return "month-first"
                samples.append(raw)
        try:
            return detect_date_order(samples)
        except AmbiguousDateOrder as exc:
            raise AmbiguousDateOrder(f"{self.path}: {exc}") from None

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
            date_order = self._scan_date_order(columns["timestamp"])

            previous: datetime | None = None
            for line_number, row in enumerate(reader, start=2):
                raw_timestamp = row[columns["timestamp"]]
                if raw_timestamp is None or not raw_timestamp.strip():
                    continue
                try:
                    timestamp = parse_timestamp(raw_timestamp, date_order)
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
