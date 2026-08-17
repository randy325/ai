"""Run configuration: one place to assemble a strategy, broker, risk and feed."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

from .broker import Commission, PaperBroker, SlippageModel
from .data import CSVFeed, DataFeed, SyntheticFeed
from .engine import TradingEngine
from .providers import LiveFeed, MarketDataFeed, Provider, build_provider
from .risk import FixedFractionSizer, PositionSizer, RiskLimits, RiskManager, VolatilitySizer
from .strategy import Strategy, TrendFilter, build_strategy


@dataclass
class RunConfig:
    """Everything needed to reproduce a run, loadable from JSON."""

    strategy: str = "sma-crossover"
    strategy_params: dict = field(default_factory=dict)
    symbol: str = "SYNTH"
    data_file: str | None = None
    provider: str | None = None
    interval: str = "1d"
    limit: int = 500
    cache_dir: str | None = ".cache/market-data"
    cache_hours: float = 12.0
    bars: int = 750
    seed: int = 7
    volatility: float = 0.25
    drift: float = 0.08

    starting_cash: float = 100_000.0
    commission_percent: float = 0.001
    commission_per_share: float = 0.0
    commission_minimum: float = 0.0
    slippage_percent: float = 0.0005

    sizer: str = "fixed"
    position_fraction: float = 0.95
    risk_per_trade: float = 0.02
    atr_multiple: float = 2.0

    max_drawdown: float | None = 0.25
    daily_loss_limit: float | None = None
    max_trades_per_day: int | None = None

    allow_short: bool = False
    max_leverage: float = 1.0
    trend_filter: int | None = None
    execute_on: str = "next_open"

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_file(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return destination

    # -- component factories -------------------------------------------------

    def build_provider(self) -> Provider:
        """The configured live-data provider, wrapped in a disk cache."""
        if not self.provider:
            raise ValueError("no provider configured")
        return build_provider(
            self.provider,
            cache_dir=self.cache_dir or None,
            cache_ttl=timedelta(hours=self.cache_hours),
        )

    def build_feed(self) -> DataFeed:
        # A data file wins over a provider: local data is explicit, and it is
        # what you want when reproducing a run exactly.
        if self.data_file:
            return CSVFeed(self.data_file, symbol=self.symbol)
        if self.provider:
            return MarketDataFeed(
                self.build_provider(), self.symbol, self.interval, self.limit
            )
        return SyntheticFeed(
            symbol=self.symbol,
            bars=self.bars,
            seed=self.seed,
            volatility=self.volatility,
            drift=self.drift,
        )

    def build_strategy(self) -> Strategy:
        params = dict(self.strategy_params)
        if self.allow_short and self.strategy != "buy-and-hold":
            params.setdefault("allow_short", True)
        strategy = build_strategy(self.strategy, **params)
        if self.trend_filter:
            strategy = TrendFilter(strategy, period=self.trend_filter)
        return strategy

    def build_broker(self) -> PaperBroker:
        return PaperBroker(
            starting_cash=self.starting_cash,
            commission=Commission(
                per_share=self.commission_per_share,
                percent=self.commission_percent,
                minimum=self.commission_minimum,
            ),
            slippage=SlippageModel(percent=self.slippage_percent),
            allow_short=self.allow_short,
            max_leverage=self.max_leverage,
        )

    def build_sizer(self) -> PositionSizer:
        if self.sizer == "fixed":
            return FixedFractionSizer(self.position_fraction)
        if self.sizer == "volatility":
            return VolatilitySizer(
                risk_per_trade=self.risk_per_trade,
                atr_multiple=self.atr_multiple,
                max_fraction=self.position_fraction,
            )
        raise ValueError(f"unknown sizer {self.sizer!r}; expected 'fixed' or 'volatility'")

    def build_risk(self) -> RiskManager:
        limits = RiskLimits(
            max_position_fraction=self.position_fraction,
            max_drawdown=self.max_drawdown,
            daily_loss_limit=self.daily_loss_limit,
            risk_per_trade=self.risk_per_trade,
            max_trades_per_day=self.max_trades_per_day,
        )
        return RiskManager(limits, self.build_sizer())

    def build_live_feed(self, warmup: int = 200, max_bars: int | None = None) -> LiveFeed:
        """A polling feed that yields each bar as it closes.

        Never cached — a cache would serve a stale bar as if it were new.
        """
        if not self.provider:
            raise ValueError("live paper trading needs --provider")
        return LiveFeed(
            provider=build_provider(self.provider),
            symbol=self.symbol,
            interval=self.interval,
            warmup=warmup,
            max_bars=max_bars,
        )

    def build_engine(self) -> TradingEngine:
        return TradingEngine(
            strategy=self.build_strategy(),
            broker=self.build_broker(),
            risk=self.build_risk(),
            execute_on=self.execute_on,
        )
