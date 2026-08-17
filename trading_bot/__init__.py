"""A backtesting and paper-trading bot built on the Python standard library.

Simulation only: nothing in this package connects to a broker or exchange, and
no code path can place a real order.

    from trading_bot import RunConfig

    config = RunConfig(strategy="sma-crossover", bars=750)
    result = config.build_engine().run(config.build_feed())
    print(result.summary())
"""

from .broker import Commission, PaperBroker, SlippageModel
from .config import RunConfig
from .data import CandleFeed, CSVFeed, DataFeed, SyntheticFeed
from .engine import BacktestResult, TradingEngine
from .metrics import Metrics, compute_metrics
from .models import (
    Candle,
    EquityPoint,
    Fill,
    Order,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    RejectionKind,
    Side,
    Signal,
    Trade,
)
from .providers import (
    PROVIDERS,
    BinanceProvider,
    FallbackProvider,
    CachingTransport,
    CoinbaseProvider,
    DataFeedError,
    LiveFeed,
    MarketDataFeed,
    MockProvider,
    Provider,
    RateLimited,
    SimulatedClock,
    StooqProvider,
    SymbolNotFound,
    Transport,
    UrllibTransport,
    YahooProvider,
    build_provider,
)
from .risk import (
    AGGRESSIVE,
    CONSERVATIVE,
    MODERATE,
    FixedFractionSizer,
    RiskLimits,
    RiskManager,
    RiskTier,
    TieredRiskManager,
    VolatilitySizer,
)
from .strategy import (
    STRATEGIES,
    BollingerReversion,
    Breakout,
    BuyAndHold,
    Ensemble,
    MACDTrend,
    RSIBreakout,
    RSIMeanReversion,
    SMACrossover,
    Strategy,
    TrendFilter,
    build_strategy,
)

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "BinanceProvider",
    "BollingerReversion",
    "Breakout",
    "BuyAndHold",
    "CSVFeed",
    "CachingTransport",
    "Candle",
    "CandleFeed",
    "CoinbaseProvider",
    "Commission",
    "DataFeed",
    "DataFeedError",
    "Ensemble",
    "EquityPoint",
    "FallbackProvider",
    "Fill",
    "FixedFractionSizer",
    "LiveFeed",
    "MACDTrend",
    "MarketDataFeed",
    "MockProvider",
    "Metrics",
    "Order",
    "OrderResult",
    "OrderStatus",
    "OrderType",
    "PROVIDERS",
    "PaperBroker",
    "Position",
    "Provider",
    "RSIBreakout",
    "RSIMeanReversion",
    "RateLimited",
    "RejectionKind",
    "RiskLimits",
    "RiskManager",
    "RiskTier",
    "RunConfig",
    "SMACrossover",
    "STRATEGIES",
    "Side",
    "Signal",
    "SimulatedClock",
    "SlippageModel",
    "StooqProvider",
    "Strategy",
    "SymbolNotFound",
    "SyntheticFeed",
    "Trade",
    "TradingEngine",
    "TieredRiskManager",
    "Transport",
    "TrendFilter",
    "UrllibTransport",
    "VolatilitySizer",
    "YahooProvider",
    "build_provider",
    "build_strategy",
    "compute_metrics",
]
