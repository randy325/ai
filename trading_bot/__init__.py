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
from .models import Candle, EquityPoint, Fill, Order, OrderType, Position, Side, Signal, Trade
from .providers import (
    PROVIDERS,
    BinanceProvider,
    CachingTransport,
    CoinbaseProvider,
    DataFeedError,
    LiveFeed,
    MarketDataFeed,
    Provider,
    RateLimited,
    StooqProvider,
    SymbolNotFound,
    Transport,
    UrllibTransport,
    YahooProvider,
    build_provider,
)
from .risk import FixedFractionSizer, RiskLimits, RiskManager, VolatilitySizer
from .strategy import (
    STRATEGIES,
    BollingerReversion,
    Breakout,
    BuyAndHold,
    MACDTrend,
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
    "EquityPoint",
    "Fill",
    "FixedFractionSizer",
    "LiveFeed",
    "MACDTrend",
    "MarketDataFeed",
    "Metrics",
    "Order",
    "OrderType",
    "PROVIDERS",
    "PaperBroker",
    "Position",
    "Provider",
    "RSIMeanReversion",
    "RateLimited",
    "RiskLimits",
    "RiskManager",
    "RunConfig",
    "SMACrossover",
    "STRATEGIES",
    "Side",
    "Signal",
    "SlippageModel",
    "StooqProvider",
    "Strategy",
    "SymbolNotFound",
    "SyntheticFeed",
    "Trade",
    "TradingEngine",
    "Transport",
    "TrendFilter",
    "UrllibTransport",
    "VolatilitySizer",
    "YahooProvider",
    "build_provider",
    "build_strategy",
    "compute_metrics",
]
