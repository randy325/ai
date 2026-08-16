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
    "BollingerReversion",
    "Breakout",
    "BuyAndHold",
    "CSVFeed",
    "Candle",
    "CandleFeed",
    "Commission",
    "DataFeed",
    "EquityPoint",
    "Fill",
    "FixedFractionSizer",
    "MACDTrend",
    "Metrics",
    "Order",
    "OrderType",
    "PaperBroker",
    "Position",
    "RSIMeanReversion",
    "RiskLimits",
    "RiskManager",
    "RunConfig",
    "SMACrossover",
    "STRATEGIES",
    "Side",
    "Signal",
    "SlippageModel",
    "Strategy",
    "SyntheticFeed",
    "Trade",
    "TradingEngine",
    "TrendFilter",
    "VolatilitySizer",
    "build_strategy",
    "compute_metrics",
]
