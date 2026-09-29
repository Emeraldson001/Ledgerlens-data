from .contracts import (
    AnomalyDetectionStrategy,
    BatchSource,
    DataSource,
    SourceConfig,
    SourceRegistry,
    SourceState,
    StreamSource,
    TradeBatchSource,
    TradeSourceConfig,
    TradeStreamSource,
)
from .data_models import AccountActivity, OrderBookEvent, Trade
from .exceptions import (
    IngestionError,
    IngestionNotFoundError,
    IngestionRateLimitError,
    IngestionTransportError,
    IngestionValidationError,
    RecordValidationError,
    SchemaDecodeError,
    SchemaValidationError,
)
from .sketches import WalletSketchBook


def _register_builtin_sources() -> None:
    """Lazy-register adapter implementations (avoids heavy imports at package level)."""
    try:
        from .adapters import HorizonHistoricalSource, HorizonSSESource, KafkaTradeSource

        SourceRegistry.register("horizon_sse", HorizonSSESource)
        SourceRegistry.register("horizon_historical", HorizonHistoricalSource)
        SourceRegistry.register("kafka", KafkaTradeSource)
    except ImportError:
        pass


__all__ = [
    "AccountActivity",
    "AnomalyDetectionStrategy",
    "BatchSource",
    "DataSource",
    "HorizonHistoricalSource",
    "HorizonSSESource",
    "IngestionError",
    "IngestionNotFoundError",
    "IngestionRateLimitError",
    "IngestionTransportError",
    "IngestionValidationError",
    "KafkaTradeSource",
    "OrderBookEvent",
    "RecordValidationError",
    "SchemaDecodeError",
    "SchemaValidationError",
    "SourceConfig",
    "SourceRegistry",
    "SourceState",
    "StreamSource",
    "Trade",
    "TradeBatchSource",
    "TradeSourceConfig",
    "TradeStreamSource",
    "WalletSketchBook",
]
