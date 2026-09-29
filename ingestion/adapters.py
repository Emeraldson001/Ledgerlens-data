"""Concrete adapter implementations wrapping existing ingestion modules.

Each adapter wraps a legacy concrete module behind the contract interfaces
defined in ``ingestion/contracts.py``.  This preserves backward compatibility
while enabling the modular framework.
"""

from __future__ import annotations

from collections.abc import Iterator

from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.contracts import (
    SourceState,
    TradeBatchSource,
    TradeStreamSource,
)
from ingestion.data_models import Trade
from utils.logging import get_logger

logger = get_logger(__name__)


class HorizonSSESource(TradeStreamSource):
    """Adapter wrapping ``ingestion.horizon_streamer.stream_trades``.

    Streams trades from Horizon SSE for configured asset pairs.
    Supports multi-region failover via the endpoint pool.
    """

    def connect(self) -> None:
        self._state = SourceState.CONNECTED
        logger.info("HorizonSSESource[%s]: connected", self.name)

    def close(self) -> None:
        self._state = SourceState.CLOSED
        logger.info("HorizonSSESource[%s]: closed", self.name)

    def stream(self) -> Iterator[Trade]:
        # Lazy import to avoid pulling in stellar_sdk at import time
        # and to keep backward compat with the existing module.
        from ingestion.horizon_streamer import stream_trades as _stream_trades

        self._state = SourceState.STREAMING
        pairs = self._trade_config.asset_pairs or config.WATCHED_ASSET_PAIRS

        for code, issuer in pairs:
            sdk_asset = SdkAsset(code, issuer) if issuer != "native" else SdkAsset.native()
            for trade in _stream_trades(sdk_asset):
                self._items_processed += 1
                yield trade


class HorizonHistoricalSource(TradeBatchSource):
    """Adapter wrapping ``ingestion.historical_loader.load_trades``.

    Bulk-loads historical trades for configured asset pairs via Horizon REST.
    """

    def connect(self) -> None:
        self._state = SourceState.CONNECTED
        logger.info("HorizonHistoricalSource[%s]: connected", self.name)

    def close(self) -> None:
        self._state = SourceState.CLOSED
        logger.info("HorizonHistoricalSource[%s]: closed", self.name)

    def fetch(self) -> Iterator[Trade]:
        from ingestion.historical_loader import load_trades as _load_trades

        self._state = SourceState.STREAMING
        pairs = self._trade_config.asset_pairs or config.WATCHED_ASSET_PAIRS
        start = self._trade_config.start_time

        for code, issuer in pairs:
            sdk_asset = SdkAsset(code, issuer) if issuer != "native" else SdkAsset.native()
            for trade in _load_trades(
                sdk_asset, sdk_asset, start_time=start, limit_per_page=self._trade_config.batch_size
            ):
                self._items_processed += 1
                yield trade


class KafkaTradeSource(TradeStreamSource):
    """Adapter wrapping ``ingestion.kafka_producer.HorizonKafkaProducer``.

    Consumes trades from a Kafka topic.
    """

    def connect(self) -> None:
        self._state = SourceState.CONNECTED
        logger.info("KafkaTradeSource[%s]: connected", self.name)

    def close(self) -> None:
        self._state = SourceState.CLOSED
        logger.info("KafkaTradeSource[%s]: closed", self.name)

    def stream(self) -> Iterator[Trade]:
        from streaming.kafka_worker import KafkaWorker

        self._state = SourceState.STREAMING
        # Delegate to the existing KafkaWorker infrastructure
        worker = KafkaWorker()
        for msg in worker.consume():
            self._items_processed += 1
            yield msg
