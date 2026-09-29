"""Tests for the ingestion contracts (modular source framework)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime

import pytest

# Import contracts directly to avoid pulling in ingestion.__init__ which
# eagerly registers adapters with heavy dependencies (stellar_sdk, confluent_kafka).
from ingestion.contracts import (  # noqa: E402
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
from ingestion.data_models import Trade  # noqa: E402


class TestSourceConfig:
    def test_default_config(self):
        cfg = SourceConfig(name="test")
        assert cfg.name == "test"
        assert cfg.enabled is True
        assert cfg.max_retries == 5

    def test_custom_config(self):
        cfg = SourceConfig(name="custom", enabled=False, max_retries=3)
        assert cfg.enabled is False
        assert cfg.max_retries == 3

    def test_tags(self):
        cfg = SourceConfig(name="tagged", tags={"env": "test", "region": "us-east"})
        assert cfg.tags["env"] == "test"


class TestTradeSourceConfig:
    def test_with_asset_pairs(self):
        cfg = TradeSourceConfig(
            name="test_trades",
            asset_pairs=[("USDC", "native"), ("BTC", "GBP")],
        )
        assert len(cfg.asset_pairs) == 2

    def test_with_start_time(self):
        t = datetime(2026, 1, 1)
        cfg = TradeSourceConfig(name="timed", start_time=t)
        assert cfg.start_time == t


class TestDataSourceLifecycle:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DataSource(SourceConfig(name="bad"))  # type: ignore[abstract]

    def test_state_transitions(self):
        class SimpleSource(DataSource[str]):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

        src = SimpleSource(SourceConfig(name="simple"))
        assert src.state == SourceState.CREATED
        assert src.name == "simple"

        src.connect()
        assert src.state == SourceState.CONNECTED

        health = src.health()
        assert health.healthy is True
        assert health.state == SourceState.CONNECTED

        src.close()
        assert src.state == SourceState.CLOSED

    def test_context_manager(self):
        class SimpleSource(DataSource[str]):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

        with SimpleSource(SourceConfig(name="ctx")) as src:
            assert src.state == SourceState.CONNECTED

        assert src.state == SourceState.CLOSED


class TestStreamSource:
    def test_stream_source_contract(self):
        class TestStream(StreamSource[int]):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

            def stream(self):
                yield 1
                yield 2
                yield 3

        src = TestStream(SourceConfig(name="int_stream"))
        src.connect()
        items = list(src.stream())
        assert items == [1, 2, 3]
        assert src.health().items_processed == 0
        src.close()


class TestBatchSource:
    def test_batch_source_contract(self):
        class TestBatch(BatchSource[str]):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

            def fetch(self):
                yield "a"
                yield "b"

        src = TestBatch(SourceConfig(name="str_batch"))
        src.connect()
        items = list(src.fetch())
        assert items == ["a", "b"]
        assert src.health().items_processed == 0
        src.close()


class TestSourceRegistry:
    def setup_method(self):
        SourceRegistry.clear()

    def test_register_and_create(self):
        class MockSource(DataSource[str]):
            def __init__(self, config):
                super().__init__(config)

            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

        SourceRegistry.register("mock", MockSource)
        assert "mock" in SourceRegistry.registered_names()

        cfg = SourceConfig(name="mock_instance")
        instance = SourceRegistry.create("mock", cfg)
        assert isinstance(instance, MockSource)
        assert instance.name == "mock_instance"

    def test_create_unknown(self):
        SourceRegistry.clear()
        with pytest.raises(KeyError):
            SourceRegistry.create("nonexistent", SourceConfig(name="x"))

    def test_registered_names(self):
        SourceRegistry.clear()

        class A(DataSource):
            def connect(self):
                pass

            def close(self):
                pass

        class B(DataSource):
            def connect(self):
                pass

            def close(self):
                pass

        SourceRegistry.register("a", A)
        SourceRegistry.register("b", B)
        assert SourceRegistry.registered_names() == ["a", "b"]

    def test_clear(self):
        class A(DataSource):
            def connect(self):
                pass

            def close(self):
                pass

        SourceRegistry.register("a", A)
        SourceRegistry.clear()
        assert SourceRegistry.registered_names() == []


class TestTradeContracts:
    def test_trade_stream_source(self):
        class MockTradeStream(TradeStreamSource):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

            def stream(self):
                yield from ()

        cfg = TradeSourceConfig(name="trade_stream")
        src = MockTradeStream(cfg)
        assert src.name == "trade_stream"
        assert src.trade_config is cfg

    def test_trade_batch_source(self):
        class MockTradeBatch(TradeBatchSource):
            def connect(self):
                self._state = SourceState.CONNECTED

            def close(self):
                self._state = SourceState.CLOSED

            def fetch(self):
                yield from ()

        cfg = TradeSourceConfig(name="trade_batch")
        src = MockTradeBatch(cfg)
        assert src.name == "trade_batch"
        assert src.trade_config is cfg


class TestAnomalyDetectionStrategy:
    def test_strategy_interface(self):
        class ThresholdStrategy(AnomalyDetectionStrategy):
            def name(self) -> str:
                return "threshold"

            def score(self, trade: Trade) -> float:
                return 0.5 if trade.base_amount > 1000 else 0.0

            def supports_batch(self) -> bool:
                return False

        strategy = ThresholdStrategy()
        assert strategy.name() == "threshold"
        assert not strategy.supports_batch()

    def test_batch_fallback(self):
        class SimpleStrategy(AnomalyDetectionStrategy):
            def name(self) -> str:
                return "simple"

            def score(self, trade: Trade) -> float:
                return 1.0

            def supports_batch(self) -> bool:
                return True

        strategy = SimpleStrategy()
        assert strategy.supports_batch()

    def test_default_score_batch(self):
        class NoBatchStrategy(AnomalyDetectionStrategy):
            def name(self) -> str:
                return "nobatch"

            def score(self, trade: Trade) -> float:
                return 0.0

            def supports_batch(self) -> bool:
                return False

        strategy = NoBatchStrategy()
        trades = []
        scores = strategy.score_batch(trades)
        assert scores == []
