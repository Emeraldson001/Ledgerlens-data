"""Formal source contracts for the modular ingestion framework.

This module defines the abstract interfaces (contracts) that every data
source in the ingestion layer must satisfy.  These contracts decouple
pipeline orchestration from specific source implementations, making it
possible to add, replace, or test sources without changing the pipeline.

Contract hierarchy
------------------
DataSource          -- base: every source has a name, config, and lifecycle
├── StreamSource    -- streaming: yields items indefinitely
├── BatchSource     -- one-shot: yields a finite collection
└── TradeSource     -- trade-specific contract (used by the scoring pipeline)

Usage
-----
    from ingestion.contracts import TradeSource, SourceRegistry

    source = SourceRegistry.get("horizon_sse")
    for trade in source.stream():
        process(trade)
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from ingestion.data_models import Trade

T = TypeVar("T")


class SourceState(enum.Enum):
    """Lifecycle state of a data source."""

    CREATED = "created"
    CONNECTED = "connected"
    STREAMING = "streaming"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class SourceHealth:
    """Health status of a data source at a point in time."""

    healthy: bool
    state: SourceState = SourceState.CREATED
    last_activity: datetime | None = None
    error_count: int = 0
    last_error: str | None = None
    items_processed: int = 0


@dataclass
class SourceConfig:
    """Base configuration for any data source.

    Subclasses can add source-specific fields.
    """

    name: str
    enabled: bool = True
    retry_on_failure: bool = True
    max_retries: int = 5
    timeout_seconds: float = 30.0
    tags: dict[str, str] = field(default_factory=dict)


class DataSource(abc.ABC, Generic[T]):
    """Abstract base contract for all ingestion data sources.

    Every source has a name, a configuration, and a lifecycle
    (connect → use → close).
    """

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._state = SourceState.CREATED
        self._items_processed = 0
        self._error_count = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> SourceConfig:
        return self._config

    @property
    def state(self) -> SourceState:
        return self._state

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the connection to the underlying data source.

        Raises ``ConnectionError`` on failure.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources held by this source."""

    def health(self) -> SourceHealth:
        """Return the current health status."""
        return SourceHealth(
            healthy=self._state not in (SourceState.FAILED, SourceState.CLOSED),
            state=self._state,
            error_count=self._error_count,
            items_processed=self._items_processed,
        )

    def __enter__(self) -> DataSource[T]:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class StreamSource(DataSource[T]):
    """A data source that yields an unbounded stream of items.

    Implementations must handle reconnection internally and should
    document their reconnection policy.
    """

    @abc.abstractmethod
    def stream(self) -> Iterator[T]:
        """Yield items from the source indefinitely.

        This is a generator that may block waiting for new data.
        Implementations should handle transient errors internally
        and only raise on fatal failures.
        """


class BatchSource(DataSource[T]):
    """A data source that yields a finite collection of items (one-shot)."""

    @abc.abstractmethod
    def fetch(self) -> Iterator[T]:
        """Yield all available items and then stop.

        The caller is responsible for iterating to completion.
        """


# ---------------------------------------------------------------------------
# Trade-specific contracts
# ---------------------------------------------------------------------------


@dataclass
class TradeSourceConfig(SourceConfig):
    """Configuration for a trade data source."""

    asset_pairs: list[tuple[str, str]] | None = None
    start_time: datetime | None = None
    batch_size: int = 200


class TradeStreamSource(StreamSource[Trade]):
    """Contract for streaming trade sources (Horizon SSE, Kafka, etc.)."""

    def __init__(self, config: TradeSourceConfig) -> None:
        super().__init__(config)
        self._trade_config = config

    @property
    def trade_config(self) -> TradeSourceConfig:
        return self._trade_config


class TradeBatchSource(BatchSource[Trade]):
    """Contract for bulk historical trade sources."""

    def __init__(self, config: TradeSourceConfig) -> None:
        super().__init__(config)
        self._trade_config = config

    @property
    def trade_config(self) -> TradeSourceConfig:
        return self._trade_config


# ---------------------------------------------------------------------------
# Anomaly detection strategy interface (#442)
# ---------------------------------------------------------------------------


class AnomalyDetectionStrategy(abc.ABC):
    """Pluggable strategy for anomaly detection on trade data.

    Implementations encapsulate a specific detection algorithm
    (Benford, ML ensemble, statistical tests, etc.) behind a
    uniform interface so the pipeline can compose or swap strategies.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abc.abstractmethod
    def score(self, trade: Trade) -> float:
        """Return an anomaly score in [0, 1] for a single trade.

        Higher scores indicate higher likelihood of anomalous behaviour.
        """

    @abc.abstractmethod
    def supports_batch(self) -> bool:
        """Return True if this strategy supports batch scoring."""

    def score_batch(self, trades: list[Trade]) -> list[float]:
        """Score a batch of trades (optional, default calls ``score`` per item)."""
        return [self.score(t) for t in trades]


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class SourceRegistry:
    """Registry of known data source implementations.

    Sources can be registered by name and discovered at runtime.
    This decouples pipeline configuration from import-time wiring.

    Usage::

        SourceRegistry.register("horizon_sse", HorizonSSESource)
        source = SourceRegistry.create("horizon_sse", config)
    """

    _registry: dict[str, type[DataSource]] = {}

    @classmethod
    def register(cls, name: str, source_cls: type[DataSource]) -> None:
        """Register a source class under *name*."""
        cls._registry[name] = source_cls

    @classmethod
    def create(cls, name: str, config: SourceConfig) -> DataSource:
        """Create a source instance by registered name."""
        source_cls = cls._registry.get(name)
        if source_cls is None:
            raise KeyError(
                f"Unknown source '{name}'. " f"Registered sources: {sorted(cls._registry)}"
            )
        return source_cls(config)

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return the list of registered source names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful in tests)."""
        cls._registry.clear()
