"""Connector contracts for ledger data providers.

The ingestion layer can use these contracts to add Horizon, archive, or
third-party ledger sources without coupling downstream pipelines to one
provider's pagination and cursor semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, TypeVar, runtime_checkable

from ingestion.data_models import AccountActivity, OrderBookEvent, Trade

LedgerRecord = Trade | OrderBookEvent | AccountActivity
RecordT = TypeVar("RecordT", bound=LedgerRecord)


@dataclass(frozen=True)
class SourceCursor:
    """Provider-neutral resume token for incremental ingestion."""

    provider: str
    position: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("SourceCursor.provider must be non-empty")
        if not self.position.strip():
            raise ValueError("SourceCursor.position must be non-empty")


@dataclass(frozen=True)
class SourceBatch(Sequence[RecordT]):
    """A validated page of ledger records plus the next resume cursor."""

    records: tuple[RecordT, ...]
    next_cursor: SourceCursor | None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    diagnostics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("SourceBatch.fetched_at must be timezone-aware")
        if any(
            not isinstance(record, (Trade, OrderBookEvent, AccountActivity))
            for record in self.records
        ):
            raise TypeError("SourceBatch.records must contain supported ledger data models")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RecordT:
        return self.records[index]


@runtime_checkable
class LedgerSourceConnector(Protocol[RecordT]):
    """Minimal contract every ledger provider connector must implement."""

    provider_name: str

    def fetch_since(
        self,
        cursor: SourceCursor | None = None,
        *,
        limit: int = 500,
    ) -> SourceBatch[RecordT]:
        """Fetch records after ``cursor`` and return a resumable batch."""


def validate_connector_batch(
    connector: LedgerSourceConnector[RecordT],
    batch: SourceBatch[RecordT],
) -> None:
    """Raise actionable diagnostics for connector contract violations."""

    if not connector.provider_name.strip():
        raise ValueError("Connector provider_name must be non-empty")
    if batch.next_cursor and batch.next_cursor.provider != connector.provider_name:
        raise ValueError(
            "Connector returned cursor for provider "
            f"{batch.next_cursor.provider!r}, expected {connector.provider_name!r}"
        )


class StaticLedgerSourceConnector(LedgerSourceConnector[RecordT]):
    """Small deterministic connector useful for tests and local fixtures."""

    def __init__(self, provider_name: str, records: Iterable[RecordT]) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name must be non-empty")
        self.provider_name = provider_name
        self._records = tuple(records)

    def fetch_since(
        self,
        cursor: SourceCursor | None = None,
        *,
        limit: int = 500,
    ) -> SourceBatch[RecordT]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        start = int(cursor.position) if cursor else 0
        end = min(start + limit, len(self._records))
        next_cursor = (
            SourceCursor(provider=self.provider_name, position=str(end))
            if end < len(self._records)
            else None
        )
        batch = SourceBatch(records=self._records[start:end], next_cursor=next_cursor)
        validate_connector_batch(self, batch)
        return batch
