"""Feature store contract for detection pipelines."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FeatureKey:
    """Stable lookup key for wallet/pair/window feature vectors."""

    wallet_id: str
    pair_id: str
    window_hours: int | None = None

    def __post_init__(self) -> None:
        if not self.wallet_id.strip():
            raise ValueError("FeatureKey.wallet_id must be non-empty")
        if not self.pair_id.strip():
            raise ValueError("FeatureKey.pair_id must be non-empty")
        if self.window_hours is not None and self.window_hours <= 0:
            raise ValueError("FeatureKey.window_hours must be positive when provided")

    @property
    def safe_id(self) -> str:
        wallet_hash = hashlib.sha256(self.wallet_id.encode()).hexdigest()[:16]
        window = self.window_hours if self.window_hours is not None else "default"
        return f"{wallet_hash}:{self.pair_id}:{window}"


@dataclass(frozen=True)
class FeatureRecord:
    """Feature vector with schema metadata required by model pipelines."""

    key: FeatureKey
    features: Mapping[str, Any]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("FeatureRecord.schema_version must be non-empty")
        if not self.features:
            raise ValueError("FeatureRecord.features must be non-empty")


@runtime_checkable
class FeatureStoreBackend(Protocol):
    """Backend contract for cache, online, or offline feature stores."""

    def get(self, key: FeatureKey) -> FeatureRecord | None:
        """Return a feature record or ``None`` on miss."""

    def put(self, record: FeatureRecord) -> None:
        """Persist a feature record."""


class InMemoryFeatureStore(FeatureStoreBackend):
    """Deterministic backend for tests, local development, and fixtures."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, FeatureRecord] = {}

    def get(self, key: FeatureKey) -> FeatureRecord | None:
        return self._records.get(key.safe_id)

    def put(self, record: FeatureRecord) -> None:
        self._records[record.key.safe_id] = record


def require_feature_schema(record: FeatureRecord, expected_schema_version: str) -> FeatureRecord:
    """Validate schema compatibility before a detector consumes features."""

    if record.schema_version != expected_schema_version:
        raise ValueError(
            "Feature schema version mismatch: "
            f"expected {expected_schema_version!r}, got {record.schema_version!r}"
        )
    return record
