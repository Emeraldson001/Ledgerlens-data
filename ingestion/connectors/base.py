"""Typed plugin contract for LedgerLens data connectors.

A "connector" is the boundary between an external data source (Horizon
REST/SSE today; a different exchange API or chain indexer tomorrow) and the
rest of the ingestion pipeline. Everything downstream of a connector
(feature engineering, scoring, ``run_pipeline.py``) only ever sees the
pydantic record types in ``ingestion.data_models`` — never a source-specific
payload shape. That boundary is what lets a new data source be added
without editing ``detection/`` or ``run_pipeline.py`` at all.

To add a connector:
    1. Subclass ``DataConnector[YourRecordType]``.
    2. Declare a class-level ``metadata = ConnectorMetadata(...)``.
    3. Implement ``load()``.
    4. Register it — see ``ingestion.connectors.registry`` for both the
       in-tree (``@register_connector``) and out-of-tree (entry point)
       paths, and ``docs/connectors.md`` for a full walkthrough.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import pandas as pd
from pydantic import BaseModel

RecordT = TypeVar("RecordT", bound=BaseModel)


class ConnectorError(Exception):
    """Base class for every connector-boundary error.

    Catching this (rather than a bare ``Exception``) is how callers such as
    ``run_pipeline.py`` or a diagnostics script distinguish "this connector
    is unusable" from an unrelated bug.
    """


class ConnectorConfigError(ConnectorError):
    """Raised when a connector's required configuration is missing or invalid."""


class ConnectorNotFoundError(ConnectorError, KeyError):
    """Raised by the registry when looking up an unregistered connector id."""


class DuplicateConnectorError(ConnectorError):
    """Raised when two connector classes try to register the same connector id."""


@dataclass(frozen=True)
class ConnectorMetadata:
    """Static, introspectable description of a connector.

    Attributes:
        connector_id: Unique key used to look the connector up in a
            ``ConnectorRegistry`` (e.g. ``"stellar-sdex-trades"``).
        record_type: The pydantic model each ``load()`` call yields.
        source: Short human-readable origin, e.g. ``"horizon-rest"``.
        description: One-line summary shown by diagnostics tooling.
        required_env: Environment variables that must be set for
            ``validate_config()`` to succeed. Most Horizon-backed connectors
            ship working defaults in ``config.py`` so this is often empty —
            it exists for connectors (in-tree or plugin) that have no safe
            default, e.g. an API key for a third-party exchange.
    """

    connector_id: str
    record_type: type[BaseModel]
    source: str
    description: str = ""
    required_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectorHealth:
    """Result of ``DataConnector.health_check()`` — never raises, always returns this."""

    connector_id: str
    ok: bool
    detail: str = ""


class DataConnector(ABC, Generic[RecordT]):
    """Base class every data connector implements.

    Subclasses must set the class attribute ``metadata`` and implement
    ``load()``. ``validate_config()``, ``to_dataframe()``, and
    ``health_check()`` have defaults that fit most sources but may be
    overridden — e.g. to validate more than "is this env var set", or to
    flatten records into a DataFrame schema an existing downstream
    consumer already expects.
    """

    metadata: ClassVar[ConnectorMetadata]

    def validate_config(self) -> None:
        """Raise ``ConnectorConfigError`` if required configuration is missing.

        Default: checks every name in ``metadata.required_env`` is a
        non-empty environment variable. This runs *before* any network
        call so a misconfigured connector fails fast with an actionable
        message instead of a confusing error deep inside ``load()``.
        """
        missing = [key for key in self.metadata.required_env if not os.environ.get(key)]
        if missing:
            raise ConnectorConfigError(
                f"Connector '{self.metadata.connector_id}' is missing required "
                f"configuration: {', '.join(missing)}. Set these (e.g. in .env — "
                f"see .env.example) before loading this connector."
            )

    @abstractmethod
    def load(self, **kwargs: object) -> Iterator[RecordT]:
        """Fetch and yield normalized records for this source.

        Implementations should raise ``ConnectorError`` (or a subclass) for
        boundary failures the caller can act on — e.g. a required kwarg
        wasn't supplied, or a source-specific resource id doesn't exist —
        rather than letting a raw source-library exception escape.
        """
        raise NotImplementedError

    def to_dataframe(self, records: Iterable[RecordT]) -> pd.DataFrame:
        """Flatten records into a DataFrame. Default: one column per field."""
        return pd.DataFrame([r.model_dump() for r in records])

    def health_check(self) -> ConnectorHealth:
        """Lightweight, non-raising status check — config validity by default.

        Safe to call from diagnostics tooling (``scripts/list_connectors.py``)
        without triggering a network call: failures are reported via the
        returned ``ConnectorHealth``, never via an exception. Subclasses that
        want to also probe the live endpoint should override this but keep
        the "never raises" contract.
        """
        try:
            self.validate_config()
        except ConnectorConfigError as exc:
            return ConnectorHealth(
                connector_id=self.metadata.connector_id, ok=False, detail=str(exc)
            )
        return ConnectorHealth(connector_id=self.metadata.connector_id, ok=True, detail="config OK")
