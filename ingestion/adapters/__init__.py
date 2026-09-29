"""Blockchain ingestion adapters with a normalized event contract.

See ``docs/ingestion_adapters.md`` for the design rationale. Public API:

- ``NormalizedEvent`` / ``NormalizedAsset`` / ``EventType`` -- the
  chain-agnostic event contract.
- ``ChainAdapter`` -- the abstract base every adapter implements.
- ``AdapterValidationError`` -- raised when a raw event can't be normalized.
- ``StellarAdapter`` / ``EvmAdapter`` -- concrete adapters shipped in this repo.
- ``AdapterRegistry`` / ``default_registry`` -- chain-name -> adapter lookup.
"""

from ingestion.adapters.base import (
    AdapterValidationError,
    ChainAdapter,
    EventType,
    NormalizedAsset,
    NormalizedEvent,
)
from ingestion.adapters.evm_adapter import EvmAdapter
from ingestion.adapters.registry import (
    AdapterNotRegisteredError,
    AdapterRegistry,
    default_registry,
)
from ingestion.adapters.stellar_adapter import StellarAdapter

__all__ = [
    "AdapterNotRegisteredError",
    "AdapterRegistry",
    "AdapterValidationError",
    "ChainAdapter",
    "default_registry",
    "EventType",
    "EvmAdapter",
    "NormalizedAsset",
    "NormalizedEvent",
    "StellarAdapter",
]
