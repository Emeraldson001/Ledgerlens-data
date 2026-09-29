"""Registry mapping chain identifiers to their :class:`ChainAdapter`.

Lets ingestion entry points (Kafka workers, backfill scripts) resolve the
correct adapter for a stream by chain name, so adding a new chain means
registering a new adapter class here rather than editing dispatch logic in
every ingestion call site.
"""

from __future__ import annotations

from typing import Any

from ingestion.adapters.base import ChainAdapter, NormalizedEvent
from utils.logging import get_logger

logger = get_logger(__name__)


class AdapterNotRegisteredError(KeyError):
    """Raised when no adapter is registered for a requested chain."""

    def __init__(self, chain: str, available: list[str]):
        self.chain = chain
        self.available = available
        super().__init__(
            f"no ChainAdapter registered for chain={chain!r}; "
            f"available: {available or '(none)'}"
        )


class AdapterRegistry:
    """Thread-unsafe-by-design (registration happens at startup) registry.

    Kept intentionally simple: registration is expected to happen once,
    during process/module init, not on the hot ingestion path -- so no
    locking overhead is added for the lookup path (`get`, `normalize`).
    """

    def __init__(self) -> None:
        self._adapters: dict[str, ChainAdapter] = {}

    def register(self, adapter: ChainAdapter) -> None:
        """Register *adapter* under its ``chain`` identifier.

        Re-registering the same chain name overwrites the previous
        adapter (useful in tests); a warning is logged so an accidental
        double-registration in production code is visible.
        """
        chain = adapter.chain.lower()
        if chain in self._adapters and self._adapters[chain] is not adapter:
            logger.warning("overwriting previously registered adapter", extra={"chain": chain})
        self._adapters[chain] = adapter

    def get(self, chain: str) -> ChainAdapter:
        adapter = self._adapters.get(chain.lower())
        if adapter is None:
            raise AdapterNotRegisteredError(chain, sorted(self._adapters))
        return adapter

    def normalize(self, chain: str, raw_event: Any) -> NormalizedEvent:
        """Convenience: look up the adapter for *chain* and normalize in one call."""
        return self.get(chain).normalize(raw_event)

    def chains(self) -> list[str]:
        return sorted(self._adapters)

    def __contains__(self, chain: str) -> bool:
        return chain.lower() in self._adapters


def _build_default_registry() -> AdapterRegistry:
    """Registers the adapters shipped with this repo.

    Import is done lazily inside the function (rather than at module top
    level) so importing ``ingestion.adapters.registry`` never triggers a
    circular import through the concrete adapter modules for callers who
    only need :class:`AdapterRegistry` itself (e.g. to build a test-only
    registry with a fake adapter).
    """
    from ingestion.adapters.evm_adapter import EvmAdapter
    from ingestion.adapters.stellar_adapter import StellarAdapter

    registry = AdapterRegistry()
    registry.register(StellarAdapter())
    registry.register(EvmAdapter())
    return registry


#: Process-wide default registry pre-populated with the adapters shipped in
#: this repo (Stellar, EVM). Ingestion entry points should use this unless
#: they need an isolated registry (e.g. for tests).
default_registry = _build_default_registry()
