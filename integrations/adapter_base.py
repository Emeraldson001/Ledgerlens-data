"""Integration adapter boundaries for external data providers.

``ingestion/horizon_fetcher.py``, ``ingestion/horizon_streamer.py``,
``ingestion/asset_metadata_fetcher.py``, and
``integrations/soroban_event_listener.py`` each talk to a different
external provider (Horizon REST, Horizon streaming, an asset metadata
service, Soroban RPC) with their own bespoke error handling and no shared
contract. That makes it hard to (a) add a new provider without
re-deriving retry/timeout/error-shape conventions, and (b) swap or add a
fallback provider, since callers are coupled to each fetcher's specific
method signatures and exception types.

This module defines a minimal, typed adapter boundary:

- ``DataProviderAdapter`` — the ABC every external-data adapter implements
  (``fetch``, ``health_check``, plus provider metadata).
- A small typed exception hierarchy (``AdapterError`` and subclasses) so
  callers can catch by category (timeout vs. rate-limit vs. auth vs.
  generic) regardless of which concrete provider raised it.
- ``AdapterResponse`` — a uniform envelope (data + source + latency +
  degraded flag) so callers don't need to know provider-specific response
  shapes to log/monitor a call.
- ``AdapterRegistry`` — name -> adapter lookup with an optional ordered
  fallback chain, so a caller can ask for "asset_metadata" and get
  automatic failover without hardcoding provider order itself.

This does not change any existing fetcher; it defines the boundary new
and existing adapters can be wrapped to satisfy incrementally.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class AdapterError(Exception):
    """Base class for all external-provider adapter errors."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AdapterTimeoutError(AdapterError):
    def __init__(self, provider: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(provider, f"request timed out after {timeout_seconds}s")


class AdapterRateLimitError(AdapterError):
    def __init__(self, provider: str, retry_after_seconds: float | None = None):
        self.retry_after_seconds = retry_after_seconds
        suffix = f", retry after {retry_after_seconds}s" if retry_after_seconds else ""
        super().__init__(provider, f"rate limited{suffix}")


class AdapterAuthError(AdapterError):
    def __init__(self, provider: str, detail: str = "authentication failed"):
        super().__init__(provider, detail)


class AdapterUnavailableError(AdapterError):
    """All configured adapters for a capability failed (used by the registry)."""

    def __init__(self, capability: str, attempted: list[str], causes: list[str]):
        self.capability = capability
        self.attempted = attempted
        self.causes = causes
        super().__init__(
            "registry",
            f"no adapter satisfied capability {capability!r}; tried {attempted} — {list(zip(attempted, causes, strict=False))}",
        )


@dataclass
class AdapterResponse:
    """Uniform envelope returned by every adapter's ``fetch``."""

    data: Any
    source: str
    fetched_at: float
    latency_ms: float
    degraded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class DataProviderAdapter(ABC):
    """Typed contract every external-data-provider integration implements.

    Concrete subclasses wrap a specific provider (Horizon, an asset
    metadata API, a Soroban RPC endpoint, ...) and are responsible for
    translating that provider's own errors into the ``AdapterError``
    hierarchy so callers have one set of exceptions to handle regardless
    of provider.
    """

    #: Stable identifier used for registry lookup and diagnostics.
    provider_name: str = "unset"

    @abstractmethod
    def fetch(self, params: dict[str, Any]) -> AdapterResponse:
        """Fetch data for the given request params. Must raise a subclass
        of AdapterError (not a provider-specific exception) on failure."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap liveness check used by the registry to skip known-bad
        adapters before attempting a full fetch."""
        raise NotImplementedError

    def timed_fetch(self, params: dict[str, Any]) -> AdapterResponse:
        """Helper for subclasses: wraps a raw fetch to fill in latency_ms
        and fetched_at consistently. Subclasses may call this from within
        their own fetch() implementation instead of computing timing by
        hand."""
        start = time.monotonic()
        response = self._do_fetch(params)
        response.latency_ms = (time.monotonic() - start) * 1000
        response.fetched_at = time.time()
        return response

    def _do_fetch(self, params: dict[str, Any]) -> AdapterResponse:
        raise NotImplementedError(
            "Override fetch() directly, or override _do_fetch() to use timed_fetch()."
        )


class AdapterRegistry:
    """Registers adapters by capability name with an ordered fallback chain.

    Example: register a primary and a backup adapter for
    "asset_metadata"; ``fetch("asset_metadata", params)`` tries them in
    registration order and returns the first success, raising
    ``AdapterUnavailableError`` (with every provider's failure reason)
    only if all of them fail.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, list[DataProviderAdapter]] = {}

    def register(self, capability: str, adapter: DataProviderAdapter) -> None:
        self._adapters.setdefault(capability, []).append(adapter)

    def get(self, capability: str) -> list[DataProviderAdapter]:
        return list(self._adapters.get(capability, []))

    def fetch(
        self, capability: str, params: dict[str, Any], skip_unhealthy: bool = True
    ) -> AdapterResponse:
        adapters = self._adapters.get(capability, [])
        if not adapters:
            raise AdapterUnavailableError(
                capability, attempted=[], causes=["no adapters registered"]
            )

        attempted: list[str] = []
        causes: list[str] = []

        for adapter in adapters:
            if skip_unhealthy:
                try:
                    if not adapter.health_check():
                        attempted.append(adapter.provider_name)
                        causes.append("failed health_check")
                        continue
                except Exception as exc:  # noqa: BLE001 - health check itself is best-effort
                    attempted.append(adapter.provider_name)
                    causes.append(f"health_check raised {type(exc).__name__}: {exc}")
                    continue

            attempted.append(adapter.provider_name)
            try:
                return adapter.fetch(params)
            except AdapterError as exc:
                causes.append(str(exc))
                continue

        raise AdapterUnavailableError(capability, attempted, causes)
