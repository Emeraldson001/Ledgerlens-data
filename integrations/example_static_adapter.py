"""A minimal, dependency-free example adapter satisfying DataProviderAdapter.

This is a reference implementation used by tests and as a template for
wrapping a real provider (e.g. Horizon, an asset metadata service). It
intentionally has no network dependency so it can serve as an executable
example of the adapter contract in ``integrations/adapter_base.py``
without requiring credentials or connectivity.

A real HTTP-backed adapter would follow the same shape: implement
``fetch``/``health_check``, and translate the underlying client's
exceptions (e.g. ``requests.Timeout``, ``requests.HTTPError`` for 401/429)
into ``AdapterTimeoutError`` / ``AdapterAuthError`` / ``AdapterRateLimitError``.
"""

from __future__ import annotations

import time
from typing import Any

from integrations.adapter_base import (
    AdapterAuthError,
    AdapterResponse,
    AdapterTimeoutError,
    DataProviderAdapter,
)


class StaticLookupAdapter(DataProviderAdapter):
    """Serves responses from an in-memory dict, simulating an external
    lookup provider (e.g. asset metadata by code)."""

    def __init__(
        self,
        provider_name: str,
        table: dict[str, Any],
        api_key: str | None = None,
        simulate_latency_ms: float = 0.0,
    ):
        self.provider_name = provider_name
        self._table = table
        self._api_key = api_key
        self._simulate_latency_ms = simulate_latency_ms
        self._healthy = True

    def health_check(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Test/ops hook to simulate the provider going down."""
        self._healthy = healthy

    def fetch(self, params: dict[str, Any]) -> AdapterResponse:
        if self._api_key is not None and params.get("api_key") != self._api_key:
            raise AdapterAuthError(self.provider_name, "invalid or missing api_key")

        key = params.get("key")
        if key is None:
            raise AdapterAuthError(self.provider_name, "missing required param 'key'")

        start = time.monotonic()
        if self._simulate_latency_ms:
            time.sleep(self._simulate_latency_ms / 1000)

        if key not in self._table:
            timeout = params.get("timeout_seconds", 5.0)
            raise AdapterTimeoutError(self.provider_name, timeout)

        latency_ms = (time.monotonic() - start) * 1000
        return AdapterResponse(
            data=self._table[key],
            source=self.provider_name,
            fetched_at=time.time(),
            latency_ms=latency_ms,
            degraded=False,
            meta={"lookup_key": key},
        )
