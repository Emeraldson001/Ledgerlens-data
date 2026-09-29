# Integration Adapter Boundaries for External Data Providers

## Why

`ingestion/horizon_fetcher.py`, `ingestion/horizon_streamer.py`,
`ingestion/asset_metadata_fetcher.py`, and
`integrations/soroban_event_listener.py` each integrate a different
external provider with its own bespoke error handling, response shape, and
no shared health/failover story. Adding a new provider means re-deriving
retry/timeout/error conventions from scratch, and there's no standard way
to add a fallback provider for a capability (e.g. a backup asset-metadata
source) without hardcoding provider-specific branching in the caller.

## What this adds

`integrations/adapter_base.py` defines the boundary:

- **`DataProviderAdapter`** (ABC) — the contract: `fetch(params) ->
  AdapterResponse`, `health_check() -> bool`, and a `provider_name`
  identifier. `timed_fetch()`/`_do_fetch()` are provided so a subclass can
  get consistent latency timing for free instead of hand-rolling it.
- **`AdapterResponse`** — uniform envelope (`data`, `source`, `fetched_at`,
  `latency_ms`, `degraded`, `meta`) so a caller logging/monitoring a fetch
  doesn't need to know the provider-specific response shape.
- **Typed exception hierarchy** — `AdapterError` and subclasses
  (`AdapterTimeoutError`, `AdapterRateLimitError`, `AdapterAuthError`,
  `AdapterUnavailableError`). Callers can catch by category regardless of
  which concrete provider raised it, instead of catching
  provider-specific exceptions (e.g. `requests.Timeout` vs. a raw Soroban
  RPC error) at every call site.
- **`AdapterRegistry`** — registers one or more adapters per named
  capability (e.g. `"asset_metadata"`) and tries them in order, skipping
  adapters that fail `health_check()`, falling back on any
  `AdapterError`, and raising `AdapterUnavailableError` (with every
  attempted provider and its failure reason) only if all adapters fail.

`integrations/example_static_adapter.py`'s `StaticLookupAdapter` is a
dependency-free reference implementation (in-memory table instead of a
real HTTP call) demonstrating the contract end-to-end, used by the test
suite and as a template for wrapping a real provider — a real adapter
follows the same shape, translating its client library's own exceptions
(e.g. `requests.Timeout`, HTTP 401/429) into the `AdapterError` hierarchy.

```python
from integrations.adapter_base import AdapterRegistry
from integrations.example_static_adapter import StaticLookupAdapter

registry = AdapterRegistry()
registry.register("asset_metadata", primary_adapter)
registry.register("asset_metadata", backup_adapter)  # automatic failover

response = registry.fetch("asset_metadata", {"key": "XLM"})
```

## Developer commands

```
pytest tests/test_adapter_base.py -v
```

Covers: the uniform response envelope, typed-error propagation on
missing/invalid params, health-check toggling, registry fallback on an
unhealthy primary, registry fallback on a raised `AdapterError`, the
aggregate `AdapterUnavailableError` (with every attempted provider and
cause) when all adapters fail, and lookup of an unregistered capability.

## Design tradeoffs

- **ABC + typed exception hierarchy, not a duck-typed convention.** The
  existing fetchers rely on implicit conventions (method names, return
  shapes). An explicit `ABC` with `@abstractmethod` gives a hard contract
  new adapters can't silently skip, and the typed exception hierarchy
  means callers stop needing `except Exception` around every external
  call to be safe.
- **Registry lives outside any individual adapter.** Fallback ordering is
  a caller/deployment concern (which provider is primary vs. backup),
  not something an adapter should know about itself, so `AdapterRegistry`
  is separate from `DataProviderAdapter`.
- **No changes to existing fetchers in this PR.** This defines the
  boundary; migrating `ingestion/horizon_fetcher.py` and
  `ingestion/asset_metadata_fetcher.py` to implement
  `DataProviderAdapter` is follow-up work, scoped separately so this
  change doesn't turn into a broad, higher-risk refactor of live
  ingestion paths.
