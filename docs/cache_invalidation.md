# Cache Invalidation Primitives for Derived Feature Data

## Overview

LedgerLens caches several layers of *derived* data on top of raw ingested
events: per-wallet feature matrices (`detection/feature_cache.py`), feature
store rows (`features/feature_store.py`), Benford window statistics, and
wallet-graph aggregates. Each of these derived values is computed from one
or more upstream sources -- raw trades, account activity, or the active
feature-engineering config version.

Previously, invalidating a derived value required each caller to remember
to call that specific cache's `invalidate()` method directly. There was no
shared way to express "this derived value depends on that upstream source"
or to cascade an upstream change through multiple dependent caches, and no
audit trail explaining why a cache entry went stale.

`detection/cache_invalidation.py` adds that as a reusable, dependency-free
primitive:

- **`DependencyGraph`** -- tracks `derived_key -> {source keys}` edges and
  computes the transitive closure of affected keys when a source changes
  (so a wallet-graph feature that depends on a per-pair Benford stat that
  depends on a raw trade is invalidated correctly through the whole chain).
- **`InvalidationRegistry`** -- pairs a `DependencyGraph` with pluggable
  evictor callbacks (one per backing cache), a fingerprint-based implicit
  staleness check, and a bounded per-key audit log.
- **`fingerprint()`** -- stable hash over arbitrary "version" inputs
  (config hash, row count, schema version) for caches where tracking every
  discrete dependency edge is impractical.
- **`CacheInvalidationError`** -- raised with the offending cache name and
  key when an evictor callback fails, instead of a bare exception.

## Design tradeoffs

- **Cascade continues past individual evictor failures.** If one backing
  cache's evictor raises, the registry still evicts the key from every
  other registered cache, logs and counts the failure
  (`cache_invalidation_errors_total`), and re-raises a single
  `CacheInvalidationError` after the full cascade completes. This favors
  eventual consistency across caches over fail-fast behavior, since a
  broken evictor in one cache should not leave stale entries in healthy
  caches.
- **Explicit dependency edges vs. fingerprints are both supported** rather
  than picking one. Discrete upstream events (a new trade for a specific
  wallet) are cheap to model as graph edges; config-wide changes are
  cheaper to model as a fingerprint comparison on read. Forcing everything
  through one mechanism would make one of the two cases awkward.
- **No new backing cache was introduced.** The registry is invalidation
  bookkeeping only -- it wraps existing caches' `invalidate`/`put` methods
  via callbacks rather than replacing `FeatureCache` or `FeatureStore`, to
  avoid an unrelated refactor of either.

## Usage

```python
from detection.cache_invalidation import InvalidationRegistry
from detection.feature_cache import FeatureCache

feature_cache = FeatureCache()
registry = InvalidationRegistry()
registry.register_cache("wallet_feature_cache", evict=feature_cache.invalidate)

# After computing and caching a wallet's feature matrix:
feature_cache.put(wallet, matrix)
registry.record_dependency(
    derived_key=wallet,
    sources={f"trade:{wallet}", "config:feature_engineering"},
    cache_name="wallet_feature_cache",
)

# When a new trade lands for that wallet:
registry.invalidate_source(f"trade:{wallet}", reason="new_trade_ingested")

# Diagnostics -- what happened to this wallet's cache entry recently?
registry.explain(wallet)
# -> [{"key": ..., "source": "trade:G...", "reason": "new_trade_ingested",
#      "cache_names": ["wallet_feature_cache"], "at": 1234.5}]
```

For config-version-driven staleness (no discrete source event to invalidate
on), use `is_stale()` on read instead:

```python
if registry.is_stale(wallet, config.feature_engineering_hash()):
    matrix = recompute(wallet)
    feature_cache.put(wallet, matrix)
    registry.record_dependency(
        wallet, sources=set(), cache_name="wallet_feature_cache",
        version_inputs=[config.feature_engineering_hash()],
    )
```

## Validation

```
pytest tests/test_cache_invalidation.py -v
```

Covers: direct and transitive dependency cascades, unrelated-source
no-ops, unregistered-cache errors, evictor-failure isolation (cascade
continues across caches when one evictor raises), the audit trail
returned by `explain()`, fingerprint-based staleness, and dependency
re-registration correctly dropping stale edges.

## Follow-up work

- Wire `InvalidationRegistry` into `detection/feature_cache.py` and
  `features/feature_store.py` call sites directly (currently opt-in via
  `register_cache`/`record_dependency`, not yet the default path for
  every cache write in the codebase).
- Expose `explain()` output through the existing forensic/incident
  reporting tooling (`detection/forensic_report.py`) for cache-staleness
  incidents.
