"""Cache invalidation primitives for derived feature data.

``FeatureCache`` (see ``detection/feature_cache.py``) and ``FeatureStore``
(see ``features/feature_store.py``) cache *derived* values -- feature
matrices, wallet-graph aggregates, Benford window statistics -- that are
computed from one or more *upstream sources*: raw trade/order-book events,
a wallet's account state, a feature-engineering config version, or an
upstream model artifact version.

Today, invalidation of those derived values is handled ad hoc: each cache
implements its own TTL and callers must remember to call ``invalidate()``
directly whenever an upstream source changes. That works for a single flat
cache, but breaks down as soon as one derived value depends on another
(e.g. a wallet-graph feature that is itself derived from per-pair Benford
statistics) -- there is no shared mechanism for propagating "this upstream
source changed" into "these N derived cache entries are now stale", and no
audit trail explaining *why* a given entry was evicted.

This module provides that shared mechanism as a small, dependency-free
primitive that any cache in the codebase can opt into:

* :class:`DependencyGraph` -- tracks which derived keys were computed from
  which source keys, and computes the transitive closure of keys to evict
  when a source changes.
* :class:`InvalidationRegistry` -- a thread-safe registry that pairs a
  ``DependencyGraph`` with a pluggable list of *evictor* callbacks (one per
  backing cache), a versioned-fingerprint staleness check, and a bounded
  audit log of invalidation events for diagnostics.
* :class:`CacheInvalidationError` -- raised when an invalidation callback
  fails, wrapping the offending key and cache name so failures are
  attributable instead of silently swallowed.
* :func:`fingerprint` -- stable hash of an arbitrary set of "version"
  inputs (config hash, upstream row count, schema version, ...), used to
  detect staleness without an explicit invalidation call.

Usage::

    from detection.cache_invalidation import InvalidationRegistry

    registry = InvalidationRegistry()
    registry.register_cache("wallet_feature_cache", evict=feature_cache.invalidate)

    # Record that the wallet-graph feature for `wallet` was derived from
    # the raw trade source `trade:wallet` and the current config version.
    registry.record_dependency(
        derived_key=wallet,
        sources={f"trade:{wallet}", "config:feature_engineering"},
        cache_name="wallet_feature_cache",
    )

    # Later, when new trades land for `wallet`:
    registry.invalidate_source(f"trade:{wallet}", reason="new_trade_ingested")

    # Diagnostics: why was a key evicted?
    registry.explain(wallet)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter

    cache_invalidation_events_total = Counter(
        "cache_invalidation_events_total",
        "Total number of derived-cache entries invalidated, by reason",
        ["reason"],
    )
    cache_invalidation_errors_total = Counter(
        "cache_invalidation_errors_total",
        "Total number of evictor callback failures during invalidation",
        ["cache_name"],
    )
except Exception:  # pragma: no cover - prometheus optional in tests
    cache_invalidation_events_total = None  # type: ignore[assignment]
    cache_invalidation_errors_total = None  # type: ignore[assignment]

#: Maximum number of invalidation events retained per derived key for
#: `explain()`. Bounded so long-running processes cannot leak memory.
MAX_AUDIT_EVENTS_PER_KEY = 20


def fingerprint(*parts: Any) -> str:
    """Compute a stable, order-sensitive hash of *parts*.

    Each part is JSON-serialized (with ``sort_keys=True`` and
    ``default=str`` so non-serializable objects such as numpy scalars or
    datetimes degrade to their ``str()`` form instead of raising) and
    concatenated before hashing, so callers can build a "version
    fingerprint" for a derived value out of arbitrary inputs -- a config
    dict, a row count, a schema version string, an upstream timestamp --
    without worrying about hashability.

    Used for *implicit* staleness detection: a caller can store the
    fingerprint alongside a cached value and compare it against a freshly
    computed fingerprint on read, invalidating without ever having called
    :meth:`InvalidationRegistry.invalidate_source` explicitly (useful when
    the set of upstream changes is too fine-grained to track as discrete
    dependency edges, e.g. "the feature engineering config changed").
    """
    blob = "\x1f".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class CacheInvalidationError(RuntimeError):
    """Raised when an evictor callback fails during cascade invalidation.

    Carries the cache name and key that failed so the caller (or the
    surrounding pipeline's error handler) can log or alert with enough
    context to find the offending cache without re-deriving it from a bare
    traceback.
    """

    def __init__(self, cache_name: str, key: str, original: Exception):
        self.cache_name = cache_name
        self.key = key
        self.original = original
        super().__init__(f"evictor for cache={cache_name!r} failed on key={key!r}: {original!r}")


@dataclass
class InvalidationEvent:
    """A single audit-log entry describing one invalidation."""

    key: str
    source: str | None
    reason: str
    cache_names: list[str] = field(default_factory=list)
    at: float = field(default_factory=time.monotonic)


class DependencyGraph:
    """Tracks derived-key -> source-key edges and computes transitive closure.

    Not thread-safe on its own -- :class:`InvalidationRegistry` wraps every
    call in a lock. Kept as a standalone class so it can be unit-tested
    (and reused) independently of the registry's cache-eviction plumbing.
    """

    def __init__(self) -> None:
        # source -> set of derived keys that depend on it
        self._forward: dict[str, set[str]] = defaultdict(set)
        # derived key -> set of sources it depends on (for cleanup on re-registration)
        self._reverse: dict[str, set[str]] = defaultdict(set)

    def add_edge(self, derived_key: str, source: str) -> None:
        self._forward[source].add(derived_key)
        self._reverse[derived_key].add(source)

    def set_dependencies(self, derived_key: str, sources: Iterable[str]) -> None:
        """Replace all dependency edges for *derived_key* with *sources*.

        Recomputing a derived value typically changes which sources it was
        built from (e.g. a wallet-graph feature now spans one more trade
        counterparty) -- calling this instead of repeated ``add_edge``
        prevents stale edges from a previous computation from keeping the
        key alive against sources it no longer actually depends on.
        """
        self.remove_key(derived_key)
        for source in sources:
            self.add_edge(derived_key, source)

    def remove_key(self, derived_key: str) -> None:
        """Drop all edges for *derived_key* (its own eviction or re-registration)."""
        old_sources = self._reverse.pop(derived_key, set())
        for source in old_sources:
            self._forward[source].discard(derived_key)
            if not self._forward[source]:
                del self._forward[source]

    def dependents_of(self, source: str) -> set[str]:
        """Return every derived key (transitively) affected by *source* changing.

        Traverses the graph breadth-first, treating each discovered derived
        key as itself a potential source (derived values may depend on
        other derived values), so a single upstream change correctly
        cascades through multi-level dependency chains instead of only
        invalidating the immediate consumers.
        """
        seen: set[str] = set()
        queue: deque[str] = deque(self._forward.get(source, set()))
        while queue:
            key = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            # A derived key can itself be a source for further-derived keys.
            queue.extend(self._forward.get(key, set()) - seen)
        return seen

    def sources_of(self, derived_key: str) -> set[str]:
        return set(self._reverse.get(derived_key, set()))

    def __len__(self) -> int:
        return len(self._reverse)


class InvalidationRegistry:
    """Coordinates dependency tracking, cache eviction, and diagnostics.

    Multiple backing caches (e.g. ``FeatureCache`` for wallet feature
    matrices, a separate cache for Benford window stats) register an
    evictor callback under a name. Derived keys record which upstream
    sources they were computed from and which cache(s) hold them.
    Invalidating a source cascades through the dependency graph and calls
    every relevant cache's evictor, recording the outcome for
    :meth:`explain`.
    """

    def __init__(self, max_audit_events_per_key: int = MAX_AUDIT_EVENTS_PER_KEY) -> None:
        self._lock = threading.RLock()
        self._graph = DependencyGraph()
        self._evictors: dict[str, Callable[[str], None]] = {}
        # derived key -> set of cache names holding an entry for it
        self._key_caches: dict[str, set[str]] = defaultdict(set)
        # derived key -> deque of recent invalidation events (bounded)
        self._audit: dict[str, deque[InvalidationEvent]] = defaultdict(
            lambda: deque(maxlen=max_audit_events_per_key)
        )
        # derived key -> last-known fingerprint, for implicit staleness checks
        self._fingerprints: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_cache(self, cache_name: str, evict: Callable[[str], None]) -> None:
        """Register an evictor callback for a backing cache.

        *evict* is called with a derived key whenever that key must be
        removed from *cache_name*. Typically this is a bound method such as
        ``feature_cache.invalidate``.
        """
        with self._lock:
            self._evictors[cache_name] = evict

    def record_dependency(
        self,
        derived_key: str,
        sources: Iterable[str],
        cache_name: str,
        version_inputs: Iterable[Any] | None = None,
    ) -> str | None:
        """Record that *derived_key* (in *cache_name*) was computed from *sources*.

        Args:
            derived_key: the cache key of the derived value (e.g. a wallet
                address).
            sources: upstream keys the value was computed from. May include
                other derived keys, forming multi-level dependency chains.
            cache_name: which registered cache holds *derived_key*. Must
                have been registered via :meth:`register_cache`.
            version_inputs: optional extra values to fold into a stored
                fingerprint (see :func:`fingerprint`) for implicit
                staleness checks via :meth:`is_stale`.

        Returns:
            The computed fingerprint if *version_inputs* was given,
            otherwise ``None``.

        Raises:
            KeyError: if *cache_name* was never registered.
        """
        with self._lock:
            if cache_name not in self._evictors:
                raise KeyError(
                    f"cache_name={cache_name!r} is not registered; call "
                    "register_cache() before record_dependency()"
                )
            self._graph.set_dependencies(derived_key, sources)
            self._key_caches[derived_key].add(cache_name)

            fp = None
            if version_inputs is not None:
                fp = fingerprint(*version_inputs)
                self._fingerprints[derived_key] = fp
            return fp

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_source(self, source: str, reason: str = "source_changed") -> set[str]:
        """Invalidate every derived key (transitively) depending on *source*.

        Returns the set of derived keys that were evicted. Does not raise
        on individual evictor failures -- each failure is logged, counted,
        and recorded in that key's audit trail, and the cascade continues
        for the remaining keys, so a broken evictor for one cache cannot
        prevent unrelated caches from being invalidated correctly. All
        failures are collected and re-raised together as a single
        :class:`CacheInvalidationError` after the cascade completes, so
        callers still learn about the failure without the cascade
        aborting halfway through.
        """
        with self._lock:
            affected = self._graph.dependents_of(source)
            errors: list[CacheInvalidationError] = []
            for key in affected:
                errors.extend(self._evict_key_locked(key, source=source, reason=reason))
            self._graph_forget_source(source)

        if errors:
            # Surface the first failure; all are already logged/audited above.
            raise errors[0]
        return affected

    def invalidate_key(self, derived_key: str, reason: str = "manual") -> None:
        """Directly invalidate a single derived key (no cascade)."""
        with self._lock:
            errors = self._evict_key_locked(derived_key, source=None, reason=reason)
        if errors:
            raise errors[0]

    def is_stale(self, derived_key: str, *version_inputs: Any) -> bool:
        """Return ``True`` if the current fingerprint of *version_inputs*
        differs from the fingerprint recorded when *derived_key* was last
        written (or if no fingerprint was ever recorded for it).

        This is the *implicit* staleness path -- unlike
        :meth:`invalidate_source`, it does not require every upstream
        change to be routed through the registry explicitly.
        """
        with self._lock:
            recorded = self._fingerprints.get(derived_key)
        if recorded is None:
            return True
        return recorded != fingerprint(*version_inputs)

    def explain(self, derived_key: str) -> list[dict[str, Any]]:
        """Return the recent invalidation audit trail for *derived_key*.

        Each entry describes what triggered the eviction (upstream source,
        human-readable reason, which caches were affected, and a
        monotonic timestamp) -- intended for debugging "why is this value
        stale / why did it just get recomputed" during an incident.
        """
        with self._lock:
            events = list(self._audit.get(derived_key, ()))
        return [
            {
                "key": e.key,
                "source": e.source,
                "reason": e.reason,
                "cache_names": list(e.cache_names),
                "at": e.at,
            }
            for e in events
        ]

    def dependents_of(self, source: str) -> set[str]:
        """Public read-only view of the dependency graph for diagnostics."""
        with self._lock:
            return self._graph.dependents_of(source)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_key_locked(
        self, derived_key: str, source: str | None, reason: str
    ) -> list[CacheInvalidationError]:
        """Evict *derived_key* from every cache it is registered in.

        Must be called with ``self._lock`` held. Returns the list of
        errors encountered (empty on full success) instead of raising, so
        :meth:`invalidate_source` can continue the cascade for other keys.
        """
        cache_names = sorted(self._key_caches.get(derived_key, ()))
        errors: list[CacheInvalidationError] = []
        succeeded: list[str] = []

        for cache_name in cache_names:
            evict = self._evictors.get(cache_name)
            if evict is None:
                continue
            try:
                evict(derived_key)
                succeeded.append(cache_name)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
                logger.warning(
                    "cache invalidation evictor failed",
                    extra={"cache_name": cache_name, "key": derived_key, "error": str(exc)},
                )
                if cache_invalidation_errors_total is not None:
                    cache_invalidation_errors_total.labels(cache_name=cache_name).inc()
                errors.append(CacheInvalidationError(cache_name, derived_key, exc))

        self._audit[derived_key].append(
            InvalidationEvent(key=derived_key, source=source, reason=reason, cache_names=succeeded)
        )
        self._fingerprints.pop(derived_key, None)
        self._key_caches.pop(derived_key, None)
        self._graph.remove_key(derived_key)

        if cache_invalidation_events_total is not None:
            cache_invalidation_events_total.labels(reason=reason).inc()

        return errors

    def _graph_forget_source(self, source: str) -> None:
        """Drop the now-empty forward edge set for *source*, if any remains."""
        # dependents_of() does not itself mutate _forward; invalidated
        # derived keys already removed their own edges via remove_key(),
        # so this is a no-op safety net for sources with zero remaining
        # dependents left dangling as empty sets.
        self._graph._forward.pop(source, None)  # noqa: SLF001 - internal cleanup


#: Process-wide default registry. Most callers can share this instance;
#: tests and multi-tenant callers that need isolation should construct
#: their own ``InvalidationRegistry()``.
default_registry = InvalidationRegistry()
