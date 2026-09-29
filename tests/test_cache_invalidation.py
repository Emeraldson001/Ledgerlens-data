"""Tests for detection.cache_invalidation (cache invalidation primitives)."""

import pytest

from detection.cache_invalidation import (
    CacheInvalidationError,
    DependencyGraph,
    InvalidationRegistry,
    fingerprint,
)

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


def test_dependents_of_direct_edge():
    graph = DependencyGraph()
    graph.add_edge(WALLET_A, f"trade:{WALLET_A}")
    assert graph.dependents_of(f"trade:{WALLET_A}") == {WALLET_A}


def test_dependents_of_transitive_chain():
    graph = DependencyGraph()
    # wallet_graph_feature depends on per_pair_stat which depends on raw trade
    graph.add_edge("per_pair_stat:XLM/USDC", "trade:XLM/USDC")
    graph.add_edge("wallet_graph_feature:GA", "per_pair_stat:XLM/USDC")

    assert graph.dependents_of("trade:XLM/USDC") == {
        "per_pair_stat:XLM/USDC",
        "wallet_graph_feature:GA",
    }


def test_set_dependencies_replaces_old_edges():
    graph = DependencyGraph()
    graph.set_dependencies(WALLET_A, {"source1"})
    graph.set_dependencies(WALLET_A, {"source2"})

    assert graph.dependents_of("source1") == set()
    assert graph.dependents_of("source2") == {WALLET_A}


def test_remove_key_clears_all_edges():
    graph = DependencyGraph()
    graph.set_dependencies(WALLET_A, {"source1", "source2"})
    graph.remove_key(WALLET_A)

    assert graph.dependents_of("source1") == set()
    assert graph.dependents_of("source2") == set()
    assert len(graph) == 0


def test_no_dependents_for_unknown_source():
    graph = DependencyGraph()
    assert graph.dependents_of("nonexistent") == set()


# ---------------------------------------------------------------------------
# fingerprint()
# ---------------------------------------------------------------------------


def test_fingerprint_stable_for_same_inputs():
    assert fingerprint({"a": 1}, "v2", 3) == fingerprint({"a": 1}, "v2", 3)


def test_fingerprint_differs_for_different_inputs():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_fingerprint_handles_non_serializable_via_str():
    class Weird:
        def __str__(self):
            return "weird-value"

    # Should not raise despite Weird() not being JSON-serializable natively.
    assert fingerprint(Weird()) == fingerprint(Weird())


# ---------------------------------------------------------------------------
# InvalidationRegistry
# ---------------------------------------------------------------------------


class _FakeCache:
    def __init__(self):
        self.store: dict[str, object] = {}
        self.invalidated: list[str] = []

    def put(self, key, value):
        self.store[key] = value

    def invalidate(self, key):
        self.invalidated.append(key)
        self.store.pop(key, None)


def test_register_cache_and_invalidate_source():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("wallet_cache", evict=cache.invalidate)

    cache.put(WALLET_A, "matrix")
    registry.record_dependency(WALLET_A, sources={f"trade:{WALLET_A}"}, cache_name="wallet_cache")

    affected = registry.invalidate_source(f"trade:{WALLET_A}", reason="new_trade_ingested")

    assert affected == {WALLET_A}
    assert cache.invalidated == [WALLET_A]


def test_invalidate_source_cascades_transitively():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)

    registry.record_dependency("per_pair_stat", sources={"trade:pair"}, cache_name="c")
    registry.record_dependency("wallet_feature", sources={"per_pair_stat"}, cache_name="c")

    affected = registry.invalidate_source("trade:pair", reason="ingest")

    assert affected == {"per_pair_stat", "wallet_feature"}


def test_invalidate_source_with_unrelated_key_is_noop():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)
    registry.record_dependency(WALLET_A, sources={"trade:A"}, cache_name="c")

    affected = registry.invalidate_source("trade:B", reason="ingest")

    assert affected == set()
    assert cache.invalidated == []


def test_record_dependency_unregistered_cache_raises():
    registry = InvalidationRegistry()
    with pytest.raises(KeyError):
        registry.record_dependency(WALLET_A, sources={"s"}, cache_name="missing")


def test_invalidate_key_direct():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)
    registry.record_dependency(WALLET_A, sources={"s"}, cache_name="c")

    registry.invalidate_key(WALLET_A, reason="manual")

    assert cache.invalidated == [WALLET_A]


def test_evictor_failure_raises_cache_invalidation_error_but_continues_cascade():
    registry = InvalidationRegistry()
    good_cache = _FakeCache()

    def _broken_evict(key):
        raise RuntimeError("boom")

    registry.register_cache("broken", evict=_broken_evict)
    registry.register_cache("good", evict=good_cache.invalidate)

    registry.record_dependency("key1", sources={"s"}, cache_name="broken")
    registry.record_dependency("key2", sources={"s"}, cache_name="good")

    with pytest.raises(CacheInvalidationError):
        registry.invalidate_source("s")

    # The healthy cache's key was still invalidated despite the other failing.
    assert good_cache.invalidated == ["key2"]


def test_explain_returns_audit_trail():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)
    registry.record_dependency(WALLET_A, sources={"trade:A"}, cache_name="c")

    registry.invalidate_source("trade:A", reason="new_trade_ingested")

    events = registry.explain(WALLET_A)
    assert len(events) == 1
    assert events[0]["reason"] == "new_trade_ingested"
    assert events[0]["source"] == "trade:A"
    assert events[0]["cache_names"] == ["c"]


def test_explain_empty_for_unknown_key():
    registry = InvalidationRegistry()
    assert registry.explain("never-seen") == []


def test_is_stale_true_when_no_fingerprint_recorded():
    registry = InvalidationRegistry()
    assert registry.is_stale(WALLET_A, "v1") is True


def test_is_stale_false_when_fingerprint_matches():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)
    registry.record_dependency(
        WALLET_A, sources={"s"}, cache_name="c", version_inputs=["config-v1", 42]
    )

    assert registry.is_stale(WALLET_A, "config-v1", 42) is False
    assert registry.is_stale(WALLET_A, "config-v2", 42) is True


def test_re_registering_dependency_replaces_stale_edges():
    registry = InvalidationRegistry()
    cache = _FakeCache()
    registry.register_cache("c", evict=cache.invalidate)

    registry.record_dependency(WALLET_A, sources={"source1"}, cache_name="c")
    registry.record_dependency(WALLET_A, sources={"source2"}, cache_name="c")

    # source1 no longer affects WALLET_A after re-registration.
    affected = registry.invalidate_source("source1")
    assert affected == set()

    affected = registry.invalidate_source("source2")
    assert affected == {WALLET_A}
