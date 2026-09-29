"""Tests for idempotent trade ingestion via the unified exactly-once dedup store.

Issue #670 replaced the previous ZSET-based, fail-open ``SeenEventCache`` with
a fail-closed implementation built on ``pipeline.exactly_once.ExactlyOnceStore``.
The graceful-degradation tests below intentionally assert the new fail-closed
contract (raise ``DedupBackendUnavailableError``) — the old fail-open behavior
("allow trade through when Redis is unavailable") is the invariant-8 bug this
issue fixes, not a regression.
"""

import pytest

from ingestion.trade_deduplicator import (
    DedupBackendUnavailableError,
    SeenEventCache,
    get_trade_dedup_cache,
)

try:
    import fakeredis

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False
    fakeredis = None


@pytest.fixture
def fake_redis_cache():
    """Create a SeenEventCache backed by a fakeredis client."""
    if not _FAKEREDIS_AVAILABLE:
        pytest.skip("fakeredis not installed")

    fake_redis_instance = fakeredis.FakeStrictRedis(decode_responses=True)

    cache = SeenEventCache(
        redis_url="redis://localhost:6379/0",
        ttl_seconds=86400,
        key_prefix="ledgerlens:trades:",
    )
    cache._backend._redis = fake_redis_instance
    cache._backend._init_error = None
    return cache


class TestSeenEventCacheDeduplication:
    def test_first_trade_not_duplicate(self, fake_redis_cache):
        assert fake_redis_cache.is_duplicate("trade-123", "paging-456", "USDC/XLM") is False

    def test_second_identical_trade_is_duplicate(self, fake_redis_cache):
        trade_id, token, pair = "trade-123", "paging-456", "USDC/XLM"
        assert fake_redis_cache.is_duplicate(trade_id, token, pair) is False
        assert fake_redis_cache.is_duplicate(trade_id, token, pair) is True

    def test_different_trades_not_duplicates(self, fake_redis_cache):
        pair = "USDC/XLM"
        fake_redis_cache.is_duplicate("trade-1", "paging-1", pair)
        assert fake_redis_cache.is_duplicate("trade-2", "paging-2", pair) is False

    def test_trade_hash_prevents_id_collision(self, fake_redis_cache):
        pair = "USDC/XLM"
        assert fake_redis_cache.is_duplicate("trade-1", "token-A", pair) is False
        assert fake_redis_cache.is_duplicate("trade-1", "token-B", pair) is False


class TestCacheOperations:
    def test_cache_trade_explicitly(self, fake_redis_cache):
        pair = "USDC/XLM"
        fake_redis_cache.cache_trade("trade-1", "token-1", pair)
        assert fake_redis_cache.get_cache_size(pair) == 1
        # Subsequent check now sees it as a duplicate.
        assert fake_redis_cache.is_duplicate("trade-1", "token-1", pair) is True

    def test_cache_size_query(self, fake_redis_cache):
        pair = "USDC/XLM"
        for i in range(5):
            fake_redis_cache.is_duplicate(f"trade-{i}", f"token-{i}", pair)
        assert fake_redis_cache.get_cache_size(pair) == 5

    def test_cache_does_not_grow_unbounded_across_pairs(self, fake_redis_cache):
        pair = "USDC/XLM"
        for i in range(100):
            fake_redis_cache.is_duplicate(f"trade-{i}", f"token-{i}", pair)
        assert fake_redis_cache.get_cache_size(pair) == 100

    def test_clear_cache_specific_pair(self, fake_redis_cache):
        pair1, pair2 = "USDC/XLM", "EUR/XLM"
        fake_redis_cache.is_duplicate("trade-1", "token-1", pair1)
        fake_redis_cache.is_duplicate("trade-2", "token-2", pair2)

        fake_redis_cache.clear_cache(pair1)

        assert fake_redis_cache.get_cache_size(pair1) == 0
        assert fake_redis_cache.get_cache_size(pair2) == 1

    def test_health_check_available(self, fake_redis_cache):
        assert fake_redis_cache.health_check() is True

    def test_health_check_unavailable(self):
        cache = SeenEventCache(redis_url="redis://nonexistent-host-for-tests:9999/0")
        assert cache.health_check() is False


class TestFailClosedDegradation:
    """Redis outage must raise, never silently allow trades through (invariant 8)."""

    def test_is_duplicate_raises_when_redis_unreachable(self):
        cache = SeenEventCache(redis_url="redis://nonexistent-host-for-tests:9999/0")
        with pytest.raises(DedupBackendUnavailableError):
            cache.is_duplicate("trade-1", "token-1", "USDC/XLM")

    def test_is_duplicate_raises_when_client_becomes_none(self, fake_redis_cache):
        fake_redis_cache._backend._redis = None
        fake_redis_cache._backend._init_error = "connection dropped mid-session"
        with pytest.raises(DedupBackendUnavailableError):
            fake_redis_cache.is_duplicate("trade-1", "token-1", "USDC/XLM")

    def test_get_cache_size_degrades_to_sentinel_not_raise(self):
        """Introspection helpers stay lenient — they are not the correctness path."""
        cache = SeenEventCache(redis_url="redis://nonexistent-host-for-tests:9999/0")
        assert cache.get_cache_size("USDC/XLM") == -1
        assert cache.clear_cache("USDC/XLM") is False


class TestAssetPairSeparation:
    def test_same_trade_different_pairs_separate(self, fake_redis_cache):
        trade_id, token = "trade-123", "paging-456"
        assert fake_redis_cache.is_duplicate(trade_id, token, "USDC/XLM") is False
        assert fake_redis_cache.is_duplicate(trade_id, token, "EUR/XLM") is False

    def test_cache_isolation_per_pair(self, fake_redis_cache):
        fake_redis_cache.is_duplicate("trade-1", "token-1", "USDC/XLM")
        assert fake_redis_cache.get_cache_size("USDC/XLM") == 1
        assert fake_redis_cache.get_cache_size("EUR/XLM") == 0


class TestConvenienceFunctions:
    def test_is_duplicate_trade_convenience(self):
        cache = get_trade_dedup_cache()
        assert cache is not None

    def test_get_trade_dedup_cache_singleton(self):
        assert get_trade_dedup_cache() is get_trade_dedup_cache()


class TestEdgeCases:
    def test_empty_paging_token_uses_trade_id(self, fake_redis_cache):
        trade_id, pair = "trade-123", "USDC/XLM"
        assert fake_redis_cache.is_duplicate(trade_id, None, pair) is False
        assert fake_redis_cache.is_duplicate(trade_id, None, pair) is True

    def test_unicode_trade_ids(self, fake_redis_cache):
        trade_id, token, pair = "trade-αβγ", "token-日本語", "USDC/XLM"
        assert fake_redis_cache.is_duplicate(trade_id, token, pair) is False
        assert fake_redis_cache.is_duplicate(trade_id, token, pair) is True

    def test_long_asset_pair_name(self, fake_redis_cache):
        pair = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
        assert fake_redis_cache.is_duplicate("trade-1", "token-1", pair) is False
        assert fake_redis_cache.is_duplicate("trade-1", "token-1", pair) is True
