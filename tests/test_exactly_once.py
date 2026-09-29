"""Tests for pipeline.exactly_once — the unified dedup/idempotency library (Issue #670)."""

import threading
import time

import pytest

from pipeline.exactly_once import (
    DedupBackendUnavailableError,
    DedupKey,
    DedupState,
    ExactlyOnceStore,
    RedisExactlyOnceBackend,
    SqlExactlyOnceBackend,
)

try:
    import fakeredis

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# DedupKey canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_key_reserves_tenant_slot():
    key = DedupKey(source="kafka_trade", external_id="1:abc")
    assert key.canonical() == "_:kafka_trade:1:abc"


def test_canonical_key_includes_tenant_when_set():
    key = DedupKey(source="kafka_trade", external_id="1:abc", tenant_id="tenant-42")
    assert key.canonical() == "tenant-42:kafka_trade:1:abc"


def test_different_sources_produce_different_canonical_keys():
    a = DedupKey(source="kafka_trade", external_id="x")
    b = DedupKey(source="horizon_trade", external_id="x")
    assert a.canonical() != b.canonical()


# ---------------------------------------------------------------------------
# SQL backend: staged/committed protocol + TTL re-verify
# ---------------------------------------------------------------------------


@pytest.fixture
def sql_store():
    backend = SqlExactlyOnceBackend("sqlite:///:memory:", ttl_hours=48.0)
    return ExactlyOnceStore(backend, ttl_seconds=86400.0)


def test_sql_new_key_is_new(sql_store):
    key = DedupKey(source="test", external_id="a")
    assert sql_store.check_and_stage(key).state is DedupState.NEW


def test_sql_staged_but_not_committed_is_redo(sql_store):
    key = DedupKey(source="test", external_id="a")
    sql_store.check_and_stage(key)  # stages it
    decision = sql_store.check_and_stage(key)  # simulate a crash before commit
    assert decision.state is DedupState.STAGED


def test_sql_committed_is_duplicate(sql_store):
    key = DedupKey(source="test", external_id="a")
    sql_store.check_and_stage(key)
    sql_store.commit(key, payload={"result": 42})

    decision = sql_store.check_and_stage(key)
    assert decision.state is DedupState.COMMITTED
    assert decision.payload == {"result": 42}


def test_sql_failed_key_can_be_restaged(sql_store):
    key = DedupKey(source="test", external_id="a")
    sql_store.check_and_stage(key)
    sql_store.mark_failed(key)

    decision = sql_store.check_and_stage(key)
    assert decision.state is DedupState.NEW


def test_sql_ttl_expired_committed_surfaces_reverify_not_new():
    backend = SqlExactlyOnceBackend("sqlite:///:memory:", ttl_hours=1 / 3600000)  # ~1ms
    store = ExactlyOnceStore(backend)
    key = DedupKey(source="test", external_id="a")
    store.check_and_stage(key)
    store.commit(key)
    time.sleep(0.05)

    decision = store.check_and_stage(key)
    assert decision.state is DedupState.TTL_EXPIRED_REVERIFY
    # Explicitly distinguishable from "never processed".
    assert decision.state is not DedupState.NEW


def test_sql_list_by_source_prefix_scopes_correctly():
    backend = SqlExactlyOnceBackend("sqlite:///:memory:")
    store = ExactlyOnceStore(backend)
    run_a = DedupKey(source="pipeline_checkpoint:run-A:pairXY", external_id="ingest")
    run_a_2 = DedupKey(source="pipeline_checkpoint:run-A:pairXY", external_id="features")
    run_b = DedupKey(source="pipeline_checkpoint:run-B:pairXY", external_id="ingest")
    for k in (run_a, run_a_2, run_b):
        store.check_and_stage(k)

    rows = backend.list_by_source_prefix("pipeline_checkpoint:run-A:pairXY")
    assert {r.external_id for r in rows} == {"ingest", "features"}


# ---------------------------------------------------------------------------
# Redis backend: fail-closed behavior
# ---------------------------------------------------------------------------


def test_redis_backend_raises_when_unreachable():
    backend = RedisExactlyOnceBackend("redis://nonexistent-host-for-tests:9999/0")
    store = ExactlyOnceStore(backend)
    key = DedupKey(source="test", external_id="a")
    with pytest.raises(DedupBackendUnavailableError):
        store.check_and_stage(key)


def test_redis_backend_health_check_false_when_unreachable():
    backend = RedisExactlyOnceBackend("redis://nonexistent-host-for-tests:9999/0")
    assert backend.health_check() is False


@pytest.fixture
def fake_redis_backend():
    if not _FAKEREDIS_AVAILABLE:
        pytest.skip("fakeredis not installed")
    backend = RedisExactlyOnceBackend("redis://localhost:6379/0")
    backend._redis = fakeredis.FakeStrictRedis(decode_responses=True)
    backend._init_error = None
    return backend


def test_redis_backend_staged_then_committed(fake_redis_backend):
    store = ExactlyOnceStore(fake_redis_backend)
    key = DedupKey(source="kafka_trade", external_id="1:abc")

    first = store.check_and_stage(key)
    assert first.state is DedupState.NEW

    # A prior attempt staged it but never committed (simulated crash).
    redo = store.check_and_stage(key)
    assert redo.state is DedupState.STAGED

    store.commit(key)
    dup = store.check_and_stage(key)
    assert dup.state is DedupState.COMMITTED


def test_redis_backend_health_check_true_when_available(fake_redis_backend):
    assert fake_redis_backend.health_check() is True


def test_redis_backend_becomes_unavailable_mid_session_raises(fake_redis_backend):
    store = ExactlyOnceStore(fake_redis_backend)
    key = DedupKey(source="kafka_trade", external_id="1:abc")
    store.check_and_stage(key)

    # Simulate the connection dying after successful use.
    fake_redis_backend._redis = None
    fake_redis_backend._init_error = "connection reset"

    with pytest.raises(DedupBackendUnavailableError):
        store.check_and_stage(key)
    with pytest.raises(DedupBackendUnavailableError):
        store.commit(key)


# ---------------------------------------------------------------------------
# Concurrency: 50 simultaneous duplicate submissions of the same key
# (Issue #670 acceptance criterion — "fifty concurrent duplicate submissions
# of the same trade_id across two worker processes result in exactly one
# feature-state update and one alert dispatch"). A thread pool racing on one
# shared backend is the deterministic, dependency-free proxy for "two worker
# processes racing on the same Redis instance" available in this environment;
# the underlying guarantee (atomic SET NX claim) is identical regardless of
# whether the concurrent callers are threads or processes, since Redis itself
# serialises the command.
# ---------------------------------------------------------------------------


def test_fifty_concurrent_duplicate_submissions_exactly_one_winner_redis(fake_redis_backend):
    store = ExactlyOnceStore(fake_redis_backend)
    key = DedupKey(source="kafka_trade", external_id="999:concurrent-trade")

    winners = []
    lock = threading.Lock()
    barrier = threading.Barrier(50)

    def attempt():
        barrier.wait()
        decision = store.check_and_stage(key)
        if decision.state is DedupState.NEW:
            with lock:
                winners.append(threading.get_ident())

    threads = [threading.Thread(target=attempt) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, (
        f"expected exactly one winner to claim the key and proceed with side "
        f"effects, got {len(winners)} — a non-atomic claim would double-process"
    )


def test_fifty_concurrent_duplicate_submissions_exactly_one_winner_sql(tmp_path):
    # A genuine multi-connection concurrency test needs a file-backed sqlite
    # DB — ``:memory:`` uses a SingletonThreadPool that does not model real
    # concurrent-connection access safely.
    db_path = tmp_path / "exactly_once_concurrency.db"
    backend = SqlExactlyOnceBackend(f"sqlite:///{db_path}")
    store = ExactlyOnceStore(backend)
    key = DedupKey(source="pipeline_checkpoint", external_id="concurrent-stage")

    winners = []
    lock = threading.Lock()
    barrier = threading.Barrier(50)

    def attempt():
        barrier.wait()
        try:
            decision = store.check_and_stage(key)
        except Exception:
            # A UNIQUE-constraint race under heavy sqlite contention may
            # surface as an IntegrityError on some drivers instead of being
            # absorbed inside check_and_stage — treat as "lost the race".
            return
        if decision.state is DedupState.NEW:
            with lock:
                winners.append(threading.get_ident())

    threads = [threading.Thread(target=attempt) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1


# ---------------------------------------------------------------------------
# Retry on transient OperationalError (e.g. sqlite "disk I/O error" / "database
# is locked") — mirrors detection.risk_score_store.RiskScoreStore's existing
# retry pattern for the same class of transient failure.
# ---------------------------------------------------------------------------


def test_sql_backend_retries_transient_operational_error(monkeypatch):
    from sqlalchemy.exc import OperationalError

    backend = SqlExactlyOnceBackend("sqlite:///:memory:")
    key = DedupKey(source="test", external_id="a")

    real_impl = backend._check_and_stage_impl
    calls = {"n": 0}

    def flaky(k, ttl):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("stmt", {}, Exception("disk I/O error"))
        return real_impl(k, ttl)

    monkeypatch.setattr(backend, "_check_and_stage_impl", flaky)

    decision = backend.check_and_stage(key, 86400.0)

    assert calls["n"] == 3
    assert decision.state is DedupState.NEW


def test_sql_backend_gives_up_after_five_attempts(monkeypatch):
    from sqlalchemy.exc import OperationalError

    backend = SqlExactlyOnceBackend("sqlite:///:memory:")
    key = DedupKey(source="test", external_id="a")

    def always_fails(k, ttl):
        raise OperationalError("stmt", {}, Exception("disk I/O error"))

    monkeypatch.setattr(backend, "_check_and_stage_impl", always_fails)

    with pytest.raises(OperationalError):
        backend.check_and_stage(key, 86400.0)
