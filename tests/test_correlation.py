"""Tests for utils/correlation.py — pipeline log correlation."""

from __future__ import annotations

import logging
import threading

import pytest

from utils.correlation import (
    CorrelationContext,
    CorrelationFilter,
    PipelineStage,
    correlation_context,
    current_correlation_id,
    current_extra,
    current_pair_id,
    current_stage,
    current_wallet,
    generate_correlation_id,
    get_correlation_fields,
    propagate_to_thread,
    snapshot,
)

# ── Correlation ID generation ────────────────────────────────────────────────


class TestCorrelationIdGeneration:
    def test_generate_returns_hex_string(self) -> None:
        cid = generate_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)

    def test_generate_is_unique(self) -> None:
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100


# ── Context variables ────────────────────────────────────────────────────────


class TestContextVars:
    def test_default_values_are_none(self) -> None:
        assert current_correlation_id() is None
        assert current_stage() is None
        assert current_pair_id() is None
        assert current_wallet() is None
        assert current_extra() is None

    def test_get_correlation_fields_empty(self) -> None:
        fields = get_correlation_fields()
        assert fields == {}


# ── Correlation context manager ─────────────────────────────────────────────


class TestCorrelationContext:
    def test_sets_correlation_id(self) -> None:
        with correlation_context("test-123"):
            assert current_correlation_id() == "test-123"
        assert current_correlation_id() is None

    def test_auto_generates_id_when_none(self) -> None:
        with correlation_context(auto_generate=True):
            cid = current_correlation_id()
            assert cid is not None
            assert len(cid) == 32

    def test_sets_all_fields(self) -> None:
        with correlation_context(
            "abc",
            stage="ingestion",
            pair_id="USDC:.../XLM:native",
            wallet="GABC...",
            extra={"key": "value"},
        ):
            assert current_correlation_id() == "abc"
            assert current_stage() == "ingestion"
            assert current_pair_id() == "USDC:.../XLM:native"
            assert current_wallet() == "GABC..."
            assert current_extra() == {"key": "value"}

    def test_cleans_up_on_exit(self) -> None:
        with correlation_context("abc", stage="detection"):
            pass
        assert current_correlation_id() is None
        assert current_stage() is None

    def test_nested_contexts(self) -> None:
        with correlation_context("outer", stage="ingestion"):
            assert current_correlation_id() == "outer"
            assert current_stage() == "ingestion"

            with correlation_context("inner", stage="detection"):
                assert current_correlation_id() == "inner"
                assert current_stage() == "detection"

            assert current_correlation_id() == "outer"
            assert current_stage() == "ingestion"

        assert current_correlation_id() is None

    def test_returns_snapshot(self) -> None:
        with correlation_context("xyz", stage="scoring") as snap:
            assert isinstance(snap, CorrelationContext)
            assert snap.correlation_id == "xyz"
            assert snap.stage == "scoring"

    def test_no_auto_generate_when_disabled(self) -> None:
        with correlation_context(None, auto_generate=False):
            assert current_correlation_id() is None

    def test_exception_safely_cleans_up(self) -> None:
        try:
            with correlation_context("fail", stage="test"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert current_correlation_id() is None
        assert current_stage() is None


# ── Correlation context as decorator ─────────────────────────────────────────


class TestCorrelationContextDecorator:
    def test_decorator_sets_context(self) -> None:
        @correlation_context.wrap(stage="detection")
        def detect():
            return current_stage()

        assert detect() == "detection"

    def test_decorator_preserves_args(self) -> None:
        @correlation_context.wrap(stage="test")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_decorator_preserves_return_value(self) -> None:
        @correlation_context.wrap(stage="test")
        def get_data():
            return {"result": 42}

        assert get_data() == {"result": 42}

    def test_decorator_cleans_up(self) -> None:
        @correlation_context.wrap(stage="ephemeral")
        def transient():
            pass

        transient()
        assert current_stage() is None

    def test_decorator_with_explicit_id(self) -> None:
        @correlation_context.wrap("custom-id", stage="scoring")
        def score():
            return current_correlation_id()

        assert score() == "custom-id"

    def test_decorator_with_pair_and_wallet(self) -> None:
        @correlation_context.wrap(
            "id-1",
            stage="scoring",
            pair_id="USDC:.../XLM:native",
            wallet="GABC...",
        )
        def score():
            return current_pair_id(), current_wallet()

        pair, wallet = score()
        assert pair == "USDC:.../XLM:native"
        assert wallet == "GABC..."


# ── Snapshot ─────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_captures_all_fields(self) -> None:
        with correlation_context("snap-1", stage="ingestion", pair_id="PAIR", wallet="W1"):
            snap = snapshot()
            assert snap.correlation_id == "snap-1"
            assert snap.stage == "ingestion"
            assert snap.pair_id == "PAIR"
            assert snap.wallet == "W1"

    def test_snapshot_outside_context(self) -> None:
        snap = snapshot()
        assert snap.correlation_id is None
        assert snap.stage is None

    def test_snapshot_is_frozen(self) -> None:
        snap = snapshot()
        with pytest.raises(AttributeError):
            snap.correlation_id = "new"  # type: ignore[misc]


# ── PipelineStage constants ─────────────────────────────────────────────────


class TestPipelineStage:
    def test_all_stages_defined(self) -> None:
        assert len(PipelineStage.ALL) == 8
        assert "ingestion" in PipelineStage.ALL
        assert "detection" in PipelineStage.ALL
        assert "streaming" in PipelineStage.ALL
        assert "scoring" in PipelineStage.ALL
        assert "alerting" in PipelineStage.ALL
        assert "monitoring" in PipelineStage.ALL
        assert "persistence" in PipelineStage.ALL
        assert "onchain" in PipelineStage.ALL

    def test_stage_constants_match(self) -> None:
        assert PipelineStage.INGESTION == "ingestion"
        assert PipelineStage.DETECTION == "detection"
        assert PipelineStage.STREAMING == "streaming"
        assert PipelineStage.SCORING == "scoring"
        assert PipelineStage.ALERTING == "alerting"
        assert PipelineStage.MONITORING == "monitoring"
        assert PipelineStage.PERSISTENCE == "persistence"
        assert PipelineStage.ONCHAIN == "onchain"


# ── CorrelationFilter ────────────────────────────────────────────────────────


class TestCorrelationFilter:
    def test_filter_adds_attributes_to_record(self) -> None:
        f = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with correlation_context("cid-1", stage="detection", pair_id="P1", wallet="W1"):
            result = f.filter(record)
        assert result is True
        assert record.correlation_id == "cid-1"  # type: ignore[attr-defined]
        assert record.stage == "detection"  # type: ignore[attr-defined]
        assert record.pair_id == "P1"  # type: ignore[attr-defined]
        assert record.wallet == "W1"  # type: ignore[attr-defined]

    def test_filter_sets_none_outside_context(self) -> None:
        f = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = f.filter(record)
        assert result is True
        assert record.correlation_id is None  # type: ignore[attr-defined]
        assert record.stage is None  # type: ignore[attr-defined]


# ── Thread propagation ──────────────────────────────────────────────────────


class TestThreadPropagation:
    def test_propagate_to_thread_inherits_id(self) -> None:
        results = []

        def worker():
            results.append(current_correlation_id())

        with correlation_context("thread-test", stage="detection"):
            t = threading.Thread(target=propagate_to_thread(worker))
            t.start()
            t.join()

        assert results == ["thread-test"]

    def test_propagate_to_thread_with_new_id(self) -> None:
        results = []

        def worker():
            results.append(current_correlation_id())

        t = threading.Thread(target=propagate_to_thread(worker, correlation_id="custom-tid"))
        t.start()
        t.join()

        assert results == ["custom-tid"]

    def test_propagate_to_thread_auto_generates(self) -> None:
        results = []

        def worker():
            cid = current_correlation_id()
            results.append(cid)

        t = threading.Thread(target=propagate_to_thread(worker))
        t.start()
        t.join()

        assert len(results) == 1
        assert results[0] is not None
        assert len(results[0]) == 32

    def test_propagate_preserves_stage(self) -> None:
        results = []

        def worker():
            results.append((current_correlation_id(), current_stage()))

        with correlation_context("x", stage="ingestion"):
            t = threading.Thread(target=propagate_to_thread(worker))
            t.start()
            t.join()

        assert results == [("x", "ingestion")]


# ── get_correlation_fields ──────────────────────────────────────────────────


class TestGetCorrelationFields:
    def test_returns_only_non_none(self) -> None:
        fields = get_correlation_fields()
        assert fields == {}

    def test_returns_all_set_fields(self) -> None:
        with correlation_context("f1", stage="scoring", pair_id="P1", wallet="W1"):
            fields = get_correlation_fields()
            assert fields == {
                "correlation_id": "f1",
                "stage": "scoring",
                "pair_id": "P1",
                "wallet": "W1",
            }

    def test_includes_extra(self) -> None:
        with correlation_context("f2", extra={"custom": "data"}):
            fields = get_correlation_fields()
            assert fields["correlation_id"] == "f2"
            assert fields["custom"] == "data"


# ── Integration: logging with correlation ────────────────────────────────────


class TestLoggingIntegration:
    def test_log_record_has_correlation_fields(self) -> None:
        from utils.logging import get_logger

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        test_logger = get_logger("test_correlation")
        test_logger.addHandler(handler)
        try:
            with correlation_context("log-test", stage="detection"):
                test_logger.info("test message")

            assert len(captured) == 1
            record = captured[0]
            assert getattr(record, "correlation_id", None) == "log-test"
            assert getattr(record, "stage", None) == "detection"
        finally:
            test_logger.removeHandler(handler)

    def test_log_without_context_has_none_fields(self) -> None:
        from utils.logging import get_logger

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record)  # type: ignore[assignment]

        test_logger = get_logger("test_no_ctx")
        test_logger.addHandler(handler)
        try:
            test_logger.info("bare message")

            assert len(captured) == 1
            record = captured[0]
            assert getattr(record, "correlation_id", None) is None
            assert getattr(record, "stage", None) is None
        finally:
            test_logger.removeHandler(handler)


# ── CorrelationContext dataclass ─────────────────────────────────────────────


class TestCorrelationContextDataclass:
    def test_defaults(self) -> None:
        ctx = CorrelationContext()
        assert ctx.correlation_id is None
        assert ctx.stage is None
        assert ctx.pair_id is None
        assert ctx.wallet is None
        assert ctx.extra is None

    def test_full_init(self) -> None:
        ctx = CorrelationContext(
            correlation_id="abc",
            stage="test",
            pair_id="P",
            wallet="W",
            extra={"k": "v"},
        )
        assert ctx.correlation_id == "abc"
        assert ctx.extra == {"k": "v"}

    def test_immutable(self) -> None:
        ctx = CorrelationContext(correlation_id="abc")
        with pytest.raises(AttributeError):
            ctx.correlation_id = "new"  # type: ignore[misc]
