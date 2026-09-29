"""Traceable acceptance tests for advanced ingestion observability work items."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry


@pytest.mark.issue("ADV-001")
def test_log_context_is_nested_and_transportable():
    from utils.logging import (
        correlation_headers,
        correlation_id_from_headers,
        get_log_context,
        log_context,
    )

    assert get_log_context() == {}
    with log_context(correlation_id="request-123", pipeline_stage="ingestion", tenant="acme"):
        assert get_log_context() == {
            "correlation_id": "request-123",
            "pipeline_stage": "ingestion",
            "tenant": "acme",
        }
        assert correlation_id_from_headers(correlation_headers()) == "request-123"
        with log_context(pipeline_stage="validation"):
            assert get_log_context()["correlation_id"] == "request-123"
            assert get_log_context()["pipeline_stage"] == "validation"
        assert get_log_context()["pipeline_stage"] == "ingestion"
    assert get_log_context() == {}


@pytest.mark.issue("ADV-002")
def test_ingestion_metrics_emit_throughput_and_typed_failures():
    from monitoring.ingestion_metrics import IngestionMetricsEmitter

    registry = CollectorRegistry()
    emitter = IngestionMetricsEmitter(registry)
    emitter.emit_success("horizon", stage="fetch", record_count=20, duration_seconds=2.0)
    emitter.emit_failure("horizon", TimeoutError("late"), stage="fetch")

    assert (
        registry.get_sample_value(
            "ledgerlens_ingestion_records_total", {"source": "horizon", "stage": "fetch"}
        )
        == 20
    )
    assert (
        registry.get_sample_value(
            "ledgerlens_ingestion_throughput_records_per_second",
            {"source": "horizon", "stage": "fetch"},
        )
        == 10
    )
    assert (
        registry.get_sample_value(
            "ledgerlens_ingestion_failures_total",
            {"source": "horizon", "stage": "fetch", "error_type": "TimeoutError"},
        )
        == 1
    )


@pytest.mark.issue("ADV-003")
def test_ingestion_exceptions_are_machine_readable():
    from ingestion.exceptions import IngestionError, RecordValidationError

    error = RecordValidationError(
        "missing trade id",
        source="horizon",
        operation="decode_trade",
        details={"field": "trade_id"},
    )

    assert isinstance(error, IngestionError)
    assert isinstance(error, ValueError)
    assert error.to_dict() == {
        "error_code": "ingestion_record_invalid",
        "error_type": "RecordValidationError",
        "message": "missing trade id",
        "retryable": False,
        "source": "horizon",
        "operation": "decode_trade",
        "details": {"field": "trade_id"},
    }
