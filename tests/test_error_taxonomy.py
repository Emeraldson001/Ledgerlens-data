"""Tests for utils/errors.py — traceable error taxonomy for data processing failures."""

import pytest

from utils.errors import (
    ConfigurationError,
    ErrorCategory,
    ExternalServiceError,
    IngestionError,
    LedgerLensError,
    StorageError,
    TransformError,
    ValidationError,
    format_diagnostic,
    wrap_errors,
)


def test_error_carries_namespaced_code_and_category():
    exc = IngestionError("002", "missing field 'amount'", context={"row": 5})
    assert exc.code == "ING-002"
    assert exc.category == ErrorCategory.INGESTION
    assert exc.context == {"row": 5}


def test_default_retryable_varies_by_category():
    assert IngestionError("001", "x").retryable is True
    assert ValidationError("001", "x").retryable is False
    assert StorageError("001", "x").retryable is True
    assert ConfigurationError("001", "x").retryable is False


def test_retryable_can_be_overridden():
    exc = IngestionError("001", "x", retryable=False)
    assert exc.retryable is False


def test_message_includes_code_context_and_remediation():
    exc = ValidationError(
        "010",
        "trade amount is negative",
        context={"wallet_id": "GABC", "amount": -5},
        remediation="Check upstream sign convention.",
    )
    text = str(exc)
    assert "[VAL-010]" in text
    assert "wallet_id='GABC'" in text
    assert "Check upstream sign convention." in text


def test_to_dict_is_json_serialisable_structure():
    exc = TransformError("003", "NaN in feature column", context={"pair": "USDC/native"})
    d = exc.to_dict()
    assert d["code"] == "XFM-003"
    assert d["category"] == "transform"
    assert d["context"] == {"pair": "USDC/native"}
    assert d["cause"] is None


def test_wrap_errors_converts_and_chains_original_exception():
    with pytest.raises(ExternalServiceError) as exc_info:
        with wrap_errors(ExternalServiceError, "005", context={"endpoint": "/trades"}):
            raise ConnectionError("connection reset by peer")

    exc = exc_info.value
    assert exc.code == "EXT-005"
    assert isinstance(exc.cause, ConnectionError)
    assert exc.__cause__ is exc.cause


def test_wrap_errors_passes_through_matching_taxonomy_type_unmodified():
    original = StorageError("099", "disk full")
    with pytest.raises(StorageError) as exc_info:
        with wrap_errors(StorageError, "001"):
            raise original

    assert exc_info.value is original
    assert exc_info.value.code == "STO-099"


def test_wrap_errors_excludes_listed_exception_types():
    with pytest.raises(KeyboardInterrupt):
        with wrap_errors(StorageError, "001", exclude=(KeyboardInterrupt,)):
            raise KeyboardInterrupt()


def test_format_diagnostic_walks_full_cause_chain():
    try:
        try:
            raise ValueError("bad row 5")
        except ValueError as inner:
            raise IngestionError(
                "001",
                "failed to parse trade batch",
                context={"batch_id": "b-42"},
                remediation="Inspect the raw Horizon payload for batch b-42.",
                cause=inner,
            ) from inner
    except IngestionError as outer:
        diagnostic = format_diagnostic(outer)

    assert "[ING-001]" in diagnostic
    assert "batch_id: 'b-42'" in diagnostic
    assert "Inspect the raw Horizon payload" in diagnostic
    assert "ValueError: bad row 5" in diagnostic


def test_base_class_defaults_apply_when_no_subclass_used():
    exc = LedgerLensError("000", "unspecified failure")
    assert exc.code == "GEN-000"
    assert exc.category == ErrorCategory.VALIDATION
    assert exc.retryable is False
