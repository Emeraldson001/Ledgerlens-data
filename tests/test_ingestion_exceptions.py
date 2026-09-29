"""Tests for the typed ingestion/validation exception hierarchy (issue #608).

Covers the shared base, the ingestion taxonomy, context carrying, the scrubbing
of raw payloads, and the inheritance guarantees that keep existing callers and
tests working.
"""

import logging

import pytest

from ingestion.amm_pool_loader import PoolNotFoundError, _validate_pool_id
from ingestion.exceptions import (
    IngestionError,
    InvalidInputError,
    RecordValidationError,
    SchemaValidationError,
    SourceUnavailableError,
    record_context,
    safe_raw,
)
from ingestion.horizon_fetcher import HorizonRateLimitExceeded
from ingestion.kafka_producer import _to_canonical_pair_id
from utils.exceptions import LedgerLensError

VALID_POOL_ID = "a" * 64


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_base_error_str_is_the_message():
    exc = LedgerLensError("something failed")
    assert str(exc) == "something failed"


def test_base_error_context_defaults_to_empty_dict():
    assert LedgerLensError("boom").context == {}


def test_base_error_copies_context_so_later_mutation_does_not_leak():
    ctx = {"source": "loader"}
    exc = LedgerLensError("boom", context=ctx)
    ctx["source"] = "mutated"

    assert exc.context == {"source": "loader"}


# ---------------------------------------------------------------------------
# Hierarchy shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [InvalidInputError, RecordValidationError, SchemaValidationError, SourceUnavailableError],
)
def test_taxonomy_descends_from_shared_base(cls):
    assert issubclass(cls, IngestionError)
    assert issubclass(cls, LedgerLensError)


def test_schema_validation_error_is_a_record_validation_error():
    assert issubclass(SchemaValidationError, RecordValidationError)


def test_invalid_input_error_is_also_a_value_error():
    """Existing ``except ValueError`` handlers and tests must keep working."""
    assert issubclass(InvalidInputError, ValueError)


def test_record_validation_error_is_not_a_key_error():
    """Pydantic-originated failures must not satisfy unrelated ``except KeyError``.

    ``ingestion/sketches.py`` uses ``except KeyError`` for hot-path control flow.
    """
    assert not issubclass(RecordValidationError, KeyError)


def test_reparented_exceptions_keep_their_original_bases():
    assert issubclass(HorizonRateLimitExceeded, SourceUnavailableError)
    assert issubclass(HorizonRateLimitExceeded, RuntimeError)
    assert issubclass(PoolNotFoundError, SourceUnavailableError)


def test_reparented_exceptions_remain_constructible_from_a_bare_message():
    """These are raised with a single positional arg in existing code."""
    assert str(HorizonRateLimitExceeded("429 exhausted")) == "429 exhausted"
    assert str(PoolNotFoundError("pool missing")) == "pool missing"


# ---------------------------------------------------------------------------
# Context carrying
# ---------------------------------------------------------------------------


def test_context_collects_source_reason_and_raw():
    exc = IngestionError(
        "bad record",
        source="mod.func",
        reason="missing field",
        raw={"trade_id": "t-1"},
    )

    assert exc.source == "mod.func"
    assert exc.reason == "missing field"
    assert exc.raw == {"trade_id": "t-1"}
    assert exc.context == {
        "source": "mod.func",
        "reason": "missing field",
        "raw": {"trade_id": "t-1"},
    }


def test_context_omits_unset_fields():
    assert IngestionError("bare").context == {}


def test_raw_payload_is_scrubbed_to_json_safe_values():
    """Mirrors ``kafka_producer._safe_raw`` so DLQ envelopes and exceptions agree."""
    exc = IngestionError("bad", source="m.f", raw={"ok": 1, "obj": object()})

    assert exc.raw["ok"] == 1
    assert isinstance(exc.raw["obj"], str)


def test_safe_raw_passes_through_primitives_and_none():
    assert safe_raw(None) is None
    assert safe_raw({"s": "a", "i": 1, "f": 1.5, "b": True, "n": None}) == {
        "s": "a",
        "i": 1,
        "f": 1.5,
        "b": True,
        "n": None,
    }


def test_context_is_loggable_as_a_logging_extra(caplog):
    """The documented convention is ``extra={"context": exc.context}``."""
    exc = IngestionError("bad record", source="mod.func", reason="missing field")

    with caplog.at_level("ERROR"):
        logging.getLogger(__name__).error("%s", exc, extra={"context": exc.context})

    assert caplog.records[-1].context == {"source": "mod.func", "reason": "missing field"}


# ---------------------------------------------------------------------------
# record_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised",
    [KeyError("id"), TypeError("bad type"), ValueError("bad value"), ZeroDivisionError("div")],
)
def test_record_context_wraps_raw_construction_failures(raised):
    with pytest.raises(RecordValidationError) as excinfo:
        with record_context("mod.func", {"trade_id": "t-1"}):
            raise raised

    exc = excinfo.value
    assert exc.source == "mod.func"
    assert exc.raw == {"trade_id": "t-1"}
    assert exc.__cause__ is raised


def test_record_context_wraps_pydantic_validation_errors():
    """Pydantic's ValidationError subclasses ValueError, so it is covered."""
    from ingestion.data_models import Trade

    with pytest.raises(RecordValidationError) as excinfo:
        with record_context("mod.func", {"trade_id": "t-1"}):
            Trade(trade_id="t-1")

    assert excinfo.value.reason


def test_record_context_does_not_downgrade_an_existing_typed_error():
    original = InvalidInputError("already typed", source="mod.func")

    with pytest.raises(InvalidInputError) as excinfo:
        with record_context("other.func", {"k": "v"}):
            raise original

    assert excinfo.value is original


def test_record_context_is_transparent_on_success():
    with record_context("mod.func", {"k": "v"}):
        result = 1 + 1
    assert result == 2


# ---------------------------------------------------------------------------
# Adoption at real call sites
# ---------------------------------------------------------------------------


def test_invalid_pool_id_raises_typed_error_still_catchable_as_value_error():
    with pytest.raises(InvalidInputError) as excinfo:
        _validate_pool_id("not-a-pool")

    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.source == "amm_pool_loader._validate_pool_id"


def test_invalid_asset_pair_raises_typed_error_still_catchable_as_value_error():
    with pytest.raises(InvalidInputError) as excinfo:
        _to_canonical_pair_id("bad code!", "native", "XLM", "native")

    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.source == "kafka_producer._to_canonical_pair_id"


def _horizon_trade_record() -> dict:
    return {
        "id": "t-1",
        "ledger_close_time": "2026-01-01T00:00:00Z",
        "base_account": "G" + "A" * 55,
        "counter_account": "G" + "B" * 55,
        "base_asset_code": "USDC",
        "base_asset_issuer": "G" + "C" * 55,
        "counter_asset_code": "",
        "counter_asset_issuer": None,
        "base_amount": "10.0",
        "counter_amount": "20.0",
        "price": {"n": "2", "d": "1"},
    }


def test_horizon_to_trade_accepts_a_well_formed_record():
    from ingestion.horizon_streamer import _to_trade

    trade = _to_trade(_horizon_trade_record())

    assert trade.trade_id == "t-1"
    assert trade.counter_asset.code == "XLM"


def test_horizon_to_trade_raises_typed_error_on_missing_field():
    from ingestion.horizon_streamer import _to_trade

    record = _horizon_trade_record()
    del record["id"]

    with pytest.raises(RecordValidationError) as excinfo:
        _to_trade(record)

    assert excinfo.value.source == "horizon_streamer._to_trade"
    assert excinfo.value.raw is not None


def test_horizon_to_trade_raises_typed_error_on_unparseable_price():
    from ingestion.horizon_streamer import _to_trade

    record = _horizon_trade_record()
    record["price"] = {"n": "1", "d": "0"}

    with pytest.raises(RecordValidationError):
        _to_trade(record)


def test_horizon_to_trade_error_does_not_satisfy_except_key_error():
    """Guards the sketches.py hot-path control-flow hazard at a real call site."""
    from ingestion.horizon_streamer import _to_trade

    record = _horizon_trade_record()
    del record["base_account"]

    try:
        _to_trade(record)
    except KeyError:  # pragma: no cover - must not be taken
        pytest.fail("RecordValidationError must not be catchable as KeyError")
    except RecordValidationError as exc:
        assert exc.source == "horizon_streamer._to_trade"
