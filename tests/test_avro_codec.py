"""Tests for Avro codec validation and schema-registry failure paths (issue #608).

The codec is the ingestion layer's first line of defence against poison-pill
Kafka messages, so its failure modes are asserted explicitly.
"""

import pytest

from ingestion.avro_codec import (
    SchemaDecodeError,
    SchemaRegistry,
    deserialize,
    load_schema,
    record_to_trade,
    serialize,
    trade_to_record,
    validate,
)
from ingestion.data_models import Asset, Trade
from ingestion.exceptions import (
    IngestionError,
    InvalidInputError,
    RecordValidationError,
    SchemaValidationError,
)


def _trade() -> Trade:
    return Trade(
        trade_id="t-1",
        ledger_close_time="2026-01-01T00:00:00Z",
        base_account="G" + "A" * 55,
        counter_account="G" + "B" * 55,
        base_asset=Asset(code="USDC", issuer="G" + "C" * 55),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=10.0,
        counter_amount=20.0,
        price=2.0,
    )


@pytest.fixture
def schema():
    return load_schema()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_trade_identity(schema):
    record = trade_to_record(_trade())
    rebuilt = record_to_trade(record)

    assert rebuilt.trade_id == "t-1"
    assert rebuilt.base_asset.code == "USDC"
    assert rebuilt.counter_asset.issuer is None


def test_serialize_accepts_a_valid_record(schema):
    assert isinstance(serialize(trade_to_record(_trade()), schema), bytes)


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------


def test_serialize_rejects_a_record_missing_fields(schema):
    with pytest.raises(SchemaValidationError) as excinfo:
        serialize({"trade_id": "t-1"}, schema)

    exc = excinfo.value
    assert exc.source == "avro_codec.serialize"
    assert exc.reason
    assert exc.raw == {"trade_id": "t-1"}


def test_validate_rejects_a_wrong_typed_record(schema):
    record = trade_to_record(_trade())
    record["base_amount"] = "not-a-number"

    with pytest.raises(SchemaValidationError) as excinfo:
        validate(record, schema)

    assert excinfo.value.source == "avro_codec.validate"


def test_schema_validation_error_is_still_a_plain_exception(schema):
    """The producer/worker DLQ paths catch broad ``Exception`` — keep that working."""
    with pytest.raises(Exception):  # noqa: B017 - asserting the broad-catch contract
        serialize({}, schema)


# ---------------------------------------------------------------------------
# record_to_trade boundary
# ---------------------------------------------------------------------------


def test_record_to_trade_rejects_a_record_missing_asset_pair():
    record = trade_to_record(_trade())
    del record["asset_pair"]

    with pytest.raises(RecordValidationError) as excinfo:
        record_to_trade(record)

    assert excinfo.value.source == "avro_codec.record_to_trade"


def test_record_to_trade_rejects_a_wrong_typed_amount():
    record = trade_to_record(_trade())
    record["base_amount"] = "not-a-number"

    with pytest.raises(RecordValidationError):
        record_to_trade(record)


def test_record_to_trade_failure_carries_the_offending_record():
    record = trade_to_record(_trade())
    del record["trade_id"]

    with pytest.raises(RecordValidationError) as excinfo:
        record_to_trade(record)

    assert excinfo.value.raw is not None
    assert "asset_pair" in excinfo.value.raw


# ---------------------------------------------------------------------------
# SchemaRegistry fingerprint lookups
# ---------------------------------------------------------------------------


def test_registry_round_trips_a_registered_schema(schema):
    registry = SchemaRegistry()
    fp = registry.register(schema)

    assert registry.get_schema(fp) == schema
    assert registry.get_version(fp) == 1


def test_registry_rejects_an_unknown_old_fingerprint(schema):
    registry = SchemaRegistry()
    known = registry.register(schema)

    with pytest.raises(InvalidInputError) as excinfo:
        registry.check_backward_compatibility(-1, known)

    assert isinstance(excinfo.value, IngestionError)
    assert "Unknown fingerprint (old)" in str(excinfo.value)


def test_registry_rejects_an_unknown_new_fingerprint(schema):
    registry = SchemaRegistry()
    known = registry.register(schema)

    with pytest.raises(InvalidInputError) as excinfo:
        registry.check_forward_compatibility(known, -1)

    assert "Unknown fingerprint (new)" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Malformed payload deserialisation (issue #682)
# ---------------------------------------------------------------------------


def test_deserialize_rejects_a_truncated_payload(schema):
    """A payload cut off mid-record raises SchemaDecodeError, not a bare exception."""
    full = serialize(trade_to_record(_trade()), schema)
    truncated = full[: len(full) // 2]

    with pytest.raises(SchemaDecodeError):
        deserialize(truncated, schema)


def test_deserialize_rejects_an_empty_payload(schema):
    """An empty payload has no bytes to decode — SchemaDecodeError is raised."""
    with pytest.raises(SchemaDecodeError):
        deserialize(b"", schema)


def test_deserialize_rejects_a_payload_encoded_against_a_different_schema(schema):
    """Bytes written with a mismatched schema do not decode under the trade schema."""
    other_schema = {
        "type": "record",
        "name": "Trade",
        "namespace": "io.ledgerlens",
        "fields": [{"name": "completely_different", "type": "string"}],
    }
    payload = serialize({"completely_different": "surprise"}, other_schema)

    with pytest.raises(SchemaDecodeError):
        deserialize(payload, schema)


def test_deserialize_wraps_failures_in_schema_decode_error(schema):
    """All malformed-payload failures surface as SchemaDecodeError (a ValueError)."""
    for bad_payload in (b"", b"\x00\x01\x02", serialize(trade_to_record(_trade()), schema)[:-1]):
        with pytest.raises(SchemaDecodeError):
            deserialize(bad_payload, schema)
        with pytest.raises(ValueError):
            deserialize(bad_payload, schema)
