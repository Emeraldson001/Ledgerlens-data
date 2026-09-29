"""Tests for validation/parsing.py — Issue #552: Robust CSV and JSON parsing contracts."""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from validation.parsing import (
    CSVParseError,
    FieldError,
    JSONParseError,
    ManipulationEvent,
    ParseResult,
    TradeRecord,
    parse_csv,
    parse_json,
    parse_known_manipulation_events,
    parse_trade_record,
)

# ---------------------------------------------------------------------------
# ParseResult helpers
# ---------------------------------------------------------------------------


class TestParseResult:
    def test_ok_when_no_errors(self):
        pr: ParseResult[dict] = ParseResult(records=[{"a": 1}])
        assert pr.ok is True

    def test_not_ok_when_errors(self):
        pr: ParseResult[dict] = ParseResult(
            errors=[FieldError(row=1, column="x", raw_value="bad", reason="wrong")]
        )
        assert pr.ok is False

    def test_raise_if_errors_raises(self):
        pr: ParseResult[dict] = ParseResult(
            errors=[FieldError(row=2, column="amount", raw_value="-1", reason="negative")]
        )
        with pytest.raises(CSVParseError):
            pr.raise_if_errors()

    def test_raise_if_errors_silent_when_ok(self):
        pr: ParseResult[dict] = ParseResult(records=[{"a": 1}])
        pr.raise_if_errors()  # should not raise

    def test_summary_contains_counts(self):
        pr: ParseResult[dict] = ParseResult(
            records=[{}], errors=[FieldError(row=1, column="c", raw_value="v", reason="r")]
        )
        s = pr.summary()
        assert "records=1" in s
        assert "errors=1" in s


# ---------------------------------------------------------------------------
# parse_csv
# ---------------------------------------------------------------------------


class TestParseCSV:
    def test_parses_simple_csv(self):
        data = "name,value\nalice,1\nbob,2\n"
        pr = parse_csv(io.StringIO(data))
        assert pr.record_count == 2
        assert pr.records[0]["name"] == "alice"

    def test_strips_whitespace_from_fields(self):
        data = "name , value\n  alice , 42 \n"
        pr = parse_csv(io.StringIO(data))
        assert pr.records[0]["name"] == "alice"
        assert pr.records[0]["value"] == "42"

    def test_required_columns_present(self):
        data = "id,amount\n1,100\n"
        pr = parse_csv(io.StringIO(data), required_columns=["id", "amount"])
        assert pr.record_count == 1

    def test_required_columns_missing_raises(self):
        data = "id,amount\n1,100\n"
        with pytest.raises(CSVParseError, match="missing_col"):
            parse_csv(io.StringIO(data), required_columns=["id", "amount", "missing_col"])

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(CSVParseError):
            parse_csv(tmp_path / "nonexistent.csv")

    def test_skips_blank_rows_by_default(self):
        data = "name,value\nalice,1\n,\nbob,2\n"
        pr = parse_csv(io.StringIO(data))
        assert pr.record_count == 2

    def test_reads_from_path(self, tmp_path):
        p = tmp_path / "trades.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        pr = parse_csv(p)
        assert pr.record_count == 2

    def test_custom_delimiter(self):
        data = "name\tvalue\nalice\t1\n"
        pr = parse_csv(io.StringIO(data), delimiter="\t")
        assert pr.records[0]["name"] == "alice"

    def test_source_label_set_from_path(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a\n1\n", encoding="utf-8")
        pr = parse_csv(p)
        assert pr.source is not None
        assert "test.csv" in pr.source


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------


class TestParseJSON:
    def test_parses_object(self):
        data = json.dumps({"key": "value", "num": 42})
        result = parse_json(io.StringIO(data))
        assert result["key"] == "value"

    def test_parses_list(self):
        data = json.dumps([1, 2, 3])
        result = parse_json(io.StringIO(data))
        assert result == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(JSONParseError, match="Invalid JSON"):
            parse_json(io.StringIO("{bad json"))

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(JSONParseError):
            parse_json(tmp_path / "missing.json")

    def test_reads_from_path(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"x": 1}), encoding="utf-8")
        result = parse_json(p)
        assert result["x"] == 1

    def test_schema_validation_passes(self):
        from pydantic import BaseModel

        class MySchema(BaseModel):
            name: str
            count: int

        data = json.dumps({"name": "test", "count": 5})
        result = parse_json(io.StringIO(data), schema=MySchema)
        assert isinstance(result, MySchema)
        assert result.count == 5

    def test_schema_validation_fails(self):
        from pydantic import BaseModel

        class MySchema(BaseModel):
            name: str
            count: int

        json.dumps({"name": "test", "count": "not_an_int_but_coercible"})
        # pydantic v2 coerces strings to int; test a genuinely invalid case
        data_bad = json.dumps({"name": "test"})  # missing required field
        with pytest.raises(JSONParseError):
            parse_json(io.StringIO(data_bad), schema=MySchema)


# ---------------------------------------------------------------------------
# parse_trade_record
# ---------------------------------------------------------------------------


class TestParseTradeRecord:
    def _valid_row(self) -> dict:
        return {
            "trade_id": "t001",
            "ledger_close_time": "2024-01-01T00:00:00Z",
            "base_account": "GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "counter_account": "GBBBBBBCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "base_asset_code": "USDC",
            "base_asset_issuer": "GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
            "counter_asset_code": "XLM",
            "counter_asset_issuer": "",
            "base_amount": "100.5",
            "counter_amount": "50.0",
            "price": "2.01",
        }

    def test_valid_row_returns_record(self):
        record, errors = parse_trade_record(self._valid_row(), row_index=1)
        assert record is not None
        assert errors == []
        assert isinstance(record, TradeRecord)
        assert record.base_amount == 100.5

    def test_invalid_float_returns_error(self):
        row = self._valid_row()
        row["base_amount"] = "not_a_number"
        record, errors = parse_trade_record(row, row_index=5)
        assert record is None
        assert any(e.column == "base_amount" for e in errors)
        assert errors[0].row == 5

    def test_invalid_account_returns_error(self):
        row = self._valid_row()
        row["base_account"] = "BADACCOUNT"
        record, errors = parse_trade_record(row, row_index=2)
        assert record is None
        assert errors

    def test_null_issuer_is_accepted(self):
        row = self._valid_row()
        row["counter_asset_issuer"] = ""
        record, errors = parse_trade_record(row)
        assert record is not None
        assert record.counter_asset_issuer is None


# ---------------------------------------------------------------------------
# parse_known_manipulation_events
# ---------------------------------------------------------------------------


class TestParseKnownManipulationEvents:
    def test_parses_real_file(self):
        path = Path("data") / "known_manipulation_events.csv"
        if not path.exists():
            pytest.skip("data/known_manipulation_events.csv not present")
        pr = parse_known_manipulation_events(path)
        assert pr.record_count > 0
        assert pr.ok
        assert all(isinstance(e, ManipulationEvent) for e in pr.records)

    def test_parses_synthetic_file(self, tmp_path):
        content = textwrap.dedent("""\
            wallet,asset_pair,campaign_start,campaign_end,label_source,label_confidence,description
            GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native,2024-01-01T00:00:00Z,2024-01-10T00:00:00Z,https://example.com,3,Test manipulation event
        """)
        p = tmp_path / "events.csv"
        p.write_text(content, encoding="utf-8")
        pr = parse_known_manipulation_events(p)
        assert pr.record_count == 1
        assert pr.records[0].label_confidence == 3

    def test_missing_required_column_raises(self, tmp_path):
        content = "wallet,asset_pair\nGAAAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ123,pair\n"
        p = tmp_path / "bad.csv"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(CSVParseError):
            parse_known_manipulation_events(p)

    def test_bad_row_collected_as_error(self, tmp_path):
        content = textwrap.dedent("""\
            wallet,asset_pair,campaign_start,campaign_end,label_source,label_confidence,description
            BADWALLET,USDC/XLM,2024-01-01T00:00:00Z,2024-01-10T00:00:00Z,src,3,desc
        """)
        p = tmp_path / "bad_row.csv"
        p.write_text(content, encoding="utf-8")
        pr = parse_known_manipulation_events(p)
        # Bad wallet should cause a row error, not a crash
        assert pr.error_count >= 1
        assert pr.record_count == 0
