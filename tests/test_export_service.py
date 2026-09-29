"""Tests for `reporting.export_service` — typed report export contracts."""

from __future__ import annotations

import csv
import io
import json

import pytest

from reporting.export_service import (
    EXPORT_REGISTRY,
    CSVExporter,
    ExportResult,
    FieldSpec,
    JSONExporter,
    NDJSONExporter,
    ReportSchema,
    SchemaValidationError,
    UnsupportedFormatError,
    export_report,
    register_exporter,
)

SCHEMA = ReportSchema(
    fields=[
        FieldSpec("wallet", str, required=True),
        FieldSpec("risk_score", (int, float), required=True),
        FieldSpec("note", str, required=False),
    ]
)

RECORDS = [
    {"wallet": "GA1", "risk_score": 85, "note": "flagged"},
    {"wallet": "GA2", "risk_score": 12.5},
]


class TestReportSchema:
    def test_valid_batch_passes(self):
        SCHEMA.validate_batch(RECORDS)  # should not raise

    def test_missing_required_field_raises_with_row_and_field(self):
        bad = [{"wallet": "GA1"}]
        with pytest.raises(SchemaValidationError) as exc:
            SCHEMA.validate_batch(bad)
        assert exc.value.row_index == 0
        assert exc.value.field_name == "risk_score"

    def test_wrong_type_raises(self):
        bad = [{"wallet": "GA1", "risk_score": "not-a-number"}]
        with pytest.raises(SchemaValidationError):
            SCHEMA.validate_batch(bad)

    def test_optional_field_may_be_absent(self):
        ok = [{"wallet": "GA1", "risk_score": 1}]
        SCHEMA.validate_batch(ok)  # should not raise

    def test_field_names(self):
        assert SCHEMA.field_names() == ["wallet", "risk_score", "note"]


class TestJSONExporter:
    def test_exports_valid_json_array(self):
        result = JSONExporter().export(RECORDS, schema=SCHEMA)
        assert isinstance(result, ExportResult)
        assert result.content_type == "application/json"
        parsed = json.loads(result.content)
        assert len(parsed) == 2

    def test_validates_against_schema(self):
        with pytest.raises(SchemaValidationError):
            JSONExporter().export([{"wallet": "GA1"}], schema=SCHEMA)

    def test_checksum_is_deterministic(self):
        r1 = JSONExporter().export(RECORDS)
        r2 = JSONExporter().export(RECORDS)
        assert r1.checksum == r2.checksum


class TestNDJSONExporter:
    def test_one_json_object_per_line(self):
        result = NDJSONExporter().export(RECORDS, schema=SCHEMA)
        lines = result.content.decode("utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["wallet"] == "GA1"

    def test_empty_batch_produces_empty_body(self):
        result = NDJSONExporter().export([])
        assert result.content == b""
        assert result.record_count == 0


class TestCSVExporter:
    def test_uses_schema_column_order(self):
        result = CSVExporter().export(RECORDS, schema=SCHEMA)
        reader = csv.reader(io.StringIO(result.content.decode("utf-8")))
        header = next(reader)
        assert header == ["wallet", "risk_score", "note"]

    def test_missing_optional_value_is_blank_cell(self):
        result = CSVExporter().export(RECORDS, schema=SCHEMA)
        rows = list(csv.DictReader(io.StringIO(result.content.decode("utf-8"))))
        assert rows[1]["note"] == ""

    def test_infers_columns_without_schema(self):
        result = CSVExporter().export([{"b": 1, "a": 2}])
        reader = csv.reader(io.StringIO(result.content.decode("utf-8")))
        header = next(reader)
        assert header == ["a", "b"]  # sorted for determinism


class TestExportReport:
    def test_dispatches_by_format(self):
        result = export_report(RECORDS, fmt="csv", schema=SCHEMA)
        assert result.content_type == "text/csv"

    def test_unknown_format_lists_available(self):
        with pytest.raises(UnsupportedFormatError) as exc:
            export_report(RECORDS, fmt="xml")
        assert "json" in str(exc.value)

    def test_register_custom_exporter(self):
        class UpperCsvExporter:
            format_name = "csv_upper"
            content_type = "text/csv"

            def export(self, records, schema=None):
                from reporting.export_service import _make_result

                body = "\n".join(str(r).upper() for r in records)
                return _make_result(body, self.content_type, "report.csv", len(records))

        register_exporter(UpperCsvExporter())
        try:
            result = export_report(RECORDS, fmt="csv_upper")
            assert b"GA1" in result.content
        finally:
            del EXPORT_REGISTRY["csv_upper"]
