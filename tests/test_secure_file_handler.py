"""Tests for ingestion.secure_file_handler — secure file handling for ledger inputs.

Covers:
- Path traversal prevention
- Extension allow-listing
- File size limits
- CSV injection sanitisation
- Null byte detection
- JSON / NDJSON ingestion
- Required column validation
- Duplicate row warnings
- End-to-end ingestion pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.secure_file_handler import (
    FileIngestionResult,
    SecureFileHandler,
    compute_file_hash,
    sanitise_cell_value,
    validate_extension,
    validate_path,
    validate_size,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_upload_dir(tmp_path: Path) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture()
def valid_csv(tmp_upload_dir: Path) -> Path:
    p = tmp_upload_dir / "ledger.csv"
    p.write_text("trade_id,amount,asset_pair\nt1,100.0,USDC/XLM\nt2,200.0,BTC/XLM\n")
    return p


@pytest.fixture()
def valid_json(tmp_upload_dir: Path) -> Path:
    p = tmp_upload_dir / "ledger.json"
    data = [
        {"trade_id": "t1", "amount": 100.0, "asset_pair": "USDC/XLM"},
        {"trade_id": "t2", "amount": 200.0, "asset_pair": "BTC/XLM"},
    ]
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def valid_ndjson(tmp_upload_dir: Path) -> Path:
    p = tmp_upload_dir / "ledger.ndjson"
    lines = [
        json.dumps({"trade_id": "t1", "amount": 100.0}),
        json.dumps({"trade_id": "t2", "amount": 200.0}),
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture()
def handler(tmp_upload_dir: Path) -> SecureFileHandler:
    return SecureFileHandler(allowed_base=tmp_upload_dir)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


class TestValidatePath:
    def test_valid_path(self, valid_csv: Path, tmp_upload_dir: Path) -> None:
        result = validate_path(valid_csv, allowed_base=tmp_upload_dir)
        assert result == valid_csv.resolve()

    def test_path_traversal_blocked(self, tmp_upload_dir: Path) -> None:
        evil = tmp_upload_dir / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="Path traversal"):
            validate_path(evil, allowed_base=tmp_upload_dir)

    def test_file_not_found(self, tmp_upload_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_path(tmp_upload_dir / "nonexistent.csv", allowed_base=tmp_upload_dir)

    def test_directory_rejected(self, tmp_upload_dir: Path) -> None:
        with pytest.raises(ValueError, match="not a regular file"):
            validate_path(tmp_upload_dir, allowed_base=tmp_upload_dir)

    def test_no_base_restriction(self, valid_csv: Path) -> None:
        result = validate_path(valid_csv)
        assert result == valid_csv.resolve()

    def test_symlink_escape_blocked(self, tmp_upload_dir: Path, tmp_path: Path) -> None:
        outside = tmp_path / "secret.csv"
        outside.write_text("trade_id,amount\nt1,100\n")
        link = tmp_upload_dir / "link.csv"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="Path traversal"):
            validate_path(link, allowed_base=tmp_upload_dir)

    def test_null_byte_path_rejected(self, tmp_upload_dir: Path) -> None:
        with pytest.raises(ValueError, match="null byte"):
            validate_path(str(tmp_upload_dir / "ledger.csv\x00"), allowed_base=tmp_upload_dir)


# ---------------------------------------------------------------------------
# Extension validation
# ---------------------------------------------------------------------------


class TestValidateExtension:
    def test_csv_allowed(self, valid_csv: Path) -> None:
        assert validate_extension(valid_csv) == ".csv"

    def test_json_allowed(self, valid_json: Path) -> None:
        assert validate_extension(valid_json) == ".json"

    def test_ndjson_allowed(self, valid_ndjson: Path) -> None:
        assert validate_extension(valid_ndjson) == ".ndjson"

    def test_exe_rejected(self, tmp_upload_dir: Path) -> None:
        evil = tmp_upload_dir / "malware.exe"
        evil.write_text("not really an exe")
        with pytest.raises(ValueError, match="not allowed"):
            validate_extension(evil)

    def test_py_rejected(self, tmp_upload_dir: Path) -> None:
        evil = tmp_upload_dir / "script.py"
        evil.write_text("import os; os.system('rm -rf /')")
        with pytest.raises(ValueError, match="not allowed"):
            validate_extension(evil)


# ---------------------------------------------------------------------------
# Size validation
# ---------------------------------------------------------------------------


class TestValidateSize:
    def test_normal_size_passes(self, valid_csv: Path) -> None:
        size = validate_size(valid_csv)
        assert size > 0

    def test_oversized_rejected(self, tmp_upload_dir: Path) -> None:
        big = tmp_upload_dir / "big.csv"
        big.write_text("trade_id,amount\n" + "t,1\n" * 100)
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_size(big, max_bytes=10)

    def test_empty_file_rejected(self, tmp_upload_dir: Path) -> None:
        empty = tmp_upload_dir / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            validate_size(empty)


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_deterministic(self, valid_csv: Path) -> None:
        h1 = compute_file_hash(valid_csv)
        h2 = compute_file_hash(valid_csv)
        assert h1 == h2

    def test_different_content_different_hash(self, valid_csv: Path, valid_json: Path) -> None:
        assert compute_file_hash(valid_csv) != compute_file_hash(valid_json)

    def test_hash_is_hex(self, valid_csv: Path) -> None:
        h = compute_file_hash(valid_csv)
        assert len(h) == 64
        int(h, 16)  # Should not raise


# ---------------------------------------------------------------------------
# Cell sanitisation
# ---------------------------------------------------------------------------


class TestSanitiseCellValue:
    def test_strips_formula_prefix_equals(self) -> None:
        assert sanitise_cell_value("=SUM(A1)") == "SUM(A1)"

    def test_strips_formula_prefix_plus(self) -> None:
        assert sanitise_cell_value("+cmd|'...'") == "cmd|'...'"

    def test_strips_formula_prefix_at(self) -> None:
        assert sanitise_cell_value("@SUM(A1)") == "SUM(A1)"

    def test_strips_tab(self) -> None:
        assert sanitise_cell_value("\tdata") == "data"

    def test_normal_value_unchanged(self) -> None:
        assert sanitise_cell_value("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert sanitise_cell_value("") == ""

    def test_null_byte_removed(self) -> None:
        assert sanitise_cell_value("hel\x00lo") == "hello"


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


class TestCSVIngestion:
    def test_valid_csv_accepted(self, handler: SecureFileHandler, valid_csv: Path) -> None:
        result = handler.ingest(valid_csv)
        assert result.accepted
        assert result.row_count == 2
        assert result.column_count == 3
        assert result.dataframe is not None
        assert "trade_id" in result.dataframe.columns

    def test_missing_required_columns(
        self, handler: SecureFileHandler, tmp_upload_dir: Path
    ) -> None:
        p = tmp_upload_dir / "bad.csv"
        p.write_text("col_a,col_b\n1,2\n")
        result = handler.ingest(p)
        assert not result.accepted
        assert "Missing required columns" in result.rejection_reason

    def test_csv_injection_sanitised(
        self, handler: SecureFileHandler, tmp_upload_dir: Path
    ) -> None:
        p = tmp_upload_dir / "inject.csv"
        p.write_text("trade_id,amount,note\nt1,100,=SUM(A1)\n")
        h = SecureFileHandler(allowed_base=tmp_upload_dir, require_columns={"trade_id", "amount"})
        result = h.ingest(p)
        assert result.accepted
        # The '=' prefix should be stripped
        assert not result.dataframe["note"].iloc[0].startswith("=")

    def test_duplicate_rows_warned(self, handler: SecureFileHandler, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "dupes.csv"
        p.write_text("trade_id,amount\nt1,100\nt1,100\n")
        result = handler.ingest(p)
        assert result.accepted
        assert any("duplicate" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# JSON ingestion
# ---------------------------------------------------------------------------


class TestJSONIngestion:
    def test_valid_json_accepted(self, handler: SecureFileHandler, valid_json: Path) -> None:
        result = handler.ingest(valid_json)
        assert result.accepted
        assert result.row_count == 2

    def test_single_object_json(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "single.json"
        p.write_text(json.dumps({"trade_id": "t1", "amount": 100.0}))
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert result.accepted
        assert result.row_count == 1

    def test_invalid_json_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "bad.json"
        p.write_text("{not valid json")
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert not result.accepted

    def test_json_scalar_records_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "scalar.json"
        p.write_text(json.dumps([{"trade_id": "t1"}, "unexpected scalar"]))
        result = SecureFileHandler(allowed_base=tmp_upload_dir).ingest(p)
        assert not result.accepted
        assert "objects" in result.rejection_reason


# ---------------------------------------------------------------------------
# NDJSON ingestion
# ---------------------------------------------------------------------------


class TestNDJSONIngestion:
    def test_valid_ndjson_accepted(self, handler: SecureFileHandler, valid_ndjson: Path) -> None:
        result = handler.ingest(valid_ndjson)
        assert result.accepted
        assert result.row_count == 2

    def test_empty_ndjson_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "empty.ndjson"
        p.write_text("\n\n\n")
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert not result.accepted

    def test_ndjson_scalar_record_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "scalar.ndjson"
        p.write_text('"unexpected scalar"\n')
        result = SecureFileHandler(allowed_base=tmp_upload_dir).ingest(p)
        assert not result.accepted
        assert "objects" in result.rejection_reason

    def test_jsonl_extension_works(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "data.jsonl"
        lines = [json.dumps({"trade_id": "t1", "amount": 100.0})]
        p.write_text("\n".join(lines) + "\n")
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert result.accepted


# ---------------------------------------------------------------------------
# Null byte detection
# ---------------------------------------------------------------------------


class TestNullByteDetection:
    def test_null_byte_in_csv_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "null.csv"
        p.write_bytes(b"trade_id,amount\x00\nt1,100\n")
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert not result.accepted
        assert "null bytes" in result.rejection_reason.lower()

    def test_null_byte_in_json_rejected(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "null.json"
        p.write_bytes(b'[{"trade_id": "t1\x00", "amount": 100}]')
        h = SecureFileHandler(allowed_base=tmp_upload_dir)
        result = h.ingest(p)
        assert not result.accepted


# ---------------------------------------------------------------------------
# Path traversal end-to-end
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_traversal_via_handler_rejected(
        self, handler: SecureFileHandler, tmp_path: Path
    ) -> None:
        # Create a file outside the upload dir
        outside = tmp_path / "secret.csv"
        outside.write_text("trade_id,amount\nt1,100\n")
        result = handler.ingest(outside)
        assert not result.accepted
        assert (
            "traversal" in result.rejection_reason.lower()
            or "outside" in result.rejection_reason.lower()
        )

    def test_disallowed_extension_via_handler(
        self, handler: SecureFileHandler, tmp_upload_dir: Path
    ) -> None:
        evil = tmp_upload_dir / "payload.sh"
        evil.write_text("#!/bin/bash\nrm -rf /")
        result = handler.ingest(evil)
        assert not result.accepted
        assert (
            "extension" in result.rejection_reason.lower()
            or "not allowed" in result.rejection_reason.lower()
        )


# ---------------------------------------------------------------------------
# Custom configuration
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_require_columns(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "custom.csv"
        p.write_text("wallet,volume\nGBLT,500\n")
        h = SecureFileHandler(
            allowed_base=tmp_upload_dir,
            require_columns={"wallet", "volume"},
        )
        result = h.ingest(p)
        assert result.accepted

    def test_no_required_columns(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "any.csv"
        p.write_text("a,b\n1,2\n")
        h = SecureFileHandler(
            allowed_base=tmp_upload_dir,
            require_columns=set(),
        )
        result = h.ingest(p)
        assert result.accepted

    def test_custom_size_limit(self, tmp_upload_dir: Path) -> None:
        p = tmp_upload_dir / "small.csv"
        p.write_text("trade_id,amount\n" + "t,1\n" * 100)
        h = SecureFileHandler(allowed_base=tmp_upload_dir, max_file_size=10)
        result = h.ingest(p)
        assert not result.accepted
        assert "size" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# FileIngestionResult
# ---------------------------------------------------------------------------


class TestFileIngestionResult:
    def test_accepted_result(self) -> None:
        r = FileIngestionResult(
            accepted=True,
            file_path="test.csv",
            file_hash="abc123",
            row_count=10,
            column_count=3,
        )
        assert r.accepted
        assert r.file_hash == "abc123"

    def test_rejected_result(self) -> None:
        r = FileIngestionResult(
            accepted=False,
            file_path="test.csv",
            rejection_reason="Too large",
        )
        assert not r.accepted
        assert r.rejection_reason == "Too large"
