"""Robust CSV and JSON parsing contracts for LedgerLens.

Issue #552 — Build robust CSV and JSON parsing contracts
=========================================================

This module provides *typed, validated* parsers for the two primary data
formats used by the LedgerLens ingestion and detection layers:

* **CSV** – tabular trade history, known manipulation events, annotation
  queues, and wallet score exports.
* **JSON** – Horizon API responses, build configs, feature ranges, schema
  evolution records, and arbitrary event payloads.

Design goals
------------
1. **Strict contracts** – every public parser declares its expected schema via
   a Pydantic model; callers receive structured Python objects, not raw dicts.
2. **Partial-success semantics** – a single bad row does *not* abort a bulk
   load.  ``ParseResult`` separates valid records from per-row errors so the
   caller can decide whether to reject the batch or continue with the clean
   subset.
3. **Actionable error messages** – ``FieldError`` names the problematic column,
   row, and raw value so contributors can pinpoint data-quality issues without
   running a debugger.
4. **Composability** – parsers are pure functions and can be composed, mocked,
   and tested independently of network I/O or the database layer.

Public API
----------
::

    from validation.parsing import (
        parse_csv,
        parse_json,
        parse_trade_record,
        parse_known_manipulation_events,
        ParseResult,
        CSVParseError,
        JSONParseError,
    )
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldError:
    """A field-level parse failure: names the row, column, raw value, and reason."""

    row: int
    column: str
    raw_value: Any
    reason: str

    def __str__(self) -> str:
        return (
            f"Row {self.row}, column '{self.column}': "
            f"{self.reason} (raw value: {self.raw_value!r})"
        )


class CSVParseError(ValueError):
    """Raised when CSV input is structurally invalid (bad delimiter, wrong encoding,
    missing required header columns, etc.).  Individual row errors are collected
    in :class:`ParseResult` rather than raised eagerly.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (file: {self.path})" if self.path else base


class JSONParseError(ValueError):
    """Raised when JSON input cannot be decoded or fails schema validation."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (file: {self.path})" if self.path else base


# ---------------------------------------------------------------------------
# Generic ParseResult container
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class ParseResult(Generic[T]):
    """Container for the outcome of a bulk parse operation.

    Attributes
    ----------
    records:
        Successfully parsed and validated objects.
    errors:
        Per-row or per-field failures that did not abort the overall parse.
    source:
        Optional label (file path, URL, …) for traceability.
    """

    records: list[T] = field(default_factory=list)
    errors: list[FieldError] = field(default_factory=list)
    source: str | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """True when every row was parsed without error."""
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def raise_if_errors(self) -> None:
        """Raise :class:`CSVParseError` when any row-level errors were collected."""
        if self.errors:
            summary = "; ".join(str(e) for e in self.errors[:5])
            suffix = f" … ({len(self.errors) - 5} more)" if len(self.errors) > 5 else ""
            raise CSVParseError(
                f"{len(self.errors)} parse error(s): {summary}{suffix}",
                path=self.source,
            )

    def summary(self) -> str:
        return (
            f"ParseResult(records={self.record_count}, errors={self.error_count}, "
            f"source={self.source!r})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Pydantic schemas for the canonical CSV/JSON shapes used in this repo
# ---------------------------------------------------------------------------


class TradeRecord(BaseModel):
    """Minimal typed representation of a single trade row (CSV / JSON)."""

    trade_id: str
    ledger_close_time: datetime
    base_account: str
    counter_account: str
    base_asset_code: str
    base_asset_issuer: str | None = None
    counter_asset_code: str
    counter_asset_issuer: str | None = None
    base_amount: float = Field(gt=0)
    counter_amount: float = Field(gt=0)
    price: float = Field(gt=0)

    @field_validator("base_account", "counter_account")
    @classmethod
    def _validate_stellar_account(cls, v: str) -> str:
        if not v.startswith("G") or len(v) < 40:
            raise ValueError(
                f"Expected a Stellar account ID starting with 'G' and ≥ 40 chars, got {v!r}"
            )
        return v

    @field_validator("ledger_close_time", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Accept ISO 8601 with or without trailing 'Z'
            return v.rstrip("Z") + "+00:00" if v.endswith("Z") else v
        return v


class ManipulationEvent(BaseModel):
    """A row from ``data/known_manipulation_events.csv``."""

    wallet: str
    asset_pair: str
    campaign_start: datetime
    campaign_end: datetime
    label_source: str
    label_confidence: int = Field(ge=1, le=3)
    description: str

    @field_validator("campaign_start", "campaign_end", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.rstrip("Z") + "+00:00" if v.endswith("Z") else v
        return v

    @field_validator("wallet")
    @classmethod
    def _validate_wallet(cls, v: str) -> str:
        if not v.startswith("G") or len(v) < 40:
            raise ValueError(f"Expected a Stellar account ID, got {v!r}")
        return v


class FeatureRangeRecord(BaseModel):
    """A single entry from ``data/feature_ranges.json``.

    The file maps feature names to ``{min, max, mean, std}`` dicts.
    """

    min: float
    max: float
    mean: float
    std: float = Field(ge=0)


# ---------------------------------------------------------------------------
# Core CSV parser
# ---------------------------------------------------------------------------


def parse_csv(
    source: str | Path | io.StringIO,
    *,
    required_columns: list[str] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    skip_blank_rows: bool = True,
    max_errors: int = 100,
) -> ParseResult[dict]:
    """Parse a CSV file or string buffer into a list of row dicts.

    Parameters
    ----------
    source:
        File path, ``pathlib.Path``, or ``io.StringIO`` containing CSV data.
    required_columns:
        Column names that *must* be present in the header row.  A
        :class:`CSVParseError` is raised if any are missing.
    delimiter:
        Field delimiter.  Defaults to ``","`` (standard CSV).
    encoding:
        File encoding (only used when *source* is a path).
    skip_blank_rows:
        When ``True`` (default), rows where every field is empty are silently
        discarded rather than counted as errors.
    max_errors:
        Stop collecting row-level errors after this many failures to prevent
        runaway error lists on very corrupt inputs.

    Returns
    -------
    ParseResult[dict]:
        Parsed row dicts in ``.records``; field-level errors in ``.errors``.

    Raises
    ------
    CSVParseError:
        When the input cannot be opened, decoded, or lacks required header
        columns (structural failures that prevent any rows from being read).
    """
    label: str | None = None

    # Resolve the source to a text stream
    if isinstance(source, (str, Path)):
        path = Path(source)
        label = str(path)
        try:
            fh: io.TextIOBase = open(path, encoding=encoding, newline="")  # noqa: WPS515
        except OSError as exc:
            raise CSVParseError(str(exc), path=label) from exc
        close_after = True
    else:
        fh = source
        close_after = False

    result: ParseResult[dict] = ParseResult(source=label)

    try:
        reader = csv.DictReader(fh, delimiter=delimiter)

        # Force the header to be read so we can check required columns.
        try:
            fieldnames = reader.fieldnames
        except csv.Error as exc:
            raise CSVParseError(f"Cannot read CSV header: {exc}", path=label) from exc

        if fieldnames is None:
            raise CSVParseError("CSV has no header row", path=label)

        if required_columns:
            missing = [c for c in required_columns if c not in fieldnames]
            if missing:
                raise CSVParseError(
                    f"Required column(s) missing from CSV header: {missing}",
                    path=label,
                )

        for _row_idx, raw_row in enumerate(reader, start=2):  # 1-indexed; row 1 = header
            if skip_blank_rows and all(v.strip() == "" for v in raw_row.values()):
                continue

            # Strip leading/trailing whitespace from every field.
            clean_row = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items()}
            result.records.append(clean_row)

            if result.error_count >= max_errors:
                break

    except CSVParseError:
        raise
    except Exception as exc:  # pragma: no cover
        raise CSVParseError(f"Unexpected CSV read error: {exc}", path=label) from exc
    finally:
        if close_after:
            fh.close()

    return result


# ---------------------------------------------------------------------------
# Core JSON parser
# ---------------------------------------------------------------------------


def parse_json(
    source: str | Path | io.StringIO,
    *,
    schema: type[BaseModel] | None = None,
    encoding: str = "utf-8",
) -> Any:
    """Parse and optionally validate a JSON file or string buffer.

    Parameters
    ----------
    source:
        File path, ``pathlib.Path``, or ``io.StringIO`` containing JSON data.
    schema:
        Optional Pydantic ``BaseModel`` subclass.  When provided, the decoded
        object is validated against the model and a validated instance is
        returned.  When ``None`` (default) the raw Python object is returned.
    encoding:
        File encoding (only used when *source* is a path).

    Returns
    -------
    Any | BaseModel:
        Validated Pydantic model instance when *schema* is given; otherwise
        the raw decoded object.

    Raises
    ------
    JSONParseError:
        When the input cannot be opened, decoded, or fails Pydantic validation.
    """
    label: str | None = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        label = str(path)
        try:
            text = path.read_text(encoding=encoding)
        except OSError as exc:
            raise JSONParseError(str(exc), path=label) from exc
    else:
        text = source.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JSONParseError(
            f"Invalid JSON: {exc.msg} at line {exc.lineno}, col {exc.colno}",
            path=label,
        ) from exc

    if schema is not None:
        try:
            if isinstance(data, dict):
                return schema.model_validate(data)
            elif isinstance(data, list):
                return [schema.model_validate(item) for item in data]
            else:
                return schema.model_validate(data)
        except ValidationError as exc:
            raise JSONParseError(
                f"JSON schema validation failed: {exc}",
                path=label,
            ) from exc

    return data


# ---------------------------------------------------------------------------
# Domain-specific parsers
# ---------------------------------------------------------------------------


def parse_trade_record(
    row: dict, *, row_index: int = 0
) -> tuple[TradeRecord | None, list[FieldError]]:
    """Validate and coerce a single trade row dict into a :class:`TradeRecord`.

    Parameters
    ----------
    row:
        Raw CSV row dict (string values from ``csv.DictReader``).
    row_index:
        1-based row number used in :class:`FieldError` messages.

    Returns
    -------
    (TradeRecord | None, list[FieldError]):
        The parsed record (or ``None`` on failure) and any field-level errors.
    """
    errors: list[FieldError] = []

    # Numeric coercion helpers
    def _float(col: str) -> float | None:
        raw = row.get(col, "")
        try:
            return float(raw)
        except (TypeError, ValueError):
            errors.append(
                FieldError(row=row_index, column=col, raw_value=raw, reason="not a valid float")
            )
            return None

    base_amount = _float("base_amount")
    counter_amount = _float("counter_amount")
    price = _float("price")

    if errors:
        return None, errors

    try:
        record = TradeRecord(
            trade_id=row.get("trade_id", ""),
            ledger_close_time=row.get("ledger_close_time", ""),
            base_account=row.get("base_account", ""),
            counter_account=row.get("counter_account", ""),
            base_asset_code=row.get("base_asset_code", ""),
            base_asset_issuer=row.get("base_asset_issuer") or None,
            counter_asset_code=row.get("counter_asset_code", ""),
            counter_asset_issuer=row.get("counter_asset_issuer") or None,
            base_amount=base_amount,  # type: ignore[arg-type]
            counter_amount=counter_amount,  # type: ignore[arg-type]
            price=price,  # type: ignore[arg-type]
        )
        return record, []
    except ValidationError as exc:
        for err in exc.errors():
            col = ".".join(str(loc) for loc in err["loc"])
            errors.append(
                FieldError(
                    row=row_index,
                    column=col,
                    raw_value=row.get(col, "<missing>"),
                    reason=err["msg"],
                )
            )
        return None, errors


def parse_known_manipulation_events(
    path: str | Path | None = None,
) -> ParseResult[ManipulationEvent]:
    """Parse ``data/known_manipulation_events.csv`` into typed :class:`ManipulationEvent` records.

    Parameters
    ----------
    path:
        Override the default data path.  Defaults to
        ``data/known_manipulation_events.csv`` relative to the repo root.

    Returns
    -------
    ParseResult[ManipulationEvent]:
        Validated records plus any per-row errors.

    Raises
    ------
    CSVParseError:
        If the file is missing, unreadable, or structurally invalid.
    """
    if path is None:
        path = Path("data") / "known_manipulation_events.csv"

    required = [
        "wallet",
        "asset_pair",
        "campaign_start",
        "campaign_end",
        "label_source",
        "label_confidence",
        "description",
    ]

    raw_result = parse_csv(path, required_columns=required)
    result: ParseResult[ManipulationEvent] = ParseResult(source=raw_result.source)

    for row_idx, row in enumerate(raw_result.records, start=2):
        try:
            event = ManipulationEvent(
                wallet=row.get("wallet", ""),
                asset_pair=row.get("asset_pair", ""),
                campaign_start=row.get("campaign_start", ""),  # type: ignore[arg-type]
                campaign_end=row.get("campaign_end", ""),  # type: ignore[arg-type]
                label_source=row.get("label_source", ""),
                label_confidence=int(row.get("label_confidence", 0)),
                description=row.get("description", ""),
            )
            result.records.append(event)
        except (ValidationError, ValueError) as exc:
            result.errors.append(
                FieldError(
                    row=row_idx,
                    column="<row>",
                    raw_value=row,
                    reason=str(exc),
                )
            )

    return result


def parse_feature_ranges(
    path: str | Path | None = None,
) -> ParseResult[tuple[str, FeatureRangeRecord]]:
    """Parse ``data/feature_ranges.json`` into typed :class:`FeatureRangeRecord` pairs.

    Returns
    -------
    ParseResult[tuple[str, FeatureRangeRecord]]:
        ``(feature_name, FeatureRangeRecord)`` pairs in ``.records``; any
        per-feature errors in ``.errors``.
    """
    if path is None:
        path = Path("data") / "feature_ranges.json"

    raw: dict = parse_json(path)
    if not isinstance(raw, dict):
        raise JSONParseError(
            "feature_ranges.json must be a JSON object mapping feature names to range dicts",
            path=str(path),
        )

    result: ParseResult[tuple[str, FeatureRangeRecord]] = ParseResult(source=str(path))

    for feature_name, range_dict in raw.items():
        try:
            record = FeatureRangeRecord.model_validate(range_dict)
            result.records.append((feature_name, record))
        except (ValidationError, TypeError) as exc:
            result.errors.append(
                FieldError(
                    row=0,
                    column=feature_name,
                    raw_value=range_dict,
                    reason=str(exc),
                )
            )

    return result
