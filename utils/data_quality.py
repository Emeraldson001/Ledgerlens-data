"""Pluggable data quality / validation framework.

Several parts of the repo do ad-hoc validation of incoming records: range
checks against ``data/feature_ranges.json``, presence checks in config
loading (``tests/test_config_validation.py``), schema checks on trade
records (``ingestion/data_models.py``, ``data/trade_avro_schema.json``).
This module provides a small, reusable, typed contract for that pattern —
compose independent ``ValidationRule`` objects, run them against a record
(or a batch of records), and get back a structured ``ValidationReport``
that tells you exactly which record and which rule failed, not just "some
row didn't validate."

Typical usage::

    from utils.data_quality import (
        DataQualityValidator, RequiredFieldRule, RangeRule, TypeRule,
    )

    validator = DataQualityValidator([
        RequiredFieldRule("wallet"),
        RequiredFieldRule("score"),
        TypeRule("score", (int, float)),
        RangeRule("score", minimum=0, maximum=100),
    ])

    report = validator.validate({"wallet": "G...", "score": 142})
    if not report.passed:
        for issue in report.issues:
            log.warning("row failed %s: %s", issue.rule_name, issue.message)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class ValidationRule(Protocol):
    """Contract every rule must satisfy.

    ``name`` should be stable and human-readable (used in diagnostics).
    ``check`` returns None on success, or a human-readable failure message
    on failure. It must not raise for "normal" validation failures — only
    for programmer errors (e.g. misconfigured rule).
    """

    name: str

    def check(self, record: dict[str, Any]) -> str | None: ...


@dataclass
class RequiredFieldRule:
    field: str
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"required:{self.field}"

    def check(self, record: dict[str, Any]) -> str | None:
        if self.field not in record or record[self.field] is None:
            return f"required field {self.field!r} is missing or null"
        return None


@dataclass
class TypeRule:
    field: str
    expected_type: type | tuple[type, ...]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        type_name = getattr(self.expected_type, "__name__", str(self.expected_type))
        self.name = f"type:{self.field}:{type_name}"

    def check(self, record: dict[str, Any]) -> str | None:
        if self.field not in record or record[self.field] is None:
            return None  # absence is RequiredFieldRule's concern, not this rule's
        value = record[self.field]
        if isinstance(value, bool) and self.expected_type in (int, float, (int, float)):
            return f"field {self.field!r} is bool, expected {self.expected_type}"
        if not isinstance(value, self.expected_type):
            return f"field {self.field!r} has type {type(value).__name__}, expected {self.expected_type}"
        return None


@dataclass
class RangeRule:
    field: str
    minimum: float | None = None
    maximum: float | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"range:{self.field}[{self.minimum},{self.maximum}]"

    def check(self, record: dict[str, Any]) -> str | None:
        if self.field not in record or record[self.field] is None:
            return None
        value = record[self.field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"field {self.field!r} is not numeric, cannot range-check"
        if self.minimum is not None and value < self.minimum:
            return f"field {self.field!r}={value} is below minimum {self.minimum}"
        if self.maximum is not None and value > self.maximum:
            return f"field {self.field!r}={value} is above maximum {self.maximum}"
        return None

    @classmethod
    def from_feature_ranges(cls, path: str = "data/feature_ranges.json") -> list[RangeRule]:
        """Build RangeRule instances from the repo's feature_ranges.json,
        if present. Returns [] if the file is missing or malformed, so
        callers can use this as an optional enrichment without special-casing
        environments where the file isn't available (e.g. minimal test
        fixtures)."""
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                ranges = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        rules = []
        for field_name, bounds in ranges.items():
            if isinstance(bounds, dict) and ("min" in bounds or "max" in bounds):
                rules.append(cls(field_name, minimum=bounds.get("min"), maximum=bounds.get("max")))
        return rules


@dataclass
class RegexRule:
    field: str
    pattern: str
    name: str = field(init=False)
    _compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"regex:{self.field}:{self.pattern}"
        self._compiled = re.compile(self.pattern)

    def check(self, record: dict[str, Any]) -> str | None:
        if self.field not in record or record[self.field] is None:
            return None
        value = record[self.field]
        if not isinstance(value, str) or not self._compiled.match(value):
            return f"field {self.field!r}={value!r} does not match pattern {self.pattern!r}"
        return None


@dataclass
class ValidationIssue:
    rule_name: str
    field: str | None
    message: str
    record_index: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    records_checked: int = 0

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def issues_for(self, field_name: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.field == field_name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "records_checked": self.records_checked,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "rule_name": i.rule_name,
                    "field": i.field,
                    "message": i.message,
                    "record_index": i.record_index,
                }
                for i in self.issues
            ],
        }


class DataQualityValidator:
    """Runs a fixed set of ValidationRule objects against records."""

    def __init__(self, rules: list[ValidationRule]):
        self.rules = rules

    def validate(self, record: dict[str, Any], record_index: int | None = None) -> ValidationReport:
        report = ValidationReport(records_checked=1)
        for rule in self.rules:
            message = rule.check(record)
            if message is not None:
                field_name = getattr(rule, "field", None)
                report.issues.append(
                    ValidationIssue(
                        rule_name=rule.name,
                        field=field_name,
                        message=message,
                        record_index=record_index,
                    )
                )
        return report

    def validate_batch(
        self, records: list[dict[str, Any]], fail_fast: bool = False
    ) -> ValidationReport:
        """Validate every record, aggregating issues with their record index.

        With ``fail_fast=True``, stops at the first record that has any
        issue (still returns the full report for that record).
        """
        aggregate = ValidationReport()
        for idx, record in enumerate(records):
            per_record = self.validate(record, record_index=idx)
            aggregate.records_checked += 1
            aggregate.issues.extend(per_record.issues)
            if fail_fast and per_record.issues:
                break
        return aggregate
