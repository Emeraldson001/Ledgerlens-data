"""Privacy-preserving data transformation utilities.

Provides a small, typed, composable pipeline for applying privacy transforms
(pseudonymization, generalization, masking) to structured records before they
leave a trust boundary (e.g. before a forensic report, log line, or export
artifact is written to disk or handed to a third party).

Design goals:
    - Each transform is a self-contained, typed unit with a documented
      contract (`FieldTransform.apply`) rather than ad-hoc inline scrubbing.
    - Every transform emits an audit entry describing *what* happened to
      *which* field and *why it is or is not reversible*, so a later
      compliance review does not have to reverse-engineer the pipeline.
    - Failures are actionable: `PrivacyTransformError` always names the
      offending field and the transform that raised it.

API::

    pipeline = PrivacyTransformPipeline([
        PseudonymizeTransform(field="wallet_address", secret_key=b"..."),
        GeneralizeNumericTransform(field="balance", bucket_size=1000),
        MaskTransform(field="email", keep_suffix=4),
    ])
    result = pipeline.apply({"wallet_address": "GA...", "balance": 15234, "email": "a@b.com"})
    result.record        # transformed record
    result.audit_log      # list[TransformAuditEntry] for compliance review
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol


class PrivacyTransformError(ValueError):
    """Raised when a transform cannot be applied to a record.

    Always carries the field name and transform name so a failure can be
    traced directly to the offending rule without inspecting the pipeline.
    """

    def __init__(self, transform: str, field_name: str, reason: str):
        self.transform = transform
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"[{transform}] field {field_name!r}: {reason}")


@dataclass(frozen=True)
class TransformAuditEntry:
    """Records that a single field was transformed, for compliance review."""

    field_name: str
    transform: str
    reversible: bool
    detail: str = ""


@dataclass(frozen=True)
class TransformResult:
    """Typed output contract for a privacy-transform pipeline run."""

    record: dict[str, Any]
    audit_log: list[TransformAuditEntry] = field(default_factory=list)

    def fields_transformed(self) -> set[str]:
        return {entry.field_name for entry in self.audit_log}


class FieldTransform(Protocol):
    """Contract every privacy transform must satisfy."""

    field_name: str

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        """Return the transformed value and an audit entry for `field_name`."""
        ...


@dataclass
class PseudonymizeTransform:
    """Replaces an identifier with a deterministic HMAC-SHA256 token.

    Deterministic pseudonymization preserves joinability (the same input
    always maps to the same token) without exposing the raw identifier.
    Not reversible without the `secret_key` used to produce the token.

    Args:
        field_name: Record key to pseudonymize.
        secret_key: HMAC key. Must be non-empty; treat as a secret credential.
        token_length: Number of hex characters kept from the digest (must be
            between 8 and 64).
    """

    field_name: str
    secret_key: bytes
    token_length: int = 32
    _name: str = field(default="pseudonymize", init=False)

    def __post_init__(self) -> None:
        if not self.secret_key:
            raise PrivacyTransformError(self._name, self.field_name, "secret_key must be non-empty")
        if not (8 <= self.token_length <= 64):
            raise PrivacyTransformError(
                self._name,
                self.field_name,
                f"token_length must be between 8 and 64, got {self.token_length}",
            )

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        if self.field_name not in record:
            raise PrivacyTransformError(self._name, self.field_name, "field missing from record")
        raw = record[self.field_name]
        if raw is None:
            raise PrivacyTransformError(self._name, self.field_name, "cannot pseudonymize None")
        digest = hmac.new(self.secret_key, str(raw).encode("utf-8"), hashlib.sha256).hexdigest()
        token = f"anon_{digest[: self.token_length]}"
        return token, TransformAuditEntry(
            field_name=self.field_name,
            transform=self._name,
            reversible=False,
            detail=f"hmac-sha256, {self.token_length} hex chars kept",
        )


@dataclass
class GeneralizeNumericTransform:
    """Buckets a numeric field to reduce re-identification precision.

    e.g. bucket_size=1000 maps 15234 -> "[15000, 16000)". This is the
    numeric-generalization half of a k-anonymity strategy: coarsening a
    quasi-identifier so fewer records are unique on that field alone.
    """

    field_name: str
    bucket_size: float
    _name: str = field(default="generalize_numeric", init=False)

    def __post_init__(self) -> None:
        if self.bucket_size <= 0:
            raise PrivacyTransformError(
                self._name, self.field_name, f"bucket_size must be > 0, got {self.bucket_size}"
            )

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        if self.field_name not in record:
            raise PrivacyTransformError(self._name, self.field_name, "field missing from record")
        raw = record[self.field_name]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise PrivacyTransformError(
                self._name, self.field_name, f"value {raw!r} is not numeric"
            ) from exc
        lower = math.floor(value / self.bucket_size) * self.bucket_size
        upper = lower + self.bucket_size
        bucket = f"[{lower:g}, {upper:g})"
        return bucket, TransformAuditEntry(
            field_name=self.field_name,
            transform=self._name,
            reversible=False,
            detail=f"bucket_size={self.bucket_size:g}",
        )


@dataclass
class GeneralizeDateTransform:
    """Truncates a date/datetime field to a coarser granularity.

    Supported granularities: "day", "month", "year".
    """

    field_name: str
    granularity: str = "day"
    _name: str = field(default="generalize_date", init=False)
    _VALID = ("day", "month", "year")

    def __post_init__(self) -> None:
        if self.granularity not in self._VALID:
            raise PrivacyTransformError(
                self._name,
                self.field_name,
                f"granularity must be one of {self._VALID}, got {self.granularity!r}",
            )

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        if self.field_name not in record:
            raise PrivacyTransformError(self._name, self.field_name, "field missing from record")
        raw = record[self.field_name]
        if isinstance(raw, str):
            try:
                raw = datetime.fromisoformat(raw)
            except ValueError as exc:
                raise PrivacyTransformError(
                    self._name, self.field_name, f"value {raw!r} is not ISO-8601"
                ) from exc
        if not isinstance(raw, (date, datetime)):
            raise PrivacyTransformError(
                self._name, self.field_name, f"value {raw!r} is not a date/datetime"
            )
        if self.granularity == "year":
            truncated = date(raw.year, 1, 1).isoformat()
        elif self.granularity == "month":
            truncated = date(raw.year, raw.month, 1).isoformat()
        else:
            truncated = (raw.date() if isinstance(raw, datetime) else raw).isoformat()
        return truncated, TransformAuditEntry(
            field_name=self.field_name,
            transform=self._name,
            reversible=False,
            detail=f"granularity={self.granularity}",
        )


@dataclass
class MaskTransform:
    """Partially masks a string field, keeping only a trailing suffix.

    e.g. keep_suffix=4 maps "alice@example.com" -> "**************.com"[-4:]
    style masking of the form "****1234". Useful for display contexts where
    a human needs to confirm identity without seeing the full value.
    """

    field_name: str
    keep_suffix: int = 4
    mask_char: str = "*"
    _name: str = field(default="mask", init=False)

    def __post_init__(self) -> None:
        if self.keep_suffix < 0:
            raise PrivacyTransformError(self._name, self.field_name, "keep_suffix must be >= 0")

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        if self.field_name not in record:
            raise PrivacyTransformError(self._name, self.field_name, "field missing from record")
        raw = record[self.field_name]
        if raw is None:
            raise PrivacyTransformError(self._name, self.field_name, "cannot mask None")
        text = str(raw)
        if len(text) <= self.keep_suffix:
            masked = self.mask_char * len(text)
        else:
            masked = self.mask_char * (len(text) - self.keep_suffix) + text[-self.keep_suffix :]
        return masked, TransformAuditEntry(
            field_name=self.field_name,
            transform=self._name,
            reversible=False,
            detail=f"keep_suffix={self.keep_suffix}",
        )


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class RedactPatternTransform:
    """Replaces a field with a fixed placeholder if it matches a regex.

    Useful as a last-line-of-defense guard (e.g. flag any field that still
    looks like an email or Stellar secret seed after upstream transforms).
    Raises `PrivacyTransformError` instead of silently passing through when
    `raise_on_match=True`, so accidental leakage fails loudly.
    """

    field_name: str
    pattern: re.Pattern[str] | str
    placeholder: str = "[REDACTED]"
    raise_on_match: bool = False
    _name: str = field(default="redact_pattern", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def apply(self, record: dict[str, Any]) -> tuple[Any, TransformAuditEntry]:
        if self.field_name not in record:
            raise PrivacyTransformError(self._name, self.field_name, "field missing from record")
        raw = record[self.field_name]
        text = "" if raw is None else str(raw)
        matched = bool(self.pattern.search(text))  # type: ignore[union-attr]
        if matched and self.raise_on_match:
            raise PrivacyTransformError(
                self._name, self.field_name, "value matched forbidden pattern"
            )
        value = self.placeholder if matched else raw
        return value, TransformAuditEntry(
            field_name=self.field_name,
            transform=self._name,
            reversible=False,
            detail=f"matched={matched}",
        )


class PrivacyTransformPipeline:
    """Applies an ordered list of `FieldTransform`s to a record.

    Fields not covered by any transform pass through unchanged. Each
    transform failure aborts the pipeline with a `PrivacyTransformError`
    that names the field and transform, rather than silently dropping data.
    """

    def __init__(self, transforms: list[FieldTransform]):
        self.transforms = transforms

    def apply(self, record: dict[str, Any]) -> TransformResult:
        output = dict(record)
        audit_log: list[TransformAuditEntry] = []
        for transform in self.transforms:
            value, entry = transform.apply(output)
            output[transform.field_name] = value
            audit_log.append(entry)
        return TransformResult(record=output, audit_log=audit_log)

    def apply_batch(self, records: list[dict[str, Any]]) -> list[TransformResult]:
        return [self.apply(record) for record in records]


def looks_like_email(value: Any) -> bool:
    """Best-effort check used by callers deciding whether to add a redact rule."""
    return bool(value) and bool(_EMAIL_RE.match(str(value)))
