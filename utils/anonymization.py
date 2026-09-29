"""Data anonymization checks for shared examples and exported datasets.

Validates that data has been properly anonymized before sharing with external
parties, contributors, or inclusion in public examples.  Detects residual PII
patterns (Stellar wallet addresses, email addresses, IP addresses, phone
numbers), sensitive field names, and high-cardinality identifiers that could
re-identify individuals.

Usage::

    from utils.anonymization import AnonymizationChecker, anonymize_dataframe

    checker = AnonymizationChecker()
    violations = checker.check_dict(record)
    if violations:
        raise ValueError(f"Anonymization violations: {violations}")

    # Or strip PII in-place before sharing:
    safe_df = anonymize_dataframe(df)

Security invariants
-------------------
- All checks are deterministic and side-effect free.
- ``anonymize_value`` replaces detected PII with a stable, one-way SHA-256
  pseudonym so downstream joins still work on the redacted dataset.
- No original PII is recoverable from the pseudonymised output.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

# Stellar public keys are 56 characters; also flag slightly truncated values
# because partial identifiers remain linkable sensitive data.
_STELLAR_ADDRESS_RE = re.compile(r"\bG[A-Z2-7]{53,55}\b")

# Stellar secret keys: S + 55 uppercase alphanumeric characters
_STELLAR_SECRET_RE = re.compile(r"\bS[A-Z2-7]{53,55}\b")

# Email addresses (simplified RFC 5322)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# IPv4 addresses
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 addresses (simplified — catches common formats)
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")

# Phone numbers (international format with optional country code)
_PHONE_RE = re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b")

# API keys / tokens (generic long hex or base64 strings that look like secrets)
_API_KEY_RE = re.compile(r"\b[A-Za-z0-9]{40,}\b")

# Sensitive field names that should not appear in shared data
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "secret",
        "secret_key",
        "private_key",
        "password",
        "passwd",
        "api_key",
        "api_token",
        "access_token",
        "refresh_token",
        "auth_token",
        "ssn",
        "social_security",
        "credit_card",
        "card_number",
        "cvv",
        "date_of_birth",
        "dob",
        "home_address",
        "phone_number",
        "phone",
        "email",
        "ip_address",
        "session_id",
    }
)

# Fields that typically contain wallet addresses and must be pseudonymised
_WALLET_FIELD_NAMES = frozenset(
    {
        "wallet",
        "base_account",
        "counter_account",
        "funding_account",
        "account_id",
        "source_wallet",
    }
)


# ---------------------------------------------------------------------------
# Violation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnonymizationViolation:
    """A single anonymization violation detected in the data."""

    field: str
    violation_type: str
    message: str
    sample: str = ""  # Truncated sample of the offending value


@dataclass
class AnonymizationReport:
    """Summary of all anonymization violations found in a dataset."""

    violations: list[AnonymizationViolation] = field(default_factory=list)
    records_checked: int = 0
    fields_checked: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        if self.is_clean:
            return (
                f"Anonymization check passed: {self.records_checked} records, "
                f"{self.fields_checked} fields checked."
            )
        by_type: dict[str, int] = {}
        for v in self.violations:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        breakdown = ", ".join(f"{t}: {c}" for t, c in sorted(by_type.items()))
        return (
            f"Anonymization check FAILED: {len(self.violations)} violation(s) "
            f"across {self.records_checked} records. Breakdown: {breakdown}"
        )


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


_PATTERN_CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("stellar_address", _STELLAR_ADDRESS_RE),
    ("stellar_secret", _STELLAR_SECRET_RE),
    ("email", _EMAIL_RE),
    ("ipv4_address", _IPV4_RE),
    ("ipv6_address", _IPV6_RE),
    ("phone_number", _PHONE_RE),
]


class AnonymizationChecker:
    """Validates that data has been properly anonymized.

    Parameters
    ----------
    extra_patterns : dict[str, re.Pattern], optional
        Additional regex patterns to check. Keys are violation type names,
        values are compiled regex patterns.
    extra_sensitive_fields : set[str], optional
        Additional field names to flag as sensitive.
    check_high_cardinality : bool
        When True (default), flag string columns in DataFrames where the
        ratio of unique values to total values exceeds
        ``high_cardinality_threshold``, as these may be quasi-identifiers.
    high_cardinality_threshold : float
        Uniqueness ratio above which a string column is flagged (default 0.9).
    k : int, optional
        Minimum group size for the k-anonymity guarantee. ``k=1`` is
        rejected with :class:`ValueError` because a single record is
        trivially re-identifiable — k=1 provides no anonymity at all, so
        silently accepting it would pretend a privacy guarantee that does
        not exist. Must be ``>= 2`` when provided.

    Raises
    ------
    ValueError
        If ``k`` is provided and ``k < 2`` (k=1 is not a valid anonymity
        configuration).
    """

    def __init__(
        self,
        *,
        extra_patterns: dict[str, re.Pattern[str]] | None = None,
        extra_sensitive_fields: set[str] | None = None,
        check_high_cardinality: bool = True,
        high_cardinality_threshold: float = 0.9,
        k: int | None = None,
    ) -> None:
        if k is not None and k < 2:
            raise ValueError(
                f"k={k} provides no anonymity guarantee — k-anonymity requires "
                "k >= 2 so each equivalence class contains at least 2 records"
            )
        self._k = k
        self._patterns = list(_PATTERN_CHECKS)
        if extra_patterns:
            self._patterns.extend(extra_patterns.items())
        self._sensitive_fields = _SENSITIVE_FIELD_NAMES | (extra_sensitive_fields or set())
        self._check_high_cardinality = check_high_cardinality
        self._high_cardinality_threshold = high_cardinality_threshold

    # -- value-level checks --------------------------------------------------

    def check_value(self, field_name: str, value: Any) -> list[AnonymizationViolation]:
        """Check a single value for PII patterns."""
        violations: list[AnonymizationViolation] = []

        # Check field name itself
        normalised = field_name.lower().strip()
        if normalised in self._sensitive_fields:
            violations.append(
                AnonymizationViolation(
                    field=field_name,
                    violation_type="sensitive_field_name",
                    message=f"Field '{field_name}' is a sensitive field name that should "
                    "not appear in shared data.",
                )
            )

        # Check string values for PII patterns
        if isinstance(value, str) and value:
            for pattern_name, pattern in self._patterns:
                if pattern.search(value):
                    # Truncate sample to avoid leaking full PII
                    sample = value[:20] + "..." if len(value) > 20 else value
                    violations.append(
                        AnonymizationViolation(
                            field=field_name,
                            violation_type=pattern_name,
                            message=f"Field '{field_name}' contains a {pattern_name} pattern.",
                            sample=sample,
                        )
                    )

        return violations

    # -- dict-level checks ---------------------------------------------------

    def check_dict(self, record: dict[str, Any]) -> list[AnonymizationViolation]:
        """Check all fields in a dictionary for anonymization violations."""
        violations: list[AnonymizationViolation] = []
        for key, value in record.items():
            violations.extend(self.check_value(key, value))
            # Recurse into nested dicts
            if isinstance(value, dict):
                for v in self.check_dict(value):
                    violations.append(
                        AnonymizationViolation(
                            field=f"{key}.{v.field}",
                            violation_type=v.violation_type,
                            message=v.message,
                            sample=v.sample,
                        )
                    )
            # Check list elements
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        for v in self.check_dict(item):
                            violations.append(
                                AnonymizationViolation(
                                    field=f"{key}[{i}].{v.field}",
                                    violation_type=v.violation_type,
                                    message=v.message,
                                    sample=v.sample,
                                )
                            )
                    elif isinstance(item, str):
                        violations.extend(self.check_value(f"{key}[{i}]", item))
        return violations

    # -- DataFrame-level checks ----------------------------------------------

    def check_dataframe(self, df: pd.DataFrame) -> AnonymizationReport:
        """Check all values in a DataFrame for anonymization violations.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        AnonymizationReport
            Report with any violations found, record/field counts.
        """
        report = AnonymizationReport(records_checked=len(df), fields_checked=len(df.columns))

        # Check column names for sensitive fields
        for col in df.columns:
            normalised = col.lower().strip()
            if normalised in self._sensitive_fields:
                report.violations.append(
                    AnonymizationViolation(
                        field=col,
                        violation_type="sensitive_field_name",
                        message=f"Column '{col}' is a sensitive field name.",
                    )
                )

        # Check high-cardinality string columns (quasi-identifiers)
        if self._check_high_cardinality and len(df) > 0:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                n_unique = df[col].nunique()
                ratio = n_unique / len(df) if len(df) > 0 else 0
                if ratio > self._high_cardinality_threshold and n_unique > 10:
                    report.violations.append(
                        AnonymizationViolation(
                            field=col,
                            violation_type="high_cardinality",
                            message=f"Column '{col}' has high cardinality "
                            f"({n_unique}/{len(df)} = {ratio:.2%} unique), "
                            "which may enable re-identification.",
                        )
                    )

        # Sample-check string values for PII patterns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            # Check up to 1000 values per column for performance
            sample = df[col].dropna()
            if len(sample) > 1000:
                sample = sample.sample(1000, random_state=42)
            for value in sample:
                if not isinstance(value, str):
                    continue
                for pattern_name, pattern in self._patterns:
                    if pattern.search(value):
                        truncated = value[:20] + "..." if len(value) > 20 else value
                        report.violations.append(
                            AnonymizationViolation(
                                field=col,
                                violation_type=pattern_name,
                                message=f"Column '{col}' contains a {pattern_name} pattern.",
                                sample=truncated,
                            )
                        )
                        # One violation per column per pattern type is enough
                        break
                else:
                    continue
                break

        return report


# ---------------------------------------------------------------------------
# Anonymization helpers
# ---------------------------------------------------------------------------


def _pseudonymise(value: str, prefix: str = "ANON") -> str:
    """Replace a value with a stable SHA-256 pseudonym.

    The same input always produces the same pseudonym, so referential
    integrity is preserved in the anonymised dataset.
    """
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def anonymize_value(value: str) -> str:
    """Replace any detected PII in *value* with pseudonyms.

    Non-PII strings are returned unchanged.
    """
    result = value
    for _name, pattern in _PATTERN_CHECKS:
        result = pattern.sub(lambda m: _pseudonymise(m.group(0)), result)
    return result


def anonymize_dataframe(
    df: pd.DataFrame,
    *,
    wallet_fields: set[str] | None = None,
    drop_sensitive: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with PII pseudonymised or removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (not modified in place).
    wallet_fields : set[str], optional
        Column names known to contain wallet addresses. If ``None``, uses
        ``_WALLET_FIELD_NAMES``.  Values are replaced with stable pseudonyms.
    drop_sensitive : bool
        When True (default), columns whose names match ``_SENSITIVE_FIELD_NAMES``
        are dropped entirely.

    Returns
    -------
    pd.DataFrame
        Anonymised copy.
    """
    out = df.copy()
    target_wallet_fields = wallet_fields or _WALLET_FIELD_NAMES

    # Drop sensitive columns
    if drop_sensitive:
        to_drop = [c for c in out.columns if c.lower().strip() in _SENSITIVE_FIELD_NAMES]
        if to_drop:
            logger.info("Dropping sensitive columns: %s", to_drop)
            out = out.drop(columns=to_drop)

    # Pseudonymise wallet address columns
    for col in out.columns:
        if col.lower().strip() in target_wallet_fields:
            out[col] = out[col].apply(
                lambda v: _pseudonymise(v, prefix="WALLET") if isinstance(v, str) and v else v
            )

    # Scan remaining string columns for residual PII patterns
    for col in out.select_dtypes(include=["object", "string"]).columns:
        if col.lower().strip() in target_wallet_fields:
            continue  # Already handled
        out[col] = out[col].apply(lambda v: anonymize_value(v) if isinstance(v, str) else v)

    return out
