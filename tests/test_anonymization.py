"""Tests for utils.anonymization — data anonymization checks for shared examples.

Covers:
- PII pattern detection (Stellar addresses, emails, IPs, phones)
- Sensitive field name detection
- High-cardinality quasi-identifier detection
- Dict and DataFrame-level validation
- Pseudonymisation stability and irreversibility
- anonymize_dataframe end-to-end pipeline
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from utils.anonymization import (
    AnonymizationChecker,
    AnonymizationReport,
    AnonymizationViolation,
    anonymize_dataframe,
    anonymize_value,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_STELLAR_ADDRESS = "GBLT2XJKNNB7DOYP3QOELK4WPU64BXFMFYXKGQP6K5FKZRZE6SYGNM"
SAMPLE_STELLAR_SECRET = "SBLT2XJKNNB7DOYP3QOELK4WPU64BXFMFYXKGQP6K5FKZRZE6SYGNM"
SAMPLE_EMAIL = "alice@example.com"
SAMPLE_IPV4 = "192.168.1.100"
SAMPLE_IPV6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
SAMPLE_PHONE = "+1-555-123-4567"


@pytest.fixture()
def checker() -> AnonymizationChecker:
    return AnonymizationChecker()


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """A DataFrame with no PII — should pass all checks."""
    return pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "amount": [100.0, 200.0, 300.0],
            "asset_pair": ["USDC/XLM", "USDC/XLM", "BTC/XLM"],
        }
    )


@pytest.fixture()
def dirty_df() -> pd.DataFrame:
    """A DataFrame with PII that should fail checks."""
    return pd.DataFrame(
        {
            "wallet": [SAMPLE_STELLAR_ADDRESS, SAMPLE_STELLAR_ADDRESS, "GNATIVE"],
            "email": [SAMPLE_EMAIL, "bob@test.org", "carol@example.net"],
            "amount": [100.0, 200.0, 300.0],
            "ip_address": [SAMPLE_IPV4, "10.0.0.1", "172.16.0.5"],
        }
    )


# ---------------------------------------------------------------------------
# Value-level detection tests
# ---------------------------------------------------------------------------


class TestCheckValue:
    def test_detects_stellar_address(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("wallet", SAMPLE_STELLAR_ADDRESS)
        types = {v.violation_type for v in violations}
        assert "stellar_address" in types

    def test_detects_stellar_secret(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("key", SAMPLE_STELLAR_SECRET)
        types = {v.violation_type for v in violations}
        assert "stellar_secret" in types

    def test_detects_email(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("contact", SAMPLE_EMAIL)
        types = {v.violation_type for v in violations}
        assert "email" in types

    def test_detects_ipv4(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("host", SAMPLE_IPV4)
        types = {v.violation_type for v in violations}
        assert "ipv4_address" in types

    def test_detects_phone(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("phone", SAMPLE_PHONE)
        types = {v.violation_type for v in violations}
        assert "phone_number" in types

    def test_clean_value_passes(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("amount", "123.45")
        assert len(violations) == 0

    def test_numeric_value_passes(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("risk_score", 75)
        assert len(violations) == 0

    def test_none_value_passes(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("optional_field", None)
        assert len(violations) == 0

    def test_empty_string_passes(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("note", "")
        assert len(violations) == 0

    def test_sensitive_field_name_flagged(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("password", "not-a-real-password")
        types = {v.violation_type for v in violations}
        assert "sensitive_field_name" in types

    def test_sensitive_field_name_case_insensitive(self, checker: AnonymizationChecker) -> None:
        violations = checker.check_value("PASSWORD", "value")
        types = {v.violation_type for v in violations}
        assert "sensitive_field_name" in types

    def test_sample_truncated(self, checker: AnonymizationChecker) -> None:
        long_value = f"user {SAMPLE_EMAIL} is the admin"
        violations = checker.check_value("note", long_value)
        email_v = [v for v in violations if v.violation_type == "email"]
        assert email_v
        assert len(email_v[0].sample) <= 23  # 20 + "..."


# ---------------------------------------------------------------------------
# Dict-level detection tests
# ---------------------------------------------------------------------------


class TestCheckDict:
    def test_detects_pii_in_flat_dict(self, checker: AnonymizationChecker) -> None:
        record = {"wallet": SAMPLE_STELLAR_ADDRESS, "amount": 100.0}
        violations = checker.check_dict(record)
        assert len(violations) > 0

    def test_detects_pii_in_nested_dict(self, checker: AnonymizationChecker) -> None:
        record = {
            "metadata": {
                "contact": SAMPLE_EMAIL,
            },
            "amount": 100.0,
        }
        violations = checker.check_dict(record)
        email_v = [v for v in violations if v.violation_type == "email"]
        assert email_v
        assert email_v[0].field == "metadata.contact"

    def test_detects_pii_in_list_of_dicts(self, checker: AnonymizationChecker) -> None:
        record = {
            "trades": [
                {"base_account": SAMPLE_STELLAR_ADDRESS},
                {"counter_account": SAMPLE_STELLAR_ADDRESS},
            ]
        }
        violations = checker.check_dict(record)
        stellar_v = [v for v in violations if v.violation_type == "stellar_address"]
        assert len(stellar_v) >= 1

    def test_detects_pii_in_list_of_strings(self, checker: AnonymizationChecker) -> None:
        record = {"emails": [SAMPLE_EMAIL, "bob@test.org"]}
        violations = checker.check_dict(record)
        email_v = [v for v in violations if v.violation_type == "email"]
        assert len(email_v) >= 1

    def test_clean_dict_passes(self, checker: AnonymizationChecker) -> None:
        record = {
            "trade_id": "t1",
            "amount": 100.0,
            "asset_pair": "USDC/XLM",
            "risk_score": 72,
        }
        violations = checker.check_dict(record)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# DataFrame-level checks
# ---------------------------------------------------------------------------


class TestCheckDataFrame:
    def test_clean_df_passes(self, checker: AnonymizationChecker, clean_df: pd.DataFrame) -> None:
        report = checker.check_dataframe(clean_df)
        assert report.is_clean
        assert report.records_checked == 3
        assert report.fields_checked == 3

    def test_dirty_df_fails(self, checker: AnonymizationChecker, dirty_df: pd.DataFrame) -> None:
        report = checker.check_dataframe(dirty_df)
        assert not report.is_clean
        types = {v.violation_type for v in report.violations}
        assert "sensitive_field_name" in types  # "email", "ip_address" columns

    def test_detects_stellar_in_df(self, checker: AnonymizationChecker) -> None:
        df = pd.DataFrame({"account": [SAMPLE_STELLAR_ADDRESS, "other"]})
        report = checker.check_dataframe(df)
        types = {v.violation_type for v in report.violations}
        assert "stellar_address" in types

    def test_high_cardinality_detection(self) -> None:
        checker = AnonymizationChecker(
            check_high_cardinality=True,
            high_cardinality_threshold=0.5,
        )
        df = pd.DataFrame({"user_id": [f"user_{i}" for i in range(100)]})
        report = checker.check_dataframe(df)
        types = {v.violation_type for v in report.violations}
        assert "high_cardinality" in types

    def test_high_cardinality_skipped_when_disabled(self) -> None:
        checker = AnonymizationChecker(check_high_cardinality=False)
        df = pd.DataFrame({"user_id": [f"user_{i}" for i in range(100)]})
        report = checker.check_dataframe(df)
        types = {v.violation_type for v in report.violations}
        assert "high_cardinality" not in types

    def test_empty_df_passes(self, checker: AnonymizationChecker) -> None:
        df = pd.DataFrame({"amount": pd.Series(dtype="float64")})
        report = checker.check_dataframe(df)
        assert report.is_clean
        assert report.records_checked == 0


# ---------------------------------------------------------------------------
# AnonymizationReport
# ---------------------------------------------------------------------------


class TestAnonymizationReport:
    def test_empty_report_is_clean(self) -> None:
        report = AnonymizationReport()
        assert report.is_clean
        assert "passed" in report.summary()

    def test_report_with_violations_not_clean(self) -> None:
        report = AnonymizationReport(
            violations=[
                AnonymizationViolation(
                    field="wallet",
                    violation_type="stellar_address",
                    message="test",
                )
            ],
            records_checked=1,
            fields_checked=1,
        )
        assert not report.is_clean
        assert "FAILED" in report.summary()
        assert "stellar_address: 1" in report.summary()


# ---------------------------------------------------------------------------
# Pseudonymisation / anonymize_value
# ---------------------------------------------------------------------------


class TestAnonymizeValue:
    def test_replaces_stellar_address(self) -> None:
        result = anonymize_value(SAMPLE_STELLAR_ADDRESS)
        assert result != SAMPLE_STELLAR_ADDRESS
        assert result.startswith("ANON-")

    def test_replaces_email(self) -> None:
        result = anonymize_value(f"Contact: {SAMPLE_EMAIL}")
        assert SAMPLE_EMAIL not in result
        assert "ANON-" in result

    def test_replaces_ipv4(self) -> None:
        result = anonymize_value(SAMPLE_IPV4)
        assert SAMPLE_IPV4 not in result

    def test_stable_pseudonym(self) -> None:
        """Same input always produces the same pseudonym."""
        a = anonymize_value(SAMPLE_STELLAR_ADDRESS)
        b = anonymize_value(SAMPLE_STELLAR_ADDRESS)
        assert a == b

    def test_different_inputs_produce_different_pseudonyms(self) -> None:
        a = anonymize_value("alice@example.com")
        b = anonymize_value("bob@example.com")
        assert a != b

    def test_clean_value_unchanged(self) -> None:
        clean = "USDC/XLM trading pair"
        assert anonymize_value(clean) == clean


# ---------------------------------------------------------------------------
# anonymize_dataframe
# ---------------------------------------------------------------------------


class TestAnonymizeDataFrame:
    def test_wallet_columns_pseudonymised(self) -> None:
        df = pd.DataFrame(
            {
                "wallet": [SAMPLE_STELLAR_ADDRESS, SAMPLE_STELLAR_ADDRESS],
                "base_account": [SAMPLE_STELLAR_ADDRESS, SAMPLE_STELLAR_ADDRESS],
                "amount": [100.0, 200.0],
            }
        )
        result = anonymize_dataframe(df)
        assert result["wallet"].iloc[0].startswith("WALLET-")
        assert result["base_account"].iloc[0].startswith("WALLET-")
        # Same address produces same pseudonym
        assert result["wallet"].iloc[0] == result["wallet"].iloc[1]
        # Amounts unchanged
        assert result["amount"].iloc[0] == 100.0

    def test_sensitive_columns_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "password": ["secret123"],
                "email": ["alice@example.com"],
                "amount": [100.0],
            }
        )
        result = anonymize_dataframe(df, drop_sensitive=True)
        assert "password" not in result.columns
        assert "email" not in result.columns
        assert "amount" in result.columns

    def test_sensitive_columns_kept_when_disabled(self) -> None:
        df = pd.DataFrame(
            {
                "email": ["alice@example.com"],
                "amount": [100.0],
            }
        )
        result = anonymize_dataframe(df, drop_sensitive=False)
        assert "email" in result.columns
        # But the value should be anonymised
        assert "alice@example.com" not in result["email"].values

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame(
            {
                "wallet": [SAMPLE_STELLAR_ADDRESS],
                "amount": [100.0],
            }
        )
        _ = anonymize_dataframe(df)
        assert df["wallet"].iloc[0] == SAMPLE_STELLAR_ADDRESS

    def test_custom_wallet_fields(self) -> None:
        df = pd.DataFrame(
            {
                "sender": [SAMPLE_STELLAR_ADDRESS],
                "receiver": [SAMPLE_STELLAR_ADDRESS],
            }
        )
        result = anonymize_dataframe(df, wallet_fields={"sender", "receiver"})
        assert result["sender"].iloc[0].startswith("WALLET-")
        assert result["receiver"].iloc[0].startswith("WALLET-")

    def test_handles_nan_values(self) -> None:
        df = pd.DataFrame(
            {
                "wallet": [SAMPLE_STELLAR_ADDRESS, None],
                "amount": [100.0, 200.0],
            }
        )
        result = anonymize_dataframe(df)
        assert result["wallet"].iloc[0].startswith("WALLET-")
        assert pd.isna(result["wallet"].iloc[1])

    def test_result_passes_checker(self) -> None:
        """After anonymization, the checker should find no PII."""
        df = pd.DataFrame(
            {
                "wallet": [SAMPLE_STELLAR_ADDRESS] * 5,
                "base_account": [SAMPLE_STELLAR_ADDRESS] * 5,
                "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        result = anonymize_dataframe(df)
        checker = AnonymizationChecker(check_high_cardinality=False)
        report = checker.check_dataframe(result)
        pii_violations = [
            v for v in report.violations if v.violation_type not in ("sensitive_field_name",)
        ]
        assert len(pii_violations) == 0


# ---------------------------------------------------------------------------
# Extra patterns / custom checker
# ---------------------------------------------------------------------------


class TestCustomChecker:
    def test_extra_patterns(self) -> None:
        custom_re = re.compile(r"\bCUSTOM-\d{6}\b")
        checker = AnonymizationChecker(extra_patterns={"custom_id": custom_re})
        violations = checker.check_value("ref", "CUSTOM-123456")
        types = {v.violation_type for v in violations}
        assert "custom_id" in types

    def test_extra_sensitive_fields(self) -> None:
        checker = AnonymizationChecker(extra_sensitive_fields={"internal_id"})
        violations = checker.check_value("internal_id", "some-value")
        types = {v.violation_type for v in violations}
        assert "sensitive_field_name" in types


# ---------------------------------------------------------------------------
# k-anonymity configuration (issue #684)
# ---------------------------------------------------------------------------


class TestKAnonymityConfig:
    def test_k_eq_1_is_rejected_with_a_clear_error(self) -> None:
        """k=1 provides no anonymity — the checker must reject it explicitly."""
        with pytest.raises(ValueError, match="k=1"):
            AnonymizationChecker(k=1)

    def test_k_below_1_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            AnonymizationChecker(k=0)

    def test_valid_k_is_accepted(self) -> None:
        checker = AnonymizationChecker(k=2)
        assert checker.check_value("amount", "123.45") == []
