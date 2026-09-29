"""Tests for `privacy.transform_utils` — the privacy-preserving transform pipeline."""

from __future__ import annotations

import pytest

from privacy.transform_utils import (
    GeneralizeDateTransform,
    GeneralizeNumericTransform,
    MaskTransform,
    PrivacyTransformError,
    PrivacyTransformPipeline,
    PseudonymizeTransform,
    RedactPatternTransform,
    looks_like_email,
)


class TestPseudonymizeTransform:
    def test_deterministic_for_same_key(self):
        t = PseudonymizeTransform(field_name="wallet", secret_key=b"secret")
        v1, _ = t.apply({"wallet": "GA123"})
        v2, _ = t.apply({"wallet": "GA123"})
        assert v1 == v2
        assert v1.startswith("anon_")

    def test_different_keys_produce_different_tokens(self):
        t1 = PseudonymizeTransform(field_name="wallet", secret_key=b"key-a")
        t2 = PseudonymizeTransform(field_name="wallet", secret_key=b"key-b")
        v1, _ = t1.apply({"wallet": "GA123"})
        v2, _ = t2.apply({"wallet": "GA123"})
        assert v1 != v2

    def test_missing_field_raises_named_error(self):
        t = PseudonymizeTransform(field_name="wallet", secret_key=b"secret")
        with pytest.raises(PrivacyTransformError) as exc:
            t.apply({"other": 1})
        assert exc.value.field_name == "wallet"

    def test_empty_key_rejected_at_construction(self):
        with pytest.raises(PrivacyTransformError):
            PseudonymizeTransform(field_name="wallet", secret_key=b"")

    def test_audit_entry_marks_irreversible(self):
        t = PseudonymizeTransform(field_name="wallet", secret_key=b"secret")
        _, entry = t.apply({"wallet": "GA123"})
        assert entry.reversible is False
        assert entry.transform == "pseudonymize"


class TestGeneralizeNumericTransform:
    def test_buckets_value(self):
        t = GeneralizeNumericTransform(field_name="balance", bucket_size=1000)
        value, _ = t.apply({"balance": 15234})
        assert value == "[15000, 16000)"

    def test_rejects_non_numeric(self):
        t = GeneralizeNumericTransform(field_name="balance", bucket_size=1000)
        with pytest.raises(PrivacyTransformError):
            t.apply({"balance": "not-a-number"})

    def test_rejects_non_positive_bucket_size(self):
        with pytest.raises(PrivacyTransformError):
            GeneralizeNumericTransform(field_name="balance", bucket_size=0)


class TestGeneralizeDateTransform:
    def test_truncates_to_month(self):
        t = GeneralizeDateTransform(field_name="ts", granularity="month")
        value, _ = t.apply({"ts": "2024-03-17T10:00:00"})
        assert value == "2024-03-01"

    def test_truncates_to_year(self):
        t = GeneralizeDateTransform(field_name="ts", granularity="year")
        value, _ = t.apply({"ts": "2024-03-17"})
        assert value == "2024-01-01"

    def test_rejects_bad_granularity(self):
        with pytest.raises(PrivacyTransformError):
            GeneralizeDateTransform(field_name="ts", granularity="fortnight")

    def test_rejects_non_date_value(self):
        t = GeneralizeDateTransform(field_name="ts")
        with pytest.raises(PrivacyTransformError):
            t.apply({"ts": "not-a-date"})


class TestMaskTransform:
    def test_keeps_trailing_suffix(self):
        t = MaskTransform(field_name="email", keep_suffix=4)
        value, _ = t.apply({"email": "alice@example.com"})
        assert value.endswith(".com")
        assert value.startswith("*")

    def test_short_values_fully_masked(self):
        t = MaskTransform(field_name="pin", keep_suffix=4)
        value, _ = t.apply({"pin": "12"})
        assert value == "**"


class TestRedactPatternTransform:
    def test_redacts_on_match(self):
        t = RedactPatternTransform(field_name="note", pattern=r"[\w.]+@[\w.]+")
        value, entry = t.apply({"note": "contact a@b.com"})
        assert value == "[REDACTED]"
        assert "matched=True" in entry.detail

    def test_raises_when_configured(self):
        t = RedactPatternTransform(field_name="note", pattern=r"SECRET", raise_on_match=True)
        with pytest.raises(PrivacyTransformError):
            t.apply({"note": "this is SECRET"})

    def test_passes_through_when_no_match(self):
        t = RedactPatternTransform(field_name="note", pattern=r"SECRET")
        value, _ = t.apply({"note": "harmless"})
        assert value == "harmless"


class TestPrivacyTransformPipeline:
    def test_applies_transforms_in_order_and_builds_audit_log(self):
        pipeline = PrivacyTransformPipeline(
            [
                PseudonymizeTransform(field_name="wallet", secret_key=b"k"),
                GeneralizeNumericTransform(field_name="balance", bucket_size=500),
                MaskTransform(field_name="email", keep_suffix=3),
            ]
        )
        result = pipeline.apply(
            {"wallet": "GA1", "balance": 1234, "email": "x@y.com", "note": "kept"}
        )
        assert result.record["note"] == "kept"
        assert result.record["wallet"].startswith("anon_")
        assert result.record["balance"] == "[1000, 1500)"
        assert result.fields_transformed() == {"wallet", "balance", "email"}
        assert len(result.audit_log) == 3

    def test_batch_apply(self):
        pipeline = PrivacyTransformPipeline(
            [GeneralizeNumericTransform(field_name="balance", bucket_size=100)]
        )
        results = pipeline.apply_batch([{"balance": 50}, {"balance": 150}])
        assert [r.record["balance"] for r in results] == ["[0, 100)", "[100, 200)"]

    def test_aborts_on_first_failure(self):
        pipeline = PrivacyTransformPipeline(
            [GeneralizeNumericTransform(field_name="missing", bucket_size=100)]
        )
        with pytest.raises(PrivacyTransformError):
            pipeline.apply({"other": 1})


def test_looks_like_email():
    assert looks_like_email("a@b.com") is True
    assert looks_like_email("not-an-email") is False
    assert looks_like_email(None) is False
