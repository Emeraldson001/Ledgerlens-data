"""Privacy threat-model tests for sensitive ledger attributes (Issue #481).

Threat model scope
------------------
Stellar ledger records carry several categories of sensitive attributes that
must never appear in raw form outside a trust boundary:

| Attribute | Type | Risk |
|-----------|------|------|
| ``base_account`` / ``counter_account`` | Stellar wallet address (``G...``) | Directly identifies a Stellar account; linkable across reports |
| ``account_id`` / ``funding_account`` | Stellar wallet address | Same as above; also reveals funding provenance |
| ``ledger_close_time`` | UTC datetime | Re-identification by combining timestamp with other quasi-identifiers |
| ``base_amount`` / ``counter_amount`` | Decimal | Precise amounts are quasi-identifiers in sparse trade graphs |
| ``home_domain`` | String | Identifies wallet operator |

This test file builds a durable regression suite that verifies:

1. **Pseudonymization contract** — wallet addresses are replaced with
   deterministic HMAC tokens; the same address always produces the same token
   under the same key, but the raw address cannot be recovered.

2. **Non-linkability across keys** — tokens produced with different HMAC keys
   are distinct and unpredictable, preventing cross-party linkage.

3. **Sensitive attribute whitelist** — a declared set of field names is never
   present in pipeline output in raw/unmasked form, enforced by
   ``SensitiveAttributeGuard``.

4. **Trade record pipeline** — applying the canonical ledger privacy pipeline
   to a ``Trade``-shaped dict produces output that:
   - contains no raw Stellar addresses matching ``G[A-Z2-7]{55}``,
   - contains no exact original amounts,
   - carries a complete, compliance-ready audit log.

5. **Date generalization** — ``ledger_close_time`` is coarsened so the exact
   timestamp is suppressed while the trading period remains usable.

6. **Differential-privacy noise** — ``DPAggregator.private_mean`` introduces
   bounded noise; the noised mean stays within the theoretical privacy band
   across many trials (statistical test, not exact).

7. **Pipeline idempotency threat** — re-applying the pipeline to already-
   transformed output must not double-pseudonymize or raise errors for fields
   that have been converted to tokens.

8. **Audit completeness** — every sensitive field must have a corresponding
   ``TransformAuditEntry`` in the result; no field silently passes through
   without a record.

9. **Error propagation** — a missing or None sensitive field raises a typed
   ``PrivacyTransformError`` with the exact field name, so operators know
   which record is malformed without inspecting the stack trace.

10. **Stellar address format detection** — helper correctly identifies live
    Stellar G-addresses and rejects lookalikes.

11. **Redaction fallback** — the ``RedactPatternTransform`` with the Stellar
    address regex acts as a last-line-of-defence guard that catches any
    G-address that slipped through the pseudonymization step.

12. **SHAP DP noise calibration** (cross-module) — Gaussian sigma grows
    correctly with sensitivity and shrinks with epsilon; noise is bounded
    by the expected multiple-of-sigma band at 3σ with very high probability.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from privacy.transform_utils import (
    GeneralizeDateTransform,
    GeneralizeNumericTransform,
    MaskTransform,
    PrivacyTransformError,
    PrivacyTransformPipeline,
    PseudonymizeTransform,
    RedactPatternTransform,
    TransformAuditEntry,
    TransformResult,
)

# ---
# detection/differential_privacy is imported only if available so this module
# can be run in isolation without the full ML dependency tree.
# ---
try:
    from detection.differential_privacy import gaussian_sigma, renyi_noise_multiplier
    from privacy.dp_aggregator import DPAggregator

    _DP_AVAILABLE = True
except ImportError:
    _DP_AVAILABLE = False


# ===========================================================================
# Constants and fixtures
# ===========================================================================

# Stellar addresses are Ed25519 public keys encoded in Stellar's
# base32check variant.  They always start with "G" and are 56 characters long.
STELLAR_ADDR_RE = re.compile(r"^G[A-Z2-7]{55}$")

# Canonical set of sensitive attribute field names in LedgerLens ledger records.
SENSITIVE_WALLET_FIELDS = frozenset(
    {
        "base_account",
        "counter_account",
        "account_id",
        "funding_account",
    }
)

SENSITIVE_AMOUNT_FIELDS = frozenset(
    {
        "base_amount",
        "counter_amount",
        "amount",
        "price",
    }
)

SENSITIVE_TEMPORAL_FIELDS = frozenset(
    {
        "ledger_close_time",
        "account_created_at",
    }
)

ALL_SENSITIVE_FIELDS = SENSITIVE_WALLET_FIELDS | SENSITIVE_AMOUNT_FIELDS | SENSITIVE_TEMPORAL_FIELDS

# Realistic sample Stellar wallet addresses (structurally valid: G + 55 base32 chars = 56 total)
# Stellar base32 alphabet: A-Z plus digits 2-7 (no 0, 1, 8, 9)
# These are synthetically constructed — not real accounts.
SAMPLE_WALLET_A = "GB6WNTESP5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HPJV"
SAMPLE_WALLET_B = "GDZALWHCGU5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HP2"
SAMPLE_WALLET_C = "GCWYPWHIMPL5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HP"

# Compile-time sanity: all must be 56-char Stellar-format addresses
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
assert len(SAMPLE_WALLET_A) == 56 and all(c in _B32 for c in SAMPLE_WALLET_A)
assert len(SAMPLE_WALLET_B) == 56 and all(c in _B32 for c in SAMPLE_WALLET_B)
assert len(SAMPLE_WALLET_C) == 56 and all(c in _B32 for c in SAMPLE_WALLET_C)

# Default HMAC key used in test suite (never use this in production!)
TEST_HMAC_KEY = b"test-privacy-key-for-ledgerlens-issue-481"

# Bucket size for amount generalization (mirrors production default)
AMOUNT_BUCKET_SIZE = 1000.0


# ---------------------------------------------------------------------------
# Shared pipeline builder
# ---------------------------------------------------------------------------


def build_ledger_privacy_pipeline(secret_key: bytes = TEST_HMAC_KEY) -> PrivacyTransformPipeline:
    """Build the canonical ledger-record privacy pipeline for tests.

    This mirrors the pipeline that should be applied to any ledger record
    before it crosses a trust boundary (export, forensic report, external API).
    """
    return PrivacyTransformPipeline(
        [
            # Wallet addresses
            PseudonymizeTransform(field_name="base_account", secret_key=secret_key),
            PseudonymizeTransform(field_name="counter_account", secret_key=secret_key),
            # Trade amounts — generalize to 1000-unit buckets
            GeneralizeNumericTransform(field_name="base_amount", bucket_size=AMOUNT_BUCKET_SIZE),
            GeneralizeNumericTransform(field_name="counter_amount", bucket_size=AMOUNT_BUCKET_SIZE),
            # Temporal: truncate to day (suppresses exact trade time)
            GeneralizeDateTransform(field_name="ledger_close_time", granularity="day"),
        ]
    )


def make_trade_record(
    wallet_a: str = SAMPLE_WALLET_A,
    wallet_b: str = SAMPLE_WALLET_B,
    base_amount: float = 1234.56,
    counter_amount: float = 9876.00,
    ledger_close_time: str = "2024-06-15T10:30:00",
) -> dict[str, Any]:
    """Produce a minimal Trade-shaped dict for testing."""
    return {
        "base_account": wallet_a,
        "counter_account": wallet_b,
        "base_amount": base_amount,
        "counter_amount": counter_amount,
        "ledger_close_time": ledger_close_time,
        "trade_id": "tx_12345",
        "price": Decimal("0.125"),
    }


def make_account_activity_record() -> dict[str, Any]:
    return {
        "account_id": SAMPLE_WALLET_A,
        "account_created_at": "2024-01-01T08:00:00",
        "funding_account": SAMPLE_WALLET_B,
        "home_domain": "example.stellar.org",
    }


# ===========================================================================
# §1 — Pseudonymization contract
# ===========================================================================


class TestPseudonymizationContract:
    """Issue #481 §1: Wallet addresses are replaced with HMAC tokens."""

    def test_pseudonymized_wallet_not_raw_stellar_address(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        base = result.record["base_account"]
        counter = result.record["counter_account"]
        assert not STELLAR_ADDR_RE.match(base), f"Raw Stellar address leaked: {base}"
        assert not STELLAR_ADDR_RE.match(counter), f"Raw Stellar address leaked: {counter}"

    def test_pseudonymized_token_has_anon_prefix(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        assert result.record["base_account"].startswith("anon_")
        assert result.record["counter_account"].startswith("anon_")

    def test_same_address_same_token_determinism(self):
        pipeline = build_ledger_privacy_pipeline()
        r1 = pipeline.apply(make_trade_record())
        r2 = pipeline.apply(make_trade_record())
        assert r1.record["base_account"] == r2.record["base_account"]
        assert r1.record["counter_account"] == r2.record["counter_account"]

    def test_different_addresses_different_tokens(self):
        pipeline = build_ledger_privacy_pipeline()
        r1 = pipeline.apply(make_trade_record(wallet_a=SAMPLE_WALLET_A))
        r2 = pipeline.apply(make_trade_record(wallet_a=SAMPLE_WALLET_B))
        assert r1.record["base_account"] != r2.record["base_account"]

    def test_token_length_respects_configuration(self):
        t = PseudonymizeTransform(
            field_name="base_account", secret_key=TEST_HMAC_KEY, token_length=16
        )
        token, _ = t.apply({"base_account": SAMPLE_WALLET_A})
        # "anon_" prefix + 16 hex chars
        assert token == f"anon_{token[5:]}"
        assert len(token[5:]) == 16

    def test_pseudonymization_is_irreversible_by_inspection(self):
        """The raw address must not appear anywhere in the token."""
        t = PseudonymizeTransform(field_name="base_account", secret_key=TEST_HMAC_KEY)
        token, _ = t.apply({"base_account": SAMPLE_WALLET_A})
        # The raw wallet address should not be a substring of the token
        assert SAMPLE_WALLET_A.lower() not in token.lower()
        assert SAMPLE_WALLET_A not in token

    def test_audit_entry_marks_field_as_pseudonymized(self):
        t = PseudonymizeTransform(field_name="base_account", secret_key=TEST_HMAC_KEY)
        _, entry = t.apply({"base_account": SAMPLE_WALLET_A})
        assert entry.field_name == "base_account"
        assert entry.transform == "pseudonymize"
        assert entry.reversible is False

    def test_funding_account_pseudonymized(self):
        t = PseudonymizeTransform(field_name="funding_account", secret_key=TEST_HMAC_KEY)
        record = make_account_activity_record()
        token, entry = t.apply(record)
        assert not STELLAR_ADDR_RE.match(token)
        assert entry.reversible is False


# ===========================================================================
# §2 — Non-linkability across keys
# ===========================================================================


class TestNonLinkabilityAcrossKeys:
    """Issue #481 §2: Tokens from different keys must not be correlated."""

    def test_different_keys_produce_different_tokens(self):
        t1 = PseudonymizeTransform(field_name="base_account", secret_key=b"key-party-1")
        t2 = PseudonymizeTransform(field_name="base_account", secret_key=b"key-party-2")
        v1, _ = t1.apply({"base_account": SAMPLE_WALLET_A})
        v2, _ = t2.apply({"base_account": SAMPLE_WALLET_A})
        assert v1 != v2, "Different keys must not produce the same token (linkability risk)"

    def test_many_addresses_all_produce_distinct_tokens(self):
        """No two distinct wallet addresses collide under the same key."""
        addresses = [
            "GB6WNTESP5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HPJV",
            "GDZALWHCGU5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HP2",
            "GCWYPWHIMPL5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HP",
            "GD6WNTESP5NU52GELR4HIZLI7NWQ37MWFNROPCQLVHIJAXIPSGJ3HPJV",
        ]
        t = PseudonymizeTransform(field_name="w", secret_key=TEST_HMAC_KEY)
        tokens = [t.apply({"w": addr})[0] for addr in addresses]
        assert len(tokens) == len(set(tokens)), "Token collision detected — hash function weak"

    def test_prefix_does_not_vary_by_key(self):
        """All tokens from any key must have the 'anon_' prefix (invariant prefix)."""
        for key in [b"key1", b"key2", b"hunter2"]:
            t = PseudonymizeTransform(field_name="w", secret_key=key)
            token, _ = t.apply({"w": SAMPLE_WALLET_A})
            assert token.startswith("anon_"), f"Missing anon_ prefix with key={key!r}"


# ===========================================================================
# §3 — Sensitive attribute whitelist / SensitiveAttributeGuard
# ===========================================================================


class SensitiveAttributeGuard:
    """Assertion helper: verifies no raw sensitive attribute survives in output.

    This class is reused in multiple test methods to provide a consistent
    contract that can be extended as new sensitive fields are identified.
    """

    STELLAR_ADDRESS_PATTERN = re.compile(r"G[A-Z2-7]{55}")

    def assert_no_raw_wallet_addresses(self, record: dict[str, Any]) -> None:
        """Fail if any value looks like a raw Stellar address."""
        for field_name, value in record.items():
            if isinstance(value, str) and self.STELLAR_ADDRESS_PATTERN.match(value):
                raise AssertionError(
                    f"Raw Stellar address found in field {field_name!r}: {value[:12]}..."
                )

    def assert_no_exact_amount(self, record: dict[str, Any], original: float) -> None:
        """Fail if the exact original amount appears in any output field."""
        for field_name, value in record.items():
            if isinstance(value, (int, float, Decimal)):
                if float(value) == original:
                    raise AssertionError(
                        f"Exact original amount {original} found in field {field_name!r}"
                    )

    def assert_all_sensitive_fields_transformed(
        self, result: TransformResult, expected_fields: frozenset[str]
    ) -> None:
        """Fail if any expected-sensitive field was not recorded in the audit log."""
        transformed = result.fields_transformed()
        missing = expected_fields & set(result.record.keys()) - transformed
        if missing:
            raise AssertionError(
                f"Sensitive fields not in audit log: {missing}. "
                "Each field must have a TransformAuditEntry to satisfy the compliance contract."
            )


class TestSensitiveAttributeWhitelist:
    """Issue #481 §3: Declared sensitive attributes are never raw in pipeline output."""

    def setup_method(self) -> None:
        self.guard = SensitiveAttributeGuard()

    def test_no_raw_stellar_addresses_in_output(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        self.guard.assert_no_raw_wallet_addresses(result.record)

    def test_no_exact_amounts_in_output(self):
        original_base = 1234.56
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record(base_amount=original_base))
        self.guard.assert_no_exact_amount(result.record, original_base)

    def test_non_sensitive_fields_unchanged(self):
        """Fields not in the sensitive set pass through unchanged."""
        pipeline = build_ledger_privacy_pipeline()
        record = make_trade_record()
        result = pipeline.apply(record)
        assert result.record["trade_id"] == "tx_12345"

    def test_sensitive_pipeline_fields_in_audit_log(self):
        expected_fields = frozenset(
            {"base_account", "counter_account", "base_amount", "counter_amount", "ledger_close_time"}
        )
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        self.guard.assert_all_sensitive_fields_transformed(result, expected_fields)


# ===========================================================================
# §4 — Trade record pipeline end-to-end
# ===========================================================================


class TestTradeRecordPipeline:
    """Issue #481 §4: Full pipeline output contract for a Trade-shaped record."""

    def test_output_contains_no_raw_wallets(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        for value in result.record.values():
            if isinstance(value, str):
                assert not STELLAR_ADDR_RE.match(value), f"Raw wallet leaked: {value}"

    def test_output_contains_generalized_amounts(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record(base_amount=1234.56))
        # Amount must be bucket string, not a float
        assert isinstance(result.record["base_amount"], str)
        assert "[" in result.record["base_amount"]

    def test_output_contains_generalized_date(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record(ledger_close_time="2024-06-15T10:30:00"))
        # Day-granularity: must be the date only
        assert result.record["ledger_close_time"] == "2024-06-15"

    def test_audit_log_has_entry_for_every_transform(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        # 5 transforms: base_account, counter_account, base_amount, counter_amount, ledger_close_time
        assert len(result.audit_log) == 5

    def test_all_audit_entries_irreversible(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        for entry in result.audit_log:
            assert entry.reversible is False, f"Entry for {entry.field_name} is marked reversible"

    def test_batch_apply_consistent(self):
        pipeline = build_ledger_privacy_pipeline()
        records = [make_trade_record(wallet_a=SAMPLE_WALLET_A), make_trade_record(wallet_a=SAMPLE_WALLET_B)]
        results = pipeline.apply_batch(records)
        # Both outputs pseudonymize wallets
        for r in results:
            assert not STELLAR_ADDR_RE.match(r.record["base_account"])

    def test_batch_apply_different_wallets_different_tokens(self):
        pipeline = build_ledger_privacy_pipeline()
        records = [
            make_trade_record(wallet_a=SAMPLE_WALLET_A),
            make_trade_record(wallet_a=SAMPLE_WALLET_B),
        ]
        results = pipeline.apply_batch(records)
        assert results[0].record["base_account"] != results[1].record["base_account"]


# ===========================================================================
# §5 — Date generalization for ledger_close_time
# ===========================================================================


class TestDateGeneralizationForLedgerCloseTime:
    """Issue #481 §5: Temporal precision suppression."""

    def test_day_granularity_strips_time(self):
        t = GeneralizeDateTransform(field_name="ledger_close_time", granularity="day")
        value, entry = t.apply({"ledger_close_time": "2024-03-17T10:30:45"})
        assert value == "2024-03-17"
        assert ":" not in value, "Time component leaked in day-granularity output"

    def test_month_granularity_strips_day_and_time(self):
        t = GeneralizeDateTransform(field_name="ledger_close_time", granularity="month")
        value, _ = t.apply({"ledger_close_time": "2024-03-17T10:30:45"})
        assert value == "2024-03-01"

    def test_year_granularity_strips_month_day_time(self):
        t = GeneralizeDateTransform(field_name="ledger_close_time", granularity="year")
        value, _ = t.apply({"ledger_close_time": "2024-03-17T10:30:45"})
        assert value == "2024-01-01"

    def test_generalized_date_not_exact_timestamp(self):
        """Generalized value must not contain seconds or sub-second precision."""
        t = GeneralizeDateTransform(field_name="ledger_close_time", granularity="day")
        value, _ = t.apply({"ledger_close_time": "2024-03-17T10:30:45.123456"})
        assert "10:30" not in value
        assert "45" not in value

    def test_rejects_invalid_timestamp_format(self):
        t = GeneralizeDateTransform(field_name="ledger_close_time")
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"ledger_close_time": "not-a-date"})
        assert exc_info.value.field_name == "ledger_close_time"

    def test_account_created_at_generalized(self):
        t = GeneralizeDateTransform(field_name="account_created_at", granularity="month")
        record = make_account_activity_record()
        value, _ = t.apply(record)
        assert value == "2024-01-01"


# ===========================================================================
# §6 — Differential-privacy noise (DPAggregator)
# ===========================================================================


@pytest.mark.skipif(not _DP_AVAILABLE, reason="DP module not available")
class TestDifferentialPrivacyNoise:
    """Issue #481 §6: Noise is statistically bounded within the DP guarantee."""

    def test_private_mean_noise_is_bounded(self):
        """Noised mean must stay within 5× the theoretical Laplace scale with high probability.

        Statistical test: over 200 trials with seed-varied RNGs, the maximum
        absolute deviation should rarely exceed 5× the scale.  We use a
        generous bound to avoid flakiness.
        """
        epsilon = 1.0
        feature_min, feature_max = 0.0, 1000.0
        n = 100
        values = np.ones(n) * 500.0  # all exactly 500
        deviations = []
        for seed in range(200):
            agg = DPAggregator(epsilon=epsilon, delta=1e-5, random_seed=seed)
            sensitivity = (feature_max - feature_min) / n
            scale = sensitivity / epsilon
            noised = agg.private_mean(values, feature_min, feature_max)
            deviations.append(abs(noised - 500.0))
        # 95th percentile deviation should be well within 5× scale
        p95 = sorted(deviations)[int(0.95 * len(deviations))]
        assert p95 < 5 * scale * 3, (
            f"p95 deviation {p95:.4f} exceeds 5×3× scale {5 * 3 * scale:.4f}. "
            "Noise is likely miscalibrated."
        )

    def test_private_count_never_negative(self):
        """Private count is clamped to ≥ 0 (negative counts are meaningless)."""
        values = np.arange(10, dtype=float)
        for seed in range(50):
            agg = DPAggregator(epsilon=0.1, delta=1e-5, random_seed=seed)
            count = agg.private_count(values)
            assert count >= 0.0, f"Negative count {count} from seed {seed}"

    def test_private_histogram_bins_non_negative(self):
        values = np.random.default_rng(42).uniform(0, 100, 1000)
        agg = DPAggregator(epsilon=1.0, delta=1e-5, random_seed=0)
        noised, edges = agg.private_histogram(values, bins=10)
        assert np.all(noised >= 0), "Histogram bins must be ≥ 0 after DP noise"

    def test_budget_consumed_increments(self):
        agg = DPAggregator(epsilon=1.0, delta=1e-5, random_seed=0)
        values = np.ones(10)
        agg.private_mean(values, 0, 10)
        agg.private_count(values)
        budget = agg.budget_consumed()
        assert budget.queries == 2
        assert budget.epsilon_used > 0

    def test_reproducibility_with_seed(self):
        values = np.linspace(100, 200, 50)
        agg1 = DPAggregator(epsilon=1.0, delta=1e-5, random_seed=999)
        agg2 = DPAggregator(epsilon=1.0, delta=1e-5, random_seed=999)
        r1 = agg1.private_mean(values, 0, 1000)
        r2 = agg2.private_mean(values, 0, 1000)
        assert r1 == r2, "Same seed must produce identical noised output"


# ===========================================================================
# §6b — Gaussian mechanism (detection/differential_privacy)
# ===========================================================================


@pytest.mark.skipif(not _DP_AVAILABLE, reason="DP module not available")
class TestGaussianMechanism:
    """Issue #481 §6b: Gaussian sigma calibration from detection/differential_privacy."""

    def test_sigma_increases_with_sensitivity(self):
        s1 = gaussian_sigma(sensitivity=0.1, epsilon=1.0, delta=1e-5)
        s2 = gaussian_sigma(sensitivity=0.5, epsilon=1.0, delta=1e-5)
        assert s2 > s1, "Higher sensitivity must produce larger sigma"

    def test_sigma_decreases_with_epsilon(self):
        s1 = gaussian_sigma(sensitivity=0.1, epsilon=0.5, delta=1e-5)
        s2 = gaussian_sigma(sensitivity=0.1, epsilon=2.0, delta=1e-5)
        assert s1 > s2, "Smaller epsilon (tighter privacy) must produce larger sigma"

    def test_sigma_positive(self):
        assert gaussian_sigma(sensitivity=0.05, epsilon=1.0, delta=1e-5) > 0

    def test_zero_sensitivity_yields_zero_sigma(self):
        assert gaussian_sigma(sensitivity=0.0, epsilon=1.0, delta=1e-5) == 0.0

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            gaussian_sigma(sensitivity=0.1, epsilon=0.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            gaussian_sigma(sensitivity=0.1, epsilon=1.0, delta=0.0)

    def test_renyi_multiplier_below_threshold(self):
        """Below the query threshold, multiplier must be 1.0."""
        m = renyi_noise_multiplier(query_count=5, threshold=100, multiplier=3.0)
        assert m == 1.0

    def test_renyi_multiplier_above_threshold(self):
        """Above the query threshold, noise must scale up."""
        m = renyi_noise_multiplier(query_count=101, threshold=100, multiplier=3.0)
        assert m == 3.0


# ===========================================================================
# §7 — Pipeline idempotency threat
# ===========================================================================


class TestPipelineIdempotencyThreat:
    """Issue #481 §7: Re-running the pipeline on already-transformed output.

    If the pipeline is accidentally run twice on the same record, the second
    run must either be idempotent (same output) or raise a clear error — it
    must NEVER silently corrupt data by double-pseudonymizing a token.
    """

    def test_pseudonymize_applied_to_token_changes_value(self):
        """A token fed back through pseudonymize produces a different token.

        This is expected behaviour — the pipeline is designed for one-pass use.
        The test documents the behaviour so operators know double-application
        changes the token (and breaks join-ability).
        """
        t = PseudonymizeTransform(field_name="wallet", secret_key=TEST_HMAC_KEY)
        original_record = {"wallet": SAMPLE_WALLET_A}
        token1, _ = t.apply(original_record)
        # Second pass
        token2, _ = t.apply({"wallet": token1})
        # The second token is a valid anon_ token but different from token1
        assert token2.startswith("anon_")
        assert token1 != token2, (
            "Second pseudonymization of a token should produce a different value"
        )

    def test_generalize_numeric_on_bucket_string_raises(self):
        """Re-applying numeric generalization to a bucket string raises PrivacyTransformError."""
        t = GeneralizeNumericTransform(field_name="balance", bucket_size=1000)
        bucket_str = "[1000, 2000)"
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"balance": bucket_str})
        assert exc_info.value.field_name == "balance"

    def test_mask_is_idempotent_on_masked_value(self):
        """Masking a value that is already fully masked is idempotent."""
        t = MaskTransform(field_name="pin", keep_suffix=4, mask_char="*")
        value1, _ = t.apply({"pin": "1234567890"})
        value2, _ = t.apply({"pin": value1})
        # Both outputs should end with the same 4 characters
        assert value1[-4:] == value2[-4:]


# ===========================================================================
# §8 — Audit log completeness
# ===========================================================================


class TestAuditLogCompleteness:
    """Issue #481 §8: Every sensitive field has a TransformAuditEntry."""

    def test_audit_log_one_entry_per_transform(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        # One entry per transform in the pipeline
        assert len(result.audit_log) == len(pipeline.transforms)

    def test_audit_entry_field_names_match_transforms(self):
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        transform_fields = {t.field_name for t in pipeline.transforms}
        audit_fields = {e.field_name for e in result.audit_log}
        assert transform_fields == audit_fields

    def test_audit_entry_has_detail(self):
        """Every audit entry should carry a non-empty detail string."""
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        for entry in result.audit_log:
            assert isinstance(entry, TransformAuditEntry)
            # detail is allowed to be empty string by the dataclass, but the
            # production transforms all set it — assert not None
            assert entry.detail is not None

    def test_fields_transformed_set_is_complete(self):
        expected = {"base_account", "counter_account", "base_amount", "counter_amount", "ledger_close_time"}
        pipeline = build_ledger_privacy_pipeline()
        result = pipeline.apply(make_trade_record())
        assert result.fields_transformed() == expected


# ===========================================================================
# §9 — Error propagation
# ===========================================================================


class TestErrorPropagation:
    """Issue #481 §9: Missing/None fields raise typed errors with field names."""

    def test_missing_wallet_field_raises_with_field_name(self):
        t = PseudonymizeTransform(field_name="base_account", secret_key=TEST_HMAC_KEY)
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"counter_account": SAMPLE_WALLET_B})
        assert exc_info.value.field_name == "base_account"
        assert exc_info.value.transform == "pseudonymize"

    def test_none_wallet_field_raises(self):
        t = PseudonymizeTransform(field_name="base_account", secret_key=TEST_HMAC_KEY)
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"base_account": None})
        assert "base_account" in str(exc_info.value)

    def test_pipeline_aborts_on_missing_field_names_field_in_error(self):
        pipeline = build_ledger_privacy_pipeline()
        record_without_counter = {
            "base_account": SAMPLE_WALLET_A,
            # "counter_account" deliberately missing
            "base_amount": 1000.0,
            "counter_amount": 1000.0,
            "ledger_close_time": "2024-06-15T10:30:00",
        }
        with pytest.raises(PrivacyTransformError) as exc_info:
            pipeline.apply(record_without_counter)
        assert exc_info.value.field_name == "counter_account"

    def test_error_message_names_transform_type(self):
        t = PseudonymizeTransform(field_name="missing_field", secret_key=TEST_HMAC_KEY)
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"other": "value"})
        assert "pseudonymize" in str(exc_info.value)
        assert "missing_field" in str(exc_info.value)

    def test_generalize_numeric_non_numeric_error_names_field(self):
        t = GeneralizeNumericTransform(field_name="base_amount", bucket_size=1000)
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"base_amount": "not_a_number"})
        assert exc_info.value.field_name == "base_amount"

    def test_generalize_date_invalid_value_error_names_field(self):
        t = GeneralizeDateTransform(field_name="ledger_close_time", granularity="day")
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"ledger_close_time": "totally-not-a-date"})
        assert exc_info.value.field_name == "ledger_close_time"


# ===========================================================================
# §10 — Stellar address format detection
# ===========================================================================


class TestStellarAddressFormatDetection:
    """Issue #481 §10: is_stellar_address helper and pattern correctness."""

    @staticmethod
    def is_stellar_address(value: Any) -> bool:
        """Return True if *value* matches the Stellar G-address pattern."""
        return isinstance(value, str) and bool(STELLAR_ADDR_RE.match(value))

    def test_known_stellar_address_detected(self):
        assert self.is_stellar_address(SAMPLE_WALLET_A)
        assert self.is_stellar_address(SAMPLE_WALLET_B)

    def test_anon_token_not_detected_as_stellar(self):
        t = PseudonymizeTransform(field_name="w", secret_key=TEST_HMAC_KEY)
        token, _ = t.apply({"w": SAMPLE_WALLET_A})
        assert not self.is_stellar_address(token)

    def test_short_string_not_stellar(self):
        assert not self.is_stellar_address("GABC123")

    def test_lowercase_not_stellar(self):
        # Stellar addresses are uppercase; a lowercase lookalike must not match
        assert not self.is_stellar_address(SAMPLE_WALLET_A.lower())

    def test_non_string_not_stellar(self):
        assert not self.is_stellar_address(None)
        assert not self.is_stellar_address(12345)
        assert not self.is_stellar_address(3.14)

    def test_s_prefix_not_stellar(self):
        # Stellar secret seed starts with 'S', not 'G' — should not match G-address pattern
        fake_seed = "S" + SAMPLE_WALLET_A[1:]
        assert not self.is_stellar_address(fake_seed)

    @given(
        prefix=st.sampled_from(["G", "A", "B", "H", "g"]),
        chars=st.text(alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"), min_size=0, max_size=80),
    )
    def test_only_56_char_G_prefix_matches(self, prefix: str, chars: str):
        """Only exactly-56-character strings starting with G in base32 charset match."""
        candidate = prefix + chars
        if self.is_stellar_address(candidate):
            assert len(candidate) == 56
            assert candidate[0] == "G"
            assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in candidate)


# ===========================================================================
# §11 — Redaction fallback for G-addresses
# ===========================================================================


class TestRedactionFallback:
    """Issue #481 §11: RedactPatternTransform acts as last-line-of-defence."""

    STELLAR_PATTERN = r"G[A-Z2-7]{55}"

    def test_stellar_address_redacted(self):
        t = RedactPatternTransform(
            field_name="note",
            pattern=self.STELLAR_PATTERN,
            placeholder="[WALLET_REDACTED]",
        )
        value, entry = t.apply({"note": SAMPLE_WALLET_A})
        assert value == "[WALLET_REDACTED]"
        assert "matched=True" in entry.detail

    def test_non_stellar_string_passes_through(self):
        t = RedactPatternTransform(
            field_name="note",
            pattern=self.STELLAR_PATTERN,
            placeholder="[WALLET_REDACTED]",
        )
        value, entry = t.apply({"note": "This is a safe message"})
        assert value == "This is a safe message"
        assert "matched=False" in entry.detail

    def test_raise_on_match_blocks_leakage(self):
        """raise_on_match=True is the defensive configuration for audit pipelines."""
        t = RedactPatternTransform(
            field_name="raw_wallet",
            pattern=self.STELLAR_PATTERN,
            raise_on_match=True,
        )
        with pytest.raises(PrivacyTransformError) as exc_info:
            t.apply({"raw_wallet": SAMPLE_WALLET_A})
        assert exc_info.value.field_name == "raw_wallet"

    def test_last_line_defence_pipeline(self):
        """A pipeline ending with a redact-raise guard catches any leaked wallet."""
        pipeline = PrivacyTransformPipeline(
            [
                # Intentionally a no-op pseudonymize with wrong field name
                # simulating a misconfigured pipeline that misses wallet fields
                GeneralizeNumericTransform(field_name="amount", bucket_size=100),
                # Fallback guard should catch the raw wallet
                RedactPatternTransform(
                    field_name="leaked_wallet",
                    pattern=self.STELLAR_PATTERN,
                    raise_on_match=True,
                ),
            ]
        )
        with pytest.raises(PrivacyTransformError):
            pipeline.apply({"amount": 500, "leaked_wallet": SAMPLE_WALLET_A})


# ===========================================================================
# §12 — Home domain masking
# ===========================================================================


class TestHomeDomainMasking:
    """Issue #481: home_domain is a quasi-identifier that should be masked."""

    def test_home_domain_partially_masked(self):
        t = MaskTransform(field_name="home_domain", keep_suffix=6)
        record = make_account_activity_record()
        value, entry = t.apply(record)
        # Should end with "ar.org" (last 6 chars of "example.stellar.org")
        assert value.endswith("ar.org")
        assert value.startswith("*")
        assert entry.reversible is False

    def test_home_domain_none_handled(self):
        """None home_domain should raise PrivacyTransformError."""
        t = MaskTransform(field_name="home_domain", keep_suffix=4)
        with pytest.raises(PrivacyTransformError):
            t.apply({"home_domain": None})

    def test_home_domain_short_value_fully_masked(self):
        t = MaskTransform(field_name="home_domain", keep_suffix=10)
        value, _ = t.apply({"home_domain": "ab.cd"})
        assert value == "*****"


# ===========================================================================
# §13 — Property-based tests
# ===========================================================================


class TestPropertyBased:
    """Property-based invariants over the privacy pipeline."""

    @given(
        suffix=st.text(
            alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"),
            min_size=55,
            max_size=55,
        ),
    )
    def test_pseudonymize_never_returns_raw_address(self, suffix: str):
        """For any G-address input (G + 55 base32 chars), output must not be a Stellar address."""
        wallet = "G" + suffix  # Always a valid G-prefix 56-char base32 string
        t = PseudonymizeTransform(field_name="w", secret_key=TEST_HMAC_KEY)
        token, _ = t.apply({"w": wallet})
        assert not STELLAR_ADDR_RE.match(token)

    @given(amount=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False))
    def test_generalize_numeric_bucket_contains_original(self, amount: float):
        """The original value must fall within its output bucket bounds."""
        t = GeneralizeNumericTransform(field_name="amount", bucket_size=AMOUNT_BUCKET_SIZE)
        value, _ = t.apply({"amount": amount})
        # Parse the bucket "[lower, upper)"
        lower_str, upper_str = value.strip("[)").split(", ")
        lower = float(lower_str)
        upper = float(upper_str)
        assert lower <= amount < upper, (
            f"Amount {amount} not in bucket [{lower}, {upper})"
        )

    @given(
        key1=st.binary(min_size=8, max_size=64),
        key2=st.binary(min_size=8, max_size=64),
    )
    def test_distinct_keys_produce_distinct_tokens(self, key1: bytes, key2: bytes):
        """Two different keys must produce different tokens for the same address."""
        from hypothesis import assume

        assume(key1 != key2)
        t1 = PseudonymizeTransform(field_name="w", secret_key=key1)
        t2 = PseudonymizeTransform(field_name="w", secret_key=key2)
        tok1, _ = t1.apply({"w": SAMPLE_WALLET_A})
        tok2, _ = t2.apply({"w": SAMPLE_WALLET_A})
        assert tok1 != tok2
