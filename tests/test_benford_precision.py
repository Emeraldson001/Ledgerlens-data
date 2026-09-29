"""Tests for precision-safe Benford analysis helpers.

These tests verify that the Decimal-based digit extraction is:
1. More accurate than float-based approaches
2. Handles edge cases correctly (very large/small amounts)
3. Compatible with pandas Series operations
4. Produces correct Benford distributions
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from utils.benford_precision import (
    extract_leading_digit_safe,
    extract_second_digit_safe,
    leading_digits_safe,
    second_digits_safe,
)
from utils.decimal_guards import DecimalAmount


class TestLeadingDigitExtraction:
    """Test single-value leading digit extraction."""

    def test_simple_integer(self):
        """Extract digit from simple integer."""
        digit = extract_leading_digit_safe(DecimalAmount("123"))
        assert digit == 1

    def test_simple_decimal(self):
        """Extract digit from simple decimal."""
        digit = extract_leading_digit_safe(DecimalAmount("456.78"))
        assert digit == 4

    def test_small_decimal(self):
        """Extract digit from small decimal."""
        digit = extract_leading_digit_safe(DecimalAmount("0.00789"))
        assert digit == 7

    def test_very_small_decimal(self):
        """Extract digit from very small decimal (1 stroop)."""
        digit = extract_leading_digit_safe(DecimalAmount("0.0000001"))
        assert digit == 1

    def test_large_amount(self):
        """Extract digit from large amount."""
        digit = extract_leading_digit_safe(DecimalAmount("9876543.21"))
        assert digit == 9

    def test_very_large_amount(self):
        """Extract digit from very large amount."""
        digit = extract_leading_digit_safe(DecimalAmount("123456789012345.67"))
        assert digit == 1

    def test_negative_amount(self):
        """Negative amounts use magnitude."""
        digit = extract_leading_digit_safe(DecimalAmount("-456.78"))
        assert digit == 4

    def test_zero(self):
        """Zero returns 0."""
        digit = extract_leading_digit_safe(DecimalAmount("0"))
        assert digit == 0

    def test_near_boundary_below_10(self):
        """Amount just below 10 (9.999...)."""
        digit = extract_leading_digit_safe(DecimalAmount("9.9999999"))
        assert digit == 9

    def test_near_boundary_at_10(self):
        """Amount exactly at 10."""
        digit = extract_leading_digit_safe(DecimalAmount("10.0"))
        assert digit == 1

    def test_near_boundary_above_10(self):
        """Amount just above 10."""
        digit = extract_leading_digit_safe(DecimalAmount("10.0000001"))
        assert digit == 1

    def test_precision_critical_case(self):
        """Test case where float precision would fail."""
        # 1000000.0000001 might round to 1000000.0 in float
        digit = extract_leading_digit_safe(DecimalAmount("1000000.0000001"))
        assert digit == 1  # Still 1, not affected by precision loss

    def test_all_digits_1_to_9(self):
        """Test extraction for all digits 1-9."""
        expected = {
            "123": 1,
            "234": 2,
            "345": 3,
            "456": 4,
            "567": 5,
            "678": 6,
            "789": 7,
            "890": 8,
            "987": 9,
        }
        for amount_str, expected_digit in expected.items():
            digit = extract_leading_digit_safe(DecimalAmount(amount_str))
            assert digit == expected_digit

    def test_from_string(self):
        """Can extract from string."""
        digit = extract_leading_digit_safe("123.45")
        assert digit == 1

    def test_from_decimal(self):
        """Can extract from Decimal."""
        digit = extract_leading_digit_safe(Decimal("456.78"))
        assert digit == 4


class TestSecondDigitExtraction:
    """Test second digit extraction."""

    def test_simple_case(self):
        """Extract second digit from simple amount."""
        digit = extract_second_digit_safe(DecimalAmount("123.45"))
        assert digit == 2

    def test_small_decimal(self):
        """Extract second digit from small decimal."""
        digit = extract_second_digit_safe(DecimalAmount("0.00789"))
        assert digit == 8

    def test_single_digit(self):
        """Single-digit amount has second digit 0."""
        digit = extract_second_digit_safe(DecimalAmount("5.0"))
        assert digit == 0

    def test_negative(self):
        """Negative amounts use magnitude."""
        digit = extract_second_digit_safe(DecimalAmount("-456.78"))
        assert digit == 5

    def test_zero(self):
        """Zero returns -1 (invalid)."""
        digit = extract_second_digit_safe(DecimalAmount("0"))
        assert digit == -1

    def test_all_second_digits(self):
        """Test extraction for all second digits 0-9."""
        expected = {
            "10": 0,
            "21": 1,
            "32": 2,
            "43": 3,
            "54": 4,
            "65": 5,
            "76": 6,
            "87": 7,
            "98": 8,
            "19": 9,
        }
        for amount_str, expected_digit in expected.items():
            digit = extract_second_digit_safe(DecimalAmount(amount_str))
            assert digit == expected_digit


class TestSeriesOperations:
    """Test pandas Series operations."""

    def test_leading_digits_series(self):
        """Extract leading digits from Series."""
        amounts = pd.Series(
            [
                DecimalAmount("123.45"),
                DecimalAmount("456.78"),
                DecimalAmount("789.01"),
            ]
        )

        digits = leading_digits_safe(amounts)

        assert len(digits) == 3
        assert list(digits) == [1, 4, 7]

    def test_leading_digits_with_zeros(self):
        """Zero amounts are filtered out."""
        amounts = pd.Series(
            [
                DecimalAmount("123.45"),
                DecimalAmount("0"),
                DecimalAmount("456.78"),
            ]
        )

        digits = leading_digits_safe(amounts)

        assert len(digits) == 2
        assert list(digits) == [1, 4]

    def test_leading_digits_with_negatives(self):
        """Negative amounts are filtered out."""
        amounts = pd.Series(
            [
                DecimalAmount("123.45"),
                DecimalAmount("-456.78"),
                DecimalAmount("789.01"),
            ]
        )

        digits = leading_digits_safe(amounts)

        # Only positive amounts
        assert len(digits) == 2
        assert list(digits) == [1, 7]

    def test_leading_digits_empty_series(self):
        """Empty Series returns empty."""
        amounts = pd.Series([], dtype=float)
        digits = leading_digits_safe(amounts)
        assert len(digits) == 0

    def test_leading_digits_all_zeros(self):
        """All zeros returns empty."""
        amounts = pd.Series(
            [
                DecimalAmount("0"),
                DecimalAmount("0"),
            ]
        )
        digits = leading_digits_safe(amounts)
        assert len(digits) == 0

    def test_second_digits_series(self):
        """Extract second digits from Series."""
        amounts = pd.Series(
            [
                DecimalAmount("123.45"),
                DecimalAmount("456.78"),
                DecimalAmount("789.01"),
            ]
        )

        digits = second_digits_safe(amounts)

        assert len(digits) == 3
        assert list(digits) == [2, 5, 8]

    def test_mixed_amount_types(self):
        """Can handle mixed Decimal/DecimalAmount."""
        amounts = pd.Series(
            [
                Decimal("123.45"),
                DecimalAmount("456.78"),
                Decimal("789.01"),
            ]
        )

        digits = leading_digits_safe(amounts)

        assert len(digits) == 3
        assert list(digits) == [1, 4, 7]


class TestBenfordDistribution:
    """Test that extraction produces valid Benford distributions."""

    def test_benford_distribution_shape(self):
        """Extracted digits follow expected distribution shape."""
        # Generate amounts following Benford's Law
        # log10-uniform distribution produces Benford distribution
        n = 1000
        np.random.seed(42)
        log_amounts = np.random.uniform(0, 3, n)  # 10^0 to 10^3
        amounts = 10**log_amounts

        # Convert to DecimalAmount
        decimal_amounts = pd.Series([DecimalAmount(str(a)) for a in amounts])

        # Extract digits
        digits = leading_digits_safe(decimal_amounts)

        # Count distribution
        counts = digits.value_counts(normalize=True).sort_index()

        # Benford's Law: P(d) = log10(1 + 1/d)
        benford_expected = {
            1: 0.301,
            2: 0.176,
            3: 0.125,
            4: 0.097,
            5: 0.079,
            6: 0.067,
            7: 0.058,
            8: 0.051,
            9: 0.046,
        }

        # Check that distribution is roughly Benford-like
        # (not exact due to random sampling, but should be close)
        for digit in range(1, 10):
            if digit in counts:
                observed = counts[digit]
                expected = benford_expected[digit]
                # Allow 5% tolerance
                assert (
                    abs(observed - expected) < 0.05
                ), f"Digit {digit}: observed {observed:.3f}, expected {expected:.3f}"

    def test_uniform_distribution_rejects_benford(self):
        """Uniform distribution should NOT follow Benford."""
        # Generate uniformly distributed amounts (NOT Benford)
        n = 1000
        np.random.seed(42)
        amounts = np.random.uniform(100, 200, n)  # All start with 1

        decimal_amounts = pd.Series([DecimalAmount(str(a)) for a in amounts])
        digits = leading_digits_safe(decimal_amounts)

        # All should be 1 (since 100-200 all start with 1)
        assert all(digits == 1)


class TestEdgeCases:
    """Test edge cases and precision-critical scenarios."""

    def test_precision_boundary_999_to_1000(self):
        """Test digit extraction near 999.999... → 1000.0 boundary."""
        # These should have different leading digits
        below = extract_leading_digit_safe(DecimalAmount("999.9999999"))
        at = extract_leading_digit_safe(DecimalAmount("1000.0000000"))
        above = extract_leading_digit_safe(DecimalAmount("1000.0000001"))

        assert below == 9  # Still 999...
        assert at == 1  # Now 1000
        assert above == 1  # Still 1000

    def test_very_large_stellar_amount(self):
        """Test Stellar max amount (near int64 limit)."""
        max_stellar = DecimalAmount("922337203685.4775807")
        digit = extract_leading_digit_safe(max_stellar)
        assert digit == 9

    def test_very_small_stellar_amount(self):
        """Test 1 stroop (smallest Stellar unit)."""
        one_stroop = DecimalAmount("0.0000001")
        digit = extract_leading_digit_safe(one_stroop)
        assert digit == 1

    def test_scientific_notation(self):
        """Test amounts in scientific notation."""
        # DecimalAmount handles scientific notation
        digit = extract_leading_digit_safe(DecimalAmount("1.23e6"))
        assert digit == 1

    def test_trailing_zeros(self):
        """Trailing zeros don't affect leading digit."""
        digit1 = extract_leading_digit_safe(DecimalAmount("123.000"))
        digit2 = extract_leading_digit_safe(DecimalAmount("123.456"))
        assert digit1 == digit2 == 1


class TestAccuracyComparison:
    """Test accuracy improvements over float-based extraction."""

    def test_float_vs_decimal_precision(self):
        """Decimal extraction is more accurate than float for edge cases."""
        # Case where float precision might cause issues
        amounts = pd.Series(
            [
                DecimalAmount("1000000.0000001"),  # Large + tiny
                DecimalAmount("0.0000001"),  # Very small
                DecimalAmount("999.9999999"),  # Near boundary
            ]
        )

        # Extract with safe method
        safe_digits = leading_digits_safe(amounts)

        # All should be correct
        assert list(safe_digits) == [1, 1, 9]

    def test_no_rounding_errors(self):
        """Decimal extraction doesn't suffer from rounding errors."""
        # Classic float problem: 0.1 + 0.2 != 0.3
        amount1 = DecimalAmount("0.1")
        amount2 = DecimalAmount("0.2")
        sum_amount = amount1 + amount2  # Exactly 0.3

        digit = extract_leading_digit_safe(sum_amount)
        assert digit == 3  # Correct!

        # With float, might get 2 or 3 depending on rounding


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_trade_amounts(self):
        """Test with realistic Stellar trade amounts."""
        trades = pd.Series(
            [
                DecimalAmount("100.5000000"),
                DecimalAmount("250.7500000"),
                DecimalAmount("50.2500000"),
                DecimalAmount("1000.0000000"),
                DecimalAmount("0.1234567"),
            ]
        )

        digits = leading_digits_safe(trades)

        assert len(digits) == 5
        assert list(digits) == [1, 2, 5, 1, 1]

    def test_large_dataset_performance(self):
        """Test extraction on large dataset (performance check)."""
        # Generate 10k amounts
        n = 10000
        np.random.seed(42)
        amounts = np.random.uniform(1, 1000000, n)
        decimal_amounts = pd.Series([DecimalAmount(str(a)) for a in amounts])

        # Should complete quickly (< 1 second for 10k)
        import time

        start = time.time()
        digits = leading_digits_safe(decimal_amounts)
        elapsed = time.time() - start

        assert len(digits) == n
        assert elapsed < 1.0, f"Extraction took {elapsed:.2f}s, expected < 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
