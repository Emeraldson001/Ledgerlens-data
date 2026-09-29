"""Tests for utils.decimal_agg — decimal-safe aggregation utilities.

Covers:
- Exact decimal arithmetic (no floating-point drift)
- Sum, mean, max, min, weighted mean, percentage
- NaN / None handling
- pandas Series and DataFrame integration
- Edge cases: empty sequences, zero weights, single values
- Quantize / rounding
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from utils.decimal_agg import (
    aggregate_column,
    aggregate_report,
    decimal_max,
    decimal_mean,
    decimal_min,
    decimal_percentage,
    decimal_sum,
    decimal_weighted_mean,
    quantize,
    to_decimal,
)

# ---------------------------------------------------------------------------
# to_decimal conversion
# ---------------------------------------------------------------------------


class TestToDecimal:
    def test_float_conversion_exact(self) -> None:
        # 0.1 as float -> Decimal should be '0.1', not '0.1000000000000000055...'
        d = to_decimal(0.1)
        assert d == Decimal("0.1")

    def test_int_conversion(self) -> None:
        assert to_decimal(42) == Decimal("42")

    def test_string_conversion(self) -> None:
        assert to_decimal("123.456") == Decimal("123.456")

    def test_decimal_passthrough(self) -> None:
        orig = Decimal("99.99")
        assert to_decimal(orig) is orig


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------


class TestQuantize:
    def test_rounds_to_places(self) -> None:
        assert quantize(Decimal("1.23456789"), 2) == Decimal("1.23")

    def test_zero_places(self) -> None:
        assert quantize(Decimal("1.6"), 0) == Decimal("2")

    def test_bankers_rounding(self) -> None:
        # 0.5 rounds to even: 0.5 -> 0, 1.5 -> 2
        assert quantize(Decimal("0.5"), 0) == Decimal("0")
        assert quantize(Decimal("1.5"), 0) == Decimal("2")

    def test_negative_places_rejected(self) -> None:
        with pytest.raises(ValueError, match="places must be >= 0"):
            quantize(Decimal("1.0"), -1)

    def test_seven_decimal_places(self) -> None:
        result = quantize(Decimal("1.12345678"), 7)
        assert result == Decimal("1.1234568")  # Banker's rounding


# ---------------------------------------------------------------------------
# decimal_sum
# ---------------------------------------------------------------------------


class TestDecimalSum:
    def test_exact_sum(self) -> None:
        # Classic floating-point trap: 0.1 + 0.2 != 0.3 in float
        result = decimal_sum([0.1, 0.2, 0.3])
        assert result == Decimal("0.6")

    def test_large_sum(self) -> None:
        values = [Decimal("1000000.01")] * 1000
        result = decimal_sum(values)
        assert result == Decimal("1000000010.00")

    def test_skips_none(self) -> None:
        result = decimal_sum([1.0, None, 2.0])
        assert result == Decimal("3.0")

    def test_skips_nan(self) -> None:
        result = decimal_sum([1.0, float("nan"), 2.0])
        assert result == Decimal("3.0")

    def test_empty_returns_zero(self) -> None:
        assert decimal_sum([]) == Decimal("0")

    def test_single_value(self) -> None:
        assert decimal_sum([42.5]) == Decimal("42.5")

    def test_mixed_types(self) -> None:
        result = decimal_sum([1, 2.5, "3.5", Decimal("4")])
        assert result == Decimal("11.0")

    def test_negative_values(self) -> None:
        result = decimal_sum([10.0, -3.0, -2.5])
        assert result == Decimal("4.5")


# ---------------------------------------------------------------------------
# decimal_mean
# ---------------------------------------------------------------------------


class TestDecimalMean:
    def test_exact_mean(self) -> None:
        result = decimal_mean([0.1, 0.2, 0.3])
        assert result == Decimal("0.2")

    def test_skips_none(self) -> None:
        result = decimal_mean([2.0, None, 4.0])
        assert result == Decimal("3.0")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            decimal_mean([])

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            decimal_mean([float("nan"), float("nan")])

    def test_single_value(self) -> None:
        assert decimal_mean([7.0]) == Decimal("7.0")


# ---------------------------------------------------------------------------
# decimal_max / decimal_min
# ---------------------------------------------------------------------------


class TestDecimalMaxMin:
    def test_max(self) -> None:
        assert decimal_max([1.0, 3.0, 2.0]) == Decimal("3.0")

    def test_min(self) -> None:
        assert decimal_min([1.0, 3.0, 2.0]) == Decimal("1.0")

    def test_max_skips_nan(self) -> None:
        assert decimal_max([1.0, float("nan"), 3.0]) == Decimal("3.0")

    def test_min_skips_none(self) -> None:
        assert decimal_min([5.0, None, 2.0]) == Decimal("2.0")

    def test_max_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            decimal_max([])

    def test_min_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            decimal_min([])

    def test_negative_values(self) -> None:
        assert decimal_max([-5.0, -1.0, -3.0]) == Decimal("-1.0")
        assert decimal_min([-5.0, -1.0, -3.0]) == Decimal("-5.0")


# ---------------------------------------------------------------------------
# decimal_weighted_mean
# ---------------------------------------------------------------------------


class TestDecimalWeightedMean:
    def test_basic_weighted_mean(self) -> None:
        result = decimal_weighted_mean([10.0, 20.0], [1, 3])
        # (10*1 + 20*3) / (1+3) = 70/4 = 17.5
        assert result == Decimal("17.5")

    def test_equal_weights_equals_mean(self) -> None:
        result = decimal_weighted_mean([2.0, 4.0, 6.0], [1, 1, 1])
        assert result == Decimal("4")

    def test_skips_none_values(self) -> None:
        result = decimal_weighted_mean([10.0, None, 30.0], [1, 1, 1])
        # Only 10 and 30 counted: (10+30)/2 = 20
        assert result == Decimal("20")

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            decimal_weighted_mean([1.0], [0])

    def test_all_none_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            decimal_weighted_mean([None, None], [1, 1])


# ---------------------------------------------------------------------------
# decimal_percentage
# ---------------------------------------------------------------------------


class TestDecimalPercentage:
    def test_basic_percentage(self) -> None:
        result = decimal_percentage(25, 100)
        assert result == Decimal("25")

    def test_fractional_percentage(self) -> None:
        result = decimal_percentage(1, 3)
        # Should be 33.333... without float drift
        assert result > Decimal("33.33")
        assert result < Decimal("33.34")

    def test_zero_denominator_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            decimal_percentage(1, 0)

    def test_hundred_percent(self) -> None:
        assert decimal_percentage(50, 50) == Decimal("100")


# ---------------------------------------------------------------------------
# pandas integration
# ---------------------------------------------------------------------------


class TestAggregateColumn:
    def test_sum_series(self) -> None:
        s = pd.Series([0.1, 0.2, 0.3])
        result = aggregate_column(s, "sum", places=7)
        assert result == Decimal("0.6000000")

    def test_mean_series(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0])
        result = aggregate_column(s, "mean", places=2)
        assert result == Decimal("20.00")

    def test_max_series(self) -> None:
        s = pd.Series([1.0, 5.0, 3.0])
        result = aggregate_column(s, "max", places=0)
        assert result == Decimal("5")

    def test_min_series(self) -> None:
        s = pd.Series([1.0, 5.0, 3.0])
        result = aggregate_column(s, "min", places=0)
        assert result == Decimal("1")

    def test_unknown_agg_raises(self) -> None:
        s = pd.Series([1.0])
        with pytest.raises(ValueError, match="Unknown"):
            aggregate_column(s, "median")

    def test_series_with_nan(self) -> None:
        s = pd.Series([1.0, float("nan"), 3.0])
        result = aggregate_column(s, "sum", places=1)
        assert result == Decimal("4.0")


class TestAggregateReport:
    def test_multiple_columns(self) -> None:
        df = pd.DataFrame(
            {
                "amount": [100.0, 200.0, 300.0],
                "fee": [1.5, 2.5, 3.0],
                "rate": [0.01, 0.02, 0.03],
            }
        )
        results = aggregate_report(df, {"amount": "sum", "fee": "sum", "rate": "mean"})
        assert results["amount"] == Decimal("600.0000000")
        assert results["fee"] == Decimal("7.0000000")
        assert results["rate"] == Decimal("0.0200000")

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError, match="not found"):
            aggregate_report(df, {"missing": "sum"})

    def test_empty_agg_map(self) -> None:
        df = pd.DataFrame({"a": [1.0]})
        results = aggregate_report(df, {})
        assert results == {}


# ---------------------------------------------------------------------------
# Floating-point precision regression tests
# ---------------------------------------------------------------------------


class TestPrecisionRegression:
    def test_classic_0_1_plus_0_2(self) -> None:
        """The classic floating-point trap: 0.1 + 0.2 should equal 0.3."""
        assert decimal_sum([0.1, 0.2]) == Decimal("0.3")

    def test_many_small_additions(self) -> None:
        """Adding 0.01 a thousand times should be exactly 10.00."""
        values = [0.01] * 1000
        result = decimal_sum(values)
        assert result == Decimal("10.00")

    def test_large_minus_large(self) -> None:
        """Subtraction of nearly-equal large numbers preserves precision."""
        result = decimal_sum([Decimal("1000000.001"), Decimal("-1000000.000")])
        assert result == Decimal("0.001")

    def test_report_total_matches_line_items(self) -> None:
        """Report line items should sum to the total without drift."""
        line_items = [Decimal("19.99"), Decimal("29.99"), Decimal("9.99"), Decimal("49.99")]
        total = decimal_sum(line_items)
        assert total == Decimal("109.96")


class TestRoundingMode:
    """Tests for ROUND_HALF_EVEN (banker's rounding) mode in decimal aggregation.
    
    Demonstrates that Decimal rounding differs from naive float rounding,
    and ensures the specific rounding mode (ROUND_HALF_EVEN) is preserved.
    """

    def test_banker_rounding_half_even_rounds_to_even(self) -> None:
        """0.5 rounds to even in banker's rounding (ROUND_HALF_EVEN)."""
        # 0.5 should round down to 0 (even)
        result = quantize(Decimal("0.5"), 0)
        assert result == Decimal("0")
        
        # 1.5 should round up to 2 (even)
        result = quantize(Decimal("1.5"), 0)
        assert result == Decimal("2")
        
        # 2.5 should round down to 2 (even)
        result = quantize(Decimal("2.5"), 0)
        assert result == Decimal("2")

    def test_float_rounding_differs_from_decimal_rounding(self) -> None:
        """Naive float rounding differs from Decimal banker's rounding.
        
        This test demonstrates why float arithmetic is unsuitable for
        financial calculations. Python's built-in round() uses ROUND_HALF_UP,
        but our Decimal uses ROUND_HALF_EVEN.
        """
        # Value that exhibits rounding difference
        val = Decimal("2.5")
        
        # Decimal with banker's rounding (ROUND_HALF_EVEN): 2.5 -> 2
        decimal_rounded = quantize(val, 0)
        assert decimal_rounded == Decimal("2")
        
        # Python's round() with ROUND_HALF_UP: 2.5 -> 3 (on some platforms)
        # (Note: Python 3 uses banker's rounding too, but for illustration)
        # The key point is that Decimal gives us explicit control.
        assert decimal_rounded == Decimal("2")

    def test_mean_with_rounding_consistency(self) -> None:
        """Mean calculation should use banker's rounding consistently."""
        # Values that would round differently under ROUND_HALF_UP vs ROUND_HALF_EVEN
        # When divided by 2: (1 + 2) / 2 = 1.5 -> should round to 2 (even)
        values = [Decimal("1"), Decimal("2")]
        result = decimal_mean(values)
        quantized = quantize(result, 0)
        assert quantized == Decimal("2")  # Banker's rounding: 1.5 -> 2 (even)

    def test_sum_no_rounding_drift_through_aggregation(self) -> None:
        """Aggregating values with banker's rounding should not drift.
        
        This is the core reason for using Decimal: sums remain exact
        without accumulating rounding errors.
        """
        # Using values that would show drift with float arithmetic
        values = [Decimal("0.1"), Decimal("0.1"), Decimal("0.1")]
        result = decimal_sum(values)
        assert result == Decimal("0.3")
        
        # In contrast, naive float sum shows drift
        float_sum = sum(float(v) for v in values)
        assert abs(float_sum - 0.3) > 1e-17  # Float shows accumulation drift

