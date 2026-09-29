"""Comprehensive test suite for numeric precision guards.

Tests cover:
- DecimalAmount arithmetic operations
- Stellar stroops conversion
- Amount validation with bounds checking
- Precision context management
- Edge cases (overflow, underflow, division by zero)
- Float conversion warnings
- Comparison operators
- Property-based tests for arithmetic properties
"""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from utils.decimal_guards import (
    STELLAR_MAX_AMOUNT,
    STELLAR_MIN_AMOUNT,
    STELLAR_PRECISION,
    AmountValidationError,
    DecimalAmount,
    StroopsConversionError,
    check_precision_loss,
    decimal_context,
    safe_divide,
    safe_float_to_decimal,
    sum_amounts,
    validate_amount,
    validate_stellar_amount,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_amounts():
    """Provide sample DecimalAmount instances for testing."""
    return {
        "small": DecimalAmount("0.0000001"),  # 1 stroop
        "medium": DecimalAmount("100.5"),
        "large": DecimalAmount("1000000.1234567"),
        "zero": DecimalAmount("0"),
        "negative": DecimalAmount("-50.25"),
    }


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Test amount validation functions."""

    def test_validate_amount_from_string(self):
        """Valid string converts to Decimal."""
        result = validate_amount("123.45")
        assert result == Decimal("123.45")
        assert isinstance(result, Decimal)

    def test_validate_amount_from_int(self):
        """Valid integer converts to Decimal."""
        result = validate_amount(100)
        assert result == Decimal("100")

    def test_validate_amount_from_decimal(self):
        """Decimal passes through unchanged."""
        decimal_value = Decimal("123.45")
        result = validate_amount(decimal_value)
        assert result == decimal_value
        assert result is decimal_value  # Same object

    def test_validate_amount_from_float_warns(self):
        """Float conversion triggers warning."""
        with pytest.warns(UserWarning, match="Converting float"):
            result = validate_amount(123.45)
            assert isinstance(result, Decimal)

    def test_validate_amount_rejects_nan(self):
        """NaN is rejected."""
        with pytest.raises(AmountValidationError, match="NaN"):
            validate_amount(Decimal("NaN"))

    def test_validate_amount_rejects_infinity(self):
        """Infinity is rejected."""
        with pytest.raises(AmountValidationError, match="Infinity"):
            validate_amount(Decimal("Infinity"))

    def test_validate_amount_rejects_negative_when_not_allowed(self):
        """Negative values rejected when allow_negative=False."""
        with pytest.raises(AmountValidationError, match="negative"):
            validate_amount("-10", allow_negative=False)

    def test_validate_amount_allows_negative_when_enabled(self):
        """Negative values allowed when allow_negative=True."""
        result = validate_amount("-10", allow_negative=True)
        assert result == Decimal("-10")

    def test_validate_amount_enforces_min_value(self):
        """Minimum value is enforced."""
        with pytest.raises(AmountValidationError, match="< minimum"):
            validate_amount("5", min_value="10")

    def test_validate_amount_enforces_max_value(self):
        """Maximum value is enforced."""
        with pytest.raises(AmountValidationError, match="> maximum"):
            validate_amount("15", max_value="10")

    def test_validate_amount_within_bounds(self):
        """Value within bounds is accepted."""
        result = validate_amount("50", min_value="0", max_value="100")
        assert result == Decimal("50")

    def test_validate_stellar_amount_valid(self):
        """Valid Stellar amount is accepted."""
        result = validate_stellar_amount("100.5000000")
        assert result == Decimal("100.5000000")

    def test_validate_stellar_amount_max_bound(self):
        """Stellar maximum bound is enforced."""
        with pytest.raises(AmountValidationError, match="> maximum"):
            validate_stellar_amount("922337203686")  # Just over max

    def test_validate_stellar_amount_min_bound(self):
        """Stellar minimum bound is enforced."""
        with pytest.raises(AmountValidationError, match="< minimum"):
            validate_stellar_amount("-922337203686")  # Just under min

    def test_validate_stellar_amount_too_many_decimals(self):
        """More than 7 decimal places is rejected."""
        with pytest.raises(AmountValidationError, match="more than 7 decimal"):
            validate_stellar_amount("100.12345678")  # 8 decimal places

    def test_validate_stellar_amount_exactly_seven_decimals(self):
        """Exactly 7 decimal places is accepted."""
        result = validate_stellar_amount("100.1234567")
        assert result == Decimal("100.1234567")


# ---------------------------------------------------------------------------
# DecimalAmount Arithmetic Tests
# ---------------------------------------------------------------------------


class TestDecimalAmountArithmetic:
    """Test arithmetic operations on DecimalAmount."""

    def test_addition(self):
        """Addition works correctly."""
        a = DecimalAmount("100.5")
        b = DecimalAmount("25.25")
        result = a + b
        assert result == DecimalAmount("125.75")
        assert isinstance(result, DecimalAmount)

    def test_addition_with_string(self):
        """Can add string values."""
        a = DecimalAmount("100")
        result = a + "50"
        assert result == DecimalAmount("150")

    def test_reverse_addition(self):
        """Reverse addition (radd) works."""
        a = DecimalAmount("100")
        result = 50 + a
        assert result == DecimalAmount("150")

    def test_subtraction(self):
        """Subtraction works correctly."""
        a = DecimalAmount("100.5")
        b = DecimalAmount("25.25")
        result = a - b
        assert result == DecimalAmount("75.25")

    def test_multiplication(self):
        """Multiplication works correctly."""
        a = DecimalAmount("10.5")
        b = DecimalAmount("2")
        result = a * b
        assert result == DecimalAmount("21")

    def test_division(self):
        """Division works correctly."""
        a = DecimalAmount("100")
        b = DecimalAmount("4")
        result = a / b
        assert result == DecimalAmount("25")

    def test_division_by_zero_raises(self):
        """Division by zero raises ZeroDivisionError."""
        a = DecimalAmount("100")
        b = DecimalAmount("0")
        with pytest.raises(ZeroDivisionError):
            _ = a / b

    def test_floor_division(self):
        """Floor division works correctly."""
        a = DecimalAmount("10")
        b = DecimalAmount("3")
        result = a // b
        assert result == DecimalAmount("3")

    def test_modulo(self):
        """Modulo operation works correctly."""
        a = DecimalAmount("10")
        b = DecimalAmount("3")
        result = a % b
        assert result == DecimalAmount("1")

    def test_power(self):
        """Power operation works correctly."""
        a = DecimalAmount("2")
        result = a**3
        assert result == DecimalAmount("8")

    def test_negation(self):
        """Negation works correctly."""
        a = DecimalAmount("100")
        result = -a
        assert result == DecimalAmount("-100")

    def test_absolute_value(self):
        """Absolute value works correctly."""
        a = DecimalAmount("-100")
        result = abs(a)
        assert result == DecimalAmount("100")

    def test_arithmetic_precision(self):
        """Arithmetic maintains precision (no float errors)."""
        # Classic float precision problem: 0.1 + 0.2 != 0.3
        a = DecimalAmount("0.1")
        b = DecimalAmount("0.2")
        result = a + b
        assert result == DecimalAmount("0.3")  # Exact!


# ---------------------------------------------------------------------------
# DecimalAmount Comparison Tests
# ---------------------------------------------------------------------------


class TestDecimalAmountComparison:
    """Test comparison operations on DecimalAmount."""

    def test_equality(self):
        """Equality comparison works."""
        a = DecimalAmount("100")
        b = DecimalAmount("100")
        assert a == b

    def test_inequality(self):
        """Inequality comparison works."""
        a = DecimalAmount("100")
        b = DecimalAmount("50")
        assert a != b

    def test_less_than(self):
        """Less than comparison works."""
        a = DecimalAmount("50")
        b = DecimalAmount("100")
        assert a < b

    def test_greater_than(self):
        """Greater than comparison works."""
        a = DecimalAmount("100")
        b = DecimalAmount("50")
        assert a > b

    def test_less_than_or_equal(self):
        """Less than or equal comparison works."""
        a = DecimalAmount("100")
        b = DecimalAmount("100")
        c = DecimalAmount("50")
        assert a <= b
        assert c <= b

    def test_greater_than_or_equal(self):
        """Greater than or equal comparison works."""
        a = DecimalAmount("100")
        b = DecimalAmount("100")
        c = DecimalAmount("150")
        assert a >= b
        assert c >= b

    def test_comparison_with_string(self):
        """Can compare with string values."""
        a = DecimalAmount("100")
        assert a == "100"
        assert a < "200"
        assert a > "50"

    def test_hash_consistency(self):
        """Hash is consistent for equal values."""
        a = DecimalAmount("100")
        b = DecimalAmount("100")
        assert hash(a) == hash(b)

    def test_can_use_in_set(self):
        """DecimalAmount can be used in sets."""
        amounts = {DecimalAmount("100"), DecimalAmount("200"), DecimalAmount("100")}
        assert len(amounts) == 2  # Duplicate removed


# ---------------------------------------------------------------------------
# Stellar Stroops Conversion Tests
# ---------------------------------------------------------------------------


class TestStroopsConversion:
    """Test Stellar stroops conversion."""

    def test_to_stroops_basic(self):
        """Basic stroops conversion works."""
        amount = DecimalAmount("100.5000000")
        stroops = amount.to_stroops()
        assert stroops == 1005000000
        assert isinstance(stroops, int)

    def test_to_stroops_small_amount(self):
        """Small amount (1 stroop) converts correctly."""
        amount = DecimalAmount("0.0000001")
        stroops = amount.to_stroops()
        assert stroops == 1

    def test_to_stroops_zero(self):
        """Zero converts to zero stroops."""
        amount = DecimalAmount("0")
        stroops = amount.to_stroops()
        assert stroops == 0

    def test_to_stroops_negative(self):
        """Negative amounts convert correctly."""
        amount = DecimalAmount("-50.5000000")
        stroops = amount.to_stroops()
        assert stroops == -505000000

    def test_to_stroops_max_value(self):
        """Maximum Stellar amount converts correctly."""
        amount = DecimalAmount(STELLAR_MAX_AMOUNT)
        stroops = amount.to_stroops()
        assert stroops == 9223372036854775807  # Max int64

    def test_to_stroops_min_value(self):
        """Minimum Stellar amount converts correctly."""
        amount = DecimalAmount(STELLAR_MIN_AMOUNT)
        stroops = amount.to_stroops()
        assert stroops == -9223372036854775807

    def test_to_stroops_rejects_too_large(self):
        """Value exceeding Stellar max is rejected."""
        amount = DecimalAmount("922337203686")  # Over max
        with pytest.raises(StroopsConversionError):
            amount.to_stroops()

    def test_to_stroops_rejects_fractional_stroops(self):
        """Fractional stroops (too many decimals) is rejected."""
        amount = DecimalAmount("100.12345678")  # 8 decimals
        with pytest.raises(StroopsConversionError, match="fractional stroops"):
            amount.to_stroops()

    def test_from_stroops_basic(self):
        """Basic from_stroops conversion works."""
        stroops = 1005000000
        amount = DecimalAmount.from_stroops(stroops)
        assert amount == DecimalAmount("100.5000000")

    def test_from_stroops_small(self):
        """Single stroop converts correctly."""
        stroops = 1
        amount = DecimalAmount.from_stroops(stroops)
        assert amount == DecimalAmount("0.0000001")

    def test_from_stroops_zero(self):
        """Zero stroops converts to zero."""
        stroops = 0
        amount = DecimalAmount.from_stroops(stroops)
        assert amount == DecimalAmount("0")

    def test_from_stroops_negative(self):
        """Negative stroops converts correctly."""
        stroops = -505000000
        amount = DecimalAmount.from_stroops(stroops)
        assert amount == DecimalAmount("-50.5000000")

    def test_from_stroops_roundtrip(self):
        """Stroops conversion roundtrips correctly."""
        original = DecimalAmount("123.4567890")
        stroops = original.to_stroops()
        recovered = DecimalAmount.from_stroops(stroops)
        assert recovered == original

    def test_from_stroops_rejects_non_integer(self):
        """Non-integer stroops is rejected."""
        with pytest.raises(StroopsConversionError, match="must be an integer"):
            DecimalAmount.from_stroops(123.45)


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Test utility functions."""

    def test_safe_divide_normal(self):
        """Normal division works."""
        result = safe_divide("100", "4")
        assert result == DecimalAmount("25")

    def test_safe_divide_by_zero_returns_default(self):
        """Division by zero returns default."""
        result = safe_divide("100", "0", default="0")
        assert result == DecimalAmount("0")

    def test_safe_divide_by_zero_uses_default_default(self):
        """Division by zero with no default returns 0."""
        result = safe_divide("100", "0")
        assert result == DecimalAmount("0")

    def test_safe_divide_custom_default(self):
        """Custom default is used on division by zero."""
        result = safe_divide("100", "0", default="999")
        assert result == DecimalAmount("999")

    def test_sum_amounts_empty(self):
        """Sum of empty list is zero."""
        result = sum_amounts([])
        assert result == DecimalAmount("0")

    def test_sum_amounts_single(self):
        """Sum of single amount works."""
        result = sum_amounts([DecimalAmount("100")])
        assert result == DecimalAmount("100")

    def test_sum_amounts_multiple(self):
        """Sum of multiple amounts works."""
        amounts = [DecimalAmount("100"), "50", Decimal("25.5")]
        result = sum_amounts(amounts)
        assert result == DecimalAmount("175.5")

    def test_sum_amounts_with_negative(self):
        """Sum handles negative values."""
        amounts = [DecimalAmount("100"), DecimalAmount("-30")]
        result = sum_amounts(amounts)
        assert result == DecimalAmount("70")

    def test_safe_float_to_decimal_warns(self):
        """safe_float_to_decimal triggers warning."""
        with pytest.warns(UserWarning, match="Converting float"):
            result = safe_float_to_decimal(123.45)
            assert isinstance(result, Decimal)

    def test_safe_float_to_decimal_handles_precision(self):
        """safe_float_to_decimal handles float precision issues."""
        # Classic float precision problem
        float_value = 0.1 + 0.2  # 0.30000000000000004
        result = safe_float_to_decimal(float_value, precision=1)
        assert result == Decimal("0.3")

    def test_check_precision_loss_within_tolerance(self):
        """check_precision_loss detects acceptable loss."""
        original = 0.1 + 0.2
        converted = Decimal("0.3")
        assert check_precision_loss(original, converted) is True

    def test_check_precision_loss_exceeds_tolerance(self):
        """check_precision_loss detects unacceptable loss."""
        original = 1.23456789
        converted = Decimal("1.23")
        assert check_precision_loss(original, converted, tolerance=Decimal("1e-5")) is False


# ---------------------------------------------------------------------------
# Precision Context Tests
# ---------------------------------------------------------------------------


class TestPrecisionContext:
    """Test precision context management."""

    def test_decimal_context_changes_precision(self):
        """Context manager changes precision."""
        with decimal_context(precision=10):
            result = DecimalAmount("1") / DecimalAmount("3")
            # Check result has 10 significant digits
            assert len(result.value.as_tuple().digits) == 10

    def test_decimal_context_restores_original(self):
        """Context manager restores original context."""
        import decimal

        original_precision = decimal.getcontext().prec

        with decimal_context(precision=10):
            pass  # Change context temporarily

        # Context should be restored
        assert decimal.getcontext().prec == original_precision

    def test_decimal_context_high_precision_warns(self):
        """Very high precision triggers warning."""
        with pytest.warns(UserWarning, match="exceeds MAX_SAFE_PRECISION"):
            with decimal_context(precision=200):
                pass


# ---------------------------------------------------------------------------
# DecimalAmount Methods Tests
# ---------------------------------------------------------------------------


class TestDecimalAmountMethods:
    """Test DecimalAmount utility methods."""

    def test_round_to_stellar_precision(self):
        """Round to Stellar precision (7 decimals)."""
        amount = DecimalAmount("100.123456789")
        rounded = amount.round()
        assert rounded == DecimalAmount("100.1234568")  # Banker's rounding

    def test_round_to_custom_precision(self):
        """Round to custom precision."""
        amount = DecimalAmount("100.123456")
        rounded = amount.round(decimal_places=2)
        assert rounded == DecimalAmount("100.12")

    def test_to_float_warns(self):
        """to_float triggers warning."""
        amount = DecimalAmount("100.5")
        with pytest.warns(UserWarning, match="lose precision"):
            float_value = amount.to_float()
            assert isinstance(float_value, float)
            assert float_value == 100.5

    def test_is_zero(self):
        """is_zero detects zero."""
        assert DecimalAmount("0").is_zero() is True
        assert DecimalAmount("0.0000000").is_zero() is True
        assert DecimalAmount("0.0000001").is_zero() is False

    def test_is_positive(self):
        """is_positive detects positive values."""
        assert DecimalAmount("100").is_positive() is True
        assert DecimalAmount("0.0000001").is_positive() is True
        assert DecimalAmount("0").is_positive() is False
        assert DecimalAmount("-100").is_positive() is False

    def test_is_negative(self):
        """is_negative detects negative values."""
        assert DecimalAmount("-100").is_negative() is True
        assert DecimalAmount("-0.0000001").is_negative() is True
        assert DecimalAmount("0").is_negative() is False
        assert DecimalAmount("100").is_negative() is False

    def test_string_representation(self):
        """String representation works."""
        amount = DecimalAmount("100.50")
        assert str(amount) == "100.50"

    def test_repr(self):
        """Developer repr works."""
        amount = DecimalAmount("100.50")
        assert repr(amount) == "DecimalAmount('100.50')"

    def test_format(self):
        """Custom formatting works."""
        amount = DecimalAmount("100.123")
        assert f"{amount:.2f}" == "100.12"


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_amount(self):
        """Very small amounts are handled correctly."""
        amount = DecimalAmount("0.0000001")  # 1 stroop
        assert amount.is_positive()
        assert not amount.is_zero()

    def test_very_large_amount(self):
        """Very large amounts within bounds are handled."""
        amount = DecimalAmount("100000000.1234567")
        assert amount.is_positive()

    def test_zero_operations(self):
        """Operations with zero work correctly."""
        zero = DecimalAmount("0")
        amount = DecimalAmount("100")

        assert zero + amount == amount
        assert amount + zero == amount
        assert amount - zero == amount
        assert zero * amount == zero
        assert amount * zero == zero

    def test_negative_operations(self):
        """Operations with negative values work."""
        pos = DecimalAmount("100")
        neg = DecimalAmount("-50")

        assert pos + neg == DecimalAmount("50")
        assert pos - neg == DecimalAmount("150")
        assert pos * neg == DecimalAmount("-5000")

    def test_division_precision(self):
        """Division maintains precision."""
        a = DecimalAmount("1")
        b = DecimalAmount("3")
        result = a / b
        # Should have many decimal places
        assert "0.333333" in str(result)

    def test_chained_operations(self):
        """Chained operations work correctly."""
        result = (
            DecimalAmount("100") + DecimalAmount("50") - DecimalAmount("25") * DecimalAmount("2")
        )
        assert result == DecimalAmount("100")  # 100 + 50 - 50

    def test_float_initialization_warns(self):
        """Float initialization triggers warning."""
        with pytest.warns(UserWarning, match="Float inputs may have precision errors"):
            amount = DecimalAmount(123.45)
            assert isinstance(amount.value, Decimal)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestArithmeticProperties:
    """Property-based tests for arithmetic operations.

    These tests verify mathematical properties that should hold for all inputs:
    - Commutativity: a + b = b + a
    - Associativity: (a + b) + c = a + (b + c)
    - Identity: a + 0 = a
    - Inverse: a - a = 0
    """

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
        b=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
    )
    def test_addition_commutative(self, a, b):
        """Addition is commutative: a + b = b + a."""
        amount_a = DecimalAmount(a)
        amount_b = DecimalAmount(b)
        assert amount_a + amount_b == amount_b + amount_a

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
        b=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
        c=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
    )
    def test_addition_associative(self, a, b, c):
        """Addition is associative: (a + b) + c = a + (b + c)."""
        amount_a = DecimalAmount(a)
        amount_b = DecimalAmount(b)
        amount_c = DecimalAmount(c)

        left = (amount_a + amount_b) + amount_c
        right = amount_a + (amount_b + amount_c)
        assert left == right

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_addition_identity(self, a):
        """Zero is the additive identity: a + 0 = a."""
        amount_a = DecimalAmount(a)
        zero = DecimalAmount("0")
        assert amount_a + zero == amount_a
        assert zero + amount_a == amount_a

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_subtraction_inverse(self, a):
        """Subtraction is the inverse of addition: a - a = 0."""
        amount_a = DecimalAmount(a)
        result = amount_a - amount_a
        assert result.is_zero()

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
        b=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        ),
    )
    def test_multiplication_commutative(self, a, b):
        """Multiplication is commutative: a * b = b * a."""
        amount_a = DecimalAmount(a)
        amount_b = DecimalAmount(b)
        assert amount_a * amount_b == amount_b * amount_a

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_multiplication_by_one(self, a):
        """One is the multiplicative identity: a * 1 = a."""
        amount_a = DecimalAmount(a)
        one = DecimalAmount("1")
        assert amount_a * one == amount_a

    @given(
        a=st.decimals(
            min_value=-1000000,
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_multiplication_by_zero(self, a):
        """Multiplication by zero gives zero: a * 0 = 0."""
        amount_a = DecimalAmount(a)
        zero = DecimalAmount("0")
        result = amount_a * zero
        assert result.is_zero()

    @given(
        a=st.decimals(
            min_value=1,  # Avoid division by zero
            max_value=1000000,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_division_inverse(self, a):
        """Division is the inverse of multiplication: (a * b) / b = a."""
        amount_a = DecimalAmount(a)
        amount_b = DecimalAmount("2")  # Safe divisor

        product = amount_a * amount_b
        quotient = product / amount_b

        # Should be equal within precision
        assert abs((quotient - amount_a).value) < Decimal("1e-10")

    @given(stroops=st.integers(min_value=-9223372036854775807, max_value=9223372036854775807))
    def test_stroops_roundtrip(self, stroops):
        """Stroops conversion roundtrips correctly."""
        amount = DecimalAmount.from_stroops(stroops)
        recovered_stroops = amount.to_stroops()
        assert recovered_stroops == stroops

    @given(
        a=st.decimals(
            min_value=-922337203685,
            max_value=922337203685,
            allow_nan=False,
            allow_infinity=False,
            places=7,
        )
    )
    def test_stellar_validation_roundtrip(self, a):
        """Stellar validation is consistent."""
        try:
            validated = validate_stellar_amount(a)
            # If validation succeeds, should be able to create DecimalAmount
            amount = DecimalAmount(validated)
            # And convert to stroops if within 7 decimals
            if amount.value.as_tuple().exponent >= -STELLAR_PRECISION:
                stroops = amount.to_stroops()
                assert isinstance(stroops, int)
        except AmountValidationError:
            # Some values correctly rejected
            pass


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test integration scenarios."""

    def test_realistic_trade_scenario(self):
        """Simulate realistic trade calculations."""
        # Trade: 100 USDC for 500 XLM
        usdc_amount = DecimalAmount("100.0000000")
        xlm_amount = DecimalAmount("500.0000000")

        # Calculate price
        price = usdc_amount / xlm_amount
        assert price == DecimalAmount("0.2")

        # Calculate fees (0.1%)
        fee_rate = DecimalAmount("0.001")
        usdc_fee = usdc_amount * fee_rate
        assert usdc_fee == DecimalAmount("0.1")

        # Net amount
        net_usdc = usdc_amount - usdc_fee
        assert net_usdc == DecimalAmount("99.9")

    def test_multiple_trades_aggregation(self):
        """Aggregate amounts from multiple trades."""
        trades = [
            DecimalAmount("100.5"),
            DecimalAmount("250.75"),
            DecimalAmount("50.25"),
        ]

        total = sum_amounts(trades)
        assert total == DecimalAmount("401.5")

        average = total / DecimalAmount(str(len(trades)))
        assert average == DecimalAmount("133.8333333333333333333333333")

    def test_precision_critical_calculation(self):
        """Test calculation where precision matters."""
        # Scenario: Detecting wash trading by comparing volumes
        volume1 = DecimalAmount("1000000.0000001")
        volume2 = DecimalAmount("1000000.0000002")

        difference = volume2 - volume1
        assert difference == DecimalAmount("0.0000001")  # Exactly 1 stroop

        # This difference would be lost with floats!
        float(volume2.value) - float(volume1.value)
        # Float may give 0.0 or incorrect value due to precision loss

    def test_benford_analysis_preparation(self):
        """Prepare amounts for Benford's Law digit analysis."""
        amounts = [
            DecimalAmount("123.45"),
            DecimalAmount("456.78"),
            DecimalAmount("789.01"),
        ]

        # Extract leading digits (for Benford analysis)
        leading_digits = []
        for amount in amounts:
            # Get first digit of integer part
            str_amount = str(amount.value).replace(".", "").lstrip("0")
            if str_amount:
                leading_digits.append(int(str_amount[0]))

        assert leading_digits == [1, 4, 7]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
