"""Numeric precision guards for financial calculations.

This module provides precision-safe arithmetic for financial data processing
in LedgerLens. Stellar uses 7 decimal places (stroops), and fraud detection
requires exact decimal arithmetic to avoid floating-point errors that could
mask anomalies or produce false positives.

Key Features
------------
- Decimal arithmetic (no floating-point errors)
- Stellar stroops conversion (7 decimal places)
- Amount validation with configurable bounds
- Overflow/underflow protection
- Precision context management
- Type-safe operations

Architecture
------------
The system uses Python's `decimal.Decimal` for all financial calculations,
with validation at data ingestion boundaries and context managers for
controlling precision throughout the pipeline.

Usage
-----
Basic decimal amounts::

    from utils.decimal_guards import DecimalAmount, validate_amount

    # Create validated amount
    amount = DecimalAmount("123.4567890")

    # Arithmetic operations
    total = amount + DecimalAmount("10.5")

    # Stellar stroops conversion
    stroops = amount.to_stroops()  # 1234567890 (integer)
    recovered = DecimalAmount.from_stroops(stroops)

Validation::

    # Validate at ingestion boundary
    validate_amount("123.45", min_value="0", max_value="1000000")

    # Custom precision
    with decimal_context(precision=28):
        result = DecimalAmount("0.1") + DecimalAmount("0.2")

Stellar Integration
-------------------
Stellar represents amounts as signed 64-bit integers in stroops (10^-7).
The valid range is [-922337203685.4775808, 922337203685.4775807].

See: https://developers.stellar.org/docs/fundamentals-and-concepts/stellar-data-structures/assets
"""

from __future__ import annotations

import decimal
import warnings
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stellar precision: 7 decimal places (1 stroop = 10^-7 XLM)
STELLAR_PRECISION = 7
STROOPS_MULTIPLIER = Decimal(10) ** STELLAR_PRECISION

# Stellar amount bounds (signed 64-bit integer in stroops)
# Maximum: 9223372036854775807 stroops = 922337203685.4775807 XLM
STELLAR_MAX_AMOUNT = Decimal("922337203685.4775807")
STELLAR_MIN_AMOUNT = Decimal("-922337203685.4775807")

# Default precision context (28 digits, same as Python default)
DEFAULT_PRECISION = 28

# Maximum safe precision to prevent memory exhaustion
MAX_SAFE_PRECISION = 100


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PrecisionError(Exception):
    """Base exception for precision-related errors."""


class AmountValidationError(PrecisionError):
    """Raised when an amount fails validation."""


class PrecisionOverflowError(PrecisionError):
    """Raised when a calculation would overflow."""


class PrecisionUnderflowError(PrecisionError):
    """Raised when a value is too small to represent."""


class StroopsConversionError(PrecisionError):
    """Raised when stroops conversion fails."""


# ---------------------------------------------------------------------------
# Precision Context Management
# ---------------------------------------------------------------------------


@contextmanager
def decimal_context(
    precision: int = DEFAULT_PRECISION,
    rounding: str = decimal.ROUND_HALF_EVEN,
):
    """Context manager for controlling decimal precision and rounding mode.

    Sets a thread-local ``decimal.Context`` for the duration of the ``with``
    block and restores the previous context on exit — including on exceptions.

    Args:
        precision: Number of significant digits for all ``Decimal`` operations
            inside the block (default: 28, matching Python's built-in default).
            Values above ``MAX_SAFE_PRECISION`` (100) emit a ``UserWarning``.
        rounding: Rounding mode applied to all ``Decimal`` operations inside
            the block (default: ``ROUND_HALF_EVEN`` — banker's rounding).

    Yields:
        decimal.Context: The newly active decimal context, in case callers need
            to inspect or further modify it.

    Raises:
        decimal.InvalidOperation: Propagated from any ``Decimal`` operation
            that produces a signalling NaN under the new context.
        decimal.Overflow: Propagated if a result exceeds the context's
            representable range.
        decimal.DivisionByZero: Propagated on exact zero division.

    Why this matters:
        All intermediate Benford statistics and ML feature aggregations must
        use a consistent precision and rounding rule; ad-hoc context changes
        elsewhere in the pipeline would produce non-reproducible scores and
        break the numeric precision guarantees described in
        ``docs/numeric_precision.md``.

    Example::

        with decimal_context(precision=10):
            result = DecimalAmount("1") / DecimalAmount("3")
            # result = 0.3333333333 (10 digits)
    """
    if precision > MAX_SAFE_PRECISION:
        warnings.warn(
            f"Precision {precision} exceeds MAX_SAFE_PRECISION ({MAX_SAFE_PRECISION}). "
            "This may cause performance issues.",
            stacklevel=2,
        )

    old_context = decimal.getcontext()
    new_context = decimal.Context(
        prec=precision,
        rounding=rounding,
        traps=[
            decimal.InvalidOperation,
            decimal.DivisionByZero,
            decimal.Overflow,
        ],
    )

    decimal.setcontext(new_context)
    try:
        yield new_context
    finally:
        decimal.setcontext(old_context)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_amount(
    value: str | int | float | Decimal,
    min_value: str | Decimal | None = None,
    max_value: str | Decimal | None = None,
    allow_negative: bool = False,
    name: str = "amount",
) -> Decimal:
    """Validate and convert an arbitrary value to ``Decimal`` at an ingestion boundary.

    Rejects ``float`` inputs with a warning (floats are accepted but converted
    via ``str`` to avoid silent precision loss), and hard-rejects NaN, infinity,
    out-of-range, and sign violations.

    Args:
        value: Value to validate.  Accepts ``str``, ``int``, ``float``, or
            ``Decimal``.  Float inputs trigger a ``UserWarning`` and are
            converted via their string representation to mitigate IEEE-754
            representation noise.
        min_value: Inclusive lower bound.  ``None`` means no lower bound.
        max_value: Inclusive upper bound.  ``None`` means no upper bound.
        allow_negative: If ``False`` (default), negative values raise
            ``AmountValidationError``.
        name: Human-readable field name embedded in error messages to aid
            debugging (e.g. ``"trade_amount"``, ``"fee"``).

    Returns:
        Validated ``Decimal`` value.

    Raises:
        AmountValidationError: If the value cannot be parsed as a ``Decimal``,
            is NaN or infinite, is negative when ``allow_negative=False``, or
            falls outside ``[min_value, max_value]``.

    Why this matters:
        Every trade amount entering the pipeline passes through this function;
        a single NaN or out-of-range value would silently corrupt Benford
        digit counts and volume aggregations, producing false anomaly scores.

    Example::

        # Validate positive amount
        amount = validate_amount("123.45", min_value="0", max_value="1000000")

        # Allow negative (for deltas)
        delta = validate_amount("-50.00", allow_negative=True)
    """
    # Convert to Decimal
    try:
        if isinstance(value, float):
            warnings.warn(
                f"Converting float to Decimal for {name}. "
                "Float inputs may have precision errors. "
                "Use string or Decimal instead.",
                stacklevel=2,
            )
            decimal_value = Decimal(str(value))
        else:
            decimal_value = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as e:
        raise AmountValidationError(
            f"Invalid {name}: {value!r} cannot be converted to Decimal"
        ) from e

    # Check for special values
    if decimal_value.is_nan():
        raise AmountValidationError(f"Invalid {name}: NaN is not allowed")

    if decimal_value.is_infinite():
        raise AmountValidationError(f"Invalid {name}: Infinity is not allowed")

    # Check sign
    if not allow_negative and decimal_value < 0:
        raise AmountValidationError(f"Invalid {name}: {decimal_value} is negative (not allowed)")

    # Check bounds
    if min_value is not None:
        min_decimal = Decimal(min_value)
        if decimal_value < min_decimal:
            raise AmountValidationError(f"Invalid {name}: {decimal_value} < minimum {min_decimal}")

    if max_value is not None:
        max_decimal = Decimal(max_value)
        if decimal_value > max_decimal:
            raise AmountValidationError(f"Invalid {name}: {decimal_value} > maximum {max_decimal}")

    return decimal_value


def validate_stellar_amount(value: str | int | float | Decimal) -> Decimal:
    """Validate that an amount conforms to Stellar's on-chain constraints.

    Stellar represents amounts as signed 64-bit integers in stroops
    (1 stroop = 10^-7 XLM).  This function enforces two invariants on top of
    the general ``validate_amount`` checks:

    1. The value falls within the signed 64-bit stroop range
       [``STELLAR_MIN_AMOUNT``, ``STELLAR_MAX_AMOUNT``]
       (i.e. [-922337203685.4775807, 922337203685.4775807]).
    2. The value has at most ``STELLAR_PRECISION`` (7) decimal places —
       fractional stroops do not exist on the network.

    ``float`` inputs are rounded to 7 decimal places before the precision
    check to mitigate IEEE-754 representation noise (e.g. ``0.1 + 0.2``).

    Args:
        value: Amount to validate.  Accepts ``str``, ``int``, ``float``, or
            ``Decimal``.

    Returns:
        Validated ``Decimal`` value, ready for stroops conversion or further
        arithmetic.

    Raises:
        AmountValidationError: If the value is NaN, infinite, outside the
            signed 64-bit stroop range, or has more than 7 decimal places.

    Why this matters:
        Submitting an amount that violates either constraint would be rejected
        by the Stellar network or silently truncated during the stroops integer
        cast, both of which would corrupt risk-score calculations and Benford
        leading-digit extraction.
    """
    decimal_value = validate_amount(
        value,
        min_value=STELLAR_MIN_AMOUNT,
        max_value=STELLAR_MAX_AMOUNT,
        allow_negative=True,
        name="stellar_amount",
    )

    if isinstance(value, float):
        quantizer = Decimal(10) ** -STELLAR_PRECISION
        decimal_value = decimal_value.quantize(quantizer, rounding=decimal.ROUND_HALF_EVEN)

    # Check decimal places
    if decimal_value.as_tuple().exponent < -STELLAR_PRECISION:
        raise AmountValidationError(
            f"Invalid stellar_amount: {decimal_value} has more than "
            f"{STELLAR_PRECISION} decimal places"
        )

    return decimal_value


# ---------------------------------------------------------------------------
# DecimalAmount Class
# ---------------------------------------------------------------------------


class DecimalAmount:
    """Precision-safe decimal amount for financial calculations.

    Wraps Python's ``Decimal`` to provide type-safe arithmetic, automatic
    validation on construction, Stellar stroops conversion, and the full set
    of comparison and arithmetic operators.  All operations return a new
    ``DecimalAmount`` so the type is preserved through expression chains.

    Attributes:
        value: The underlying ``Decimal`` value (read-only property).

    Why this matters:
        Using plain ``float`` for trade amounts introduces IEEE-754 rounding
        errors that skew Benford leading-digit distributions and produce
        non-reproducible volume aggregations.  ``DecimalAmount`` makes exact
        arithmetic the default and surfaces unsafe conversions (e.g. from
        ``float``) with explicit warnings, satisfying the numeric precision
        guarantees in ``docs/numeric_precision.md``.

    Example::

        amount1 = DecimalAmount("100.50")
        amount2 = DecimalAmount("25.25")

        total = amount1 + amount2  # DecimalAmount("125.75")
        ratio = amount1 / amount2  # DecimalAmount("3.9801980198...")

        # Stellar conversion
        stroops = amount1.to_stroops()  # 1005000000
        recovered = DecimalAmount.from_stroops(stroops)  # "100.5000000"
    """

    __slots__ = ("_value",)

    def __init__(self, value: str | int | Decimal | DecimalAmount):
        """Initialize a ``DecimalAmount``, validating the input immediately.

        Args:
            value: Amount value.  Accepts ``str``, ``int``, ``Decimal``, or
                another ``DecimalAmount``.  ``float`` is also accepted but
                emits a ``UserWarning`` and is converted via its string
                representation to mitigate IEEE-754 noise.

        Raises:
            AmountValidationError: If ``value`` is NaN, infinite, or cannot
                be parsed as a ``Decimal``.
        """
        if isinstance(value, DecimalAmount):
            self._value = value._value
        elif isinstance(value, float):
            warnings.warn(
                "DecimalAmount initialized with float. "
                "Float inputs may have precision errors. "
                "Use string or Decimal instead.",
                stacklevel=2,
            )
            self._value = Decimal(str(value))
        else:
            # DecimalAmount is also used for deltas, balances, and
            # counterfactual differences, so signed values are valid here.
            self._value = validate_amount(value, allow_negative=True, name="amount")

    @property
    def value(self) -> Decimal:
        """Get the underlying Decimal value."""
        return self._value

    # -----------------------------------------------------------------------
    # Arithmetic Operations
    # -----------------------------------------------------------------------

    def __add__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Add two amounts."""
        other_value = self._to_decimal(other)
        return DecimalAmount(self._value + other_value)

    def __radd__(self, other: str | int | Decimal) -> DecimalAmount:
        """Reverse add."""
        return self.__add__(other)

    def __sub__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Subtract two amounts."""
        other_value = self._to_decimal(other)
        return DecimalAmount(self._value - other_value)

    def __rsub__(self, other: str | int | Decimal) -> DecimalAmount:
        """Reverse subtract."""
        other_value = self._to_decimal(other)
        return DecimalAmount(other_value - self._value)

    def __mul__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Multiply two amounts."""
        other_value = self._to_decimal(other)
        return DecimalAmount(self._value * other_value)

    def __rmul__(self, other: str | int | Decimal) -> DecimalAmount:
        """Reverse multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Divide two amounts."""
        other_value = self._to_decimal(other)
        if other_value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return DecimalAmount(self._value / other_value)

    def __rtruediv__(self, other: str | int | Decimal) -> DecimalAmount:
        """Reverse divide."""
        other_value = self._to_decimal(other)
        if self._value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return DecimalAmount(other_value / self._value)

    def __floordiv__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Floor division."""
        other_value = self._to_decimal(other)
        if other_value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return DecimalAmount(self._value // other_value)

    def __mod__(self, other: DecimalAmount | str | int | Decimal) -> DecimalAmount:
        """Modulo operation."""
        other_value = self._to_decimal(other)
        if other_value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")
        return DecimalAmount(self._value % other_value)

    def __pow__(self, exponent: int | Decimal) -> DecimalAmount:
        """Power operation."""
        if isinstance(exponent, DecimalAmount):
            exponent = exponent._value
        elif not isinstance(exponent, (int, Decimal)):
            exponent = Decimal(exponent)
        return DecimalAmount(self._value**exponent)

    def __neg__(self) -> DecimalAmount:
        """Negation."""
        return DecimalAmount(-self._value)

    def __pos__(self) -> DecimalAmount:
        """Positive (no-op)."""
        return DecimalAmount(self._value)

    def __abs__(self) -> DecimalAmount:
        """Absolute value."""
        return DecimalAmount(abs(self._value))

    # -----------------------------------------------------------------------
    # Comparison Operations
    # -----------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, (DecimalAmount, str, int, Decimal)):
            return NotImplemented
        other_value = self._to_decimal(other)
        return self._value == other_value

    def __ne__(self, other: object) -> bool:
        """Inequality comparison."""
        return not self.__eq__(other)

    def __lt__(self, other: DecimalAmount | str | int | Decimal) -> bool:
        """Less than."""
        other_value = self._to_decimal(other)
        return self._value < other_value

    def __le__(self, other: DecimalAmount | str | int | Decimal) -> bool:
        """Less than or equal."""
        other_value = self._to_decimal(other)
        return self._value <= other_value

    def __gt__(self, other: DecimalAmount | str | int | Decimal) -> bool:
        """Greater than."""
        other_value = self._to_decimal(other)
        return self._value > other_value

    def __ge__(self, other: DecimalAmount | str | int | Decimal) -> bool:
        """Greater than or equal."""
        other_value = self._to_decimal(other)
        return self._value >= other_value

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(self._value)

    # -----------------------------------------------------------------------
    # Stellar Stroops Conversion
    # -----------------------------------------------------------------------

    def to_stroops(self) -> int:
        """Convert to Stellar stroops (integer with 7 decimal places).

        Returns:
            Integer stroops value

        Raises:
            StroopsConversionError: If conversion fails or value out of range

        Example::

            amount = DecimalAmount("100.5000000")
            stroops = amount.to_stroops()  # 1005000000
        """
        # Validate first
        try:
            validate_stellar_amount(self._value)
        except AmountValidationError as e:
            if self._value.as_tuple().exponent < -STELLAR_PRECISION:
                raise StroopsConversionError(
                    f"Cannot convert {self._value} to stroops: fractional stroops are not allowed"
                ) from e
            raise StroopsConversionError(f"Cannot convert {self._value} to stroops: {e}") from e

        # Convert to stroops
        stroops_decimal = self._value * STROOPS_MULTIPLIER
        stroops_int = int(stroops_decimal)

        # Verify no precision loss
        if Decimal(stroops_int) != stroops_decimal:
            raise StroopsConversionError(
                f"Precision loss in stroops conversion: {self._value} " f"has fractional stroops"
            )

        return stroops_int

    @classmethod
    def from_stroops(cls, stroops: int) -> DecimalAmount:
        """Create DecimalAmount from Stellar stroops.

        Args:
            stroops: Integer stroops value

        Returns:
            DecimalAmount

        Raises:
            StroopsConversionError: If conversion fails

        Example::

            stroops = 1005000000
            amount = DecimalAmount.from_stroops(stroops)  # "100.5000000"
        """
        if not isinstance(stroops, int):
            raise StroopsConversionError(
                f"Stroops must be an integer, got {type(stroops).__name__}"
            )

        decimal_value = Decimal(stroops) / STROOPS_MULTIPLIER

        # Validate result
        try:
            validate_stellar_amount(decimal_value)
        except AmountValidationError as e:
            raise StroopsConversionError(f"Invalid stroops value {stroops}: {e}") from e

        return cls(decimal_value)

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------

    def round(self, decimal_places: int = STELLAR_PRECISION) -> DecimalAmount:
        """Round to the specified number of decimal places using banker's rounding.

        Args:
            decimal_places: Number of decimal places to retain
                (default: ``STELLAR_PRECISION`` = 7, matching Stellar's stroop
                resolution).

        Returns:
            New ``DecimalAmount`` rounded to ``decimal_places`` digits.

        Why this matters:
            Rounding to exactly 7 places before a stroops conversion prevents
            ``StroopsConversionError`` from fractional-stroop values that arise
            from intermediate arithmetic, ensuring pipeline outputs stay within
            Stellar's representable range.
        """
        quantizer = Decimal(10) ** -decimal_places
        rounded = self._value.quantize(quantizer, rounding=decimal.ROUND_HALF_EVEN)
        return DecimalAmount(rounded)

    # INTENTIONAL PRECISION BOUNDARY (see Issue #778): the only place in this
    # module (and utils/decimal_agg.py) where a Decimal is deliberately
    # converted to float. This is an explicit escape hatch for legacy code that
    # still expects a native ``float``; it is NOT a silent Decimal->float
    # conversion on the happy path and every other public boundary returns
    # Decimal. The precision loss is surfaced to callers via a warning.
    def to_float(self) -> float:
        """Convert to ``float``, emitting a precision-loss warning.

        Returns:
            ``float`` representation of the underlying ``Decimal`` value.
            Precision beyond ~15–17 significant digits is silently discarded
            by IEEE-754.

        Warning:
            Provided only for compatibility with legacy code and third-party
            libraries that do not accept ``Decimal``.  Avoid in any code path
            that feeds Benford analysis, volume aggregation, or risk scoring —
            the rounding error introduced here can shift leading digits and
            produce spurious anomaly signals.

        Why this matters:
            Calling this method in a hot path reintroduces the exact precision
            hazards that ``DecimalAmount`` was designed to eliminate; the
            warning serves as an auditable signal in logs that a precision
            boundary was crossed.
        """
        warnings.warn(
            "Converting DecimalAmount to float may lose precision. "
            "Use Decimal operations when possible.",
            stacklevel=2,
        )
        return float(self._value)

    def is_zero(self) -> bool:
        """Return ``True`` if the amount is exactly zero.

        Why this matters:
            Used as a denominator guard before division (e.g. in ratio
            features); exact ``Decimal`` zero comparison avoids the
            ``== 0.0`` float pitfall where very small non-zero values
            compare equal to zero.
        """
        return self._value == 0

    def is_positive(self) -> bool:
        """Return ``True`` if the amount is strictly greater than zero.

        Why this matters:
            Distinguishes genuine positive trade volume from zero or
            negative deltas when filtering the feature matrix, preventing
            sign errors from propagating into Benford digit extraction.
        """
        return self._value > 0

    def is_negative(self) -> bool:
        """Return ``True`` if the amount is strictly less than zero.

        Why this matters:
            Flags net-negative balances and counterfactual deltas so
            callers can handle them explicitly rather than silently folding
            them into unsigned volume sums.
        """
        return self._value < 0

    @staticmethod
    def _to_decimal(value: DecimalAmount | str | int | Decimal) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, DecimalAmount):
            return value._value
        elif isinstance(value, Decimal):
            return value
        else:
            return Decimal(value)

    # -----------------------------------------------------------------------
    # String Representation
    # -----------------------------------------------------------------------

    def __str__(self) -> str:
        """String representation."""
        return str(self._value)

    def __repr__(self) -> str:
        """Developer representation."""
        return f"DecimalAmount('{self._value}')"

    def __format__(self, format_spec: str) -> str:
        """Custom formatting."""
        return format(self._value, format_spec)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def safe_divide(
    numerator: DecimalAmount | str | int | Decimal,
    denominator: DecimalAmount | str | int | Decimal,
    default: DecimalAmount | str | int | Decimal | None = None,
) -> DecimalAmount:
    """Divide two amounts, returning a safe default when the denominator is zero.

    Args:
        numerator: The dividend.  Accepts ``DecimalAmount``, ``str``, ``int``,
            or ``Decimal``.
        denominator: The divisor.  Accepts the same types as ``numerator``.
        default: Value to return when ``denominator`` is zero.  Defaults to
            ``DecimalAmount("0")`` when ``None``.

    Returns:
        ``numerator / denominator`` as a ``DecimalAmount``, or ``default`` if
        the denominator is exactly zero.

    Why this matters:
        Several ML features — counterparty concentration ratio,
        volume-to-unique-counterparty ratio, net asset flow deviation —
        divide by trade counts or total volumes that can legitimately be zero
        for wallets with no activity in a given time window.  A bare ``/``
        would raise ``ZeroDivisionError`` and abort scoring; this guard
        returns a neutral value that keeps the pipeline running.

    Example::

        # Safe division
        result = safe_divide("100", "0", default="0")  # Returns DecimalAmount("0")

        # Normal division
        result = safe_divide("100", "5")  # Returns DecimalAmount("20")
    """
    if not isinstance(numerator, DecimalAmount):
        numerator = DecimalAmount(numerator)

    if not isinstance(denominator, DecimalAmount):
        denominator = DecimalAmount(denominator)

    if denominator.is_zero():
        if default is None:
            return DecimalAmount("0")
        return DecimalAmount(default) if not isinstance(default, DecimalAmount) else default

    return numerator / denominator


def sum_amounts(amounts: list[DecimalAmount | str | int | Decimal]) -> DecimalAmount:
    """Sum a list of amounts with exact ``Decimal`` precision.

    Iterates the list and accumulates into a ``DecimalAmount("0")`` starting
    value, converting each element on the fly if needed.

    Args:
        amounts: List of amounts to sum.  Each element may be a
            ``DecimalAmount``, ``str``, ``int``, or ``Decimal``.  An empty
            list returns ``DecimalAmount("0")``.

    Returns:
        The exact sum of all elements as a ``DecimalAmount``.

    Why this matters:
        Volume aggregations across thousands of trades are the input to
        Benford analysis; ``sum()`` on Python floats accumulates rounding
        error that shifts leading-digit distributions and produces
        false-positive anomaly scores.  Using ``Decimal`` accumulation
        keeps the digit counts exact regardless of list length.

    Example::

        amounts = [DecimalAmount("100.50"), "25.25", Decimal("10.10")]
        total = sum_amounts(amounts)  # DecimalAmount("135.85")
    """
    if not amounts:
        return DecimalAmount("0")

    total = DecimalAmount("0")
    for amount in amounts:
        if not isinstance(amount, DecimalAmount):
            amount = DecimalAmount(amount)
        total = total + amount

    return total


def safe_float_to_decimal(value: float, precision: int = STELLAR_PRECISION) -> Decimal:
    """Convert a ``float`` to ``Decimal`` with controlled, bounded precision.

    Mitigates IEEE-754 representation noise by:

    1. Converting the float to its string representation (avoids
       ``Decimal(0.1)`` expanding to ``0.1000000000000000055511151...``).
    2. Rounding the result to ``precision`` decimal places using banker's
       rounding.

    Always emits a ``UserWarning`` so call-sites are visible in logs.

    Args:
        value: Float value to convert.
        precision: Number of decimal places to round to after conversion
            (default: ``STELLAR_PRECISION`` = 7).

    Returns:
        ``Decimal`` rounded to ``precision`` places.

    Raises:
        decimal.InvalidOperation: If ``value`` is NaN or infinite and the
            active decimal context traps ``InvalidOperation``.

    Why this matters:
        External data sources (Horizon API JSON, CSV exports) sometimes
        deliver amounts as JSON numbers which Python parses as ``float``.
        Calling this function at the ingestion boundary limits the precision
        hazard to a single, auditable conversion point instead of letting
        float noise propagate through the entire feature pipeline.

    Example::

        # Float precision issue
        value = 0.1 + 0.2  # 0.30000000000000004

        # Safe conversion
        decimal_value = safe_float_to_decimal(value, precision=2)  # Decimal("0.30")
    """
    warnings.warn(
        "Converting float to Decimal. Float inputs may have precision errors. "
        "Use string or Decimal input for exact precision.",
        stacklevel=2,
    )

    # Convert to string to avoid float representation errors
    str_value = f"{value:.{precision + 2}f}"  # Extra digits for rounding
    decimal_value = Decimal(str_value)

    # Round to desired precision
    quantizer = Decimal(10) ** -precision
    return decimal_value.quantize(quantizer, rounding=decimal.ROUND_HALF_EVEN)


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


def check_precision_loss(
    original: float | Decimal,
    converted: Decimal,
    tolerance: Decimal = Decimal("1e-10"),
) -> bool:
    """Check whether a float-to-``Decimal`` conversion introduced unacceptable precision loss.

    Computes ``|Decimal(str(original)) - converted|`` and compares it against
    ``tolerance``.  Returns ``True`` when the loss is within the tolerance
    (i.e. the conversion is acceptable), ``False`` when it exceeds it.

    Args:
        original: The source value before conversion — typically a ``float``
            read from an external data source.
        converted: The ``Decimal`` produced by the conversion under test.
        tolerance: Maximum acceptable absolute difference between
            ``Decimal(str(original))`` and ``converted``
            (default: ``Decimal("1e-10")``).

    Returns:
        ``True`` if ``|original - converted| <= tolerance``, ``False``
        otherwise.

    Why this matters:
        Benford's Law analysis extracts the *leading digit* of each trade
        amount; a precision loss as small as 10^-8 on a value like
        ``0.09999...`` vs ``0.1`` changes the leading digit from 9 to 1,
        flipping its contribution in the chi-square statistic.  Use this
        function in ingestion tests and validation scripts to confirm that
        float→Decimal conversions stay within safe bounds.

    Example::

        original = 0.1 + 0.2  # 0.30000000000000004
        converted = Decimal("0.3")
        is_ok = check_precision_loss(original, converted)  # True (within tolerance)
    """
    original_decimal = Decimal(str(original))
    difference = abs(original_decimal - converted)
    return difference <= tolerance
