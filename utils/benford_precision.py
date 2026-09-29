"""Precision-safe Benford analysis helpers.

This module provides precision-safe digit extraction for Benford's Law analysis
using Decimal arithmetic instead of float operations. This ensures exact digit
extraction without floating-point precision errors.

Key Features
------------
- Extract leading digits without float precision loss
- Support for large and small amounts with exact arithmetic
- Compatible with pandas Series for bulk operations
- Maintains compatibility with existing benford_engine.py

Usage
-----
Basic usage::

    from utils.benford_precision import extract_leading_digit_safe
    from utils.decimal_guards import DecimalAmount

    amount = DecimalAmount("123.456")
    digit = extract_leading_digit_safe(amount)  # Returns 1

Pandas Series usage::

    import pandas as pd
    from utils.benford_precision import leading_digits_safe

    amounts = pd.Series([
        DecimalAmount("123.45"),
        DecimalAmount("456.78"),
        DecimalAmount("789.01"),
    ])
    digits = leading_digits_safe(amounts)  # Series([1, 4, 7])

Migration from benford_engine.py::

    # Before (float-based, precision loss)
    from detection.benford_engine import leading_digits
    digits = leading_digits(amounts)  # Uses np.log10, float division

    # After (Decimal-based, exact)
    from utils.benford_precision import leading_digits_safe
    digits = leading_digits_safe(amounts)  # Uses Decimal, exact

Why This Matters
----------------
Float precision errors can corrupt Benford analysis:

- Large amounts: 1000000.0000001 vs 1000000.0 (float may round)
- Small amounts: 0.0000001 (float may lose precision)
- Digit boundaries: 999.999... vs 1000.0 (rounding changes leading digit!)

With Decimal arithmetic, digit extraction is always exact.
"""

from decimal import Decimal

import pandas as pd

from utils.decimal_guards import DecimalAmount, validate_amount
from utils.logging import get_logger

logger = get_logger(__name__)


def extract_leading_digit_safe(amount: DecimalAmount | Decimal | str) -> int:
    """Extract the first significant digit from an amount (precision-safe).

    This function uses Decimal arithmetic to extract the leading digit without
    floating-point precision errors. It handles:
    - Positive and negative amounts (uses magnitude)
    - Very large amounts (e.g., 1000000.0000001)
    - Very small amounts (e.g., 0.0000001)
    - Zero (returns 0)

    Parameters
    ----------
    amount : DecimalAmount | Decimal | str
        Amount to extract digit from

    Returns
    -------
    int
        Leading digit (1-9), or 0 if amount is zero

    Examples
    --------
    >>> extract_leading_digit_safe(DecimalAmount("123.45"))
    1
    >>> extract_leading_digit_safe(DecimalAmount("0.00789"))
    7
    >>> extract_leading_digit_safe(DecimalAmount("-456.78"))
    4
    >>> extract_leading_digit_safe(DecimalAmount("0"))
    0

    Notes
    -----
    Benford's Law applies to the magnitude of nonzero values, so negative
    amounts are treated as their absolute value. Zero amounts return 0
    (caller should filter these out before analysis).
    """
    # Convert to Decimal if needed
    if isinstance(amount, DecimalAmount):
        decimal_value = amount.value
    elif isinstance(amount, str):
        decimal_value = validate_amount(amount)
    else:
        decimal_value = amount

    # Take absolute value (Benford applies to magnitudes)
    decimal_value = abs(decimal_value)

    # Handle zero
    if decimal_value == 0:
        return 0

    # Convert to string and extract first non-zero digit
    # This avoids float precision issues with log10
    str_value = str(decimal_value)

    # Remove negative sign, decimal point, and leading zeros
    str_value = str_value.lstrip("-").replace(".", "").lstrip("0")

    if not str_value:
        return 0

    # First character is the leading digit
    leading_digit = int(str_value[0])

    # Sanity check (should be 1-9)
    if not 1 <= leading_digit <= 9:
        logger.warning(f"Invalid leading digit {leading_digit} for amount {amount}")
        return 0

    return leading_digit


def extract_second_digit_safe(amount: DecimalAmount | Decimal | str) -> int:
    """Extract the second significant digit from an amount (precision-safe).

    Similar to extract_leading_digit_safe, but returns the second digit.
    Used for second-digit Benford analysis.

    Parameters
    ----------
    amount : DecimalAmount | Decimal | str
        Amount to extract digit from

    Returns
    -------
    int
        Second digit (0-9), or -1 if amount has only one significant digit

    Examples
    --------
    >>> extract_second_digit_safe(DecimalAmount("123.45"))
    2
    >>> extract_second_digit_safe(DecimalAmount("0.00789"))
    8
    >>> extract_second_digit_safe(DecimalAmount("5.0"))
    0

    Notes
    -----
    If the amount has only one significant digit (e.g., "5"), the second
    digit is considered 0. If the amount is zero or invalid, returns -1.
    """
    # Convert to Decimal if needed
    if isinstance(amount, DecimalAmount):
        decimal_value = amount.value
    elif isinstance(amount, str):
        decimal_value = validate_amount(amount)
    else:
        decimal_value = amount

    # Take absolute value
    decimal_value = abs(decimal_value)

    # Handle zero
    if decimal_value == 0:
        return -1

    # Convert to string and extract second digit
    str_value = str(decimal_value)
    str_value = str_value.lstrip("-").replace(".", "").lstrip("0")

    if len(str_value) < 2:
        # Only one significant digit, second is 0
        return 0

    # Second character is the second digit
    second_digit = int(str_value[1])

    return second_digit


def leading_digits_safe(amounts: pd.Series) -> pd.Series:
    """Extract leading digits from a pandas Series of amounts (precision-safe).

    This is a drop-in replacement for benford_engine.leading_digits() that
    uses Decimal arithmetic for exact digit extraction.

    Parameters
    ----------
    amounts : pd.Series
        Series of amounts (DecimalAmount, Decimal, or convertible values)

    Returns
    -------
    pd.Series
        Series of leading digits (1-9), with zeros and negatives filtered out

    Examples
    --------
    >>> amounts = pd.Series([
    ...     DecimalAmount("123.45"),
    ...     DecimalAmount("456.78"),
    ...     DecimalAmount("0.00789"),
    ... ])
    >>> leading_digits_safe(amounts)
    0    1
    1    4
    2    7
    dtype: int64

    Notes
    -----
    - Zero and negative amounts are filtered out (Benford's Law applies to
      positive magnitudes only)
    - Returns empty Series if all amounts are zero/negative
    - Much more accurate than float-based log10 approach for edge cases
    """
    # Filter positive amounts only (Benford applies to positive values)
    positive_amounts = amounts[amounts > 0]

    if positive_amounts.empty:
        return pd.Series([], dtype=int)

    # Extract leading digits using precision-safe function
    try:
        digits = positive_amounts.apply(extract_leading_digit_safe)
    except Exception as e:
        logger.error(f"Error extracting leading digits: {e}")
        # Fallback to empty series
        return pd.Series([], dtype=int)

    # Filter out any zeros (shouldn't happen, but defensive)
    digits = digits[digits > 0]

    return digits


def second_digits_safe(amounts: pd.Series) -> pd.Series:
    """Extract second digits from a pandas Series of amounts (precision-safe).

    This is a drop-in replacement for benford_engine.second_digits() that
    uses Decimal arithmetic for exact digit extraction.

    Parameters
    ----------
    amounts : pd.Series
        Series of amounts (DecimalAmount, Decimal, or convertible values)

    Returns
    -------
    pd.Series
        Series of second digits (0-9), with zeros and negatives filtered out

    Examples
    --------
    >>> amounts = pd.Series([
    ...     DecimalAmount("123.45"),
    ...     DecimalAmount("456.78"),
    ...     DecimalAmount("0.00789"),
    ... ])
    >>> second_digits_safe(amounts)
    0    2
    1    5
    2    8
    dtype: int64

    Notes
    -----
    - Zero and negative amounts are filtered out
    - Single-digit amounts have second digit of 0
    - Returns empty Series if all amounts are zero/negative
    """
    # Filter positive amounts only
    positive_amounts = amounts[amounts > 0]

    if positive_amounts.empty:
        return pd.Series([], dtype=int)

    # Extract second digits using precision-safe function
    try:
        digits = positive_amounts.apply(extract_second_digit_safe)
    except Exception as e:
        logger.error(f"Error extracting second digits: {e}")
        return pd.Series([], dtype=int)

    # Filter out invalid values (-1)
    digits = digits[digits >= 0]

    return digits


def verify_digit_extraction_accuracy(
    amounts: pd.Series,
    float_based_digits: pd.Series,
    safe_digits: pd.Series,
) -> dict:
    """Compare float-based vs Decimal-based digit extraction to find discrepancies.

    This diagnostic function compares the results of float-based (log10) and
    Decimal-based digit extraction to identify cases where float precision
    causes incorrect digit extraction.

    Parameters
    ----------
    amounts : pd.Series
        Original amounts
    float_based_digits : pd.Series
        Digits extracted using float arithmetic (e.g., np.log10)
    safe_digits : pd.Series
        Digits extracted using Decimal arithmetic

    Returns
    -------
    dict
        Diagnostic report with:
        - total_compared: Number of amounts compared
        - discrepancies: Number of mismatches
        - discrepancy_rate: Percentage of mismatches
        - examples: List of (amount, float_digit, safe_digit) tuples

    Examples
    --------
    >>> amounts = pd.Series([1000000.0000001, 999.9999999])
    >>> float_digits = leading_digits(amounts)  # May have errors
    >>> safe_digits = leading_digits_safe(amounts)  # Exact
    >>> report = verify_digit_extraction_accuracy(amounts, float_digits, safe_digits)
    >>> print(f"Discrepancy rate: {report['discrepancy_rate']:.2%}")
    """
    # Align indices
    common_idx = float_based_digits.index.intersection(safe_digits.index)
    float_aligned = float_based_digits.loc[common_idx]
    safe_aligned = safe_digits.loc[common_idx]
    amounts_aligned = amounts.loc[common_idx]

    # Find mismatches
    mismatches = float_aligned != safe_aligned

    # Collect examples
    examples = []
    if mismatches.any():
        mismatch_idx = mismatches[mismatches].index
        for idx in mismatch_idx[:10]:  # Limit to 10 examples
            examples.append(
                {
                    "amount": amounts_aligned.loc[idx],
                    "float_digit": int(float_aligned.loc[idx]),
                    "safe_digit": int(safe_aligned.loc[idx]),
                }
            )

    report = {
        "total_compared": len(common_idx),
        "discrepancies": int(mismatches.sum()),
        "discrepancy_rate": (
            float(mismatches.sum() / len(common_idx)) if len(common_idx) > 0 else 0.0
        ),
        "examples": examples,
    }

    return report


def log_discrepancy_report(report: dict) -> None:
    """Log a formatted discrepancy report.

    Parameters
    ----------
    report : dict
        Report from verify_digit_extraction_accuracy()
    """
    if report["discrepancies"] == 0:
        logger.info(
            f"No discrepancies found in {report['total_compared']} amounts "
            "(float and Decimal digit extraction agree)"
        )
    else:
        logger.warning(
            f"Found {report['discrepancies']} discrepancies "
            f"({report['discrepancy_rate']:.2%}) in {report['total_compared']} amounts"
        )

        if report["examples"]:
            logger.warning("Example discrepancies:")
            for ex in report["examples"]:
                logger.warning(
                    f"  Amount: {ex['amount']}, "
                    f"Float digit: {ex['float_digit']}, "
                    f"Decimal digit: {ex['safe_digit']}"
                )
