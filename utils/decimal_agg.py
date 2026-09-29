"""Decimal-safe aggregation utilities for financial reports.

Standard IEEE-754 floating-point arithmetic accumulates rounding errors that
are unacceptable in financial reporting (e.g. a portfolio total that is off by
one cent, or a reconciliation that fails due to epsilon drift).

This module provides aggregation functions that use Python's ``decimal.Decimal``
type internally so that sums, means, and percentage calculations are exact to a
configurable precision.  Results are returned as ``Decimal`` (for downstream
Decimal-aware code) **or** as ``float`` via convenience wrappers, after the
precision-safe computation is complete.

Usage::

    from utils.decimal_agg import decimal_sum, decimal_mean, decimal_weighted_mean
    import pandas as pd

    amounts = pd.Series([0.1, 0.2, 0.3])
    total = decimal_sum(amounts)           # Decimal('0.6') exactly
    avg   = decimal_mean(amounts)          # Decimal('0.2')

    # DataFrame column aggregation
    from utils.decimal_agg import aggregate_column, aggregate_report
    totals = aggregate_report(df, {"amount": "sum", "fee": "sum", "rate": "mean"})

Precision
---------
All internal arithmetic uses ``decimal.ROUND_HALF_EVEN`` (banker's rounding)
with 28-digit precision (Python default).  The ``quantize`` helper rounds a
final result to a specified number of decimal places for display.

Thread safety
-------------
Each function creates its own ``decimal.localcontext`` so concurrent calls
from the streaming pipeline do not interfere with each other.
"""

from __future__ import annotations

import decimal
from collections.abc import Iterable
from decimal import Decimal, localcontext

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Internal precision — 28 digits is Python's default and plenty for Stellar
# amounts (which have at most 7 decimal places).
_INTERNAL_PRECISION = 28

# Default display rounding
_DEFAULT_DISPLAY_PLACES = 7

# Rounding mode: banker's rounding (round half to even) minimises bias
_ROUNDING = decimal.ROUND_HALF_EVEN


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert a numeric value to Decimal safely.

    For ``float`` inputs the conversion goes through ``str`` first to avoid
    the well-known ``Decimal(0.1) == Decimal('0.1000000000000000055...')``
    surprise.

    Note:
        This is the single, intentional input boundary that accepts ``float``
        (Issue #778). Every public function in this module converts any
        incoming ``float`` here and returns ``Decimal``; no function returns a
        ``float``. Keeping this conversion in one place prevents a silent
        ``Decimal``->``float`` loss at a *return* boundary anywhere else.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def quantize(value: Decimal, places: int = _DEFAULT_DISPLAY_PLACES) -> Decimal:
    """Round *value* to *places* decimal digits using banker's rounding.
    
    Uses ROUND_HALF_EVEN (banker's rounding) which rounds 0.5 to the nearest
    even number, minimizing bias in aggregations.
    """
    if places < 0:
        raise ValueError(f"places must be >= 0, got {places}")
    exp = Decimal(10) ** -places
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_PRECISION
        ctx.rounding = _ROUNDING
        return value.quantize(exp)


# ---------------------------------------------------------------------------
# Core aggregation functions
# ---------------------------------------------------------------------------


def decimal_sum(values: Iterable[float | int | str | Decimal]) -> Decimal:
    """Return the exact sum of *values* as a Decimal.

    Uses ROUND_HALF_EVEN (banker's rounding) for internal arithmetic.
    
    ``NaN`` and ``None`` entries are silently skipped (matching pandas
    behaviour for ``Series.sum(skipna=True)``).
    """
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_PRECISION
        ctx.rounding = _ROUNDING
        total = Decimal(0)
        for v in values:
            if v is None:
                continue
            d = to_decimal(v)
            if d.is_nan():
                continue
            total += d
        return total


def decimal_mean(values: Iterable[float | int | str | Decimal]) -> Decimal:
    """Return the exact arithmetic mean of *values* as a Decimal.

    Uses ROUND_HALF_EVEN (banker's rounding) for internal arithmetic.
    
    Raises ``ValueError`` if no valid (non-NaN, non-None) values are found.
    """
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_PRECISION
        ctx.rounding = _ROUNDING
        total = Decimal(0)
        count = 0
        for v in values:
            if v is None:
                continue
            d = to_decimal(v)
            if d.is_nan():
                continue
            total += d
            count += 1
        if count == 0:
            raise ValueError("Cannot compute mean of empty sequence.")
        return total / count


def decimal_max(values: Iterable[float | int | str | Decimal]) -> Decimal:
    """Return the maximum of *values* as a Decimal.

    Raises ``ValueError`` if no valid values are found.
    """
    result: Decimal | None = None
    for v in values:
        if v is None:
            continue
        d = to_decimal(v)
        if d.is_nan():
            continue
        if result is None or d > result:
            result = d
    if result is None:
        raise ValueError("Cannot compute max of empty sequence.")
    return result


def decimal_min(values: Iterable[float | int | str | Decimal]) -> Decimal:
    """Return the minimum of *values* as a Decimal.

    Raises ``ValueError`` if no valid values are found.
    """
    result: Decimal | None = None
    for v in values:
        if v is None:
            continue
        d = to_decimal(v)
        if d.is_nan():
            continue
        if result is None or d < result:
            result = d
    if result is None:
        raise ValueError("Cannot compute min of empty sequence.")
    return result


def decimal_weighted_mean(
    values: Iterable[float | int | str | Decimal],
    weights: Iterable[float | int | str | Decimal],
) -> Decimal:
    """Return the weighted mean using exact Decimal arithmetic.

    Each ``(value, weight)`` pair contributes ``value * weight`` to the
    numerator and ``weight`` to the denominator.  Pairs where either the
    value or the weight is ``None`` / ``NaN`` are skipped.

    Raises ``ValueError`` if total weight is zero.
    """
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_PRECISION
        ctx.rounding = _ROUNDING
        numerator = Decimal(0)
        denominator = Decimal(0)
        for v, w in zip(values, weights, strict=False):
            if v is None or w is None:
                continue
            dv = to_decimal(v)
            dw = to_decimal(w)
            if dv.is_nan() or dw.is_nan():
                continue
            numerator += dv * dw
            denominator += dw
        if denominator == 0:
            raise ValueError("Total weight is zero; cannot compute weighted mean.")
        return numerator / denominator


def decimal_percentage(
    part: float | int | str | Decimal, whole: float | int | str | Decimal
) -> Decimal:
    """Return ``(part / whole) * 100`` as a Decimal percentage.

    Raises ``ValueError`` if *whole* is zero.
    """
    d_part = to_decimal(part)
    d_whole = to_decimal(whole)
    if d_whole == 0:
        raise ValueError("Cannot compute percentage with zero denominator.")
    with localcontext() as ctx:
        ctx.prec = _INTERNAL_PRECISION
        ctx.rounding = _ROUNDING
        return (d_part / d_whole) * Decimal(100)


# ---------------------------------------------------------------------------
# pandas integration
# ---------------------------------------------------------------------------

_AGG_FUNCTIONS = {
    "sum": decimal_sum,
    "mean": decimal_mean,
    "max": decimal_max,
    "min": decimal_min,
}


def aggregate_column(
    series: pd.Series,
    agg: str = "sum",
    places: int = _DEFAULT_DISPLAY_PLACES,
) -> Decimal:
    """Aggregate a pandas Series using decimal-safe arithmetic.

    Parameters
    ----------
    series : pd.Series
        Numeric column to aggregate.
    agg : str
        Aggregation function name: ``"sum"``, ``"mean"``, ``"max"``, or ``"min"``.
    places : int
        Decimal places for the result.

    Returns
    -------
    Decimal
        Aggregated, quantized result.
    """
    func = _AGG_FUNCTIONS.get(agg)
    if func is None:
        raise ValueError(f"Unknown aggregation '{agg}'. Choose from: {sorted(_AGG_FUNCTIONS)}")
    raw = func(series.dropna())
    return quantize(raw, places)


def aggregate_report(
    df: pd.DataFrame,
    column_aggs: dict[str, str],
    places: int = _DEFAULT_DISPLAY_PLACES,
) -> dict[str, Decimal]:
    """Aggregate multiple columns of a report DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The report data.
    column_aggs : dict[str, str]
        Mapping of column name to aggregation function name.
    places : int
        Decimal places for each result.

    Returns
    -------
    dict[str, Decimal]
        Column-name to aggregated Decimal value.

    Raises
    ------
    KeyError
        If a column is not found in the DataFrame.
    """
    results: dict[str, Decimal] = {}
    for col, agg in column_aggs.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        results[col] = aggregate_column(df[col], agg, places)
    return results
