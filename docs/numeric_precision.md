# Numeric Precision Guards for Financial Calculations

**Status:** ✅ Implemented  
**Issue:** #483 (Stellar Wave Advanced Build)  
**Author:** Product Labo Team  
**Last Updated:** 2026-07-29

## Overview

This document describes the numeric precision guard system for LedgerLens-data, a comprehensive solution for handling financial calculations with exact decimal arithmetic. The system prevents precision loss errors common with floating-point arithmetic and provides Stellar blockchain-specific features for handling stroops (7-decimal precision).

### Why This Matters

Floating-point arithmetic has inherent precision issues that are unacceptable for financial calculations:

```python
# ❌ Float precision problem
0.1 + 0.2 == 0.3  # False! Result is 0.30000000000000004

# ❌ Loss of precision in large numbers
amount = 1000000.0000001
amount == 1000000.0  # May be True due to float precision limits

# ✅ Decimal arithmetic is exact
from decimal import Decimal
Decimal('0.1') + Decimal('0.2') == Decimal('0.3')  # True!
```

For fraud detection systems analyzing blockchain transactions, even tiny precision errors can:
- Cause false positives/negatives in anomaly detection
- Break Benford's Law analysis (requires exact digit extraction)
- Corrupt volume aggregations and statistical analysis
- Violate Stellar network constraints (7-decimal stroops precision)

## Architecture

### Core Components

```
utils/decimal_guards.py          # Core precision system
├── DecimalAmount                 # Type-safe wrapper for Decimal
├── validate_amount()             # Amount validation with bounds
├── validate_stellar_amount()     # Stellar-specific validation
├── decimal_context()             # Precision context manager
├── safe_divide()                 # Division with zero protection
├── sum_amounts()                 # Precision-safe summation
└── safe_float_to_decimal()       # Safe float conversion

tests/test_decimal_guards.py     # Comprehensive test suite (80+ tests)
scripts/validate_precision.py    # CLI validation tool
docs/numeric_precision.md        # This documentation
```

### Design Principles

1. **Decimal over float**: All financial calculations use Python's `decimal.Decimal` type
2. **Type safety**: `DecimalAmount` wrapper provides type-safe operations
3. **Automatic validation**: All inputs validated for NaN, Infinity, bounds
4. **Stellar integration**: Native stroops conversion with 7-decimal precision
5. **Explicit float handling**: Warns on float input, converts via string
6. **Property-based testing**: Hypothesis tests verify arithmetic properties

## Quick Start

### Basic Usage

```python
from utils.decimal_guards import DecimalAmount

# Create amounts (from string, int, or Decimal)
price = DecimalAmount("100.50")
quantity = DecimalAmount(5)

# Arithmetic operations (all precision-safe)
total = price * quantity  # DecimalAmount('502.50')
fee = total * DecimalAmount("0.001")  # 0.1% fee
net = total - fee

# Comparisons
if total > DecimalAmount("500"):
    print("Large transaction")

# Stellar stroops conversion
stroops = price.to_stroops()  # 1005000000 (int)
recovered = DecimalAmount.from_stroops(stroops)  # Back to DecimalAmount
```

### Stellar Blockchain Integration

```python
from utils.decimal_guards import (
    DecimalAmount,
    validate_stellar_amount,
    STELLAR_MAX_AMOUNT,
    STELLAR_PRECISION,
)

# Validate Stellar amounts (7 decimal max, bounds checking)
amount = validate_stellar_amount("123.4567890")  # ✅ OK: 7 decimals
amount = validate_stellar_amount("123.45678901")  # ❌ Error: 8 decimals

# Stroops conversion (Stellar's base unit: 1 stroop = 0.0000001 XLM)
xlm_amount = DecimalAmount("100.5000000")
stroops = xlm_amount.to_stroops()  # 1005000000

# Reverse conversion
recovered = DecimalAmount.from_stroops(stroops)  # DecimalAmount('100.5000000')

# Bounds checking (int64 limits)
max_xlm = DecimalAmount(STELLAR_MAX_AMOUNT)  # 922337203685.4775807
max_stroops = max_xlm.to_stroops()  # 9223372036854775807 (max int64)
```

### Validation Functions

```python
from utils.decimal_guards import validate_amount, validate_stellar_amount

# Basic validation
amount = validate_amount("100.50")  # Returns Decimal

# With bounds
amount = validate_amount(
    "50.00",
    min_value="0",
    max_value="100",
    allow_negative=False,
)

# Stellar-specific (7 decimals, int64 bounds)
stellar_amount = validate_stellar_amount("100.5000000")

# Validation errors
validate_amount("NaN")  # ❌ AmountValidationError: NaN not allowed
validate_amount("-10", allow_negative=False)  # ❌ Negative not allowed
validate_stellar_amount("100.12345678")  # ❌ Too many decimals
```

### Safe Utility Functions

```python
from utils.decimal_guards import safe_divide, sum_amounts, safe_float_to_decimal

# Division with zero protection
result = safe_divide("100", "0", default="999")  # Returns DecimalAmount('999')

# Precision-safe summation
amounts = [DecimalAmount("100"), "50.5", Decimal("25")]
total = sum_amounts(amounts)  # DecimalAmount('175.5')

# Float conversion (with warning)
decimal_value = safe_float_to_decimal(123.45, precision=2)  # Rounds to 2 decimals
```

### Precision Context Management

```python
from utils.decimal_guards import decimal_context

# Temporarily change precision and rounding
with decimal_context(precision=10, rounding="ROUND_DOWN"):
    result = DecimalAmount("1") / DecimalAmount("3")
    # Result has 10 significant digits, rounded down

# Context automatically restores after block
```

## Integration Guide

### Migrating Existing Code

#### Step 1: Identify Financial Variables

Find all variables holding financial values:
- `amount`, `volume`, `price`, `balance`, `value`, `total`, `fee`, etc.
- Stellar-specific: `stroops`, `xlm_amount`, etc.

#### Step 2: Replace float with DecimalAmount

**Before:**
```python
def calculate_fee(amount: float, rate: float) -> float:
    return amount * rate

total = 100.50
fee = calculate_fee(total, 0.001)  # Float precision issues
```

**After:**
```python
from utils.decimal_guards import DecimalAmount

def calculate_fee(amount: DecimalAmount, rate: DecimalAmount) -> DecimalAmount:
    return amount * rate

total = DecimalAmount("100.50")
fee = calculate_fee(total, DecimalAmount("0.001"))  # Exact!
```

#### Step 3: Update Data Models

**Before:**
```python
from dataclasses import dataclass

@dataclass
class Trade:
    amount: float
    price: float
```

**After:**
```python
from dataclasses import dataclass
from decimal import Decimal
from utils.decimal_guards import validate_stellar_amount

@dataclass
class Trade:
    amount: Decimal
    price: Decimal
    
    def __post_init__(self):
        # Validate on initialization
        self.amount = validate_stellar_amount(self.amount)
        self.price = validate_stellar_amount(self.price)
```

#### Step 4: Update Database Interactions

Store as Decimal or integer stroops, never as float:

```python
# Option 1: Store as Decimal string
df["amount"] = df["amount"].apply(lambda x: str(DecimalAmount(x)))

# Option 2: Store as integer stroops (Stellar-specific)
df["amount_stroops"] = df["amount"].apply(
    lambda x: DecimalAmount(x).to_stroops()
)
```

### Integration with Benford Analysis

For Benford's Law fraud detection, exact digit extraction is critical:

```python
from utils.decimal_guards import DecimalAmount

def extract_leading_digit(amount: DecimalAmount) -> int:
    """Extract first significant digit for Benford analysis."""
    # Get string representation without decimal point
    amount_str = str(abs(amount.value)).replace(".", "").lstrip("0")
    
    if not amount_str:
        return 0
    
    return int(amount_str[0])

# Usage
amounts = [DecimalAmount("123.45"), DecimalAmount("456.78")]
leading_digits = [extract_leading_digit(amt) for amt in amounts]
# [1, 4] - exact digits for Benford distribution analysis
```

### Integration with Data Ingestion

```python
from utils.decimal_guards import DecimalAmount, validate_stellar_amount
import pandas as pd

def process_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw trade data with precision validation."""
    
    # Convert amount columns to DecimalAmount
    for col in ["amount", "price", "volume"]:
        if col in df.columns:
            # Validate each value
            df[col] = df[col].apply(
                lambda x: validate_stellar_amount(x) if pd.notna(x) else None
            )
    
    # Calculate derived values with precision
    if "amount" in df.columns and "price" in df.columns:
        df["total_value"] = df.apply(
            lambda row: (
                DecimalAmount(row["amount"]) * DecimalAmount(row["price"])
                if pd.notna(row["amount"]) and pd.notna(row["price"])
                else None
            ),
            axis=1,
        )
    
    return df
```

## Validation CLI Tool

The `validate_precision.py` CLI tool scans codebases and datasets for precision issues.

### Usage

```bash
# Scan entire codebase
python -m scripts.validate_precision

# Scan specific modules
python -m scripts.validate_precision --modules detection ingestion

# Validate a dataset
python -m scripts.validate_precision --validate-dataset data/trades.parquet

# JSON output for CI integration
python -m scripts.validate_precision --json > precision_report.json

# Quiet mode (summary only)
python -m scripts.validate_precision --quiet
```

### What It Detects

1. **Float arithmetic on financial values**
   - Binary operations (+, -, *, /) on variables with financial keywords
   - Missing Decimal imports

2. **Dangerous conversions**
   - `float()` calls on financial values
   - `round()` without Decimal context

3. **Type annotation issues**
   - Financial variables annotated as `float`

4. **Dataset issues**
   - Float64 columns for financial data
   - Excess precision (>7 decimals for Stellar)

### Exit Codes

- `0`: No issues found
- `1`: Warnings found (non-critical)
- `2`: Errors found (critical precision issues)

### Example Output

```
================================================================================
Numeric Precision Validation Report
================================================================================

Summary
  Total issues: 12
  Errors:   3
  Warnings: 8
  Info:     1

Issues by Type
  float_arithmetic: 5
  float_conversion: 3
  float_annotation: 3
  float_column: 1

Files with Most Issues
    5 issues: detection/benford_engine.py
    4 issues: ingestion/data_models.py
    3 issues: detection/volume_analysis.py

Detailed Issues

✗ ERROR [float_conversion]
  File: detection/benford_engine.py:45
  Converting financial value to float (precision loss)
  💡 Suggestion: Use DecimalAmount or Decimal instead

⚠ WARNING [float_arithmetic]
  File: ingestion/data_models.py:23
  Arithmetic operation on potential financial value without Decimal import
  💡 Suggestion: Import and use DecimalAmount from utils.decimal_guards
```

## Performance Considerations

### Decimal vs Float Performance

Decimal arithmetic is slower than float arithmetic:

```python
# Approximate performance comparison
Float operations:     ~50-100 ns per operation
Decimal operations:   ~500-1000 ns per operation  (10x slower)
DecimalAmount:        ~600-1200 ns per operation  (validation overhead)
```

### When Performance Matters

1. **Use Decimal for financial calculations** (always)
   - Correctness > performance for monetary values
   - Errors from float precision cost more than Decimal overhead

2. **Optimize hot paths**
   - Batch operations when possible
   - Cache converted values
   - Use Decimal directly if DecimalAmount overhead matters

3. **Consider stroops for storage**
   - Store as int64 stroops in databases (fast, compact)
   - Convert to DecimalAmount only for calculations
   - Example: 100.5 XLM → 1005000000 stroops (int64)

### Benchmarking

```python
import timeit
from decimal import Decimal
from utils.decimal_guards import DecimalAmount

# Float arithmetic (baseline)
float_time = timeit.timeit(
    "a + b",
    setup="a, b = 100.5, 50.25",
    number=1000000,
)

# Decimal arithmetic
decimal_time = timeit.timeit(
    "a + b",
    setup="from decimal import Decimal; a, b = Decimal('100.5'), Decimal('50.25')",
    number=1000000,
)

# DecimalAmount arithmetic
decimalamount_time = timeit.timeit(
    "a + b",
    setup="from utils.decimal_guards import DecimalAmount; a, b = DecimalAmount('100.5'), DecimalAmount('50.25')",
    number=1000000,
)

print(f"Float:         {float_time:.3f}s (baseline)")
print(f"Decimal:       {decimal_time:.3f}s ({decimal_time/float_time:.1f}x slower)")
print(f"DecimalAmount: {decimalamount_time:.3f}s ({decimalamount_time/float_time:.1f}x slower)")
```

## Testing

### Running Tests

```bash
# Run all precision tests
pytest tests/test_decimal_guards.py -v

# Run with Hypothesis statistics
pytest tests/test_decimal_guards.py -v --hypothesis-show-statistics

# Run specific test class
pytest tests/test_decimal_guards.py::TestValidation -v

# Run property-based tests only
pytest tests/test_decimal_guards.py::TestArithmeticProperties -v
```

### Test Coverage

The test suite includes 80+ test cases covering:

1. **Validation Tests** (15 tests)
   - String/int/Decimal/float conversion
   - NaN/Infinity rejection
   - Bounds checking
   - Stellar-specific validation

2. **Arithmetic Tests** (12 tests)
   - All operators (+, -, *, /, //, %, **)
   - Reverse operations (radd, rsub, etc.)
   - Precision maintenance

3. **Comparison Tests** (8 tests)
   - Equality, inequality, ordering
   - Hashing, set usage

4. **Stroops Conversion Tests** (12 tests)
   - to_stroops, from_stroops
   - Roundtrips, edge cases
   - Max/min values, fractional stroops

5. **Utility Tests** (10 tests)
   - safe_divide, sum_amounts
   - safe_float_to_decimal
   - check_precision_loss

6. **Context Tests** (3 tests)
   - Precision changes
   - Context restoration

7. **Method Tests** (10 tests)
   - Rounding, float conversion
   - is_zero/positive/negative
   - Formatting

8. **Edge Cases** (10 tests)
   - Very small/large values
   - Zero operations
   - Negative operations
   - Chained operations

9. **Property-Based Tests** (10 tests with Hypothesis)
   - Commutativity: a + b = b + a
   - Associativity: (a + b) + c = a + (b + c)
   - Identity: a + 0 = a, a * 1 = a
   - Inverse: a - a = 0
   - Stroops roundtrip

10. **Integration Tests** (4 tests)
    - Realistic trade scenarios
    - Volume aggregation
    - Precision-critical calculations
    - Benford analysis preparation

## API Reference

### DecimalAmount Class

```python
class DecimalAmount:
    """Type-safe wrapper for Decimal with validation and Stellar support."""
    
    def __init__(self, value: str | int | Decimal | float) -> None:
        """Create DecimalAmount with validation."""
    
    # Arithmetic operations
    def __add__(self, other) -> DecimalAmount: ...
    def __sub__(self, other) -> DecimalAmount: ...
    def __mul__(self, other) -> DecimalAmount: ...
    def __truediv__(self, other) -> DecimalAmount: ...
    def __floordiv__(self, other) -> DecimalAmount: ...
    def __mod__(self, other) -> DecimalAmount: ...
    def __pow__(self, other) -> DecimalAmount: ...
    
    # Comparison operations
    def __eq__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    
    # Stellar stroops conversion
    def to_stroops(self) -> int:
        """Convert to Stellar stroops (7-decimal precision)."""
    
    @classmethod
    def from_stroops(cls, stroops: int) -> DecimalAmount:
        """Create from Stellar stroops."""
    
    # Utility methods
    def round(self, decimal_places: int = STELLAR_PRECISION) -> DecimalAmount:
        """Round to specified decimal places (default 7 for Stellar)."""
    
    def to_float(self) -> float:
        """Convert to float (with precision loss warning)."""
    
    def is_zero(self) -> bool:
        """Check if value is zero."""
    
    def is_positive(self) -> bool:
        """Check if value is positive."""
    
    def is_negative(self) -> bool:
        """Check if value is negative."""
```

### Validation Functions

```python
def validate_amount(
    value: str | int | Decimal | float,
    allow_negative: bool = True,
    min_value: str | Decimal | None = None,
    max_value: str | Decimal | None = None,
) -> Decimal:
    """Validate and convert amount to Decimal."""

def validate_stellar_amount(value: str | int | Decimal) -> Decimal:
    """Validate amount for Stellar blockchain (7 decimals, int64 bounds)."""
```

### Utility Functions

```python
def safe_divide(
    numerator: str | Decimal,
    denominator: str | Decimal,
    default: str | Decimal = "0",
) -> DecimalAmount:
    """Divide with zero protection."""

def sum_amounts(amounts: list[str | Decimal | DecimalAmount]) -> DecimalAmount:
    """Sum multiple amounts with precision."""

def safe_float_to_decimal(
    value: float,
    precision: int | None = None,
) -> Decimal:
    """Convert float to Decimal safely (with warning)."""

def check_precision_loss(
    original: float,
    converted: Decimal,
    tolerance: Decimal = Decimal("1e-10"),
) -> bool:
    """Check if float→Decimal conversion lost precision."""
```

### Context Manager

```python
@contextmanager
def decimal_context(
    precision: int = 28,
    rounding: str = "ROUND_HALF_EVEN",
) -> Iterator[None]:
    """Temporarily change Decimal precision and rounding mode."""
```

### Constants

```python
STELLAR_PRECISION = 7  # Stellar uses 7 decimal places
STROOPS_MULTIPLIER = Decimal(10) ** 7  # 10,000,000
STELLAR_MAX_AMOUNT = Decimal("922337203685.4775807")  # Max int64 stroops
STELLAR_MIN_AMOUNT = Decimal("-922337203685.4775807")  # Min int64 stroops
MAX_SAFE_PRECISION = 100  # Maximum recommended precision
DEFAULT_ROUNDING = "ROUND_HALF_EVEN"  # Banker's rounding
```

### Exceptions

```python
class AmountValidationError(ValueError):
    """Raised when amount validation fails."""

class PrecisionOverflowError(ValueError):
    """Raised when amount exceeds safe precision bounds."""

class StroopsConversionError(ValueError):
    """Raised when stroops conversion fails."""

class PrecisionUnderflowError(ValueError):
    """Raised when amount is too small to represent."""
```

## Troubleshooting

### Common Issues

#### Issue: "Float inputs may have precision errors"

**Cause:** Creating `DecimalAmount` from float

**Solution:**
```python
# ❌ Avoid
amount = DecimalAmount(100.5)  # Warning!

# ✅ Use string instead
amount = DecimalAmount("100.5")  # No warning
```

#### Issue: "Converting float to Decimal may lose precision"

**Cause:** Using `to_float()` method

**Solution:**
```python
# ❌ Avoid float conversion
float_value = amount.to_float()  # Warning!

# ✅ Keep as DecimalAmount
# Use DecimalAmount throughout calculations
```

#### Issue: "Fractional stroops not allowed"

**Cause:** Amount has more than 7 decimal places

**Solution:**
```python
# ❌ Too many decimals
amount = DecimalAmount("100.12345678")  # 8 decimals
stroops = amount.to_stroops()  # Error!

# ✅ Round to 7 decimals first
amount = DecimalAmount("100.12345678").round(7)  # Now 7 decimals
stroops = amount.to_stroops()  # OK
```

#### Issue: "Amount exceeds maximum Stellar value"

**Cause:** Amount larger than int64 max when converted to stroops

**Solution:**
```python
from utils.decimal_guards import STELLAR_MAX_AMOUNT

# ❌ Too large
huge_amount = DecimalAmount("1000000000000")  # > max
stroops = huge_amount.to_stroops()  # Error!

# ✅ Check bounds first
if amount.value <= STELLAR_MAX_AMOUNT:
    stroops = amount.to_stroops()
else:
    # Handle overflow
    pass
```

### Debugging Tips

1. **Enable logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check precision context**
   ```python
   from decimal import getcontext
   print(getcontext())  # Shows current precision and rounding
   ```

3. **Validate intermediate results**
   ```python
   result = amount1 + amount2
   assert isinstance(result, DecimalAmount)
   assert result.value.is_finite()
   ```

4. **Use validation CLI**
   ```bash
   python -m scripts.validate_precision --modules detection
   ```

## Future Enhancements

Potential improvements for future iterations:

1. **Performance Optimizations**
   - C extension for hot paths
   - Cython compilation
   - NumPy integration for bulk operations

2. **Additional Features**
   - Currency-aware amounts (USD, EUR, XLM)
   - Exchange rate conversions
   - Tax calculation helpers

3. **Tooling**
   - Pre-commit hook for precision validation
   - IDE plugin for real-time warnings
   - Auto-fix for simple issues

4. **Integration**
   - SQLAlchemy custom types
   - Pandas extension types
   - Arrow/Parquet optimized storage

## References

- [Python Decimal Module](https://docs.python.org/3/library/decimal.html)
- [Stellar Network Precision](https://developers.stellar.org/docs/glossary#stroop)
- [IEEE 754 Floating Point](https://en.wikipedia.org/wiki/IEEE_754)
- [Benford's Law](https://en.wikipedia.org/wiki/Benford%27s_law)
- [Hypothesis Testing](https://hypothesis.readthedocs.io/)

## Support

For issues or questions:
- GitHub Issues: [LedgerLens-data/issues](https://github.com/product-labo/Ledgerlens-data/issues)
- Documentation: `docs/numeric_precision.md`
- Tests: `tests/test_decimal_guards.py`
- Code: `utils/decimal_guards.py`

---

**License:** MIT  
**Maintainer:** Product Labo Team  
**Status:** Production Ready
