# Numeric Precision Guards

**Location:** `utils/decimal_guards.py`, `utils/benford_precision.py`  
**Status:** Production Ready  
**Issue:** #483

## Quick Start

```python
from utils.decimal_guards import DecimalAmount

# Create amounts (always use strings for exact values)
price = DecimalAmount("100.50")
quantity = DecimalAmount("5")

# All operations are precision-safe
total = price * quantity  # DecimalAmount('502.50')
fee = total * DecimalAmount("0.001")  # 0.1% fee
net = total - fee

# Stellar stroops conversion (7-decimal precision)
stroops = price.to_stroops()  # 1005000000 (int)
recovered = DecimalAmount.from_stroops(stroops)  # Back to DecimalAmount
```

## Why Use This?

❌ **Float arithmetic has precision errors:**
```python
0.1 + 0.2 == 0.3  # False!
1000000.0000001 == 1000000.0  # May be True (precision loss)
```

✅ **Decimal arithmetic is exact:**
```python
DecimalAmount("0.1") + DecimalAmount("0.2") == DecimalAmount("0.3")  # True!
```

## Core Functions

### DecimalAmount Class
Type-safe wrapper for Decimal with validation and Stellar support.

### Validation Functions
- `validate_amount()` - General validation with bounds
- `validate_stellar_amount()` - Stellar-specific (7 decimals, int64 bounds)

### Utility Functions
- `safe_divide()` - Division with zero protection
- `sum_amounts()` - Precision-safe summation
- `safe_float_to_decimal()` - Safe float conversion

### Benford Analysis
- `extract_leading_digit_safe()` - Exact digit extraction
- `leading_digits_safe()` - pandas Series support

## Documentation

See **`docs/numeric_precision.md`** for complete documentation:
- Architecture overview
- Migration guide
- API reference
- Performance considerations
- Troubleshooting

## Testing

```bash
# Run tests
pytest tests/test_decimal_guards.py tests/test_benford_precision.py -v

# Validate codebase
python -m scripts.validate_precision

# Run benchmarks
python -m scripts.benchmark_precision
```

## Performance

Decimal arithmetic is ~10-15x slower than float, but provides exact precision.
**For financial calculations, correctness is more important than performance.**

## Migration

Replace float with DecimalAmount in financial code:

```python
# Before
amount: float = 100.50

# After
amount: DecimalAmount = DecimalAmount("100.50")
```

See `docs/numeric_precision.md` for complete migration guide.
