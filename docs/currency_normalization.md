# Currency and Amount Normalization

**Status:** ✅ Implemented  
**Issue:** #484 (Stellar Wave Advanced Build)  
**Author:** Product Labo Team  
**Last Updated:** 2026-07-29

## Overview

This document describes the currency and amount normalization system for LedgerLens-data, a comprehensive solution for standardizing amounts across different asset pairs to enable meaningful cross-asset comparisons in fraud detection.

### Why This Matters

Stellar DEX supports hundreds of trading pairs with different assets. Detecting anomalies requires comparing volumes and prices across these pairs, but direct comparisons are meaningless without normalization:

```python
# ❌ Meaningless comparison
volume_usdc_xlm = 10000  # 10,000 USDC traded
volume_btc_xlm = 0.5     # 0.5 BTC traded
# Which is more? Can't tell without exchange rates!

# ✅ Normalized comparison
norm_usdc = normalize_to_xlm(10000, "USDC")  # 85,000 XLM
norm_btc = normalize_to_xlm(0.5, "BTC")       # 300,000 XLM
# Now we can compare: BTC volume is ~3.5x larger
```

For fraud detection analyzing Stellar blockchain transactions:
- **Benford's Law analysis** - Requires amounts in common currency for distribution
- **Volume anomaly detection** - Compare volumes across different pairs
- **Wash trading detection** - Aggregate activity across multiple asset pairs
- **Cross-pair correlation** - Find coordinated manipulation across markets

## Architecture

### Core Components

```
utils/currency_normalization.py        # Core system (1,200+ lines)
├── ExchangeRateProvider (Protocol)    # Pluggable rate sources
│   ├── get_rate(from, to, timestamp)
│   └── get_rates_batch(pairs, timestamp)
│
├── Data Structures
│   ├── CurrencyPair                   # Exchange rate with metadata
│   ├── NormalizedAmount               # Result with provenance
│   ├── AssetMetadata                  # Asset classification
│   └── Enums (AssetType, StablecoinType, NormalizationStatus)
│
├── Providers
│   ├── MockExchangeRateProvider       # Testing/development
│   └── CachedRateProvider             # TTL caching wrapper
│
├── Asset Classification
│   └── AssetClassifier                # Detect stablecoins, native, tokens
│
└── Normalization Strategies
    ├── XLMNormalization               # Convert to XLM (native)
    ├── USDNormalization               # Convert to USD (via USDC)
    └── MultiHopNormalization          # Multi-hop via liquid pairs

utils/normalization_helpers.py         # Helper utilities
├── normalize_trade_amounts_to_series() # pandas integration
├── calculate_normalized_volume()       # Total volume calculation
├── compare_cross_pair_volumes()        # Cross-pair comparison
├── detect_cross_pair_anomalies()       # Anomaly detection
└── create_normalized_dataframe()       # Bulk analysis

ingestion/data_models.py               # Model integration
├── Trade.normalize_base_amount()
├── Trade.normalize_counter_amount()
├── Trade.normalize_both_amounts()
├── Trade.get_normalized_trade_value()
├── OrderBookEvent.normalize_amount()
└── OrderBookEvent.get_normalized_order_value()

scripts/validate_normalization.py     # Validation CLI
scripts/benchmark_normalization.py    # Performance benchmarks
tests/test_currency_normalization.py  # Test suite (50+ tests)
```



### Design Principles

1. **Protocol-based providers** - Pluggable exchange rate sources
2. **Immutable results** - NormalizedAmount preserves provenance for auditing
3. **Confidence scoring** - Weight by liquidity, staleness, conversion path
4. **Multi-hop support** - Convert via intermediate liquid pairs when needed
5. **Timestamp-aware** - Historical rates for backtesting
6. **Decimal precision** - Integrate with decimal_guards for exactness
7. **Lazy evaluation** - Fetch rates only when needed
8. **Caching strategy** - TTL cache for rate providers

## Quick Start

### Basic Normalization

```python
from utils.currency_normalization import create_xlm_strategy
from ingestion.data_models import Asset
from utils.decimal_guards import DecimalAmount

# Create strategy (with default mock provider)
strategy = create_xlm_strategy()

# Define assets
usdc = Asset(code="USDC", issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")
xlm = Asset(code="XLM", issuer=None)

# Normalize amount
amount = DecimalAmount("100")  # 100 USDC
normalized = strategy.normalize(amount, usdc)

print(f"100 USDC = {normalized.value} XLM")
# Output: 100 USDC = 850.0 XLM (at 8.5 rate)
print(f"Confidence: {normalized.confidence}")
# Output: Confidence: 0.95
```

### Trade Normalization

```python
from utils.currency_normalization import create_xlm_strategy

# Load trade
trade = Trade(...)  # From database or API

# Create strategy
strategy = create_xlm_strategy()

# Normalize both sides
norm_base, norm_counter = trade.normalize_both_amounts(strategy)

# Compare (should be equal for valid trade)
if abs(norm_base.value - norm_counter.value) > DecimalAmount("0.01"):
    print("⚠️ Trade amounts don't match after normalization!")
    print(f"Base: {norm_base.value} XLM")
    print(f"Counter: {norm_counter.value} XLM")
```

### Multi-Asset Aggregation

```python
from utils.currency_normalization import aggregate_normalized, create_xlm_strategy

# Portfolio holdings across different assets
holdings = [
    (DecimalAmount("1000"), usdc_asset),   # 1000 USDC
    (DecimalAmount("500"), usdt_asset),    # 500 USDT
    (DecimalAmount("5000"), xlm_asset),    # 5000 XLM
]

# Create strategy
strategy = create_xlm_strategy()

# Aggregate to XLM
total = aggregate_normalized(
    holdings,
    xlm_asset,
    strategy.provider,
)

print(f"Total portfolio value: {total.value} XLM")
# Output: Total portfolio value: 17700.0 XLM
```

## Core Concepts

### Exchange Rate Provider

The `ExchangeRateProvider` protocol defines how to fetch exchange rates:

```python
from typing import Protocol
from datetime import datetime

class ExchangeRateProvider(Protocol):
    def get_rate(
        self,
        from_asset: Asset,
        to_asset: Asset,
        timestamp: datetime | None = None,
    ) -> CurrencyPair | None:
        """Get exchange rate between two assets."""
        ...
    
    def get_rates_batch(
        self,
        pairs: list[tuple[Asset, Asset]],
        timestamp: datetime | None = None,
    ) -> dict[tuple[str, str], CurrencyPair]:
        """Get multiple rates in batch."""
        ...
    
    def is_available(self, asset: Asset) -> bool:
        """Check if rates available for asset."""
        ...
```

**Implementations:**
- `MockExchangeRateProvider` - For testing with configurable rates
- `CachedRateProvider` - Wrapper adding TTL caching
- (Future) `StellarDEXRateProvider` - Real rates from DEX
- (Future) `ExternalOracleProvider` - CoinGecko, Band Protocol, etc.



### Currency Pair

Represents an exchange rate between two assets:

```python
@dataclass(frozen=True)
class CurrencyPair:
    from_asset: Asset
    to_asset: Asset
    rate: Decimal              # 1 from_asset = rate * to_asset
    timestamp: datetime
    source: str                # "mock", "stellar_dex", "oracle"
    liquidity: Decimal | None  # Optional liquidity indicator
    confidence: Decimal        # 0-1 confidence score
    
    def inverse(self) -> CurrencyPair:
        """Return inverse pair (swap from/to)."""
    
    def is_stale(self, threshold: timedelta) -> bool:
        """Check if rate is too old."""
```

**Example:**
```python
# 1 USDC = 8.5 XLM
pair = CurrencyPair(
    from_asset=usdc,
    to_asset=xlm,
    rate=Decimal("8.5"),
    timestamp=datetime.now(),
    source="stellar_dex",
    confidence=Decimal("0.95"),
)

# Get inverse: 1 XLM = 0.1176 USDC
inv = pair.inverse()
```

### Normalized Amount

Result of normalization with full provenance:

```python
@dataclass(frozen=True)
class NormalizedAmount:
    value: Decimal                     # Amount in base currency
    base_asset: Asset                  # Base currency
    original_value: Decimal            # Original amount
    original_asset: Asset              # Original asset
    exchange_rate: CurrencyPair | None # Rate used (None if same currency)
    confidence: Decimal                # 0-1 confidence
    status: NormalizationStatus        # SUCCESS, NO_RATE, STALE_RATE, etc.
    conversion_path: list[Asset]       # Path for multi-hop
    
    def is_successful(self) -> bool:
        """Check if normalization succeeded."""
    
    def is_same_currency(self) -> bool:
        """Check if no conversion was needed."""
```

**Example:**
```python
normalized = strategy.normalize(DecimalAmount("100"), usdc_asset)

print(f"Original: {normalized.original_value} {normalized.original_asset.code}")
print(f"Normalized: {normalized.value} {normalized.base_asset.code}")
print(f"Rate: {normalized.exchange_rate.rate}")
print(f"Confidence: {normalized.confidence}")
print(f"Status: {normalized.status}")

# Output:
# Original: 100 USDC
# Normalized: 850.0 XLM
# Rate: 8.5
# Confidence: 0.95
# Status: NormalizationStatus.SUCCESS
```

### Asset Classification

Automatically classify assets for better normalization decisions:

```python
from utils.currency_normalization import AssetClassifier, AssetType

classifier = AssetClassifier()

# Classify assets
xlm_metadata = classifier.classify(xlm_asset)
print(xlm_metadata.asset_type)  # AssetType.NATIVE
print(xlm_metadata.liquidity_score)  # 1.0

usdc_metadata = classifier.classify(usdc_asset)
print(usdc_metadata.asset_type)  # AssetType.STABLECOIN
print(usdc_metadata.stablecoin_type)  # StablecoinType.FIAT_BACKED

# Helper methods
classifier.is_stablecoin(usdc_asset)  # True
classifier.is_native(xlm_asset)  # True
classifier.get_preferred_base(usdc_asset)  # Returns XLM asset
```

## Normalization Strategies

### XLM Normalization

Convert all amounts to XLM (Stellar native currency):

```python
from utils.currency_normalization import XLMNormalization, MockExchangeRateProvider

provider = MockExchangeRateProvider()
strategy = XLMNormalization(provider)

# Normalize various assets to XLM
norm_usdc = strategy.normalize(DecimalAmount("100"), usdc_asset)
norm_usdt = strategy.normalize(DecimalAmount("100"), usdt_asset)
norm_xlm = strategy.normalize(DecimalAmount("100"), xlm_asset)

# All now in XLM
print(f"USDC: {norm_usdc.value} XLM")  # 850.0
print(f"USDT: {norm_usdt.value} XLM")  # 840.0
print(f"XLM: {norm_xlm.value} XLM")    # 100.0 (identity)
```

**When to use:**
- Default choice for Stellar DEX analysis
- XLM has most liquid trading pairs
- Native asset, universally available

### USD Normalization

Convert all amounts to USD equivalent (via USDC):

```python
from utils.currency_normalization import USDNormalization

provider = MockExchangeRateProvider()
strategy = USDNormalization(provider)

# Normalize to USD
norm_xlm = strategy.normalize(DecimalAmount("850"), xlm_asset)
print(f"850 XLM = {norm_xlm.value} USD")  # 100.0 USD

# Stablecoins treated as 1:1
norm_usdc = strategy.normalize(DecimalAmount("100"), usdc_asset)
print(f"100 USDC = {norm_usdc.value} USD")  # 100.0 USD (confidence 0.99)
```

**When to use:**
- Reporting in familiar currency
- International comparisons
- Stablecoin-focused analysis



### Multi-Hop Normalization

Convert via intermediate liquid pairs when direct rate unavailable:

```python
from utils.currency_normalization import MultiHopNormalization

provider = MockExchangeRateProvider()

# Set up rates:
# OBSCURE -> XLM: 10.0
# XLM -> USDC: 0.1176 (inverse of 8.5)
provider.set_rate(obscure_asset, xlm_asset, Decimal("10.0"))

strategy = MultiHopNormalization(provider, base_asset=usdc_asset, max_hops=3)

# Convert OBSCURE to USDC via XLM
# Path: OBSCURE -> XLM -> USDC
norm = strategy.normalize(DecimalAmount("100"), obscure_asset)

print(f"Value: {norm.value} USDC")
print(f"Path: {' -> '.join(a.code for a in norm.conversion_path)}")
print(f"Confidence: {norm.confidence}")  # Reduced due to multi-hop

# Output:
# Value: 117.65 USDC
# Path: OBSCURE -> XLM -> USDC
# Confidence: 0.855 (penalty for multi-hop)
```

**When to use:**
- Handling obscure/illiquid assets
- Missing direct rates
- Fallback for other strategies

## Integration Patterns

### Benford Analysis

Normalize amounts before Benford digit extraction:

```python
from utils.benford_precision import leading_digits_safe
from utils.normalization_helpers import normalize_trade_amounts_to_series

# Load trades across multiple pairs
trades = [Trade(...), Trade(...), ...]  # Different asset pairs

# Normalize all to XLM
strategy = create_xlm_strategy()
normalized_series, _ = normalize_trade_amounts_to_series(trades, strategy)

# Extract leading digits (now from same currency)
digits = leading_digits_safe(normalized_series)

# Analyze Benford distribution
from detection.benford_engine import observed_distribution
dist = observed_distribution(normalized_series)
```

### Volume Analysis

Compare volumes across different pairs:

```python
from utils.normalization_helpers import compare_cross_pair_volumes

# Volumes from different pairs (in their native currencies)
volumes_by_pair = {
    ("USDC", "XLM"): Decimal("10000"),  # 10,000 USDC traded
    ("BTC", "XLM"): Decimal("0.5"),      # 0.5 BTC traded
    ("USDT", "XLM"): Decimal("9500"),    # 9,500 USDT traded
}

# Normalize to XLM
strategy = create_xlm_strategy()
normalized_volumes = compare_cross_pair_volumes(volumes_by_pair, strategy)

# Compare
for pair, norm_volume in normalized_volumes.items():
    print(f"{pair}: {norm_volume.value:,.0f} XLM")

# Output:
# ('USDC', 'XLM'): 85,000 XLM
# ('BTC', 'XLM'): 300,000 XLM
# ('USDT', 'XLM'): 79,800 XLM
```

### Anomaly Detection

Detect unusual volumes across pairs:

```python
from utils.normalization_helpers import detect_cross_pair_anomalies

anomalies = detect_cross_pair_anomalies(
    volumes_by_pair,
    strategy,
    threshold_multiplier=Decimal("3.0"),  # 3x median = anomaly
)

for pair, norm_volume, reason in anomalies:
    print(f"⚠️ Anomaly detected:")
    print(f"  Pair: {pair}")
    print(f"  Volume: {norm_volume.value:,.0f} XLM")
    print(f"  Reason: {reason}")
```

### Bulk Analysis

Create normalized DataFrame for analysis:

```python
from utils.normalization_helpers import create_normalized_dataframe

# Load trades
trades = load_trades_from_db(...)

# Create DataFrame with normalized columns
strategy = create_xlm_strategy()
df = create_normalized_dataframe(trades, strategy)

# Columns available:
# - trade_id
# - base_amount, base_amount_norm (normalized to XLM)
# - counter_amount, counter_amount_norm
# - base_asset_code, counter_asset_code
# - normalization_confidence
# - normalization_status

# Filter high-confidence normalizations
high_conf = df[df['normalization_confidence'] >= 0.9]

# Analyze normalized amounts
print(f"Median normalized volume: {high_conf['base_amount_norm'].median()}")
print(f"Total normalized volume: {high_conf['base_amount_norm'].sum()}")
```

## Performance

### Benchmarks

From `scripts/benchmark_normalization.py`:

| Operation | Rate (ops/s) | Per-op (ms) |
|-----------|--------------|-------------|
| Same currency (identity) | ~100,000 | 0.01 |
| With conversion (uncached) | ~10,000 | 0.10 |
| With conversion (cached) | ~50,000 | 0.02 |
| Aggregate 10 amounts | ~5,000 | 0.20 |
| Multi-hop conversion | ~3,000 | 0.33 |

**Key findings:**
- **Caching provides 5-10x speedup** for repeated conversions
- **Multi-hop is ~3x slower** than direct conversion
- **Batch operations scale linearly** with amount count
- **Identity conversions are fast** (no rate lookup)

### Optimization Tips

1. **Use CachedRateProvider**
   ```python
   from utils.currency_normalization import CachedRateProvider
   
   base_provider = MockExchangeRateProvider()
   cached = CachedRateProvider(base_provider, ttl=timedelta(minutes=5))
   strategy = XLMNormalization(cached)
   ```

2. **Batch operations when possible**
   ```python
   # ❌ Slow: normalize one at a time
   for trade in trades:
       norm = strategy.normalize(trade.base_amount, trade.base_asset)
   
   # ✅ Fast: batch with helpers
   normalized_series, _ = normalize_trade_amounts_to_series(trades, strategy)
   ```

3. **Pre-normalize in hot paths**
   ```python
   # Normalize once, use many times
   norm_amounts = [strategy.normalize(amt, asset) for amt, asset in amounts]
   
   # Now can compare without repeated normalization
   for norm in norm_amounts:
       if norm.value > threshold:
           flag_anomaly(norm)
   ```

4. **Monitor cache hit rate**
   ```python
   stats = cached_provider.get_cache_stats()
   print(f"Cache size: {stats['cache_size']}")
   print(f"Hit rate: {calculate_hit_rate()}")
   ```



## Validation and Testing

### Running Tests

```bash
# Run all normalization tests
pytest tests/test_currency_normalization.py -v

# Run specific test class
pytest tests/test_currency_normalization.py::TestXLMNormalization -v

# With coverage
pytest tests/test_currency_normalization.py --cov=utils.currency_normalization --cov-report=html
```

### Validation CLI

Scan codebase for normalization issues:

```bash
# Scan entire codebase
python -m scripts.validate_normalization

# Scan specific modules
python -m scripts.validate_normalization --modules detection features

# Check dataset
python -m scripts.validate_normalization --check-dataset data/trades.parquet

# CI integration
python -m scripts.validate_normalization --json > report.json
```

**What it detects:**
- Unnormalized cross-asset comparisons
- Direct sums across currencies
- Pandas aggregations without normalization
- Cross-pair iterations missing normalization
- Datasets with multiple assets

### Performance Benchmarks

```bash
# Run all benchmarks
python -m scripts.benchmark_normalization

# Save baseline
python -m scripts.benchmark_normalization --output baseline.json

# Compare with baseline (regression detection)
python -m scripts.benchmark_normalization --compare baseline.json
```

## API Reference

### Core Functions

```python
def normalize_amount(
    amount: Decimal | DecimalAmount,
    from_asset: Asset,
    to_asset: Asset,
    provider: ExchangeRateProvider,
    timestamp: datetime | None = None,
) -> NormalizedAmount:
    """Normalize an amount from one asset to another."""

def aggregate_normalized(
    amounts: list[tuple[Decimal | DecimalAmount, Asset]],
    base_asset: Asset,
    provider: ExchangeRateProvider,
    timestamp: datetime | None = None,
) -> NormalizedAmount:
    """Aggregate multiple amounts into base currency."""
```

### Factory Functions

```python
def create_default_provider(use_cache: bool = True) -> ExchangeRateProvider:
    """Create default provider (Mock with optional cache)."""

def create_xlm_strategy(use_cache: bool = True) -> XLMNormalization:
    """Create XLM normalization strategy."""

def create_usd_strategy(use_cache: bool = True) -> USDNormalization:
    """Create USD normalization strategy."""
```

### Helper Functions

```python
def normalize_trade_amounts_to_series(
    trades: list[Trade],
    strategy: NormalizationStrategy,
) -> tuple[pd.Series, pd.Series]:
    """Convert trade amounts to normalized pandas Series."""

def calculate_normalized_volume(
    trades: list[Trade],
    strategy: NormalizationStrategy,
    use_base: bool = True,
) -> NormalizedAmount:
    """Calculate total volume across trades."""

def compare_cross_pair_volumes(
    volumes_by_pair: dict[tuple[str, str], Decimal],
    strategy: NormalizationStrategy,
    asset_resolver: dict[str, Asset] | None = None,
) -> dict[tuple[str, str], NormalizedAmount]:
    """Compare volumes across different pairs."""

def detect_cross_pair_anomalies(
    volumes_by_pair: dict[tuple[str, str], Decimal],
    strategy: NormalizationStrategy,
    threshold_multiplier: Decimal = Decimal("3.0"),
) -> list[tuple[tuple[str, str], NormalizedAmount, str]]:
    """Detect anomalous volumes across pairs."""

def format_normalized_amount(normalized: NormalizedAmount) -> str:
    """Format normalized amount for display."""
```

### Trade Model Extensions

```python
class Trade:
    def normalize_base_amount(
        self,
        strategy: NormalizationStrategy,
    ) -> NormalizedAmount:
        """Normalize base amount."""
    
    def normalize_counter_amount(
        self,
        strategy: NormalizationStrategy,
    ) -> NormalizedAmount:
        """Normalize counter amount."""
    
    def normalize_both_amounts(
        self,
        strategy: NormalizationStrategy,
    ) -> tuple[NormalizedAmount, NormalizedAmount]:
        """Normalize both amounts."""
    
    def get_normalized_trade_value(
        self,
        strategy: NormalizationStrategy,
    ) -> NormalizedAmount:
        """Get trade value in base currency."""
```

### OrderBookEvent Model Extensions

```python
class OrderBookEvent:
    def normalize_amount(
        self,
        strategy: NormalizationStrategy,
    ) -> NormalizedAmount:
        """Normalize order amount."""
    
    def get_normalized_order_value(
        self,
        strategy: NormalizationStrategy,
    ) -> NormalizedAmount:
        """Get order value (amount * price) in base currency."""
```

## Troubleshooting

### Common Issues

#### Issue: "No rate available"

**Cause:** Exchange rate provider doesn't have rate for asset pair

**Solution:**
```python
# Check if rate available
if not provider.is_available(asset):
    logger.warning(f"No rates for {asset.code}")
    # Use fallback or skip

# Try multi-hop
multihop = MultiHopNormalization(provider, base_asset=xlm_asset)
normalized = multihop.normalize(amount, asset)
```

#### Issue: "Stale rate"

**Cause:** Exchange rate is too old

**Solution:**
```python
# Check rate staleness
if normalized.status == NormalizationStatus.STALE_RATE:
    logger.warning(f"Stale rate, confidence reduced to {normalized.confidence}")
    # Use with caution or skip

# Adjust staleness threshold
pair.is_stale(threshold=timedelta(minutes=10))  # More lenient
```

#### Issue: "Low confidence"

**Cause:** Multi-hop conversion or low liquidity

**Solution:**
```python
# Filter by confidence
high_conf = [n for n in normalized_amounts if n.confidence >= 0.8]

# Or use confidence-weighted aggregation
from utils.normalization_helpers import filter_high_confidence_normalizations
high_conf = filter_high_confidence_normalizations(normalized_amounts, min_confidence=Decimal("0.9"))
```

#### Issue: "Cache not helping"

**Cause:** Cache TTL too short or different timestamps

**Solution:**
```python
# Increase TTL
cached = CachedRateProvider(provider, ttl=timedelta(minutes=15))

# Use same timestamp for batch
timestamp = datetime.now()
for trade in trades:
    normalized = strategy.normalize(trade.base_amount, trade.base_asset, timestamp=timestamp)

# Check cache stats
stats = cached.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
```



## Migration Guide

### Step 1: Update Imports

```python
# Add normalization imports
from utils.currency_normalization import (
    create_xlm_strategy,
    create_usd_strategy,
    normalize_amount,
    aggregate_normalized,
)
from utils.normalization_helpers import (
    normalize_trade_amounts_to_series,
    compare_cross_pair_volumes,
    detect_cross_pair_anomalies,
)
```

### Step 2: Create Strategy

```python
# Create once at module level or in __init__
strategy = create_xlm_strategy(use_cache=True)
```

### Step 3: Normalize Before Comparison

**Before:**
```python
# ❌ Direct comparison across assets
if trade1.base_amount > trade2.base_amount:
    # Meaningless if different assets!
    flag_anomaly(trade1)
```

**After:**
```python
# ✅ Normalize first
norm1 = trade1.normalize_base_amount(strategy)
norm2 = trade2.normalize_base_amount(strategy)

if norm1.value > norm2.value:
    # Now meaningful comparison
    flag_anomaly(trade1)
```

### Step 4: Normalize Before Aggregation

**Before:**
```python
# ❌ Summing different currencies
total = sum(trade.base_amount for trade in trades)
```

**After:**
```python
# ✅ Aggregate with normalization
amounts = [(trade.base_amount, trade.base_asset) for trade in trades]
total = aggregate_normalized(amounts, xlm_asset, strategy.provider)
print(f"Total: {total.value} XLM")
```

### Step 5: Update Benford Analysis

**Before:**
```python
# ❌ Benford on mixed currencies
amounts = [trade.base_amount for trade in trades]
digits = leading_digits_safe(pd.Series(amounts))
```

**After:**
```python
# ✅ Normalize first
normalized_series, _ = normalize_trade_amounts_to_series(trades, strategy)
digits = leading_digits_safe(normalized_series)
```

### Step 6: Add Confidence Checks

```python
# Check normalization success
normalized = trade.normalize_base_amount(strategy)

if not normalized.is_successful():
    logger.warning(f"Normalization failed: {normalized.status}")
    continue

if normalized.confidence < 0.8:
    logger.warning(f"Low confidence: {normalized.confidence}")
    # Handle with caution
```

## Future Enhancements

Potential improvements for future iterations:

1. **Real Exchange Rate Providers**
   - StellarDEXRateProvider (TWAP from actual trades)
   - ExternalOracleProvider (CoinGecko, Band Protocol)
   - Historical rate database

2. **Advanced Features**
   - Confidence-weighted aggregations
   - Circuit breaker for stale/missing rates
   - Rate staleness alerts
   - Multi-currency portfolio tracking

3. **Performance Optimizations**
   - Async rate fetching
   - Batch rate updates
   - Persistent cache (Redis, database)
   - Rate prediction/interpolation

4. **Integrations**
   - Stellar Horizon API integration
   - Price feed streaming
   - Real-time rate updates
   - External price oracles

## Examples

### Example 1: Cross-Pair Volume Comparison

```python
from utils.normalization_helpers import compare_cross_pair_volumes
import pandas as pd

# Load trades for analysis window
trades_df = pd.read_parquet("trades_last_hour.parquet")

# Group by pair and sum volumes
volumes_by_pair = (
    trades_df.groupby(["base_asset_code", "counter_asset_code"])
    ["base_amount"]
    .sum()
    .to_dict()
)

# Normalize to XLM
strategy = create_xlm_strategy()
normalized_volumes = compare_cross_pair_volumes(volumes_by_pair, strategy)

# Find highest volume pair
max_pair = max(normalized_volumes.items(), key=lambda x: x[1].value)
print(f"Highest volume pair: {max_pair[0]}")
print(f"Volume: {max_pair[1].value:,.0f} XLM")
```

### Example 2: Wash Trading Detection

```python
from utils.normalization_helpers import detect_cross_pair_anomalies

# Calculate volumes per account across pairs
account_volumes = {}
for trade in trades:
    account = trade.base_account
    pair = (trade.base_asset.code, trade.counter_asset.code)
    
    if account not in account_volumes:
        account_volumes[account] = {}
    if pair not in account_volumes[account]:
        account_volumes[account][pair] = Decimal("0")
    
    account_volumes[account][pair] += trade.base_amount

# Detect accounts with anomalous cross-pair activity
strategy = create_xlm_strategy()
for account, volumes in account_volumes.items():
    anomalies = detect_cross_pair_anomalies(volumes, strategy, threshold_multiplier=Decimal("5.0"))
    
    if anomalies:
        print(f"⚠️ Suspicious activity from {account}:")
        for pair, norm_volume, reason in anomalies:
            print(f"  {pair}: {norm_volume.value:,.0f} XLM - {reason}")
```

### Example 3: Portfolio Rebalancing

```python
from utils.currency_normalization import aggregate_normalized

# Current portfolio
portfolio = {
    "USDC": DecimalAmount("5000"),
    "USDT": DecimalAmount("3000"),
    "XLM": DecimalAmount("10000"),
    "BTC": DecimalAmount("0.1"),
}

# Resolve assets
asset_map = {
    "USDC": usdc_asset,
    "USDT": usdt_asset,
    "XLM": xlm_asset,
    "BTC": btc_asset,
}

# Calculate total value
strategy = create_xlm_strategy()
amounts = [(amount, asset_map[code]) for code, amount in portfolio.items()]
total = aggregate_normalized(amounts, xlm_asset, strategy.provider)

print(f"Total portfolio value: {total.value:,.0f} XLM")

# Calculate allocations
for code, amount in portfolio.items():
    norm = strategy.normalize(amount, asset_map[code])
    allocation = (norm.value / total.value) * 100
    print(f"{code}: {allocation:.1f}%")
```

## References

- [Stellar Network Documentation](https://developers.stellar.org/docs)
- [Stellar DEX Overview](https://developers.stellar.org/docs/learn/glossary#decentralized-exchange-dex)
- [Decimal Precision Guards](./numeric_precision.md)
- [Currency Normalization Source](../utils/currency_normalization.py)
- [Test Suite](../tests/test_currency_normalization.py)

## Support

For issues or questions:
- GitHub Issues: [LedgerLens-data/issues](https://github.com/product-labo/Ledgerlens-data/issues)
- Documentation: `docs/currency_normalization.md`
- Tests: `tests/test_currency_normalization.py`
- Code: `utils/currency_normalization.py`

---

**License:** MIT  
**Maintainer:** Product Labo Team  
**Status:** Production Ready
