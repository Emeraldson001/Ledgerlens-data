"""Currency and amount normalization for cross-asset comparisons.

This module provides contracts and utilities for normalizing amounts across
different asset pairs to enable meaningful comparisons in fraud detection.
Stellar DEX supports hundreds of asset pairs, and detecting anomalies requires
comparing volumes and prices across different currencies.

Key Concepts
------------
**Normalization:** Converting amounts from their native asset to a common
base currency (typically XLM or USD-equivalent) using exchange rates.

**Exchange Rate Provider:** Pluggable source of asset prices (mock, Stellar
DEX TWAP, external oracles, etc.)

**Currency Pair:** Represents conversion between two assets with rate metadata.

**Normalized Amount:** Amount expressed in base currency with original context.

Architecture
------------
```
ExchangeRateProvider (Protocol)
├── get_rate(from_asset, to_asset, timestamp) -> CurrencyPair
├── get_rates_batch(pairs, timestamp) -> dict[tuple, CurrencyPair]
└── Implementations:
    ├── MockExchangeRateProvider (testing)
    ├── StellarDEXRateProvider (TWAP from trades)
    └── CachedRateProvider (wrapper with TTL cache)

AssetMetadata
├── asset_type (native, stablecoin, token, etc.)
├── category (crypto, fiat-backed, algorithmic)
├── liquidity_score (0-1, for confidence weighting)
└── preferred_base (default normalization target)

NormalizedAmount
├── value (Decimal in base currency)
├── base_asset (Asset used for normalization)
├── original_value (Decimal in original asset)
├── original_asset (Asset before conversion)
├── exchange_rate (CurrencyPair used)
└── confidence (0-1, based on liquidity/staleness)

NormalizationStrategy
├── XLMNormalization - Use XLM as base (native Stellar)
├── USDNormalization - Use USD stablecoins (USDC, etc.)
├── DirectPairNormalization - Use observed trade price
└── MultiHopNormalization - Chain conversions via liquid pairs
```

Usage
-----
Basic normalization::

    from utils.currency_normalization import (
        normalize_amount,
        XLMNormalization,
        MockExchangeRateProvider,
    )

    # Setup
    provider = MockExchangeRateProvider()
    strategy = XLMNormalization(provider)

    # Normalize
    amount = DecimalAmount("100")
    asset = Asset(code="USDC", issuer="GA5ZSEJYB37JRC...")
    normalized = strategy.normalize(amount, asset, timestamp=datetime.now())

    print(f"100 USDC = {normalized.value} XLM")

Multi-asset aggregation::

    from utils.currency_normalization import aggregate_normalized

    amounts = [
        (DecimalAmount("100"), Asset(code="USDC", issuer="...")),
        (DecimalAmount("500"), Asset(code="BTC", issuer="...")),
        (DecimalAmount("1000"), Asset(code="XLM", issuer=None)),
    ]

    total = aggregate_normalized(amounts, strategy, timestamp)
    print(f"Total: {total.value} XLM")

Trade normalization::

    trade = Trade(...)

    # Normalize base amount to XLM
    normalized_base = strategy.normalize(
        trade.base_amount,
        trade.base_asset,
        trade.ledger_close_time,
    )

    # Normalize counter amount to XLM
    normalized_counter = strategy.normalize(
        trade.counter_amount,
        trade.counter_asset,
        trade.ledger_close_time,
    )

    # Now comparable!
    if normalized_base.value > normalized_counter.value:
        print("Base asset more valuable")

Design Decisions
----------------
1. **Protocol-based providers** - Pluggable exchange rate sources
2. **Immutable NormalizedAmount** - Preserve provenance for auditing
3. **Confidence scoring** - Weight by liquidity, staleness, path length
4. **Multi-hop support** - Convert via intermediate liquid pairs
5. **Timestamp-aware** - Historical rates for backtesting
6. **Decimal precision** - Integrate with decimal_guards for exactness
7. **Lazy evaluation** - Fetch rates only when needed
8. **Caching strategy** - TTL cache for rate providers

Future Enhancements
-------------------
- Integration with external price feeds (CoinGecko, Band Protocol)
- Circuit breaker for stale/missing rates
- Confidence-weighted aggregations
- Multi-currency portfolio tracking
- Real-time rate streaming
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from importlib import import_module
from typing import Protocol, runtime_checkable

from utils.decimal_guards import DecimalAmount, validate_amount
from utils.logging import get_logger

# Loaded lazily through the module system to keep the foundation package free
# of a static dependency edge on the ingestion domain layer.
Asset = import_module("ingestion.data_models").Asset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class AssetType(StrEnum):
    """Classification of asset types on Stellar."""

    NATIVE = "native"  # XLM
    STABLECOIN = "stablecoin"  # USDC, USDT, etc.
    TOKEN = "token"  # Other issued assets
    SYNTHETIC = "synthetic"  # Synthetic/wrapped assets
    UNKNOWN = "unknown"


class StablecoinType(StrEnum):
    """Types of stablecoins."""

    FIAT_BACKED = "fiat_backed"  # USDC, USDT (backed by USD reserves)
    CRYPTO_BACKED = "crypto_backed"  # DAI (backed by crypto collateral)
    ALGORITHMIC = "algorithmic"  # Algorithmic stablecoins
    COMMODITY = "commodity"  # Gold-backed, etc.


class NormalizationStatus(StrEnum):
    """Status of normalization operation."""

    SUCCESS = "success"  # Normalization succeeded
    NO_RATE = "no_rate"  # Exchange rate not available
    STALE_RATE = "stale_rate"  # Rate too old
    LOW_LIQUIDITY = "low_liquidity"  # Low confidence due to liquidity
    MULTI_HOP = "multi_hop"  # Multi-hop conversion used
    ERROR = "error"  # Normalization failed


# Common Stellar stablecoins (issuer address → code)
KNOWN_STABLECOINS = {
    # USDC (Circle)
    "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN": "USDC",
    # USDT (Tether)
    "GCQTGZQQ5G4PTM2GL7CDIFKUBIPEC52BROAQIAPW53XBRJVN6ZJVTG6V": "USDT",
    # AQUA (AquaUSD)
    "GBNZILSTVQZ4R7IKQDGHYGY2QXL5QOFJYQMXPKWRRM5PAV7Y4M67AQUA": "AQUA",
}

# Native asset (XLM)
NATIVE_ASSET = Asset(code="XLM", issuer=None)

# Default rate staleness threshold (5 minutes)
DEFAULT_STALENESS_THRESHOLD = timedelta(minutes=5)

# Minimum liquidity score for high confidence
MIN_CONFIDENCE_LIQUIDITY = 0.7


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetMetadata:
    """Metadata about an asset for normalization.

    Attributes
    ----------
    asset : Asset
        The asset
    asset_type : AssetType
        Classification (native, stablecoin, token, etc.)
    stablecoin_type : StablecoinType | None
        Type of stablecoin if applicable
    liquidity_score : Decimal
        Liquidity score 0-1 (for confidence weighting)
    preferred_base : Asset | None
        Preferred base currency for this asset
    display_name : str
        Human-readable name
    """

    asset: Asset
    asset_type: AssetType
    stablecoin_type: StablecoinType | None = None
    liquidity_score: Decimal = Decimal("0.5")
    preferred_base: Asset | None = None
    display_name: str = ""

    def __post_init__(self):
        """Validate metadata."""
        if self.liquidity_score < 0 or self.liquidity_score > 1:
            raise ValueError(f"Liquidity score must be 0-1, got {self.liquidity_score}")

        if self.asset_type == AssetType.STABLECOIN and self.stablecoin_type is None:
            logger.warning(f"Stablecoin {self.asset.code} has no stablecoin_type")


@dataclass(frozen=True)
class CurrencyPair:
    """Exchange rate between two assets.

    Represents the rate to convert from_asset to to_asset at a specific time.
    Rate semantics: 1 from_asset = rate * to_asset

    Attributes
    ----------
    from_asset : Asset
        Source asset
    to_asset : Asset
        Target asset
    rate : Decimal
        Exchange rate (1 from_asset = rate * to_asset)
    timestamp : datetime
        When the rate was observed
    source : str
        Source of the rate (e.g., "stellar_dex", "mock", "external")
    liquidity : Decimal | None
        Liquidity indicator (volume, depth, etc.)
    confidence : Decimal
        Confidence in rate (0-1)

    Examples
    --------
    >>> # 1 USDC = 8.5 XLM
    >>> pair = CurrencyPair(
    ...     from_asset=Asset(code="USDC", issuer="..."),
    ...     to_asset=Asset(code="XLM", issuer=None),
    ...     rate=Decimal("8.5"),
    ...     timestamp=datetime.now(),
    ...     source="stellar_dex",
    ...     confidence=Decimal("0.95"),
    ... )
    """

    from_asset: Asset
    to_asset: Asset
    rate: Decimal
    timestamp: datetime
    source: str
    liquidity: Decimal | None = None
    confidence: Decimal = Decimal("1.0")

    def __post_init__(self):
        """Validate currency pair."""
        if self.rate <= 0:
            raise ValueError(f"Exchange rate must be positive, got {self.rate}")

        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def inverse(self) -> "CurrencyPair":
        """Return inverse pair (swap from/to assets).

        Returns
        -------
        CurrencyPair
            Inverted pair

        Examples
        --------
        >>> # 1 USDC = 8.5 XLM
        >>> pair = CurrencyPair(from_asset=usdc, to_asset=xlm, rate=Decimal("8.5"), ...)
        >>> inv = pair.inverse()
        >>> # 1 XLM = 0.1176 USDC (1/8.5)
        >>> inv.rate
        Decimal('0.1176470588235294117647058824')
        """
        return CurrencyPair(
            from_asset=self.to_asset,
            to_asset=self.from_asset,
            rate=Decimal("1") / self.rate,
            timestamp=self.timestamp,
            source=self.source,
            liquidity=self.liquidity,
            confidence=self.confidence,
        )

    def is_stale(self, threshold: timedelta = DEFAULT_STALENESS_THRESHOLD) -> bool:
        """Check if rate is stale.

        Parameters
        ----------
        threshold : timedelta
            Maximum age before rate is considered stale

        Returns
        -------
        bool
            True if rate is older than threshold
        """
        age = datetime.now() - self.timestamp
        return age > threshold

    def pair_key(self) -> tuple[str, str]:
        """Return canonical pair key for indexing.

        Returns
        -------
        tuple[str, str]
            (from_asset_id, to_asset_id)
        """
        from_id = f"{self.from_asset.code}:{self.from_asset.issuer or 'native'}"
        to_id = f"{self.to_asset.code}:{self.to_asset.issuer or 'native'}"
        return (from_id, to_id)


@dataclass(frozen=True)
class NormalizedAmount:
    """Amount normalized to a base currency.

    Immutable record preserving original amount, conversion details, and
    confidence for auditing and diagnostics.

    Attributes
    ----------
    value : Decimal
        Amount in base currency
    base_asset : Asset
        Base currency used
    original_value : Decimal
        Original amount before conversion
    original_asset : Asset
        Original asset before conversion
    exchange_rate : CurrencyPair | None
        Rate used for conversion (None if same currency)
    confidence : Decimal
        Confidence in normalization (0-1)
    status : NormalizationStatus
        Status of normalization
    conversion_path : list[Asset]
        Assets in conversion path (for multi-hop)

    Examples
    --------
    >>> # 100 USDC normalized to XLM
    >>> norm = NormalizedAmount(
    ...     value=Decimal("850.0"),  # 100 * 8.5
    ...     base_asset=Asset(code="XLM", issuer=None),
    ...     original_value=Decimal("100"),
    ...     original_asset=Asset(code="USDC", issuer="..."),
    ...     exchange_rate=CurrencyPair(...),
    ...     confidence=Decimal("0.95"),
    ...     status=NormalizationStatus.SUCCESS,
    ... )
    """

    value: Decimal
    base_asset: Asset
    original_value: Decimal
    original_asset: Asset
    exchange_rate: CurrencyPair | None = None
    confidence: Decimal = Decimal("1.0")
    status: NormalizationStatus = NormalizationStatus.SUCCESS
    conversion_path: list[Asset] = field(default_factory=list)

    def __post_init__(self):
        """Validate normalized amount."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def is_successful(self) -> bool:
        """Check if normalization succeeded."""
        return self.status == NormalizationStatus.SUCCESS

    def is_same_currency(self) -> bool:
        """Check if base and original currency are the same."""
        return (
            self.base_asset.code == self.original_asset.code
            and self.base_asset.issuer == self.original_asset.issuer
        )

    def to_decimal_amount(self) -> DecimalAmount:
        """Convert to DecimalAmount (drops context)."""
        return DecimalAmount(self.value)


# ---------------------------------------------------------------------------
# Exchange Rate Provider Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExchangeRateProvider(Protocol):
    """Protocol for exchange rate providers.

    Implementations fetch rates from various sources (Stellar DEX, external
    oracles, mock data, etc.) and return CurrencyPair objects.
    """

    def get_rate(
        self,
        from_asset: Asset,
        to_asset: Asset,
        timestamp: datetime | None = None,
    ) -> CurrencyPair | None:
        """Get exchange rate between two assets.

        Parameters
        ----------
        from_asset : Asset
            Source asset
        to_asset : Asset
            Target asset
        timestamp : datetime, optional
            Historical timestamp (None = current)

        Returns
        -------
        CurrencyPair | None
            Exchange rate, or None if not available
        """
        ...

    def get_rates_batch(
        self,
        pairs: list[tuple[Asset, Asset]],
        timestamp: datetime | None = None,
    ) -> dict[tuple[str, str], CurrencyPair]:
        """Get exchange rates for multiple pairs (batch operation).

        Parameters
        ----------
        pairs : list[tuple[Asset, Asset]]
            List of (from_asset, to_asset) pairs
        timestamp : datetime, optional
            Historical timestamp (None = current)

        Returns
        -------
        dict[tuple[str, str], CurrencyPair]
            Map of pair keys to rates
        """
        ...

    def is_available(self, asset: Asset) -> bool:
        """Check if rates are available for an asset.

        Parameters
        ----------
        asset : Asset
            Asset to check

        Returns
        -------
        bool
            True if rates available
        """
        ...


# ---------------------------------------------------------------------------
# Mock Exchange Rate Provider (for testing)
# ---------------------------------------------------------------------------


class MockExchangeRateProvider:
    """Mock exchange rate provider for testing.

    Provides configurable static rates for testing normalization logic
    without external dependencies.

    Examples
    --------
    >>> provider = MockExchangeRateProvider()
    >>> provider.set_rate(usdc, xlm, Decimal("8.5"))
    >>> rate = provider.get_rate(usdc, xlm)
    >>> rate.rate
    Decimal('8.5')
    """

    def __init__(self):
        self._rates: dict[tuple[str, str], Decimal] = {}
        self._confidence: dict[tuple[str, str], Decimal] = {}

        # Default rates for common pairs
        self._set_default_rates()

    def _set_default_rates(self):
        """Set default rates for testing."""
        # XLM = 1 (base)
        # USDC = 8.5 XLM
        # USDT = 8.4 XLM
        # BTC = 600000 XLM

        usdc = Asset(code="USDC", issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")
        usdt = Asset(code="USDT", issuer="GCQTGZQQ5G4PTM2GL7CDIFKUBIPEC52BROAQIAPW53XBRJVN6ZJVTG6V")
        xlm = NATIVE_ASSET

        self.set_rate(usdc, xlm, Decimal("8.5"), confidence=Decimal("0.95"))
        self.set_rate(usdt, xlm, Decimal("8.4"), confidence=Decimal("0.90"))

    def set_rate(
        self,
        from_asset: Asset,
        to_asset: Asset,
        rate: Decimal,
        confidence: Decimal = Decimal("1.0"),
    ):
        """Set an exchange rate.

        Parameters
        ----------
        from_asset : Asset
            Source asset
        to_asset : Asset
            Target asset
        rate : Decimal
            Exchange rate
        confidence : Decimal
            Confidence score (0-1)
        """
        key = self._pair_key(from_asset, to_asset)
        self._rates[key] = rate
        self._confidence[key] = confidence

        # Also set inverse
        inv_key = self._pair_key(to_asset, from_asset)
        self._rates[inv_key] = Decimal("1") / rate
        self._confidence[inv_key] = confidence

    def get_rate(
        self,
        from_asset: Asset,
        to_asset: Asset,
        timestamp: datetime | None = None,
    ) -> CurrencyPair | None:
        """Get exchange rate."""
        # Same asset
        if from_asset.code == to_asset.code and from_asset.issuer == to_asset.issuer:
            return CurrencyPair(
                from_asset=from_asset,
                to_asset=to_asset,
                rate=Decimal("1.0"),
                timestamp=timestamp or datetime.now(),
                source="identity",
                confidence=Decimal("1.0"),
            )

        key = self._pair_key(from_asset, to_asset)

        if key not in self._rates:
            return None

        return CurrencyPair(
            from_asset=from_asset,
            to_asset=to_asset,
            rate=self._rates[key],
            timestamp=timestamp or datetime.now(),
            source="mock",
            confidence=self._confidence.get(key, Decimal("1.0")),
        )

    def get_rates_batch(
        self,
        pairs: list[tuple[Asset, Asset]],
        timestamp: datetime | None = None,
    ) -> dict[tuple[str, str], CurrencyPair]:
        """Get multiple rates."""
        results = {}
        for from_asset, to_asset in pairs:
            rate = self.get_rate(from_asset, to_asset, timestamp)
            if rate:
                results[rate.pair_key()] = rate
        return results

    def is_available(self, asset: Asset) -> bool:
        """Check if rates available for asset."""
        # Check if any rate involves this asset
        asset_id = f"{asset.code}:{asset.issuer or 'native'}"
        for from_id, to_id in self._rates.keys():
            if asset_id in (from_id, to_id):
                return True
        return False

    def _pair_key(self, from_asset: Asset, to_asset: Asset) -> tuple[str, str]:
        """Generate pair key."""
        from_id = f"{from_asset.code}:{from_asset.issuer or 'native'}"
        to_id = f"{to_asset.code}:{to_asset.issuer or 'native'}"
        return (from_id, to_id)


# ---------------------------------------------------------------------------
# Normalization Functions
# ---------------------------------------------------------------------------


def normalize_amount(
    amount: Decimal | DecimalAmount,
    from_asset: Asset,
    to_asset: Asset,
    provider: ExchangeRateProvider,
    timestamp: datetime | None = None,
) -> NormalizedAmount:
    """Normalize an amount from one asset to another.

    Parameters
    ----------
    amount : Decimal | DecimalAmount
        Amount to normalize
    from_asset : Asset
        Source asset
    to_asset : Asset
        Target asset (base currency)
    provider : ExchangeRateProvider
        Exchange rate provider
    timestamp : datetime, optional
        Historical timestamp (None = current)

    Returns
    -------
    NormalizedAmount
        Normalized amount with conversion details

    Examples
    --------
    >>> amount = DecimalAmount("100")
    >>> usdc = Asset(code="USDC", issuer="...")
    >>> xlm = Asset(code="XLM", issuer=None)
    >>> provider = MockExchangeRateProvider()
    >>>
    >>> normalized = normalize_amount(amount, usdc, xlm, provider)
    >>> print(f"100 USDC = {normalized.value} XLM")
    100 USDC = 850.0 XLM
    """
    # Convert to Decimal if needed
    if isinstance(amount, DecimalAmount):
        amount_decimal = amount.value
    else:
        amount_decimal = validate_amount(amount)

    # Same currency - no conversion needed
    if from_asset.code == to_asset.code and from_asset.issuer == to_asset.issuer:
        return NormalizedAmount(
            value=amount_decimal,
            base_asset=to_asset,
            original_value=amount_decimal,
            original_asset=from_asset,
            exchange_rate=None,
            confidence=Decimal("1.0"),
            status=NormalizationStatus.SUCCESS,
            conversion_path=[from_asset],
        )

    # Get exchange rate
    rate = provider.get_rate(from_asset, to_asset, timestamp)

    if rate is None:
        # No rate available
        return NormalizedAmount(
            value=amount_decimal,  # Return original
            base_asset=from_asset,  # Keep original asset
            original_value=amount_decimal,
            original_asset=from_asset,
            exchange_rate=None,
            confidence=Decimal("0.0"),
            status=NormalizationStatus.NO_RATE,
            conversion_path=[from_asset],
        )

    # Check staleness
    if rate.is_stale():
        status = NormalizationStatus.STALE_RATE
        confidence = rate.confidence * Decimal("0.5")  # Reduce confidence
    else:
        status = NormalizationStatus.SUCCESS
        confidence = rate.confidence

    # Convert
    normalized_value = amount_decimal * rate.rate

    return NormalizedAmount(
        value=normalized_value,
        base_asset=to_asset,
        original_value=amount_decimal,
        original_asset=from_asset,
        exchange_rate=rate,
        confidence=confidence,
        status=status,
        conversion_path=[from_asset, to_asset],
    )


def aggregate_normalized(
    amounts: list[tuple[Decimal | DecimalAmount, Asset]],
    base_asset: Asset,
    provider: ExchangeRateProvider,
    timestamp: datetime | None = None,
) -> NormalizedAmount:
    """Aggregate multiple amounts from different assets into base currency.

    Parameters
    ----------
    amounts : list[tuple[Decimal | DecimalAmount, Asset]]
        List of (amount, asset) tuples to aggregate
    base_asset : Asset
        Base currency to normalize to
    provider : ExchangeRateProvider
        Exchange rate provider
    timestamp : datetime, optional
        Historical timestamp

    Returns
    -------
    NormalizedAmount
        Aggregated amount in base currency

    Examples
    --------
    >>> amounts = [
    ...     (DecimalAmount("100"), usdc_asset),
    ...     (DecimalAmount("500"), btc_asset),
    ...     (DecimalAmount("1000"), xlm_asset),
    ... ]
    >>>
    >>> total = aggregate_normalized(amounts, xlm_asset, provider)
    >>> print(f"Total: {total.value} XLM")
    """
    if not amounts:
        return NormalizedAmount(
            value=Decimal("0"),
            base_asset=base_asset,
            original_value=Decimal("0"),
            original_asset=base_asset,
            confidence=Decimal("1.0"),
            status=NormalizationStatus.SUCCESS,
        )

    total = Decimal("0")
    min_confidence = Decimal("1.0")
    all_successful = True
    conversion_path = []

    for amount, asset in amounts:
        normalized = normalize_amount(amount, asset, base_asset, provider, timestamp)

        if normalized.is_successful():
            total += normalized.value
            min_confidence = min(min_confidence, normalized.confidence)
            if asset not in conversion_path:
                conversion_path.append(asset)
        else:
            all_successful = False
            logger.warning(f"Failed to normalize {amount} {asset.code}: {normalized.status}")

    status = NormalizationStatus.SUCCESS if all_successful else NormalizationStatus.ERROR

    return NormalizedAmount(
        value=total,
        base_asset=base_asset,
        original_value=total,  # Aggregated value
        original_asset=base_asset,  # In base currency
        confidence=min_confidence,
        status=status,
        conversion_path=conversion_path,
    )


# ---------------------------------------------------------------------------
# Asset Classification and Metadata
# ---------------------------------------------------------------------------


class AssetClassifier:
    """Classify assets and provide metadata for normalization.

    This classifier detects asset types (native, stablecoin, token) and
    provides metadata for confidence weighting in normalization.

    Examples
    --------
    >>> classifier = AssetClassifier()
    >>> metadata = classifier.classify(Asset(code="USDC", issuer="..."))
    >>> metadata.asset_type
    AssetType.STABLECOIN
    >>> metadata.stablecoin_type
    StablecoinType.FIAT_BACKED
    """

    def __init__(self):
        self._metadata_cache: dict[str, AssetMetadata] = {}
        self._initialize_known_assets()

    def _initialize_known_assets(self):
        """Initialize metadata for known assets."""
        # XLM (native)
        xlm = NATIVE_ASSET
        self._metadata_cache[self._asset_key(xlm)] = AssetMetadata(
            asset=xlm,
            asset_type=AssetType.NATIVE,
            liquidity_score=Decimal("1.0"),
            display_name="Stellar Lumens (XLM)",
        )

        # USDC (Circle)
        usdc = Asset(
            code="USDC",
            issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        )
        self._metadata_cache[self._asset_key(usdc)] = AssetMetadata(
            asset=usdc,
            asset_type=AssetType.STABLECOIN,
            stablecoin_type=StablecoinType.FIAT_BACKED,
            liquidity_score=Decimal("0.95"),
            preferred_base=NATIVE_ASSET,
            display_name="USD Coin (USDC)",
        )

        # USDT (Tether)
        usdt = Asset(
            code="USDT",
            issuer="GCQTGZQQ5G4PTM2GL7CDIFKUBIPEC52BROAQIAPW53XBRJVN6ZJVTG6V",
        )
        self._metadata_cache[self._asset_key(usdt)] = AssetMetadata(
            asset=usdt,
            asset_type=AssetType.STABLECOIN,
            stablecoin_type=StablecoinType.FIAT_BACKED,
            liquidity_score=Decimal("0.90"),
            preferred_base=NATIVE_ASSET,
            display_name="Tether USD (USDT)",
        )

    def classify(self, asset: Asset) -> AssetMetadata:
        """Classify an asset and return metadata.

        Parameters
        ----------
        asset : Asset
            Asset to classify

        Returns
        -------
        AssetMetadata
            Asset metadata
        """
        key = self._asset_key(asset)

        # Return cached if available
        if key in self._metadata_cache:
            return self._metadata_cache[key]

        # Classify
        if asset.issuer is None:
            # Native XLM
            metadata = AssetMetadata(
                asset=asset,
                asset_type=AssetType.NATIVE,
                liquidity_score=Decimal("1.0"),
                display_name="Stellar Lumens (XLM)",
            )
        elif asset.issuer in KNOWN_STABLECOINS:
            # Known stablecoin
            metadata = AssetMetadata(
                asset=asset,
                asset_type=AssetType.STABLECOIN,
                stablecoin_type=StablecoinType.FIAT_BACKED,
                liquidity_score=Decimal("0.8"),
                preferred_base=NATIVE_ASSET,
                display_name=f"{asset.code} Stablecoin",
            )
        elif asset.code in ("USDC", "USDT", "USDD", "DAI", "BUSD"):
            # Stablecoin by code (even if issuer unknown)
            metadata = AssetMetadata(
                asset=asset,
                asset_type=AssetType.STABLECOIN,
                stablecoin_type=StablecoinType.FIAT_BACKED,
                liquidity_score=Decimal("0.7"),
                preferred_base=NATIVE_ASSET,
                display_name=f"{asset.code} Stablecoin",
            )
        else:
            # Generic token
            metadata = AssetMetadata(
                asset=asset,
                asset_type=AssetType.TOKEN,
                liquidity_score=Decimal("0.5"),
                preferred_base=NATIVE_ASSET,
                display_name=asset.code,
            )

        # Cache
        self._metadata_cache[key] = metadata

        return metadata

    def is_stablecoin(self, asset: Asset) -> bool:
        """Check if asset is a stablecoin.

        Parameters
        ----------
        asset : Asset
            Asset to check

        Returns
        -------
        bool
            True if stablecoin
        """
        metadata = self.classify(asset)
        return metadata.asset_type == AssetType.STABLECOIN

    def is_native(self, asset: Asset) -> bool:
        """Check if asset is native XLM.

        Parameters
        ----------
        asset : Asset
            Asset to check

        Returns
        -------
        bool
            True if native XLM
        """
        return asset.issuer is None and asset.code == "XLM"

    def get_preferred_base(self, asset: Asset) -> Asset:
        """Get preferred base currency for an asset.

        Parameters
        ----------
        asset : Asset
            Asset

        Returns
        -------
        Asset
            Preferred base currency (typically XLM)
        """
        metadata = self.classify(asset)
        return metadata.preferred_base or NATIVE_ASSET

    def _asset_key(self, asset: Asset) -> str:
        """Generate cache key for asset."""
        return f"{asset.code}:{asset.issuer or 'native'}"


# ---------------------------------------------------------------------------
# Cached Exchange Rate Provider (Wrapper)
# ---------------------------------------------------------------------------


class CachedRateProvider:
    """Wrapper that adds TTL caching to any ExchangeRateProvider.

    Caches exchange rates for a configurable TTL to reduce provider calls
    and improve performance for bulk operations.

    Parameters
    ----------
    provider : ExchangeRateProvider
        Underlying rate provider
    ttl : timedelta
        Time-to-live for cached rates

    Examples
    --------
    >>> base_provider = MockExchangeRateProvider()
    >>> cached = CachedRateProvider(base_provider, ttl=timedelta(minutes=5))
    >>>
    >>> # First call fetches from base provider
    >>> rate1 = cached.get_rate(usdc, xlm)
    >>>
    >>> # Second call returns cached (if within TTL)
    >>> rate2 = cached.get_rate(usdc, xlm)
    """

    def __init__(
        self,
        provider: ExchangeRateProvider,
        ttl: timedelta = timedelta(minutes=5),
    ):
        self.provider = provider
        self.ttl = ttl
        self._cache: dict[tuple[str, str, str], CurrencyPair] = {}
        self._cache_times: dict[tuple[str, str, str], datetime] = {}

    def get_rate(
        self,
        from_asset: Asset,
        to_asset: Asset,
        timestamp: datetime | None = None,
    ) -> CurrencyPair | None:
        """Get exchange rate (with caching)."""
        # Generate cache key
        ts_key = timestamp.isoformat() if timestamp else "current"
        cache_key = (
            f"{from_asset.code}:{from_asset.issuer or 'native'}",
            f"{to_asset.code}:{to_asset.issuer or 'native'}",
            ts_key,
        )

        # Check cache
        if cache_key in self._cache:
            cached_time = self._cache_times[cache_key]
            age = datetime.now() - cached_time

            if age < self.ttl:
                # Cache hit
                return self._cache[cache_key]
            else:
                # Cache expired
                del self._cache[cache_key]
                del self._cache_times[cache_key]

        # Fetch from provider
        rate = self.provider.get_rate(from_asset, to_asset, timestamp)

        if rate:
            # Cache result
            self._cache[cache_key] = rate
            self._cache_times[cache_key] = datetime.now()

        return rate

    def get_rates_batch(
        self,
        pairs: list[tuple[Asset, Asset]],
        timestamp: datetime | None = None,
    ) -> dict[tuple[str, str], CurrencyPair]:
        """Get multiple rates (with caching)."""
        # Check which pairs are cached
        uncached_pairs = []
        results = {}

        for from_asset, to_asset in pairs:
            rate = self.get_rate(from_asset, to_asset, timestamp)
            if rate:
                results[rate.pair_key()] = rate
            else:
                uncached_pairs.append((from_asset, to_asset))

        # Fetch uncached from provider
        if uncached_pairs:
            uncached_results = self.provider.get_rates_batch(uncached_pairs, timestamp)

            # Cache and add to results
            for pair_key, rate in uncached_results.items():
                cache_key = (
                    pair_key[0],
                    pair_key[1],
                    timestamp.isoformat() if timestamp else "current",
                )
                self._cache[cache_key] = rate
                self._cache_times[cache_key] = datetime.now()
                results[pair_key] = rate

        return results

    def is_available(self, asset: Asset) -> bool:
        """Check if rates available for asset."""
        return self.provider.is_available(asset)

    def clear_cache(self):
        """Clear all cached rates."""
        self._cache.clear()
        self._cache_times.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns
        -------
        dict
            Cache size, hit rate, etc.
        """
        return {
            "cache_size": len(self._cache),
            "oldest_entry": (min(self._cache_times.values()) if self._cache_times else None),
            "newest_entry": (max(self._cache_times.values()) if self._cache_times else None),
        }


# ---------------------------------------------------------------------------
# Normalization Strategies
# ---------------------------------------------------------------------------


class NormalizationStrategy:
    """Base class for normalization strategies.

    Strategies define how to convert amounts to a common base currency.
    Different strategies use different base currencies (XLM, USD, etc.).
    """

    def __init__(self, provider: ExchangeRateProvider):
        self.provider = provider
        self.classifier = AssetClassifier()

    def normalize(
        self,
        amount: Decimal | DecimalAmount,
        asset: Asset,
        timestamp: datetime | None = None,
    ) -> NormalizedAmount:
        """Normalize an amount to the strategy's base currency.

        Parameters
        ----------
        amount : Decimal | DecimalAmount
            Amount to normalize
        asset : Asset
            Asset of the amount
        timestamp : datetime, optional
            Historical timestamp

        Returns
        -------
        NormalizedAmount
            Normalized amount
        """
        raise NotImplementedError

    def get_base_asset(self) -> Asset:
        """Get the base currency for this strategy.

        Returns
        -------
        Asset
            Base currency
        """
        raise NotImplementedError


class XLMNormalization(NormalizationStrategy):
    """Normalize all amounts to XLM (Stellar native currency).

    This is the most natural choice for Stellar DEX, as XLM is the native
    asset and has the most liquid trading pairs.

    Examples
    --------
    >>> provider = MockExchangeRateProvider()
    >>> strategy = XLMNormalization(provider)
    >>>
    >>> # Normalize USDC to XLM
    >>> amount = DecimalAmount("100")
    >>> usdc = Asset(code="USDC", issuer="...")
    >>> normalized = strategy.normalize(amount, usdc)
    >>> print(f"100 USDC = {normalized.value} XLM")
    """

    def get_base_asset(self) -> Asset:
        """Return XLM as base."""
        return NATIVE_ASSET

    def normalize(
        self,
        amount: Decimal | DecimalAmount,
        asset: Asset,
        timestamp: datetime | None = None,
    ) -> NormalizedAmount:
        """Normalize amount to XLM."""
        return normalize_amount(
            amount,
            asset,
            self.get_base_asset(),
            self.provider,
            timestamp,
        )


class USDNormalization(NormalizationStrategy):
    """Normalize all amounts to USD equivalent (via stablecoins).

    Uses USDC as the primary USD proxy. Falls back to USDT or other
    stablecoins if USDC rate not available.

    Examples
    --------
    >>> provider = MockExchangeRateProvider()
    >>> strategy = USDNormalization(provider)
    >>>
    >>> # Normalize XLM to USD
    >>> amount = DecimalAmount("850")
    >>> xlm = Asset(code="XLM", issuer=None)
    >>> normalized = strategy.normalize(amount, xlm)
    >>> print(f"850 XLM = {normalized.value} USD")
    """

    def __init__(self, provider: ExchangeRateProvider):
        super().__init__(provider)

        # Preferred USD stablecoin (USDC)
        self.usd_asset = Asset(
            code="USDC",
            issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        )

    def get_base_asset(self) -> Asset:
        """Return USDC as USD proxy."""
        return self.usd_asset

    def normalize(
        self,
        amount: Decimal | DecimalAmount,
        asset: Asset,
        timestamp: datetime | None = None,
    ) -> NormalizedAmount:
        """Normalize amount to USD (via USDC)."""
        # If already USD stablecoin, treat as 1:1
        if self.classifier.is_stablecoin(asset):
            if isinstance(amount, DecimalAmount):
                amount_decimal = amount.value
            else:
                amount_decimal = validate_amount(amount)

            return NormalizedAmount(
                value=amount_decimal,
                base_asset=self.usd_asset,
                original_value=amount_decimal,
                original_asset=asset,
                exchange_rate=None,
                confidence=Decimal("0.99"),  # Slight discount for stablecoin peg risk
                status=NormalizationStatus.SUCCESS,
                conversion_path=[asset],
            )

        # Convert via exchange rate
        return normalize_amount(
            amount,
            asset,
            self.get_base_asset(),
            self.provider,
            timestamp,
        )


class MultiHopNormalization(NormalizationStrategy):
    """Multi-hop normalization via intermediate liquid pairs.

    If direct rate not available, attempts to find conversion path through
    liquid intermediate assets (typically XLM or major stablecoins).

    Examples
    --------
    >>> provider = MockExchangeRateProvider()
    >>> strategy = MultiHopNormalization(provider, base_asset=usdc)
    >>>
    >>> # Convert obscure token to USD via XLM
    >>> # TOKEN -> XLM -> USDC
    >>> amount = DecimalAmount("1000")
    >>> token = Asset(code="OBSCURE", issuer="...")
    >>> normalized = strategy.normalize(amount, token)
    """

    def __init__(
        self,
        provider: ExchangeRateProvider,
        base_asset: Asset | None = None,
        max_hops: int = 3,
    ):
        super().__init__(provider)
        self.base_asset = base_asset or NATIVE_ASSET
        self.max_hops = max_hops

        # Liquid intermediate assets to try
        self.liquid_assets = [
            NATIVE_ASSET,  # XLM
            Asset(code="USDC", issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"),
            Asset(code="USDT", issuer="GCQTGZQQ5G4PTM2GL7CDIFKUBIPEC52BROAQIAPW53XBRJVN6ZJVTG6V"),
        ]

    def get_base_asset(self) -> Asset:
        """Return configured base asset."""
        return self.base_asset

    def normalize(
        self,
        amount: Decimal | DecimalAmount,
        asset: Asset,
        timestamp: datetime | None = None,
    ) -> NormalizedAmount:
        """Normalize with multi-hop conversion."""
        # Try direct conversion first
        direct = normalize_amount(
            amount,
            asset,
            self.base_asset,
            self.provider,
            timestamp,
        )

        if direct.is_successful():
            return direct

        # Try multi-hop via liquid intermediates
        best_result = direct
        best_confidence = Decimal("0.0")

        for intermediate in self.liquid_assets:
            # Skip if intermediate is source or target
            if intermediate.code == asset.code and intermediate.issuer == asset.issuer:
                continue
            if (
                intermediate.code == self.base_asset.code
                and intermediate.issuer == self.base_asset.issuer
            ):
                continue

            # Try: asset -> intermediate -> base
            hop1 = normalize_amount(
                amount,
                asset,
                intermediate,
                self.provider,
                timestamp,
            )

            if not hop1.is_successful():
                continue

            hop2 = normalize_amount(
                hop1.value,
                intermediate,
                self.base_asset,
                self.provider,
                timestamp,
            )

            if not hop2.is_successful():
                continue

            # Success! Combine results
            combined_confidence = (
                hop1.confidence * hop2.confidence * Decimal("0.9")
            )  # Penalty for multi-hop

            if combined_confidence > best_confidence:
                if isinstance(amount, DecimalAmount):
                    amount_decimal = amount.value
                else:
                    amount_decimal = validate_amount(amount)

                best_result = NormalizedAmount(
                    value=hop2.value,
                    base_asset=self.base_asset,
                    original_value=amount_decimal,
                    original_asset=asset,
                    exchange_rate=hop2.exchange_rate,
                    confidence=combined_confidence,
                    status=NormalizationStatus.MULTI_HOP,
                    conversion_path=[asset, intermediate, self.base_asset],
                )
                best_confidence = combined_confidence

        return best_result


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def create_default_provider(use_cache: bool = True) -> ExchangeRateProvider:
    """Create default exchange rate provider for testing/development.

    Parameters
    ----------
    use_cache : bool
        Whether to wrap with caching

    Returns
    -------
    ExchangeRateProvider
        Provider instance
    """
    provider = MockExchangeRateProvider()

    if use_cache:
        provider = CachedRateProvider(provider, ttl=timedelta(minutes=5))

    return provider


def create_xlm_strategy(use_cache: bool = True) -> XLMNormalization:
    """Create XLM normalization strategy with default provider.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached provider

    Returns
    -------
    XLMNormalization
        Strategy instance
    """
    provider = create_default_provider(use_cache=use_cache)
    return XLMNormalization(provider)


def create_usd_strategy(use_cache: bool = True) -> USDNormalization:
    """Create USD normalization strategy with default provider.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached provider

    Returns
    -------
    USDNormalization
        Strategy instance
    """
    provider = create_default_provider(use_cache=use_cache)
    return USDNormalization(provider)


def format_normalized_amount(normalized: NormalizedAmount) -> str:
    """Format normalized amount for display.

    Parameters
    ----------
    normalized : NormalizedAmount
        Normalized amount

    Returns
    -------
    str
        Formatted string

    Examples
    --------
    >>> norm = NormalizedAmount(...)
    >>> print(format_normalized_amount(norm))
    100.00 USDC → 850.00 XLM (confidence: 95%)
    """
    if normalized.is_same_currency():
        return f"{normalized.value:.2f} {normalized.base_asset.code}"

    conversion_arrow = " → ".join(asset.code for asset in normalized.conversion_path)

    if not conversion_arrow:
        conversion_arrow = f"{normalized.original_asset.code} → {normalized.base_asset.code}"

    confidence_pct = float(normalized.confidence * 100)

    return (
        f"{normalized.original_value:.2f} {normalized.original_asset.code} "
        f"→ {normalized.value:.2f} {normalized.base_asset.code} "
        f"({conversion_arrow}, confidence: {confidence_pct:.0f}%)"
    )
