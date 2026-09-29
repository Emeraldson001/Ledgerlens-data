"""Helper utilities for currency normalization in detection pipelines.

This module provides convenience functions for common normalization tasks
in fraud detection and analysis workflows.

Usage
-----
Normalize trades in a detection pipeline::

    from utils.normalization_helpers import normalize_trades_bulk
    import pandas as pd

    # DataFrame with trades
    trades_df = pd.DataFrame([...])

    # Normalize all to XLM
    normalized_df = normalize_trades_bulk(
        trades_df,
        base_currency="XLM",
        columns=["base_amount", "counter_amount"],
    )

    # Now all amounts in XLM for cross-asset comparison
    benford_analysis(normalized_df["base_amount_xlm"])

Compare volumes across asset pairs::

    from utils.normalization_helpers import compare_cross_pair_volumes

    # Volumes from different pairs
    volumes = {
        ("USDC", "XLM"): Decimal("10000"),
        ("BTC", "XLM"): Decimal("0.5"),
        ("USDT", "XLM"): Decimal("9500"),
    }

    # Normalize and compare
    normalized_volumes = compare_cross_pair_volumes(volumes, strategy)

    # Detect anomalies
    for pair, norm_volume in normalized_volumes.items():
        if norm_volume.value > threshold:
            flag_anomaly(pair, norm_volume)
"""

from decimal import Decimal
from importlib import import_module
from typing import Any

import pandas as pd

from utils.currency_normalization import (
    NATIVE_ASSET,
    NormalizationStrategy,
    NormalizedAmount,
    aggregate_normalized,
)
from utils.logging import get_logger

_data_models = import_module("ingestion.data_models")
Asset = _data_models.Asset
Trade = _data_models.Trade

logger = get_logger(__name__)


def normalize_trade_amounts_to_series(
    trades: list[Trade],
    strategy: NormalizationStrategy,
) -> tuple[pd.Series, pd.Series]:
    """Normalize trade amounts to pandas Series for analysis.

    Parameters
    ----------
    trades : list[Trade]
        List of trades
    strategy : NormalizationStrategy
        Normalization strategy

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (normalized_base_amounts, normalized_counter_amounts)

    Examples
    --------
    >>> trades = [Trade(...), Trade(...), ...]
    >>> strategy = create_xlm_strategy()
    >>> base_series, counter_series = normalize_trade_amounts_to_series(trades, strategy)
    >>>
    >>> # Use in Benford analysis
    >>> from utils.benford_precision import leading_digits_safe
    >>> digits = leading_digits_safe(base_series)
    """
    base_amounts = []
    counter_amounts = []

    for trade in trades:
        norm_base = trade.normalize_base_amount(strategy)
        norm_counter = trade.normalize_counter_amount(strategy)

        if norm_base.is_successful():
            base_amounts.append(norm_base.value)
        else:
            logger.warning(
                f"Failed to normalize trade {trade.trade_id} base amount: {norm_base.status}"
            )
            base_amounts.append(None)

        if norm_counter.is_successful():
            counter_amounts.append(norm_counter.value)
        else:
            logger.warning(
                f"Failed to normalize trade {trade.trade_id} counter amount: {norm_counter.status}"
            )
            counter_amounts.append(None)

    return pd.Series(base_amounts), pd.Series(counter_amounts)


def calculate_normalized_volume(
    trades: list[Trade],
    strategy: NormalizationStrategy,
    use_base: bool = True,
) -> NormalizedAmount:
    """Calculate total trading volume across trades.

    Parameters
    ----------
    trades : list[Trade]
        List of trades
    strategy : NormalizationStrategy
        Normalization strategy
    use_base : bool
        If True, use base_amount; if False, use counter_amount

    Returns
    -------
    NormalizedAmount
        Total volume in base currency

    Examples
    --------
    >>> trades = [Trade(...), Trade(...), ...]
    >>> strategy = create_xlm_strategy()
    >>> volume = calculate_normalized_volume(trades, strategy)
    >>> print(f"Total volume: {volume.value} XLM")
    """
    amounts = []

    for trade in trades:
        if use_base:
            amount = trade.base_amount
            asset = trade.base_asset
        else:
            amount = trade.counter_amount
            asset = trade.counter_asset

        amounts.append((amount, asset))

    return aggregate_normalized(
        amounts,
        strategy.get_base_asset(),
        strategy.provider,
        timestamp=trades[0].ledger_close_time if trades else None,
    )


def compare_cross_pair_volumes(
    volumes_by_pair: dict[tuple[str, str], Decimal],
    strategy: NormalizationStrategy,
    asset_resolver: dict[str, Asset] | None = None,
) -> dict[tuple[str, str], NormalizedAmount]:
    """Compare trading volumes across different asset pairs.

    Parameters
    ----------
    volumes_by_pair : dict[tuple[str, str], Decimal]
        Map of (base_code, counter_code) to volume
    strategy : NormalizationStrategy
        Normalization strategy
    asset_resolver : dict[str, Asset], optional
        Map of asset code to Asset object (for issuer resolution)

    Returns
    -------
    dict[tuple[str, str], NormalizedAmount]
        Map of pair to normalized volume

    Examples
    --------
    >>> volumes = {
    ...     ("USDC", "XLM"): Decimal("10000"),
    ...     ("BTC", "XLM"): Decimal("0.5"),
    ... }
    >>> strategy = create_xlm_strategy()
    >>> normalized = compare_cross_pair_volumes(volumes, strategy)
    >>>
    >>> # Compare
    >>> for pair, norm_volume in normalized.items():
    ...     print(f"{pair}: {norm_volume.value} XLM")
    """
    if asset_resolver is None:
        asset_resolver = {}

    normalized_volumes = {}

    for (base_code, counter_code), volume in volumes_by_pair.items():
        # Resolve assets
        if base_code in asset_resolver:
            base_asset = asset_resolver[base_code]
        elif base_code == "XLM":
            base_asset = NATIVE_ASSET
        else:
            # Assume token without known issuer
            base_asset = Asset(code=base_code, issuer="UNKNOWN")

        # Normalize
        normalized = strategy.normalize(volume, base_asset)
        normalized_volumes[(base_code, counter_code)] = normalized

    return normalized_volumes


def detect_cross_pair_anomalies(
    volumes_by_pair: dict[tuple[str, str], Decimal],
    strategy: NormalizationStrategy,
    threshold_multiplier: Decimal = Decimal("3.0"),
) -> list[tuple[tuple[str, str], NormalizedAmount, str]]:
    """Detect anomalous volumes across asset pairs.

    Identifies pairs with volumes significantly higher than median.

    Parameters
    ----------
    volumes_by_pair : dict[tuple[str, str], Decimal]
        Map of pair to volume
    strategy : NormalizationStrategy
        Normalization strategy
    threshold_multiplier : Decimal
        Multiplier for median (default 3x = anomaly)

    Returns
    -------
    list[tuple[tuple[str, str], NormalizedAmount, str]]
        List of (pair, normalized_volume, reason)

    Examples
    --------
    >>> volumes = {...}
    >>> strategy = create_xlm_strategy()
    >>> anomalies = detect_cross_pair_anomalies(volumes, strategy)
    >>>
    >>> for pair, volume, reason in anomalies:
    ...     print(f"Anomaly: {pair} - {reason}")
    """
    # Normalize all volumes
    normalized_volumes = compare_cross_pair_volumes(volumes_by_pair, strategy)

    # Get values
    values = [norm.value for norm in normalized_volumes.values() if norm.is_successful()]

    if not values:
        return []

    # Calculate median
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median = sorted_values[n // 2]

    # Detect anomalies
    anomalies = []
    threshold = median * threshold_multiplier

    for pair, normalized in normalized_volumes.items():
        if not normalized.is_successful():
            anomalies.append((pair, normalized, "Normalization failed"))
        elif normalized.value > threshold:
            reason = f"Volume {normalized.value} exceeds threshold {threshold} ({threshold_multiplier}x median)"
            anomalies.append((pair, normalized, reason))
        elif normalized.confidence < Decimal("0.5"):
            reason = f"Low confidence ({normalized.confidence}) in normalized volume"
            anomalies.append((pair, normalized, reason))

    return anomalies


def create_normalized_dataframe(
    trades: list[Trade],
    strategy: NormalizationStrategy,
) -> pd.DataFrame:
    """Create DataFrame with normalized amounts for analysis.

    Parameters
    ----------
    trades : list[Trade]
        List of trades
    strategy : NormalizationStrategy
        Normalization strategy

    Returns
    -------
    pd.DataFrame
        DataFrame with original and normalized amounts

    Examples
    --------
    >>> trades = [Trade(...), ...]
    >>> strategy = create_xlm_strategy()
    >>> df = create_normalized_dataframe(trades, strategy)
    >>>
    >>> # Columns: trade_id, base_amount, base_amount_norm, base_asset,
    >>> #          counter_amount, counter_amount_norm, counter_asset,
    >>> #          normalization_confidence
    """
    rows = []

    for trade in trades:
        norm_base = trade.normalize_base_amount(strategy)
        norm_counter = trade.normalize_counter_amount(strategy)

        row = {
            "trade_id": trade.trade_id,
            "ledger_close_time": trade.ledger_close_time,
            "base_amount": trade.base_amount,
            "base_amount_norm": norm_base.value if norm_base.is_successful() else None,
            "base_asset_code": trade.base_asset.code,
            "counter_amount": trade.counter_amount,
            "counter_amount_norm": norm_counter.value if norm_counter.is_successful() else None,
            "counter_asset_code": trade.counter_asset.code,
            "normalization_confidence": min(norm_base.confidence, norm_counter.confidence),
            "normalization_status": (
                "success"
                if norm_base.is_successful() and norm_counter.is_successful()
                else "failed"
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def filter_high_confidence_normalizations(
    normalized_amounts: list[NormalizedAmount],
    min_confidence: Decimal = Decimal("0.8"),
) -> list[NormalizedAmount]:
    """Filter normalized amounts by confidence threshold.

    Parameters
    ----------
    normalized_amounts : list[NormalizedAmount]
        List of normalized amounts
    min_confidence : Decimal
        Minimum confidence threshold

    Returns
    -------
    list[NormalizedAmount]
        Filtered list with high confidence only

    Examples
    --------
    >>> normalized_amounts = [...]
    >>> high_confidence = filter_high_confidence_normalizations(
    ...     normalized_amounts,
    ...     min_confidence=Decimal("0.9"),
    ... )
    >>> print(f"High confidence: {len(high_confidence)}/{len(normalized_amounts)}")
    """
    return [
        norm
        for norm in normalized_amounts
        if norm.is_successful() and norm.confidence >= min_confidence
    ]


def calculate_normalization_success_rate(
    normalized_amounts: list[NormalizedAmount],
) -> dict[str, Any]:
    """Calculate statistics on normalization success.

    Parameters
    ----------
    normalized_amounts : list[NormalizedAmount]
        List of normalized amounts

    Returns
    -------
    dict
        Statistics: success_rate, avg_confidence, failure_reasons

    Examples
    --------
    >>> normalized_amounts = [...]
    >>> stats = calculate_normalization_success_rate(normalized_amounts)
    >>> print(f"Success rate: {stats['success_rate']:.1%}")
    >>> print(f"Avg confidence: {stats['avg_confidence']:.2f}")
    """
    total = len(normalized_amounts)
    if total == 0:
        return {
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "failure_reasons": {},
        }

    successful = [norm for norm in normalized_amounts if norm.is_successful()]
    failed = [norm for norm in normalized_amounts if not norm.is_successful()]

    success_rate = len(successful) / total

    avg_confidence = (
        sum(norm.confidence for norm in successful) / len(successful)
        if successful
        else Decimal("0.0")
    )

    failure_reasons = {}
    for norm in failed:
        reason = norm.status.value
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "success_rate": float(success_rate),
        "avg_confidence": float(avg_confidence),
        "failure_reasons": failure_reasons,
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
    }
