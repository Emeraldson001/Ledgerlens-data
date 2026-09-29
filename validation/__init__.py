"""LedgerLens validation package.

Provides reusable parsing contracts and reconciliation checks for the
ingestion and feature pipeline layers.

Sub-modules
-----------
parsing        – #552: CSV / JSON parsing contracts with typed output,
                 field-level coercion, and structured error reporting.
reconciliation – #554: Cross-layer reconciliation checks that assert
                 derived records are consistent with their raw sources.
"""

from validation.parsing import (
    CSVParseError,
    JSONParseError,
    ParseResult,
    parse_csv,
    parse_json,
    parse_known_manipulation_events,
    parse_trade_record,
)
from validation.reconciliation import (
    ReconciliationError,
    ReconciliationReport,
    reconcile_features,
    reconcile_trade_counts,
    reconcile_wallet_scores,
)

__all__ = [
    # parsing
    "CSVParseError",
    "JSONParseError",
    "ParseResult",
    "parse_csv",
    "parse_json",
    "parse_known_manipulation_events",
    "parse_trade_record",
    # reconciliation
    "ReconciliationError",
    "ReconciliationReport",
    "reconcile_features",
    "reconcile_trade_counts",
    "reconcile_wallet_scores",
]
