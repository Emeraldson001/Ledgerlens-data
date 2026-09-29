"""Package-boundary contracts for the LedgerLens platform.

Formal types and protocols that define the interfaces between packages.
Every cross-package data flow should be typed through this module so
that implementations can be composed, tested, and swapped independently.
"""

from contracts.alerting import AlertChannel, AlertEvent
from contracts.feature_store import FeatureStore
from contracts.risk_score import RiskScore, Scorer
from contracts.source import TradeSource

__all__ = [
    "AlertChannel",
    "AlertEvent",
    "FeatureStore",
    "RiskScore",
    "Scorer",
    "TradeSource",
]
