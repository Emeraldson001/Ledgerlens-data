"""Risk-score contracts for cross-package communication.

The ``RiskScore`` TypedDict is the single most important cross-package
data structure in the platform — produced by the detection layer and
consumed by streaming, integrations, alerts, and reporting.

The ``Scorer`` protocol defines the interface that every scoring
component (production, shadow-deployment, ensemble) must satisfy.
"""

from __future__ import annotations

import typing
from typing import Protocol, TypedDict, runtime_checkable


class RiskScore(TypedDict, total=False):
    """The canonical risk-score shape flowing from detection to consumers.

    All fields are optional via ``total=False`` so that producers can
    include only the fields they populate and consumers can safely
    access fields via ``.get()``.

    Required in practice
    --------------------
    - ``wallet`` — wallet public key
    - ``asset_pair`` — e.g. ``"USDC_native"``
    - ``score`` — integer risk score 0–100
    - ``benford_flag`` — whether Benford analysis flagged
    - ``ml_flag`` — whether ML model flagged
    - ``confidence`` — confidence level 0–100
    - ``timestamp`` — unix seconds
    """

    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    timestamp: int

    # Extended fields
    propagated_risk: float
    ring_id: int | None
    score_lower: float
    score_upper: float
    coverage_guarantee: float
    replay_model_version: str
    model_name: str
    feature_contributions: dict[str, float]


@runtime_checkable
class Scorer(Protocol):
    """Interface for a component that produces risk scores.

    Implementations include ``RiskScorer``, ``ShadowDeploymentScorer``,
    and ensemble wrappers.

    Usage::

        def score(self, feature_row: pd.Series, **kwargs: Any) -> RiskScore:
            ...
    """

    def score(self, feature_row: typing.Any, **kwargs: typing.Any) -> RiskScore:
        """Score a single feature row and return a ``RiskScore`` dict."""
