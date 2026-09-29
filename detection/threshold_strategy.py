"""
Threshold Strategy Framework for Ledgerlens-data

This module provides a framework for determining the optimal risk score threshold
for flagging transactions or entities. It supports three strategies:

1. StaticStrategy: Uses a fixed threshold defined in the configuration or provided directly.
   Best for simple baselines or regulatory requirements that mandate a specific cutoff.
2. StatisticalStrategy: Sweeps a grid of candidate thresholds to maximize a target metric
   (e.g., F1-score) subject to constraints like a minimum recall floor.
   Best for offline evaluation or recalibration using historical labeled data.
3. AdaptiveStrategy: Uses a Multi-Armed Bandit (MAB) reinforcement learning agent
   to dynamically select thresholds based on context and feedback.
   Best for online production environments where the optimal threshold may drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ThresholdResult",
    "ThresholdDiagnostics",
    "ThresholdStrategy",
    "StaticStrategy",
    "StatisticalStrategy",
    "AdaptiveStrategy",
    "build_strategy",
]


@dataclass
class ThresholdResult:
    threshold: float
    strategy_name: str
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdDiagnostics:
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    sweep_curve: list[dict[str, float]]
    recommended_threshold: float
    alerts_impacted: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ThresholdStrategy(ABC):
    @abstractmethod
    def select_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThresholdResult:
        pass

    @abstractmethod
    def diagnostics(self, scores: np.ndarray, labels: np.ndarray) -> ThresholdDiagnostics:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class StaticStrategy(ThresholdStrategy):
    def __init__(self, threshold: float | None = None):
        if threshold is None:
            # config value is 0-100, we need 0-1
            self._threshold = config.RISK_SCORE_FLAG_THRESHOLD / 100.0
        else:
            self._threshold = threshold

    @property
    def name(self) -> str:
        return "static"

    def select_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThresholdResult:
        return ThresholdResult(threshold=self._threshold, strategy_name=self.name)

    def diagnostics(self, scores: np.ndarray, labels: np.ndarray) -> ThresholdDiagnostics:
        preds = (scores >= self._threshold).astype(int)
        p = float(precision_score(labels, preds, zero_division=0))
        r = float(recall_score(labels, preds, zero_division=0))
        f = float(f1_score(labels, preds, zero_division=0))

        alerts_impacted = int(preds.sum())

        sweep_curve = [
            {
                "threshold": self._threshold,
                "precision": p,
                "recall": r,
                "f1": f,
                "alert_count": alerts_impacted,
            }
        ]

        return ThresholdDiagnostics(
            precision_at_threshold=p,
            recall_at_threshold=r,
            f1_at_threshold=f,
            sweep_curve=sweep_curve,
            recommended_threshold=self._threshold,
            alerts_impacted=0,  # Recommended is current, so impact is 0
        )


class StatisticalStrategy(ThresholdStrategy):
    def __init__(
        self,
        grid: list[float] | None = None,
        target_metric: str = "f1",
        recall_floor: float | None = 0.85,
    ):
        self._grid = grid if grid is not None else [i / 100.0 for i in range(1, 100)]
        self._target_metric = target_metric
        self._recall_floor = recall_floor

    @property
    def name(self) -> str:
        return "statistical"

    def select_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThresholdResult:
        if labels is None:
            logger.warning(
                "StatisticalStrategy requires labels to select optimal threshold. Using fallback/first grid value."
            )
            return ThresholdResult(threshold=self._grid[0], strategy_name=self.name)

        diagnostics = self.diagnostics(scores, labels)
        return ThresholdResult(threshold=diagnostics.recommended_threshold, strategy_name=self.name)

    def diagnostics(self, scores: np.ndarray, labels: np.ndarray) -> ThresholdDiagnostics:
        sweep_curve = []
        best_threshold = self._grid[0]
        best_metric_val = -1.0

        fallback_threshold = self._grid[0]
        highest_recall = -1.0

        for th in self._grid:
            preds = (scores >= th).astype(int)
            p = float(precision_score(labels, preds, zero_division=0))
            r = float(recall_score(labels, preds, zero_division=0))
            f = float(f1_score(labels, preds, zero_division=0))
            alert_count = int(preds.sum())

            sweep_curve.append(
                {
                    "threshold": th,
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "alert_count": alert_count,
                }
            )

            if r > highest_recall:
                highest_recall = r
                fallback_threshold = th

            metric_val = (
                f
                if self._target_metric == "f1"
                else (p if self._target_metric == "precision" else r)
            )

            if self._recall_floor is None or r >= self._recall_floor:
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_threshold = th

        # If no threshold meets the recall floor, use the one with highest recall
        if best_metric_val == -1.0:
            logger.warning(
                f"No threshold met the recall floor of {self._recall_floor}. Falling back to highest recall."
            )
            best_threshold = fallback_threshold

        # Get stats for best threshold
        preds_best = (scores >= best_threshold).astype(int)
        p_best = float(precision_score(labels, preds_best, zero_division=0))
        r_best = float(recall_score(labels, preds_best, zero_division=0))
        f_best = float(f1_score(labels, preds_best, zero_division=0))
        alerts_best = int(preds_best.sum())

        # Calculate impact assuming current threshold is 0.5 for demonstration if no other context
        current_preds = (scores >= 0.5).astype(int)
        current_alerts = int(current_preds.sum())

        return ThresholdDiagnostics(
            precision_at_threshold=p_best,
            recall_at_threshold=r_best,
            f1_at_threshold=f_best,
            sweep_curve=sweep_curve,
            recommended_threshold=best_threshold,
            alerts_impacted=alerts_best - current_alerts,
        )


class AdaptiveStrategy(ThresholdStrategy):
    def __init__(self, agent: Any = None, state_path: str = "data/threshold_agent.json"):
        from detection.threshold_rl import ThresholdAgent

        if agent is None:
            # Try to load, fallback to new if not found/error
            try:
                self._agent = ThresholdAgent.load(state_path) if state_path else ThresholdAgent()
            except Exception:
                self._agent = ThresholdAgent()
        else:
            self._agent = agent

    @property
    def name(self) -> str:
        return "adaptive"

    def select_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThresholdResult:
        ctx = context or {}
        # agent returns 0-100 scale
        selected_arm = self._agent.select_threshold(ctx)
        threshold = selected_arm / 100.0

        return ThresholdResult(
            threshold=threshold, strategy_name=self.name, metadata={"selected_arm": selected_arm}
        )

    def diagnostics(self, scores: np.ndarray, labels: np.ndarray) -> ThresholdDiagnostics:
        # Get stats at current chosen threshold from RL agent
        # We simulate a generic context for diagnostic purposes
        selected_arm = self._agent.select_threshold({})
        threshold = selected_arm / 100.0

        preds = (scores >= threshold).astype(int)
        p = float(precision_score(labels, preds, zero_division=0))
        r = float(recall_score(labels, preds, zero_division=0))
        f = float(f1_score(labels, preds, zero_division=0))
        alert_count = int(preds.sum())

        sweep_curve = [
            {
                "threshold": threshold,
                "precision": p,
                "recall": r,
                "f1": f,
                "alert_count": alert_count,
            }
        ]

        metadata = {
            "q_values": getattr(self._agent, "q_values", None),
            "arm_counts": getattr(self._agent, "arm_counts", None),
        }

        return ThresholdDiagnostics(
            precision_at_threshold=p,
            recall_at_threshold=r,
            f1_at_threshold=f,
            sweep_curve=sweep_curve,
            recommended_threshold=threshold,
            alerts_impacted=0,
            metadata=metadata,
        )


def build_strategy(strategy_name: str, **kwargs) -> ThresholdStrategy:
    if strategy_name == "static":
        return StaticStrategy(**kwargs)
    elif strategy_name == "statistical":
        return StatisticalStrategy(**kwargs)
    elif strategy_name == "adaptive":
        return AdaptiveStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")
