"""Threshold Tuning Workflows for Anomaly Alerts — Issue #535.

Provides automated, data-driven workflows for setting and tuning the
LedgerLens alert threshold (``RISK_SCORE_FLAG_THRESHOLD``) so operators
can replace the hard-coded default of 70 with a value calibrated to their
precision/recall requirements.

Three tuning strategies are available:

``precision_recall``
    Sweeps threshold values and picks the one that maximises F1 (or
    a user-supplied β-weighted F-score) on a labelled evaluation set.

``false_positive_budget``
    Finds the highest threshold (fewest alerts) that keeps the false-positive
    rate at or below a user-supplied budget (e.g. FPR ≤ 0.05).

``cost_sensitive``
    Minimises a configurable cost function
    ``cost = FP_weight * FP + FN_weight * FN``
    to balance the operational cost of alert fatigue against missed detections.

Results are written to ``reports/threshold_tuning/`` as JSON and Markdown.

This module is consumed by ``scripts/tune_alert_thresholds.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path("reports/threshold_tuning")

# Candidate threshold grid
DEFAULT_THRESHOLD_GRID = list(range(40, 96, 5))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ThresholdEvalPoint:
    threshold: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    f_beta: float
    fpr: float
    cost: float


@dataclass
class TuningResult:
    strategy: str
    recommended_threshold: int
    eval_points: list[ThresholdEvalPoint] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "recommended_threshold": self.recommended_threshold,
            "parameters": self.parameters,
            "generated_at": self.generated_at
            or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "eval_points": [
                {
                    "threshold": p.threshold,
                    "tp": p.tp,
                    "fp": p.fp,
                    "tn": p.tn,
                    "fn": p.fn,
                    "precision": round(p.precision, 4),
                    "recall": round(p.recall, 4),
                    "f1": round(p.f1, 4),
                    "f_beta": round(p.f_beta, 4),
                    "fpr": round(p.fpr, 4),
                    "cost": round(p.cost, 4),
                }
                for p in self.eval_points
            ],
        }


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: int,
    beta: float = 1.0,
    fp_weight: float = 1.0,
    fn_weight: float = 1.0,
) -> ThresholdEvalPoint:
    """Compute classification metrics at a given threshold."""
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    denom_beta = (1 + beta**2) * precision + recall
    f_beta = ((1 + beta**2) * precision * recall) / denom_beta if denom_beta > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    cost = fp_weight * fp + fn_weight * fn

    return ThresholdEvalPoint(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        f_beta=f_beta,
        fpr=fpr,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Tuning strategies
# ---------------------------------------------------------------------------


class ThresholdTuner:
    """Data-driven threshold tuning for LedgerLens anomaly alerts.

    Parameters
    ----------
    scores:
        1-D array of model risk scores (0–100) for the evaluation set.
    labels:
        1-D binary array of ground-truth labels (1 = wash-trade, 0 = clean).
    threshold_grid:
        Candidate threshold values to evaluate (default: 40, 45, …, 95).
    """

    def __init__(
        self,
        scores: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
        threshold_grid: list[int] | None = None,
    ) -> None:
        self._scores = np.asarray(scores, dtype=float)
        self._labels = np.asarray(labels, dtype=int)
        self._grid = threshold_grid or DEFAULT_THRESHOLD_GRID

        if len(self._scores) != len(self._labels):
            raise ValueError(
                f"scores and labels must have the same length "
                f"({len(self._scores)} vs {len(self._labels)})"
            )
        if len(self._scores) == 0:
            raise ValueError("scores/labels arrays must not be empty")

    # ------------------------------------------------------------------
    # Strategy: precision_recall
    # ------------------------------------------------------------------

    def tune_precision_recall(self, beta: float = 1.0) -> TuningResult:
        """Pick the threshold that maximises the F-beta score.

        beta=1 → F1 (equal precision/recall weight)
        beta<1 → precision-favoured (fewer false alerts)
        beta>1 → recall-favoured (fewer missed wash trades)
        """
        eval_points = [
            _compute_metrics(self._scores, self._labels, t, beta=beta) for t in self._grid
        ]
        best = max(eval_points, key=lambda p: p.f_beta)
        return TuningResult(
            strategy="precision_recall",
            recommended_threshold=best.threshold,
            eval_points=eval_points,
            parameters={"beta": beta},
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    # ------------------------------------------------------------------
    # Strategy: false_positive_budget
    # ------------------------------------------------------------------

    def tune_false_positive_budget(self, max_fpr: float = 0.05) -> TuningResult:
        """Find the lowest threshold where FPR ≤ max_fpr.

        Among all threshold values that satisfy the FPR constraint, the one
        that maximises recall is selected (i.e. we catch as many wash trades
        as possible without exceeding the FP budget).
        """
        eval_points = [_compute_metrics(self._scores, self._labels, t) for t in self._grid]
        feasible = [p for p in eval_points if p.fpr <= max_fpr]
        if not feasible:
            # Relax: return the threshold with the minimum FPR
            logger.warning(
                "No threshold satisfies FPR ≤ %.3f — returning minimum-FPR threshold",
                max_fpr,
            )
            best = min(eval_points, key=lambda p: p.fpr)
        else:
            best = max(feasible, key=lambda p: p.recall)

        return TuningResult(
            strategy="false_positive_budget",
            recommended_threshold=best.threshold,
            eval_points=eval_points,
            parameters={"max_fpr": max_fpr},
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    # ------------------------------------------------------------------
    # Strategy: cost_sensitive
    # ------------------------------------------------------------------

    def tune_cost_sensitive(self, fp_weight: float = 1.0, fn_weight: float = 5.0) -> TuningResult:
        """Minimise a cost function ``FP_weight * FP + FN_weight * FN``.

        The asymmetric default (FN costs 5× more than FP) reflects the
        LedgerLens context where missing a wash-trade ring is worse than
        a false alert.
        """
        eval_points = [
            _compute_metrics(
                self._scores,
                self._labels,
                t,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            for t in self._grid
        ]
        best = min(eval_points, key=lambda p: p.cost)
        return TuningResult(
            strategy="cost_sensitive",
            recommended_threshold=best.threshold,
            eval_points=eval_points,
            parameters={"fp_weight": fp_weight, "fn_weight": fn_weight},
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    # ------------------------------------------------------------------
    # Convenience: run all strategies
    # ------------------------------------------------------------------

    def tune_all(
        self,
        beta: float = 1.0,
        max_fpr: float = 0.05,
        fp_weight: float = 1.0,
        fn_weight: float = 5.0,
    ) -> dict[str, TuningResult]:
        return {
            "precision_recall": self.tune_precision_recall(beta=beta),
            "false_positive_budget": self.tune_false_positive_budget(max_fpr=max_fpr),
            "cost_sensitive": self.tune_cost_sensitive(fp_weight=fp_weight, fn_weight=fn_weight),
        }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def save_tuning_report(
    result: TuningResult,
    output_dir: Path = REPORTS_DIR,
    also_markdown: bool = True,
) -> Path:
    """Write JSON (and optionally Markdown) report for a tuning result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"threshold_tuning_{result.strategy}_{ts}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Threshold tuning report → %s", json_path)

    if also_markdown:
        md_path = output_dir / f"threshold_tuning_{result.strategy}_{ts}.md"
        md_path.write_text(_render_markdown(result))
        logger.info("Threshold tuning markdown → %s", md_path)

    return json_path


def _render_markdown(result: TuningResult) -> str:
    d = result.to_dict()
    lines = [
        "# LedgerLens Threshold Tuning Report",
        "",
        f"**Strategy:** `{d['strategy']}`",
        f"**Recommended threshold:** **{d['recommended_threshold']}**",
        f"**Parameters:** {json.dumps(d['parameters'])}",
        f"**Generated:** {d['generated_at']}",
        "",
        "## Evaluation Grid",
        "",
        "| Threshold | Precision | Recall | F1 | F-beta | FPR | Cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in d["eval_points"]:
        marker = " ← recommended" if p["threshold"] == d["recommended_threshold"] else ""
        lines.append(
            f"| {p['threshold']}{marker} | {p['precision']:.3f} | {p['recall']:.3f} "
            f"| {p['f1']:.3f} | {p['f_beta']:.3f} | {p['fpr']:.3f} | {p['cost']:.1f} |"
        )
    return "\n".join(lines)
