"""
benchmarks/contracts.py — Typed contracts for the benchmark subsystem.

All public types used by datasets.py, runner.py, and external callers live here
so the contracts can be imported without pulling in heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkDataset:
    """A single benchmark scenario with labelled trade data.

    Attributes:
        name: Short machine-readable identifier (e.g. ``"benford_baseline"``).
        description: Human-readable description of the scenario.
        trades: DataFrame of trade records.  Must contain the columns defined
            in ``REQUIRED_TRADE_COLUMNS``.  Validated on construction via
            :func:`validate`.
        labels: Boolean Series, same index as *trades*, where ``True`` means
            wash trade / anomalous.
        metadata: Arbitrary key-value tags (scenario type, expected difficulty,
            etc.).  Not used by the runner but surfaced in diagnostics.
    """

    name: str
    description: str
    trades: pd.DataFrame
    labels: pd.Series
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` with a clear diagnostic if the dataset is malformed."""
        missing = REQUIRED_TRADE_COLUMNS - set(self.trades.columns)
        if missing:
            raise ValueError(
                f"BenchmarkDataset '{self.name}': trades DataFrame is missing required "
                f"columns: {sorted(missing)}. "
                f"Present columns: {sorted(self.trades.columns)}."
            )
        if len(self.trades) != len(self.labels):
            raise ValueError(
                f"BenchmarkDataset '{self.name}': trades has {len(self.trades)} rows "
                f"but labels has {len(self.labels)} entries."
            )
        if self.labels.dtype != bool:
            raise ValueError(
                f"BenchmarkDataset '{self.name}': labels must be a boolean Series "
                f"(got dtype={self.labels.dtype})."
            )


# Minimum columns required in every benchmark trade DataFrame
REQUIRED_TRADE_COLUMNS: frozenset[str] = frozenset(
    [
        "wallet_id",
        "asset_pair",
        "amount",
        "timestamp",
    ]
)


@dataclass
class BenchmarkResult:
    """Metrics produced by running one detector against one dataset.

    Attributes:
        dataset_name: Name of the :class:`BenchmarkDataset` used.
        detector_name: Name/label of the detector under test.
        precision: Fraction of predicted positives that are true positives.
        recall: Fraction of true positives that were predicted as positive.
        f1: Harmonic mean of precision and recall.
        auc_roc: Area under the ROC curve (0–1).
        runtime_seconds: Wall-clock time to run the detector on the dataset.
        extra: Optional dict of additional detector-specific metrics.
        error: If not ``None``, the detector raised this exception; all metric
            fields will be ``None``.
    """

    dataset_name: str
    detector_name: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc_roc: float | None = None
    runtime_seconds: float | None = None
    extra: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def passed(self, min_f1: float = 0.0, min_auc_roc: float = 0.0) -> bool:
        """Return ``True`` if the result meets the minimum quality thresholds."""
        if self.error is not None:
            return False
        f1_ok = self.f1 is not None and self.f1 >= min_f1
        auc_ok = self.auc_roc is not None and self.auc_roc >= min_auc_roc
        return f1_ok and auc_ok

    def as_dict(self) -> dict[str, object]:
        """Serialise to a plain dict suitable for JSON or tabular output."""
        return {
            "dataset": self.dataset_name,
            "detector": self.detector_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "runtime_s": self.runtime_seconds,
            "error": self.error,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------


class DetectorProtocol(Protocol):
    """Structural protocol that any detector must satisfy to run benchmarks.

    A compliant detector accepts a DataFrame of trades and returns a Series of
    float scores (higher = more suspicious), one per row.
    """

    def __call__(self, trades: pd.DataFrame) -> pd.Series:
        """Score every trade row.  Returns a float Series, same index as *trades*."""
        ...


# Convenience alias — use this in type annotations instead of the Protocol
DetectorCallable = Callable[[pd.DataFrame], pd.Series]
