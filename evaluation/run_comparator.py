"""Evaluation report comparisons across model runs — Issue #534.

Provides a ``ModelRunComparator`` that loads ``metrics.json`` files from
multiple training runs and produces:

* A unified diff of per-metric values between runs.
* Regression detection: flags metrics that dropped by more than a configurable
  tolerance from a *baseline* run.
* A human-readable text summary and a machine-readable JSON report.

The comparator is intentionally schema-flexible: any numeric scalar found in
``metrics.json`` is eligible for comparison.  The exact metric keys vary across
runs (e.g. as new metrics are added), and the comparator handles missing keys
gracefully by marking them as ``"added"`` or ``"removed"``.

Components
----------
``RunMetrics``
    Lightweight dataclass wrapping a parsed ``metrics.json`` file.

``MetricDiff``
    Per-metric comparison result.

``RegressionFlag``
    Populated when a metric change exceeds the regression tolerance.

``ComparisonReport``
    Aggregated output of :meth:`ModelRunComparator.compare`.

``ModelRunComparator``
    Orchestrates loading and comparison; entry point for all use cases.

Usage example
-------------
>>> comparator = ModelRunComparator(
...     runs_dir=Path("models"),
... )
>>> report = comparator.compare(
...     baseline_run="run_20240601",
...     candidate_run="run_20240701",
...     regression_tolerance=0.01,
... )
>>> print(report.summary())
>>> report.save(Path("reports/run_comparison.json"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RunMetrics",
    "MetricDiff",
    "RegressionFlag",
    "ComparisonReport",
    "ModelRunComparator",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Parsed metrics.json from a single model training run.

    Parameters
    ----------
    run_id:
        Unique identifier for the run (directory or filename stem).
    metrics:
        Flat dict of metric name → numeric value (or nested dict — the
        comparator flattens nested structures automatically).
    source_path:
        Path from which the metrics were loaded.
    raw:
        Original unparsed JSON dict (preserved for reference).
    """

    run_id: str
    metrics: dict[str, float]
    source_path: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | str, run_id: str | None = None) -> RunMetrics:
        """Load metrics from a JSON file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not valid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")

        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a JSON object at {path}; got {type(raw).__name__}")

        rid = run_id or path.parent.name or path.stem
        metrics = _flatten_metrics(raw)
        return cls(run_id=rid, metrics=metrics, source_path=str(path), raw=raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_path": self.source_path,
            "metrics": self.metrics,
        }


def _flatten_metrics(
    obj: Any,
    prefix: str = "",
    sep: str = ".",
) -> dict[str, float]:
    """Recursively flatten *obj* into a dot-separated dict of numeric values.

    Non-numeric leaf values (strings, booleans, lists, etc.) are silently
    skipped — only ``int`` and ``float`` scalars are included.
    """
    result: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}{sep}{k}" if prefix else k
            result.update(_flatten_metrics(v, prefix=full_key, sep=sep))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        result[prefix] = float(obj)
    return result


# ---------------------------------------------------------------------------
# Diff types
# ---------------------------------------------------------------------------


@dataclass
class MetricDiff:
    """Comparison result for a single metric between two runs."""

    metric: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None  # candidate - baseline; None if one run is missing
    delta_pct: float | None  # relative change %; None if baseline is 0 or missing
    status: str  # "improved" | "regressed" | "unchanged" | "added" | "removed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline_value,
            "candidate": self.candidate_value,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "status": self.status,
        }


@dataclass
class RegressionFlag:
    """Records a detected regression for a single metric."""

    metric: str
    baseline_value: float
    candidate_value: float
    delta: float
    delta_pct: float | None
    tolerance: float  # configured tolerance used for this flag

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline_value,
            "candidate": self.candidate_value,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "tolerance": self.tolerance,
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Aggregated output of :class:`ModelRunComparator.compare`."""

    baseline_run_id: str
    candidate_run_id: str
    diffs: list[MetricDiff] = field(default_factory=list)
    regressions: list[RegressionFlag] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @property
    def improved(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.status == "improved"]

    @property
    def regressed_diffs(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.status == "regressed"]

    @property
    def added(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.status == "added"]

    @property
    def removed(self) -> list[MetricDiff]:
        return [d for d in self.diffs if d.status == "removed"]

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def summary(self, max_rows: int = 40) -> str:
        """Return a human-readable summary string."""
        lines = [
            "ModelRunComparison",
            f"  baseline  : {self.baseline_run_id}",
            f"  candidate : {self.candidate_run_id}",
            f"  generated : {self.generated_at}",
            f"  metrics   : {len(self.diffs)} total  "
            f"(improved={len(self.improved)}, regressed={len(self.regressed_diffs)}, "
            f"added={len(self.added)}, removed={len(self.removed)})",
            f"  regressions flagged: {len(self.regressions)}",
            "",
        ]

        if self.regressions:
            lines.append("  ── REGRESSIONS ──")
            for r in self.regressions:
                pct = f"{r.delta_pct:+.2f}%" if r.delta_pct is not None else "N/A"
                lines.append(
                    f"  ✗ {r.metric}: {r.baseline_value:.6g} → {r.candidate_value:.6g} "
                    f"(Δ={r.delta:+.6g}, {pct}, tolerance={r.tolerance:.6g})"
                )
            lines.append("")

        if self.improved:
            lines.append("  ── IMPROVEMENTS ──")
            for d in self.improved[:max_rows]:
                pct = f"{d.delta_pct:+.2f}%" if d.delta_pct is not None else ""
                lines.append(
                    f"  ✓ {d.metric}: {d.baseline_value:.6g} → {d.candidate_value:.6g} "
                    f"(Δ={d.delta:+.6g} {pct})"
                )
            if len(self.improved) > max_rows:
                lines.append(f"  ... and {len(self.improved) - max_rows} more")
            lines.append("")

        if self.added:
            lines.append(f"  ── ADDED ({len(self.added)}) ──")
            for d in self.added[:max_rows]:
                lines.append(f"  + {d.metric} = {d.candidate_value:.6g}")
            lines.append("")

        if self.removed:
            lines.append(f"  ── REMOVED ({len(self.removed)}) ──")
            for d in self.removed[:max_rows]:
                lines.append(f"  - {d.metric} = {d.baseline_value:.6g}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "summary": {
                "total": len(self.diffs),
                "improved": len(self.improved),
                "regressed": len(self.regressed_diffs),
                "added": len(self.added),
                "removed": len(self.removed),
                "regressions_flagged": len(self.regressions),
            },
            "regressions": [r.to_dict() for r in self.regressions],
            "diffs": [d.to_dict() for d in self.diffs],
        }

    def save(self, path: Path | str) -> Path:
        """Write the report as a JSON file.  Parent directories are created."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Comparison report saved to %s", path)
        return path


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class ModelRunComparator:
    """Compare metrics.json files across model training runs.

    Two usage patterns are supported:

    **Directory-based**
    ::

        comparator = ModelRunComparator(runs_dir=Path("models"))
        report = comparator.compare("run_20240601", "run_20240701")

    Expects ``<runs_dir>/<run_id>/metrics.json`` for each run.

    **Path-based (direct)**
    ::

        comparator = ModelRunComparator()
        report = comparator.compare_paths(
            baseline_path=Path("models/run_A/metrics.json"),
            candidate_path=Path("models/run_B/metrics.json"),
        )

    Parameters
    ----------
    runs_dir:
        Root directory under which run sub-directories are found.  If
        ``None``, only path-based comparison is available.
    metrics_filename:
        Filename of the metrics file within each run directory.
        Defaults to ``"metrics.json"``.
    higher_is_better:
        Set of metric name substrings where *higher* values are better
        (e.g. AUC, F1).  Metrics not matching any entry in this set are
        assumed to be error metrics where *lower* is better.
        Defaults to a standard set for ML classification.
    lower_is_better:
        Explicit set of metric name substrings where *lower* is better.
        Takes precedence over *higher_is_better*.
    """

    _DEFAULT_HIGHER_IS_BETTER: frozenset[str] = frozenset(
        {"auc", "f1", "precision", "recall", "accuracy", "score", "ap", "pr_auc"}
    )
    _DEFAULT_LOWER_IS_BETTER: frozenset[str] = frozenset(
        {"loss", "error", "mse", "mae", "rmse", "bce", "cross_entropy"}
    )

    def __init__(
        self,
        runs_dir: Path | str | None = None,
        metrics_filename: str = "metrics.json",
        higher_is_better: frozenset[str] | None = None,
        lower_is_better: frozenset[str] | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else None
        self.metrics_filename = metrics_filename
        self._higher = higher_is_better or self._DEFAULT_HIGHER_IS_BETTER
        self._lower = lower_is_better or self._DEFAULT_LOWER_IS_BETTER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        baseline_run: str,
        candidate_run: str,
        regression_tolerance: float = 0.01,
        metadata: dict[str, Any] | None = None,
    ) -> ComparisonReport:
        """Compare two named runs under ``self.runs_dir``.

        Parameters
        ----------
        baseline_run:
            Sub-directory name of the baseline run (e.g. ``"run_20240601"``).
        candidate_run:
            Sub-directory name of the candidate run.
        regression_tolerance:
            Maximum allowed degradation before a metric is flagged as a
            regression.  Absolute delta for metrics in ``lower_is_better``,
            absolute delta for ``higher_is_better`` metrics.  Default ``0.01``
            (1 percentage point for AUC/F1).
        metadata:
            Arbitrary metadata stored in the report.

        Returns
        -------
        ComparisonReport
        """
        if self.runs_dir is None:
            raise RuntimeError("runs_dir not set; use compare_paths() instead")

        baseline_path = self.runs_dir / baseline_run / self.metrics_filename
        candidate_path = self.runs_dir / candidate_run / self.metrics_filename

        return self.compare_paths(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            baseline_run_id=baseline_run,
            candidate_run_id=candidate_run,
            regression_tolerance=regression_tolerance,
            metadata=metadata,
        )

    def compare_paths(
        self,
        baseline_path: Path | str,
        candidate_path: Path | str,
        baseline_run_id: str | None = None,
        candidate_run_id: str | None = None,
        regression_tolerance: float = 0.01,
        metadata: dict[str, Any] | None = None,
    ) -> ComparisonReport:
        """Compare metrics from two explicit paths.

        Parameters
        ----------
        baseline_path:
            Path to the baseline ``metrics.json``.
        candidate_path:
            Path to the candidate ``metrics.json``.
        baseline_run_id / candidate_run_id:
            Optional human labels for the runs; inferred from the path if not
            provided.
        regression_tolerance:
            See :meth:`compare`.
        metadata:
            Arbitrary metadata stored in the report.
        """
        baseline = RunMetrics.load(baseline_path, run_id=baseline_run_id)
        candidate = RunMetrics.load(candidate_path, run_id=candidate_run_id)

        diffs = self._compute_diffs(baseline.metrics, candidate.metrics)
        regressions = self._detect_regressions(diffs, regression_tolerance)

        return ComparisonReport(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            diffs=diffs,
            regressions=regressions,
            metadata={
                "baseline_source": baseline.source_path,
                "candidate_source": candidate.source_path,
                "regression_tolerance": regression_tolerance,
                **(metadata or {}),
            },
        )

    def list_runs(self) -> list[str]:
        """Return the list of run IDs found under ``runs_dir``.

        Raises
        ------
        RuntimeError
            If ``runs_dir`` is not set.
        """
        if self.runs_dir is None:
            raise RuntimeError("runs_dir not set")
        if not self.runs_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.runs_dir.iterdir()
            if p.is_dir() and (p / self.metrics_filename).exists()
        )

    def compare_all(
        self,
        regression_tolerance: float = 0.01,
    ) -> list[ComparisonReport]:
        """Compare each consecutive pair of runs (sorted by name).

        Useful for spotting metric drift over a sequence of retraining runs.
        Returns an empty list if fewer than two runs exist.
        """
        runs = self.list_runs()
        reports: list[ComparisonReport] = []
        for i in range(len(runs) - 1):
            report = self.compare(
                baseline_run=runs[i],
                candidate_run=runs[i + 1],
                regression_tolerance=regression_tolerance,
            )
            reports.append(report)
        return reports

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_diffs(
        self,
        baseline: dict[str, float],
        candidate: dict[str, float],
    ) -> list[MetricDiff]:
        all_keys = sorted(set(baseline) | set(candidate))
        diffs: list[MetricDiff] = []

        for key in all_keys:
            b_val = baseline.get(key)
            c_val = candidate.get(key)

            if b_val is None and c_val is not None:
                diffs.append(
                    MetricDiff(
                        metric=key,
                        baseline_value=None,
                        candidate_value=c_val,
                        delta=None,
                        delta_pct=None,
                        status="added",
                    )
                )
                continue

            if b_val is not None and c_val is None:
                diffs.append(
                    MetricDiff(
                        metric=key,
                        baseline_value=b_val,
                        candidate_value=None,
                        delta=None,
                        delta_pct=None,
                        status="removed",
                    )
                )
                continue

            # Both present
            assert b_val is not None and c_val is not None
            delta = c_val - b_val
            delta_pct: float | None = None
            if b_val != 0:
                delta_pct = (delta / abs(b_val)) * 100.0

            status = self._classify_status(key, delta)
            diffs.append(
                MetricDiff(
                    metric=key,
                    baseline_value=b_val,
                    candidate_value=c_val,
                    delta=delta,
                    delta_pct=delta_pct,
                    status=status,
                )
            )

        return diffs

    def _classify_status(self, metric_key: str, delta: float) -> str:
        """Return ``"improved"``, ``"regressed"``, or ``"unchanged"``."""
        if delta == 0.0:
            return "unchanged"

        key_lower = metric_key.lower()

        # Check explicit lower-is-better first
        if any(s in key_lower for s in self._lower):
            return "improved" if delta < 0 else "regressed"

        # Check higher-is-better
        if any(s in key_lower for s in self._higher):
            return "improved" if delta > 0 else "regressed"

        # Default: no direction known — just report as unchanged / improved
        return "improved" if delta > 0 else "regressed"

    def _detect_regressions(
        self, diffs: list[MetricDiff], tolerance: float
    ) -> list[RegressionFlag]:
        """Return regression flags for metrics that degraded beyond *tolerance*."""
        flags: list[RegressionFlag] = []
        for diff in diffs:
            if diff.status != "regressed":
                continue
            if diff.baseline_value is None or diff.candidate_value is None:
                continue
            if diff.delta is None:
                continue
            if abs(diff.delta) > tolerance:
                flags.append(
                    RegressionFlag(
                        metric=diff.metric,
                        baseline_value=diff.baseline_value,
                        candidate_value=diff.candidate_value,
                        delta=diff.delta,
                        delta_pct=diff.delta_pct,
                        tolerance=tolerance,
                    )
                )
        return flags
