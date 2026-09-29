"""
ci_metrics/contracts.py — Typed contracts for the CI regression monitoring subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class MetricSnapshot:
    """A single named metric value captured during a CI run.

    Attributes:
        name: Metric identifier (e.g. ``"test_pass_rate"``, ``"benford_f1"``).
        value: Numeric value.
        unit: Optional unit string for display (e.g. ``"ratio"``, ``"seconds"``).
        higher_is_better: Direction of improvement; controls regression logic.
    """

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True


@dataclass
class CIRunRecord:
    """All metrics captured for a single CI pipeline run.

    Attributes:
        run_id: Unique run identifier (e.g. GitHub Actions ``GITHUB_RUN_ID``).
        commit_sha: Git commit SHA at which the run was triggered.
        branch: Branch name.
        timestamp_utc: ISO-8601 UTC timestamp string (``YYYY-MM-DDTHH:MM:SSZ``).
        metrics: List of :class:`MetricSnapshot` objects captured during the run.
        python_version: Python interpreter version string.
        extra: Arbitrary key-value metadata (workflow name, runner OS, etc.).
    """

    run_id: str
    commit_sha: str
    branch: str
    timestamp_utc: str
    metrics: list[MetricSnapshot] = field(default_factory=list)
    python_version: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "run_id": self.run_id,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "timestamp_utc": self.timestamp_utc,
            "python_version": self.python_version,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "higher_is_better": m.higher_is_better,
                }
                for m in self.metrics
            ],
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CIRunRecord:
        """Deserialise from a plain dict produced by :meth:`as_dict`."""
        metrics = [
            MetricSnapshot(
                name=str(m["name"]),
                value=float(m["value"]),  # type: ignore[arg-type]
                unit=str(m.get("unit", "")),
                higher_is_better=bool(m.get("higher_is_better", True)),
            )
            for m in (data.get("metrics") or [])
        ]
        return cls(
            run_id=str(data["run_id"]),
            commit_sha=str(data["commit_sha"]),
            branch=str(data["branch"]),
            timestamp_utc=str(data["timestamp_utc"]),
            metrics=metrics,
            python_version=str(data.get("python_version", "")),
            extra=dict(data.get("extra", {})),  # type: ignore[arg-type]
        )


@dataclass
class RegressionAlert:
    """A single regression signal detected by :class:`RegressionDetector`.

    Attributes:
        metric_name: Name of the regressed metric.
        latest_value: Value in the most recent run.
        baseline_value: Reference value (rolling mean of prior runs).
        delta_pct: Percentage change from baseline (negative = degradation for
            higher-is-better metrics).
        severity: ``"warning"`` or ``"critical"``.
        message: Human-readable summary for CI log output.
    """

    metric_name: str
    latest_value: float
    baseline_value: float
    delta_pct: float
    severity: Literal["warning", "critical"]
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric_name,
            "latest": self.latest_value,
            "baseline": self.baseline_value,
            "delta_pct": round(self.delta_pct, 2),
            "severity": self.severity,
            "message": self.message,
        }
