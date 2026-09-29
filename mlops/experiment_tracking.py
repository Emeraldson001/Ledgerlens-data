"""Reproducible local experiment tracking.

This module intentionally uses append-only JSONL files so model development
metadata remains easy to diff, archive, and inspect in CI without requiring an
external service.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ExperimentRun:
    """Immutable metadata for one model-development run."""

    name: str
    params: dict[str, Any]
    feature_schema_hash: str
    dataset_sha256: str
    git_sha: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ExperimentRun.name must be non-empty")
        if not self.feature_schema_hash.strip():
            raise ValueError("ExperimentRun.feature_schema_hash must be non-empty")
        if not self.dataset_sha256.strip():
            raise ValueError("ExperimentRun.dataset_sha256 must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("ExperimentRun.created_at must be timezone-aware")

    @property
    def run_id(self) -> str:
        payload = {
            "dataset_sha256": self.dataset_sha256,
            "feature_schema_hash": self.feature_schema_hash,
            "git_sha": self.git_sha,
            "name": self.name,
            "params": self.params,
        }
        return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:16]

    def to_record(
        self, metrics: dict[str, float], artifacts: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "params": self.params,
            "metrics": metrics,
            "artifacts": artifacts or {},
            "feature_schema_hash": self.feature_schema_hash,
            "dataset_sha256": self.dataset_sha256,
            "git_sha": self.git_sha,
        }


class JsonlExperimentTracker:
    """Append-only experiment tracker with deterministic run identifiers."""

    def __init__(self, path: str | Path = "models/experiments.jsonl") -> None:
        self.path = Path(path)

    def log_run(
        self,
        run: ExperimentRun,
        metrics: dict[str, float],
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not metrics:
            raise ValueError("metrics must be non-empty")
        invalid_metrics = [
            name for name, value in metrics.items() if not isinstance(value, int | float)
        ]
        if invalid_metrics:
            raise TypeError(f"metrics must be numeric: {invalid_metrics}")

        record = run.to_record(
            metrics={k: float(v) for k, v in metrics.items()}, artifacts=artifacts
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_stable_json(record) + "\n")
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
