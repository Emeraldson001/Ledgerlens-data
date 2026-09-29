"""Feature Compatibility Checks Across Model Versions — Issue #532.

Provides a systematic compatibility check between the feature schema of a
model version (as recorded in its ``model_metadata.json``) and an incoming
feature row or dataset.  This extends the existing single-version hash check
in ``detection/model_inference.py`` to support:

1. **Multi-version comparison** — compare the feature schemas of two or more
   model versions and identify added, removed, and renamed (type-shifted)
   features.

2. **Forward compatibility** — determine whether a feature row produced by a
   newer feature pipeline is safe to score with an older model (missing
   required columns are flagged; extra columns are warned).

3. **Backward compatibility** — determine whether a feature row from an older
   pipeline can be scored by a newer model (newly added required features
   that are absent get flagged).

4. **Schema diff report** — produce a human-readable diff so PR reviewers can
   see exactly what changed between two training runs.

This module is consumed by ``scripts/check_feature_compat.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------


def load_metadata(path: Path | str) -> dict[str, Any]:
    """Load a ``model_metadata.json`` file.

    Accepts either the exact file path or the directory that contains it.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "model_metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"model_metadata.json not found at: {p}")
    return json.loads(p.read_text())


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Reproduce the hash used by model_training.py to fingerprint the feature schema."""
    return "sha256:" + hashlib.sha256(json.dumps(sorted(feature_columns)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Compatibility result data structures
# ---------------------------------------------------------------------------


@dataclass
class CompatibilityIssue:
    severity: str  # "error" | "warning" | "info"
    code: str  # machine-readable code, e.g. "MISSING_FEATURE"
    feature: str  # affected feature name (or "" for schema-level issues)
    message: str


@dataclass
class CompatibilityReport:
    """Result of comparing a source feature schema against a target model's schema."""

    source_label: str  # e.g. "current_pipeline" or version name
    target_label: str  # e.g. "v2.1.0" or model dir path
    source_features: list[str]
    target_features: list[str]
    issues: list[CompatibilityIssue] = field(default_factory=list)
    generated_at: str = ""

    # Derived
    @property
    def added(self) -> list[str]:
        """Features in target but NOT in source (newly required by the model)."""
        return sorted(set(self.target_features) - set(self.source_features))

    @property
    def removed(self) -> list[str]:
        """Features in source but NOT in target (dropped from the model schema)."""
        return sorted(set(self.source_features) - set(self.target_features))

    @property
    def common(self) -> list[str]:
        return sorted(set(self.source_features) & set(self.target_features))

    @property
    def is_compatible(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at
            or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source_label": self.source_label,
            "target_label": self.target_label,
            "is_compatible": self.is_compatible,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "n_source_features": len(self.source_features),
            "n_target_features": len(self.target_features),
            "n_common_features": len(self.common),
            "features_added_in_target": self.added,
            "features_removed_from_target": self.removed,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "feature": i.feature,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------


class FeatureCompatibilityChecker:
    """Check feature schema compatibility between a source and a target model version.

    Parameters
    ----------
    target_metadata:
        The ``model_metadata.json`` dict for the model you want to score with.
    source_features:
        The list of feature columns produced by the current pipeline (or the
        source model's ``feature_columns`` list for a version-to-version diff).
    source_label:
        Human-readable name for the source (e.g. ``"current_pipeline"``).
    target_label:
        Human-readable name for the target model version.
    strict:
        When ``True``, extra columns in the source that are not in the target
        schema are also raised as errors (not just warnings).  Default ``False``.
    """

    def __init__(
        self,
        target_metadata: dict[str, Any],
        source_features: list[str],
        source_label: str = "source",
        target_label: str = "target",
        strict: bool = False,
    ) -> None:
        self._target_meta = target_metadata
        self._source_features = list(source_features)
        self._source_label = source_label
        self._target_label = target_label
        self._strict = strict

        target_cols = target_metadata.get("feature_columns")
        if not target_cols:
            raise ValueError(f"target metadata for '{target_label}' has no 'feature_columns' key")
        self._target_features: list[str] = list(target_cols)

    def check(self) -> CompatibilityReport:
        report = CompatibilityReport(
            source_label=self._source_label,
            target_label=self._target_label,
            source_features=self._source_features,
            target_features=self._target_features,
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

        source_set = set(self._source_features)
        target_set = set(self._target_features)

        # Schema hash match check
        source_hash = compute_feature_schema_hash(self._source_features)
        target_hash = self._target_meta.get("feature_schema_hash", "")
        if source_hash == target_hash:
            report.issues.append(
                CompatibilityIssue(
                    severity="info",
                    code="SCHEMA_HASH_MATCH",
                    feature="",
                    message=(
                        f"Feature schema hashes match exactly — "
                        f"'{self._source_label}' is fully compatible with "
                        f"'{self._target_label}'."
                    ),
                )
            )
            return report

        # Features present in target but missing from source (ERRORS — model can't score)
        missing_from_source = target_set - source_set
        for feat in sorted(missing_from_source):
            report.issues.append(
                CompatibilityIssue(
                    severity="error",
                    code="MISSING_FEATURE",
                    feature=feat,
                    message=(
                        f"Feature '{feat}' is required by model '{self._target_label}' "
                        f"but absent from '{self._source_label}'. "
                        "Add this feature to the pipeline or use a compatible model version."
                    ),
                )
            )

        # Features present in source but not in target (warnings — extra columns are ignored)
        extra_in_source = source_set - target_set
        severity = "error" if self._strict else "warning"
        for feat in sorted(extra_in_source):
            report.issues.append(
                CompatibilityIssue(
                    severity=severity,
                    code="EXTRA_FEATURE",
                    feature=feat,
                    message=(
                        f"Feature '{feat}' is present in '{self._source_label}' "
                        f"but not required by '{self._target_label}'. "
                        "It will be ignored during scoring."
                    ),
                )
            )

        # Flag version metadata for informational context
        trained_at = self._target_meta.get("trained_at", "unknown")
        python_ver = self._target_meta.get("python_version", "unknown")
        ll_ver = self._target_meta.get("ledgerlens_version", "unknown")
        report.issues.append(
            CompatibilityIssue(
                severity="info",
                code="VERSION_INFO",
                feature="",
                message=(
                    f"Target model '{self._target_label}' trained at {trained_at}, "
                    f"Python {python_ver}, LedgerLens {ll_ver}."
                ),
            )
        )

        return report


# ---------------------------------------------------------------------------
# Multi-version diff
# ---------------------------------------------------------------------------


@dataclass
class MultiVersionDiff:
    """Pairwise schema diff across multiple model versions."""

    versions: list[str]
    pairwise: list[dict[str, Any]] = field(default_factory=list)
    feature_timeline: dict[str, dict[str, str]] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at
            or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "versions": self.versions,
            "pairwise_diffs": self.pairwise,
            "feature_timeline": self.feature_timeline,
        }


def diff_model_versions(
    metadatas: list[tuple[str, dict[str, Any]]],
) -> MultiVersionDiff:
    """Compute pairwise schema diffs across a list of (label, metadata) pairs.

    ``feature_timeline`` records for each feature which versions it appeared in
    (``"present"``) or not (``"absent"``), making it easy to spot when a feature
    was introduced or removed.
    """
    if len(metadatas) < 2:
        raise ValueError("Need at least 2 model versions to diff")

    versions = [label for label, _ in metadatas]
    all_features: set[str] = set()
    for _, meta in metadatas:
        all_features.update(meta.get("feature_columns", []))

    # Build timeline: feature → {version_label: "present"/"absent"}
    timeline: dict[str, dict[str, str]] = {}
    for feat in sorted(all_features):
        timeline[feat] = {}
        for label, meta in metadatas:
            cols = set(meta.get("feature_columns", []))
            timeline[feat][label] = "present" if feat in cols else "absent"

    # Pairwise diffs (consecutive version pairs)
    pairwise: list[dict[str, Any]] = []
    for i in range(len(metadatas) - 1):
        a_label, a_meta = metadatas[i]
        b_label, b_meta = metadatas[i + 1]
        try:
            checker = FeatureCompatibilityChecker(
                target_metadata=b_meta,
                source_features=a_meta.get("feature_columns", []),
                source_label=a_label,
                target_label=b_label,
            )
            report = checker.check()
            pairwise.append(report.to_dict())
        except Exception as exc:  # noqa: BLE001
            pairwise.append(
                {
                    "source_label": a_label,
                    "target_label": b_label,
                    "error": str(exc),
                }
            )

    return MultiVersionDiff(
        versions=versions,
        pairwise=pairwise,
        feature_timeline=timeline,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


# ---------------------------------------------------------------------------
# Convenience: check current pipeline against active model
# ---------------------------------------------------------------------------


def check_current_pipeline(
    model_dir: str | Path = "models",
    pipeline_features: list[str] | None = None,
) -> CompatibilityReport:
    """Check the current pipeline's feature schema against the active model.

    If ``pipeline_features`` is not supplied, the list is derived from
    ``detection/model_training.py::FEATURE_COLUMNS_EXCLUDE`` and the
    synthetic dataset's columns (best-effort).
    """
    model_dir = Path(model_dir)

    # Load active model metadata
    try:
        metadata = load_metadata(model_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Active model metadata not found — run model training first: {exc}"
        ) from exc

    # Derive pipeline features if not supplied
    if pipeline_features is None:
        pipeline_features = _infer_pipeline_features(metadata)

    checker = FeatureCompatibilityChecker(
        target_metadata=metadata,
        source_features=pipeline_features,
        source_label="current_pipeline",
        target_label=str(model_dir),
    )
    return checker.check()


def _infer_pipeline_features(metadata: dict[str, Any]) -> list[str]:
    """Best-effort: infer the current pipeline's feature columns."""
    try:
        import pandas as pd

        from detection.feature_engineering import build_feature_matrix

        synth_path = Path("data/synthetic_dataset.parquet")
        if synth_path.exists():
            df = pd.read_parquet(synth_path)
            feat_df = build_feature_matrix(df)
            from detection.model_training import FEATURE_COLUMNS_EXCLUDE

            return [c for c in feat_df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not infer pipeline features dynamically: %s", exc)

    # Fall back to what the metadata recorded
    return list(metadata.get("feature_columns", []))
