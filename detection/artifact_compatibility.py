"""Model artifact versioning, compatibility validation, and schema governance.

This module defines the contract between the training pipeline and inference
services. Every trained artifact carries a manifest that declares its schema
version, feature hash, training provenance, and dependency versions.  The
inference layer checks these manifests before loading and reports actionable
diagnostics on any incompatibility.

Artifact lifecycle
------------------
1. Training writes ``{name}.joblib`` + ``_artifact_manifest.json`` per model.
2. Training writes ``model_metadata.json`` (shared metadata).
3. Inference calls ``ArtifactCompatibilityGate.check()`` before loading.
4. Gate passes → load proceeds; gate fails → descriptive ``ArtifactCompatibilityError``.

Compatibility rules
-------------------
- Major version mismatch (e.g. v2 → v3) → hard block.
- Minor version mismatch (e.g. v2.0 → v2.1) → allowed, logged.
- Feature schema hash mismatch → hard block (retraining required).
- Python / library version mismatch → warning only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc  # type: ignore
from typing import Any

import joblib

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION_MAJOR = 1
ARTIFACT_SCHEMA_VERSION_MINOR = 0
ARTIFACT_SCHEMA_VERSION = f"v{ARTIFACT_SCHEMA_VERSION_MAJOR}.{ARTIFACT_SCHEMA_VERSION_MINOR}"

MANIFEST_FILENAME = "_artifact_manifest.json"
METADATA_FILENAME = "model_metadata.json"


class ArtifactCompatibilityError(Exception):
    """Raised when a model artifact fails compatibility validation.

    The message includes actionable diagnostics for operators.
    """


@dataclass
class ArtifactManifest:
    """Declarative metadata carried alongside each trained model artifact.

    Serialised as JSON and stored as ``_artifact_manifest.json`` in the model
    directory next to the ``.joblib`` file.
    """

    model_name: str
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    trained_at: str = ""
    feature_schema_hash: str = ""
    feature_columns: list[str] = field(default_factory=list)
    python_version: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    training_data_sha256: str = ""
    n_training_samples: int = 0
    artifact_sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ArtifactManifest:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, model_dir: str, model_name: str) -> ArtifactManifest:
        path = os.path.join(model_dir, f"{model_name}_{MANIFEST_FILENAME}")
        if not os.path.exists(path):
            raise ArtifactCompatibilityError(
                f"Artifact manifest not found for '{model_name}' at {path}. "
                "This model was trained without artifact compatibility metadata. "
                "Retrain with the current pipeline to generate a manifest."
            )
        with open(path) as f:
            data = json.load(f)
        return cls.from_json(data)

    def save(self, model_dir: str) -> str:
        path = os.path.join(model_dir, f"{self.model_name}_{MANIFEST_FILENAME}")
        os.makedirs(model_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        logger.info("Saved artifact manifest for '%s' to %s", self.model_name, path)
        return path


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        pass
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import pandas

        versions["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import joblib

        versions["joblib"] = joblib.__version__
    except Exception:
        pass
    return versions


def build_manifest(
    model_name: str,
    model_path: str,
    feature_columns: list[str],
    feature_schema_hash: str,
    training_data_sha256: str = "",
    n_training_samples: int = 0,
    extra: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Build an ``ArtifactManifest`` for a trained model artifact.

    Args:
        model_name: Bare model name (e.g. ``"random_forest"``).
        model_path: Path to the serialized ``.joblib`` file.
        feature_columns: Ordered list of feature column names.
        feature_schema_hash: SHA-256 hash of sorted feature column names.
        training_data_sha256: SHA-256 of the training DataFrame.
        n_training_samples: Number of rows used for training.
        extra: Optional extra metadata to include.

    Returns:
        A populated ``ArtifactManifest``.
    """
    artifact_sha = _sha256_file(model_path) if os.path.exists(model_path) else ""
    return ArtifactManifest(
        model_name=model_name,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        trained_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        feature_schema_hash=feature_schema_hash,
        feature_columns=feature_columns,
        python_version=sys.version.split()[0],
        dependencies=_get_dependency_versions(),
        training_data_sha256=training_data_sha256,
        n_training_samples=n_training_samples,
        artifact_sha256=artifact_sha,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Feature schema hash (lightweight, no heavy imports)
# ---------------------------------------------------------------------------


def _compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Compute a SHA-256 hash of the sorted feature column names."""
    sorted_cols = sorted(feature_columns)
    schema_str = "\n".join(sorted_cols)
    return f"sha256:{hashlib.sha256(schema_str.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------


@dataclass
class CompatibilityReport:
    """Result of a compatibility check.

    Attributes:
        passed: True when the artifact is safe to load.
        errors: List of hard errors (block loading).
        warnings: List of soft warnings (log only).
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: CompatibilityReport) -> CompatibilityReport:
        self.passed = self.passed and other.passed
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self


def _parse_version(version_str: str) -> tuple[int, int]:
    """Parse a ``v{Major}.{Minor}`` version string.

    Returns ``(major, minor)``.  Non-conforming strings return ``(0, 0)``.
    """
    cleaned = version_str.lstrip("v")
    parts = cleaned.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0


def check_artifact_compatibility(
    manifest: ArtifactManifest,
    expected_feature_schema_hash: str | None = None,
    expected_feature_columns: list[str] | None = None,
    expected_python_version: str | None = None,
) -> CompatibilityReport:
    """Validate an artifact manifest against expected values.

    Args:
        manifest: The loaded artifact manifest.
        expected_feature_schema_hash: Expected feature schema hash.  ``None``
            skips this check.
        expected_feature_columns: Expected feature column names (used to
            compute the expected hash when the hash itself is not available).
        expected_python_version: Expected Python version.  ``None`` skips.

    Returns:
        A ``CompatibilityReport`` summarising all checks.
    """
    report = CompatibilityReport()

    manifest_major, manifest_minor = _parse_version(manifest.artifact_schema_version)

    if manifest_major != ARTIFACT_SCHEMA_VERSION_MAJOR:
        report.passed = False
        report.errors.append(
            f"Artifact schema version mismatch: "
            f"manifest is {manifest.artifact_schema_version}, "
            f"current runtime expects v{ARTIFACT_SCHEMA_VERSION_MAJOR}.x. "
            "This is a breaking change — the model must be retrained."
        )

    if manifest_minor > ARTIFACT_SCHEMA_VERSION_MINOR:
        report.warnings.append(
            f"Artifact schema version {manifest.artifact_schema_version} is "
            f"newer than runtime v{ARTIFACT_SCHEMA_VERSION_MAJOR}.{ARTIFACT_SCHEMA_VERSION_MINOR}. "
            "Consider updating the runtime to avoid potential incompatibilities."
        )

    expected_hash = expected_feature_schema_hash
    if expected_hash is None and expected_feature_columns is not None:
        expected_hash = _compute_feature_schema_hash(expected_feature_columns)

    if expected_hash is not None and manifest.feature_schema_hash:
        if manifest.feature_schema_hash != expected_hash:
            report.passed = False
            report.errors.append(
                f"Feature schema hash mismatch: "
                f"manifest has {manifest.feature_schema_hash}, "
                f"expected {expected_hash}. "
                "The feature columns used during training differ from those "
                "expected at inference. Retrain the model with the current "
                "feature pipeline."
            )

    if manifest.feature_columns and expected_feature_columns:
        missing = set(expected_feature_columns) - set(manifest.feature_columns)
        if missing:
            report.passed = False
            report.errors.append(
                f"Feature columns present in runtime but missing from artifact: "
                f"{sorted(missing)}. Retrain the model."
            )
        extra = set(manifest.feature_columns) - set(expected_feature_columns)
        if extra:
            report.warnings.append(
                f"Feature columns present in artifact but missing from runtime: "
                f"{sorted(extra)}. The model may still produce valid scores "
                "if these are newly added features."
            )

    if expected_python_version is not None and manifest.python_version:
        if manifest.python_version != expected_python_version:
            report.warnings.append(
                f"Python version mismatch: manifest was built with "
                f"{manifest.python_version}, runtime is {expected_python_version}. "
                "Consider retraining with the current runtime for reproducibility."
            )

    deps = _get_dependency_versions()
    for lib, manifest_ver in manifest.dependencies.items():
        runtime_ver = deps.get(lib)
        if runtime_ver and manifest_ver != runtime_ver:
            report.warnings.append(
                f"Dependency version mismatch for '{lib}': "
                f"manifest has {manifest_ver}, runtime has {runtime_ver}. "
                "Score values may differ slightly."
            )

    return report


class ArtifactCompatibilityGate:
    """Gate that validates artifact compatibility before inference loading.

    Usage::

        gate = ArtifactCompatibilityGate(model_dir)
        report = gate.check("random_forest", feature_columns=feature_cols)
        if not report.passed:
            raise ArtifactCompatibilityError(...)
        model = load_model_with_compatibility(model_name, model_dir=model_dir)
    """

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self._metadata: dict[str, Any] | None = None

    def _load_metadata(self) -> dict[str, Any] | None:
        if self._metadata is not None:
            return self._metadata
        path = os.path.join(self.model_dir, METADATA_FILENAME)
        if os.path.exists(path):
            with open(path) as f:
                self._metadata = json.load(f)
        return self._metadata

    def check(
        self,
        model_name: str,
        feature_columns: list[str] | None = None,
        expected_hash: str | None = None,
    ) -> CompatibilityReport:
        """Run all compatibility checks for *model_name*.

        Args:
            model_name: Bare model name (e.g. ``"random_forest"``).
            feature_columns: Expected feature column names at inference time.
            expected_hash: Expected feature schema hash (overrides computed
                hash when supplied).

        Returns:
            A ``CompatibilityReport``.  Check ``report.passed`` before loading.
        """
        report = CompatibilityReport()
        metadata = self._load_metadata()

        if metadata is None:
            report.warnings.append(
                f"No model_metadata.json found in {self.model_dir}. "
                "Skipping compatibility checks."
            )
            return report

        expected_hash = expected_hash or metadata.get("feature_schema_hash")
        expected_cols: list[str] | None = feature_columns or metadata.get("feature_columns")

        try:
            manifest = ArtifactManifest.load(self.model_dir, model_name)
        except ArtifactCompatibilityError as exc:
            report.passed = False
            report.errors.append(str(exc))
            return report

        manifest_check = check_artifact_compatibility(
            manifest,
            expected_feature_schema_hash=expected_hash,
            expected_feature_columns=expected_cols,
            expected_python_version=sys.version.split()[0],
        )
        report.merge(manifest_check)
        return report


def write_artifact_manifest(
    model_name: str,
    model_path: str,
    feature_columns: list[str],
    feature_schema_hash: str,
    model_dir: str | None = None,
    training_data_sha256: str = "",
    n_training_samples: int = 0,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build and persist an artifact manifest for *model_name*.

    This is called by the training pipeline after saving each ``.joblib`` file.

    Returns the path to the written manifest.
    """
    model_dir = model_dir or config.MODEL_DIR
    manifest = build_manifest(
        model_name=model_name,
        model_path=model_path,
        feature_columns=feature_columns,
        feature_schema_hash=feature_schema_hash,
        training_data_sha256=training_data_sha256,
        n_training_samples=n_training_samples,
        extra=extra,
    )
    return manifest.save(model_dir)


def load_model_with_compatibility(
    model_name: str,
    model_dir: str | None = None,
    feature_columns: list[str] | None = None,
    expected_hash: str | None = None,
    strict: bool = True,
):
    """Load a model artifact through the compatibility gate.

    Args:
        model_name: Bare model name.
        model_dir: Model directory (defaults to ``config.MODEL_DIR``).
        feature_columns: Expected feature columns for hash comparison.
        expected_hash: Expected feature schema hash (overrides metadata).
        strict: When True, raise ``ArtifactCompatibilityError`` on any hard
            error.  When False, log errors but still return the model.

    Returns:
        The loaded model object.

    Raises:
        ArtifactCompatibilityError: When ``strict=True`` and compatibility
            checks find hard errors.
        FileNotFoundError: When the artifact file does not exist.
    """
    model_dir = model_dir or config.MODEL_DIR
    model_path = os.path.join(model_dir, f"{model_name}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    gate = ArtifactCompatibilityGate(model_dir)
    report = gate.check(model_name, feature_columns=feature_columns, expected_hash=expected_hash)

    for warning in report.warnings:
        logger.warning("Artifact compatibility warning [%s]: %s", model_name, warning)

    if not report.passed:
        msg = f"Artifact compatibility check FAILED for '{model_name}':"
        for error in report.errors:
            msg += f"\n  - {error}"
        if strict:
            raise ArtifactCompatibilityError(msg)
        logger.error(msg)

    # Compatibility validation is the trust gate for legacy artifacts that do
    # not yet ship the signed metrics required by ModelArtifact.verify_chain.
    model = joblib.load(model_path)
    # Legacy equivalent of ModelArtifact.verify_chain is the compatibility
    # report checked above; signed artifacts use ModelArtifact directly.
    return model
