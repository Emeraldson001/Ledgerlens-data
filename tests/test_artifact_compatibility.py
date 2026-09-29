"""Tests for the artifact compatibility gate and manifest system."""

"""Tests for the artifact compatibility gate and manifest system."""

import json
import os
import sys

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

# Import directly without going through detection.__init__ to avoid pulling in
# every transitive dependency of the full package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.artifact_compatibility import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_SCHEMA_VERSION_MAJOR,
    ArtifactCompatibilityError,
    ArtifactCompatibilityGate,
    ArtifactManifest,
    _parse_version,
    build_manifest,
    check_artifact_compatibility,
    load_model_with_compatibility,
    write_artifact_manifest,
)


class TestVersionParsing:
    def test_parse_full_version(self):
        assert _parse_version("v1.0") == (1, 0)

    def test_parse_major_only(self):
        assert _parse_version("v2") == (2, 0)

    def test_parse_no_v_prefix(self):
        assert _parse_version("1.5") == (1, 5)

    def test_parse_invalid(self):
        assert _parse_version("invalid") == (0, 0)

    def test_parse_empty(self):
        assert _parse_version("") == (0, 0)


class TestArtifactManifest:
    def test_build_manifest(self):
        manifest = ArtifactManifest(
            model_name="random_forest",
            feature_schema_hash="sha256:abc123",
            feature_columns=["feat_a", "feat_b"],
        )
        assert manifest.model_name == "random_forest"
        assert manifest.artifact_schema_version == ARTIFACT_SCHEMA_VERSION

    def test_round_trip_json(self, tmp_path):
        original = ArtifactManifest(
            model_name="xgboost",
            artifact_schema_version="v1.0",
            trained_at="2026-01-01T00:00:00Z",
            feature_schema_hash="sha256:def456",
            feature_columns=["col1", "col2"],
            python_version="3.11.0",
            dependencies={"scikit-learn": "1.4.0"},
            n_training_samples=1000,
        )
        data = original.to_json()
        restored = ArtifactManifest.from_json(data)
        assert restored.model_name == original.model_name
        assert restored.feature_schema_hash == original.feature_schema_hash
        assert restored.n_training_samples == original.n_training_samples

    def test_save_and_load(self, tmp_path):
        manifest = ArtifactManifest(
            model_name="lightgbm",
            feature_schema_hash="sha256:789ghi",
            feature_columns=["col_a"],
        )
        manifest.save(str(tmp_path))

        loaded = ArtifactManifest.load(str(tmp_path), "lightgbm")
        assert loaded.model_name == "lightgbm"
        assert loaded.feature_schema_hash == "sha256:789ghi"

    def test_load_missing_manifest(self, tmp_path):
        with pytest.raises(ArtifactCompatibilityError):
            ArtifactManifest.load(str(tmp_path), "nonexistent")


class TestCheckArtifactCompatibility:
    def test_perfect_match(self):
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:match",
            feature_columns=["a", "b"],
            python_version=sys.version.split()[0],
        )
        report = check_artifact_compatibility(
            manifest,
            expected_feature_schema_hash="sha256:match",
            expected_feature_columns=["a", "b"],
            expected_python_version=sys.version.split()[0],
        )
        assert report.passed
        assert len(report.errors) == 0

    def test_major_version_mismatch(self):
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version="v99.0",
            feature_schema_hash="sha256:x",
        )
        report = check_artifact_compatibility(manifest)
        assert not report.passed
        assert any("schema version mismatch" in e.lower() for e in report.errors)

    def test_feature_hash_mismatch(self):
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:old_hash",
            feature_columns=["a", "b"],
        )
        report = check_artifact_compatibility(
            manifest,
            expected_feature_schema_hash="sha256:new_hash",
        )
        assert not report.passed
        assert any("hash mismatch" in e.lower() for e in report.errors)

    def test_missing_feature_columns(self):
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:x",
            feature_columns=["a"],
        )
        report = check_artifact_compatibility(
            manifest,
            expected_feature_columns=["a", "b"],
        )
        assert not report.passed
        assert any("missing" in e.lower() for e in report.errors)

    def test_extra_feature_columns_warning(self):
        common_hash = "sha256:abc123def456"
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash=common_hash,
            feature_columns=["a", "b", "c"],
        )
        report = check_artifact_compatibility(
            manifest,
            expected_feature_schema_hash=common_hash,
            expected_feature_columns=["a", "b"],
        )
        assert report.passed
        assert any("missing from runtime" in w.lower() for w in report.warnings)

    def test_minor_version_ahead(self):
        manifest = ArtifactManifest(
            model_name="rf",
            artifact_schema_version=f"v{ARTIFACT_SCHEMA_VERSION_MAJOR}.99",
            feature_schema_hash="sha256:x",
        )
        report = check_artifact_compatibility(manifest)
        assert report.passed
        assert any("newer" in w.lower() for w in report.warnings)


class TestArtifactCompatibilityGate:
    def test_gate_with_valid_manifest(self, tmp_path):
        model_dir = str(tmp_path)

        meta = {
            "feature_schema_hash": "sha256:abc",
            "feature_columns": ["x", "y"],
            "trained_at": "2026-01-01T00:00:00Z",
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f)

        manifest = ArtifactManifest(
            model_name="random_forest",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:abc",
            feature_columns=["x", "y"],
        )
        manifest.save(model_dir)

        gate = ArtifactCompatibilityGate(model_dir)
        report = gate.check("random_forest")
        assert report.passed, f"Gate failed: {report.errors}"

    def test_gate_without_metadata(self, tmp_path):
        gate = ArtifactCompatibilityGate(str(tmp_path))
        report = gate.check("random_forest")
        assert report.passed

    def test_gate_without_manifest(self, tmp_path):
        model_dir = str(tmp_path)
        meta = {
            "feature_schema_hash": "sha256:abc",
            "feature_columns": ["x", "y"],
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f)

        gate = ArtifactCompatibilityGate(model_dir)
        report = gate.check("random_forest")
        assert not report.passed
        assert any("manifest not found" in e.lower() for e in report.errors)


class TestBuildManifest:
    def test_build_manifest_with_real_file(self, tmp_path):
        model_path = os.path.join(str(tmp_path), "test.joblib")
        joblib.dump(RandomForestClassifier(), model_path)

        manifest = build_manifest(
            model_name="test_model",
            model_path=model_path,
            feature_columns=["a", "b"],
            feature_schema_hash="sha256:test",
            n_training_samples=500,
        )
        assert manifest.model_name == "test_model"
        assert manifest.artifact_sha256 != ""
        assert manifest.n_training_samples == 500
        assert "scikit-learn" in manifest.dependencies

    def test_manifest_persistence(self, tmp_path):
        model_path = os.path.join(str(tmp_path), "rf.joblib")
        joblib.dump(RandomForestClassifier(), model_path)

        written = write_artifact_manifest(
            model_name="rf",
            model_path=model_path,
            feature_columns=["a", "b"],
            feature_schema_hash="sha256:xyz",
            model_dir=str(tmp_path),
            n_training_samples=100,
        )
        assert os.path.exists(written)

        loaded = ArtifactManifest.load(str(tmp_path), "rf")
        assert loaded.artifact_sha256 != ""


class TestLoadModelWithCompatibility:
    def test_load_with_compatibility_success(self, tmp_path):
        model_dir = str(tmp_path)
        model = RandomForestClassifier()
        model.fit([[0, 0], [1, 1]], [0, 1])

        meta = {
            "feature_schema_hash": "sha256:abc",
            "feature_columns": ["x", "y"],
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f)

        model_path = os.path.join(model_dir, "test_model.joblib")
        joblib.dump(model, model_path)

        manifest = ArtifactManifest(
            model_name="test_model",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:abc",
            feature_columns=["x", "y"],
        )
        manifest.save(model_dir)

        loaded = load_model_with_compatibility(
            "test_model",
            model_dir=model_dir,
            strict=True,
        )
        assert loaded is not None

    def test_load_with_compatibility_blocks_mismatch(self, tmp_path):
        model_dir = str(tmp_path)
        model = RandomForestClassifier()
        model.fit([[0, 0], [1, 1]], [0, 1])

        meta = {
            "feature_schema_hash": "sha256:old_hash",
            "feature_columns": ["x", "y"],
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f)

        model_path = os.path.join(model_dir, "bad.joblib")
        joblib.dump(model, model_path)

        manifest = ArtifactManifest(
            model_name="bad",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:old_hash",
            feature_columns=["x", "y"],
        )
        manifest.save(model_dir)

        with pytest.raises(ArtifactCompatibilityError):
            load_model_with_compatibility(
                "bad",
                model_dir=model_dir,
                expected_hash="sha256:different_hash",
                strict=True,
            )

    def test_load_nonexistent_model(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_with_compatibility(
                "nonexistent",
                model_dir=str(tmp_path),
                strict=True,
            )

    def test_load_non_strict_fallback(self, tmp_path):
        model_dir = str(tmp_path)
        model = RandomForestClassifier()
        model.fit([[0, 0], [1, 1]], [0, 1])

        meta = {
            "feature_schema_hash": "sha256:hash_a",
            "feature_columns": ["x", "y"],
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(meta, f)

        model_path = os.path.join(model_dir, "ns.joblib")
        joblib.dump(model, model_path)

        manifest = ArtifactManifest(
            model_name="ns",
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash="sha256:hash_a",
            feature_columns=["x", "y"],
        )
        manifest.save(model_dir)

        loaded = load_model_with_compatibility(
            "ns",
            model_dir=model_dir,
            expected_hash="sha256:wrong",
            strict=False,
        )
        assert loaded is not None
