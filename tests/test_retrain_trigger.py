"""Tests for scripts/retrain_if_drifted.py.

Validates promotion gate logic, archive creation, exit codes, and the
end-to-end drift → retrain → promote flow via mocked dependencies.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sklearn.dummy import DummyClassifier

from config import config
from scripts.retrain_if_drifted import (
    archive_current_models,
    should_promote,
)


def _configure_governance_for_immediate_promotion(tmp_path, monkeypatch):
    """Wire up MODEL_PROMOTION_*/signing config so --no-shadow's gated
    promotion (detection.model_governance.promote_candidate) succeeds for
    the automated retrain-pipeline actor, using a throwaway keypair."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_path = str(tmp_path / "signing_key.pem")
    with open(private_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    public_path = str(tmp_path / "public_key.pem")
    with open(public_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    monkeypatch.setattr(config, "MODEL_SIGNING_PRIVATE_KEY_PATH", private_path)
    monkeypatch.setattr(config, "TRUSTED_SIGNING_PUBLIC_KEY_PATH", public_path)
    monkeypatch.setattr(config, "MODEL_PROMOTION_SECRET", "test-secret")
    monkeypatch.setattr(config, "MODEL_PROMOTION_AUTHORIZED_ACTORS", "retrain-pipeline")
    monkeypatch.setattr(config, "MODEL_PROMOTION_SYSTEM_ACTOR", "retrain-pipeline")
    monkeypatch.setattr(config, "RISK_SCORE_DB_URL", f"sqlite:///{tmp_path}/retrain_gov.db")


@pytest.fixture
def temp_model_dir(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def sample_old_metrics():
    return {
        "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
        "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
        "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
    }


@pytest.fixture
def sample_model_metadata():
    return {
        "trained_at": "2026-06-01T00:00:00Z",
        "feature_columns": ["feat_a", "feat_b"],
        "feature_schema_hash": "sha256:abc123",
        "model_names": ["random_forest", "xgboost", "lightgbm"],
        "feature_distributions": {
            "feat_a": {
                "bin_edges": [0.0, 1.0, 2.0, 3.0],
                "expected_proportions": [0.25, 0.25, 0.25, 0.25],
            },
            "feat_b": {
                "bin_edges": [0.0, 1.0, 2.0, 3.0],
                "expected_proportions": [0.25, 0.25, 0.25, 0.25],
            },
        },
    }


class TestArchiveCurrentModels:
    def test_archive_created_with_correct_permissions(self, temp_model_dir):
        model_files = ["random_forest.joblib", "model_metadata.json", "metrics.json"]
        for fname in model_files:
            with open(os.path.join(temp_model_dir, fname), "w") as f:
                f.write("test")

        archive_path = archive_current_models(temp_model_dir)
        assert os.path.exists(archive_path)
        assert oct(os.stat(archive_path).st_mode & 0o777) == "0o750"

        for fname in model_files:
            assert os.path.exists(os.path.join(archive_path, fname))

    def test_archive_created_before_promotion(self, temp_model_dir):
        model_files = ["random_forest.joblib", "model_metadata.json", "metrics.json"]
        for fname in model_files:
            with open(os.path.join(temp_model_dir, fname), "w") as f:
                f.write("test")

        archive_path = archive_current_models(temp_model_dir)
        assert os.path.isdir(archive_path)

        for fname in model_files:
            assert os.path.exists(os.path.join(temp_model_dir, fname))
            assert os.path.exists(os.path.join(archive_path, fname))


class TestShouldPromote:
    def test_promotion_gate_allows_identical_metrics(self, sample_old_metrics):
        promote, reason = should_promote(sample_old_metrics, sample_old_metrics)
        assert promote is True

    def test_promotion_gate_allows_improvement(self, sample_old_metrics):
        new_metrics = {
            name: {k: v * 1.05 if k in ("auc_roc", "f1") else v for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is True

    def test_promotion_gate_allows_minor_regression(self, sample_old_metrics):
        new_metrics = {
            name: {k: (v - 0.005 if k in ("auc_roc", "f1") else v) for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is True

    def test_promotion_gate_blocks_regression(self, sample_old_metrics):
        new_metrics = {
            name: {k: (v - 0.02 if k in ("auc_roc", "f1") else v) for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "AUC-ROC" in reason or "F1" in reason

    def test_promotion_gate_blocks_single_model_regression(self, sample_old_metrics):
        new_metrics = dict(sample_old_metrics)
        new_metrics["xgboost"] = {
            "auc_roc": 0.80,
            "pr_auc": 0.88,
            "f1": 0.84,
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "xgboost" in reason

    def test_promotion_blocks_missing_model(self, sample_old_metrics):
        new_metrics = {
            "random_forest": sample_old_metrics["random_forest"],
            "xgboost": sample_old_metrics["xgboost"],
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "lightgbm" in reason


class TestRetrainScriptExitCodes:
    def _run_retrain_main(self, argv: list[str]) -> int:
        from scripts.retrain_if_drifted import main

        return main(argv)

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_0_no_drift(self, mock_load_metadata, mock_get_feature_data, temp_model_dir):
        """No drift → exit code 0."""
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)
        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 500),
            }
        )

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 0

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    @patch("scripts.retrain_if_drifted.load_training_data")
    @patch("scripts.retrain_if_drifted.train_models")
    @patch("scripts.retrain_if_drifted.load_metrics")
    def test_exit_code_2_retrained_and_promoted(
        self,
        mock_load_metrics,
        mock_train_models,
        mock_load_training_data,
        mock_load_metadata,
        mock_get_feature_data,
        temp_model_dir,
        tmp_path,
        monkeypatch,
    ):
        """Drift detected, retrained, promoted immediately via --no-shadow → exit code 2."""
        _configure_governance_for_immediate_promotion(tmp_path, monkeypatch)
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)

        # Shifted distribution → triggers drift
        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(10, 20, 500),
            }
        )

        mock_load_training_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 100),
                "label": [1, 0] * 50,
            }
        )

        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

        mock_results = {
            name: {"model": dummy, "metrics": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88}}
            for name in ["random_forest", "xgboost", "lightgbm"]
        }
        mock_train_models.return_value = {
            "results": mock_results,
            "feature_columns": ["feat_a"],
            "feature_distributions": {},
            "n_train": 80,
            "n_test": 20,
        }

        mock_load_metrics.side_effect = [
            {
                "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
                "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
                "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
            },
            {
                "random_forest": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
                "xgboost": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
                "lightgbm": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
            },
        ]

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
                "--retrain-data-path",
                "/fake/path.parquet",
                "--no-shadow",
            ]
        )
        assert code == 2

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    @patch("scripts.retrain_if_drifted.load_training_data")
    @patch("scripts.retrain_if_drifted.train_models")
    @patch("scripts.retrain_if_drifted.load_metrics")
    def test_exit_code_3_retrained_not_promoted(
        self,
        mock_load_metrics,
        mock_train_models,
        mock_load_training_data,
        mock_load_metadata,
        mock_get_feature_data,
        temp_model_dir,
        tmp_path,
        monkeypatch,
    ):
        """Drift detected, retrained, NOT promoted (regression) via --no-shadow → exit code 3."""
        _configure_governance_for_immediate_promotion(tmp_path, monkeypatch)
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)

        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(10, 20, 500),
            }
        )

        mock_load_training_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 100),
                "label": [1, 0] * 50,
            }
        )

        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

        mock_results = {
            name: {"model": dummy, "metrics": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78}}
            for name in ["random_forest", "xgboost", "lightgbm"]
        }
        mock_train_models.return_value = {
            "results": mock_results,
            "feature_columns": ["feat_a"],
            "feature_distributions": {},
            "n_train": 80,
            "n_test": 20,
        }

        mock_load_metrics.side_effect = [
            {
                "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
                "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
                "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
            },
            {
                "random_forest": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
                "xgboost": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
                "lightgbm": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
            },
        ]

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
                "--retrain-data-path",
                "/fake/path.parquet",
                "--no-shadow",
            ]
        )
        assert code == 3

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_1_missing_metadata(
        self, mock_load_metadata, mock_get_feature_data, temp_model_dir
    ):
        """Missing model_metadata.json → exit code 1."""
        mock_load_metadata.return_value = None

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 1

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_1_missing_distributions(
        self, mock_load_metadata, mock_get_feature_data, temp_model_dir
    ):
        """model_metadata.json without feature_distributions → exit code 1."""
        mock_load_metadata.return_value = {"trained_at": "2026-01-01"}

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 1


class TestRetrainEndToEnd:
    def test_archive_created_during_retrain(
        self, tmp_path, sample_old_metrics, sample_model_metadata
    ):
        """Archive directory is populated during a triggered retrain."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(sample_model_metadata, f)
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(sample_old_metrics, f)
        for name in ["random_forest.joblib", "xgboost.joblib", "lightgbm.joblib"]:
            with open(os.path.join(model_dir, name), "w") as f:
                f.write("test")

        archive_path = archive_current_models(model_dir)
        assert os.path.isdir(archive_path)
        for name in ["random_forest.joblib", "model_metadata.json", "metrics.json"]:
            assert os.path.exists(os.path.join(archive_path, name))


# ---------------------------------------------------------------------------
# Grand 2 (issue #671, Task D): --check-shadow / --no-shadow must not raise
# AttributeError, and the shadow-deploy branch must be reachable and
# exercised end-to-end (default path with neither flag starts a shadow
# deployment rather than promoting immediately or silently doing nothing).
# ---------------------------------------------------------------------------


def _dummy_training_output(feature_columns, auc=0.95):
    dummy = DummyClassifier(strategy="constant", constant=1)
    dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    results = {
        name: {"model": dummy, "metrics": {"auc_roc": auc, "pr_auc": auc, "f1": auc}}
        for name in ["random_forest", "xgboost", "lightgbm"]
    }
    return {
        "results": results,
        "feature_columns": feature_columns,
        "feature_distributions": {},
        "n_train": 80,
        "n_test": 20,
    }


class TestShadowDeployment:
    def test_check_shadow_flags_do_not_raise_attribute_error(self, temp_model_dir):
        """The historical bug: args.check_shadow/args.no_shadow were read in
        main() but never defined by parse_args(), so *every* invocation
        crashed with AttributeError regardless of which flags were passed."""
        from scripts.retrain_if_drifted import main

        assert main(["--model-dir", temp_model_dir, "--check-shadow"]) == 0

    def test_default_invocation_starts_shadow_deployment_not_immediate_promotion(
        self, temp_model_dir, sample_old_metrics, tmp_path, monkeypatch
    ):
        """Neither --no-shadow nor --check-shadow: a freshly retrained
        candidate must go to shadow (exit 4), not be promoted immediately
        (exit 2) — the dead-code bug being fixed here made immediate
        promotion the *only* reachable outcome regardless of flags."""
        monkeypatch.setattr(config, "RISK_SCORE_DB_URL", f"sqlite:///{tmp_path}/gov.db")

        model_metadata = {
            "feature_distributions": {
                "feat_a": {"bin_edges": [0.0, 0.5, 1.0], "expected_proportions": [0.5, 0.5]},
                "feat_b": {"bin_edges": [0.0, 0.5, 1.0], "expected_proportions": [0.5, 0.5]},
            }
        }
        with open(os.path.join(temp_model_dir, "model_metadata.json"), "w") as f:
            json.dump(model_metadata, f)
        with open(os.path.join(temp_model_dir, "metrics.json"), "w") as f:
            json.dump(sample_old_metrics, f)

        rng = np.random.default_rng(1)
        training_output = _dummy_training_output(["feat_a", "feat_b"])

        with (
            patch("scripts.retrain_if_drifted.get_feature_data") as mock_feat,
            patch("scripts.retrain_if_drifted.train_models", return_value=training_output),
            patch(
                "scripts.retrain_if_drifted.load_training_data",
                return_value=pd.DataFrame(
                    {
                        "feat_a": rng.uniform(0, 1, 50),
                        "feat_b": rng.uniform(0, 1, 50),
                        "label": [0, 1] * 25,
                    }
                ),
            ),
        ):
            mock_feat.return_value = pd.DataFrame(
                {"feat_a": rng.uniform(10, 20, 500), "feat_b": rng.uniform(10, 20, 500)}
            )

            from scripts.retrain_if_drifted import main

            code = main(
                [
                    "--model-dir",
                    temp_model_dir,
                    "--retrain-data-path",
                    "/fake.parquet",
                    "--lookback-days",
                    "7",
                ]
            )

        assert code == 4
        state_path = os.path.join(temp_model_dir, "shadow_deployment_state.json")
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)

        from detection.persistence import ModelVersionRecord, get_engine, get_session_factory

        sf = get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
        with sf() as session:
            row = session.query(ModelVersionRecord).filter_by(version_id=state["version_id"]).one()
            assert row.status == "shadow"

        # Production model files must be untouched — only a candidate dir
        # was written alongside them.
        assert os.path.exists(os.path.join(temp_model_dir, "metrics.json"))
        with open(os.path.join(temp_model_dir, "metrics.json")) as f:
            assert json.load(f) == sample_old_metrics

    def _write_bundle(self, path, feature_columns, auc):
        os.makedirs(path, exist_ok=True)
        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        import joblib

        metrics = {}
        for name in ["random_forest", "xgboost", "lightgbm"]:
            joblib.dump(dummy, os.path.join(path, f"{name}.joblib"))
            metrics[name] = {"auc_roc": auc, "pr_auc": auc, "f1": auc}
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    def _shadow_setup(self, tmp_path, monkeypatch, drift_rate):
        _configure_governance_for_immediate_promotion(tmp_path, monkeypatch)
        model_dir = str(tmp_path / "models")
        candidate_dir = str(tmp_path / "models_new")
        self._write_bundle(model_dir, ["feat_a"], auc=0.90)
        self._write_bundle(candidate_dir, ["feat_a"], auc=0.95)

        from datetime import UTC, datetime, timedelta

        shadow_start = (
            datetime.now(UTC) - timedelta(hours=config.SHADOW_PERIOD_HOURS + 1)
        ).isoformat()
        shadow_state = {
            "version_id": "shadow-version-under-test",
            "candidate_dir": candidate_dir,
            "shadow_start": shadow_start,
            "drift_rate": drift_rate,
            "drift_events": 1 if drift_rate else 0,
            "total_shadow_requests": 100,
        }
        with open(os.path.join(model_dir, "shadow_deployment_state.json"), "w") as f:
            json.dump(shadow_state, f)

        from detection import model_governance
        from detection.persistence import get_engine, get_session_factory

        sf = get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
        model_governance.record_shadow_start(
            "shadow-version-under-test", candidate_dir, metrics=None, session_factory=sf
        )
        return model_dir, candidate_dir, sf

    def test_check_shadow_promotes_when_drift_low(self, tmp_path, monkeypatch):
        model_dir, candidate_dir, sf = self._shadow_setup(tmp_path, monkeypatch, drift_rate=0.01)

        from scripts.retrain_if_drifted import main

        code = main(["--model-dir", model_dir, "--check-shadow"])
        assert code == 5

        # The candidate directory is removed once its contents have been
        # published to production by the gated promotion path.
        assert not os.path.exists(candidate_dir)
        assert not os.path.exists(os.path.join(model_dir, "shadow_deployment_state.json"))

        from detection.persistence import ModelVersionRecord

        with sf() as session:
            shadow_row = (
                session.query(ModelVersionRecord)
                .filter_by(version_id="shadow-version-under-test")
                .one()
            )
            assert shadow_row.status == "archived"
            production_rows = session.query(ModelVersionRecord).filter_by(status="production").all()
            assert len(production_rows) == 1

    def test_check_shadow_rolls_back_when_drift_high(self, tmp_path, monkeypatch):
        model_dir, candidate_dir, sf = self._shadow_setup(tmp_path, monkeypatch, drift_rate=0.5)

        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            original_prod_bytes = f.read()

        from scripts.retrain_if_drifted import main

        code = main(["--model-dir", model_dir, "--check-shadow"])
        assert code == 6

        assert not os.path.exists(candidate_dir)
        assert not os.path.exists(os.path.join(model_dir, "shadow_deployment_state.json"))
        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == original_prod_bytes, "production must be untouched on rollback"

        from detection.persistence import ModelVersionRecord

        with sf() as session:
            shadow_row = (
                session.query(ModelVersionRecord)
                .filter_by(version_id="shadow-version-under-test")
                .one()
            )
            assert shadow_row.status == "rolled_back"
            assert shadow_row.shadow_drift_rate == 0.5
