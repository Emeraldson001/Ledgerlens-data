"""Tests for detection/model_training.py — provenance, poisoning detection."""

import json
import os

import pandas as pd
import pytest

from detection.model_compatibility import compute_feature_contract_hash
from detection.model_training import (
    MODEL_REGISTRY,
    compute_feature_schema_hash,
    detect_label_poisoning,
    save_models,
    save_training_artifacts,
    sha256_dataframe,
    split_features_labels,
    train_models,
)
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_output():
    df = generate_synthetic_dataset(n_wallets=60, seed=1)
    return train_models(df, test_size=0.3, random_state=1), df


def test_split_features_labels_excludes_wallet_and_label():
    df = generate_synthetic_dataset(n_wallets=10, seed=1)
    X, y = split_features_labels(df)
    assert "wallet" not in X.columns
    assert "label" not in X.columns
    assert len(X) == len(y)


def test_train_models_returns_metrics_for_each_model(trained_output):
    output, _ = trained_output
    results = output["results"]
    assert set(results) == set(MODEL_REGISTRY)
    for result in results.values():
        base_keys = {"auc_roc", "pr_auc", "f1"}
        conformal_keys = {
            "conformal_empirical_coverage",
            "conformal_q_hat",
            "calibration_split_size",
        }
        assert base_keys.issubset(set(result["metrics"]))
        assert conformal_keys.issubset(set(result["metrics"]))
        assert 0.0 <= result["metrics"]["auc_roc"] <= 1.0
        assert 0.0 <= result["metrics"]["conformal_empirical_coverage"] <= 1.0


def test_train_models_returns_held_out_split(trained_output):
    output, _ = trained_output
    assert len(output["X_test"]) == output["n_test"]
    assert len(output["y_test"]) == output["n_test"]
    assert "label" not in output["X_test"].columns


def test_save_models_and_training_artifacts(tmp_path, trained_output):
    output, _ = trained_output
    results = output["results"]
    model_dir = str(tmp_path)

    save_models(results, model_dir)
    for name in MODEL_REGISTRY:
        assert os.path.exists(os.path.join(model_dir, f"{name}.joblib"))

    save_training_artifacts(output, "data/synthetic.parquet", model_dir)
    assert os.path.exists(os.path.join(model_dir, "metrics.json"))
    assert os.path.exists(os.path.join(model_dir, "model_metadata.json"))

    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)
    assert set(MODEL_REGISTRY).issubset(set(metrics))

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)
    expected_hash = compute_feature_schema_hash(output["feature_columns"])
    assert meta["feature_schema_hash"] == expected_hash
    assert meta["feature_contract_version"] == 1
    assert meta["feature_dtypes"] == output["feature_dtypes"]
    assert meta["feature_contract_hash"] == compute_feature_contract_hash(
        output["feature_columns"],
        output["feature_dtypes"],
    )


# ---------------------------------------------------------------------------
# Provenance: SHA-256 of training data
# ---------------------------------------------------------------------------


def test_training_data_sha256_changes_when_row_added():
    df = generate_synthetic_dataset(n_wallets=20, seed=5)
    sha1 = sha256_dataframe(df)

    extra = df.iloc[[0]].copy()
    extra["wallet"] = "GNEW"
    df2 = pd.concat([df, extra], ignore_index=True)
    sha2 = sha256_dataframe(df2)

    assert sha1 != sha2


# ---------------------------------------------------------------------------
# Label poisoning detection
# ---------------------------------------------------------------------------


def test_detect_label_poisoning_returns_true_when_ratio_shifts(tmp_path):
    baseline_path = str(tmp_path / "baseline.json")
    # Write a baseline with ~10% wash-trade ratio
    baseline_ratio = 0.10
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": baseline_ratio}, f)

    # Current distribution: ~30% wash trades → shift = 0.20 > 0.15 threshold
    distribution = {0: 70, 1: 30}
    assert detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=0.15)


def test_detect_label_poisoning_returns_false_when_ratio_ok(tmp_path):
    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.20}, f)

    distribution = {0: 82, 1: 18}  # 18% — shift < 15%
    assert not detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=0.15)


@pytest.mark.parametrize(
    "bad_threshold",
    [
        15,  # Issue #740's exact motivating mix-up: 15 (percent) instead of 0.15
        1.5,
        0,
        -0.1,
    ],
)
def test_detect_label_poisoning_rejects_out_of_range_threshold(tmp_path, bad_threshold):
    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.10}, f)

    distribution = {0: 70, 1: 30}
    with pytest.raises(ValueError, match="POISON_LABEL_RATIO_THRESHOLD must be a fraction in \\(0, 1\\]"):
        detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=bad_threshold)


def test_detect_label_poisoning_accepts_boundary_threshold_of_one(tmp_path):
    """1.0 is the inclusive upper bound — a 100% ratio shift is a valid (if
    extreme) configuration, not a mix-up like 15 vs 0.15."""
    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.0}, f)

    distribution = {0: 0, 1: 100}
    assert not detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=1.0)


def test_detect_label_poisoning_rejects_bad_config_default(monkeypatch):
    """The same validation applies when the bad value comes from config
    (POISON_LABEL_RATIO_THRESHOLD) rather than an explicit override."""
    import detection.model_training as mt

    monkeypatch.setattr(mt.config, "POISON_LABEL_RATIO_THRESHOLD", 15)
    with pytest.raises(ValueError, match="POISON_LABEL_RATIO_THRESHOLD must be a fraction in \\(0, 1\\]"):
        detect_label_poisoning({0: 70, 1: 30})


def test_detect_label_poisoning_creates_baseline_when_missing(tmp_path):
    baseline_path = str(tmp_path / "new_baseline.json")
    assert not os.path.exists(baseline_path)

    distribution = {0: 90, 1: 10}
    result = detect_label_poisoning(distribution, baseline_path=baseline_path)
    assert result is False
    assert os.path.exists(baseline_path)


def test_detect_label_poisoning_aborts_training(tmp_path, monkeypatch):
    """When poisoning is detected, no .pkl / .joblib files should be written."""
    import detection.model_training as mt

    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.05}, f)

    # Patch baseline path and threshold so poisoning is always detected
    monkeypatch.setattr(mt, "LABEL_DISTRIBUTION_BASELINE_PATH", baseline_path)
    monkeypatch.setattr(mt.config, "POISON_LABEL_RATIO_THRESHOLD", 0.05)
    monkeypatch.setattr(mt.config, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(mt.config, "MODEL_SIGNING_PRIVATE_KEY_PATH", "")

    # Build a minimal dataset with a high wash-trade ratio (60%)
    df = generate_synthetic_dataset(n_wallets=40, seed=7)
    # Override labels so ratio = 60%
    df["label"] = [1 if i % 5 != 0 else 0 for i in range(len(df))]

    # Simulate main() with a temp data file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        df.to_parquet(tmp_file.name)
        tmp_file_path = tmp_file.name

    monkeypatch.setattr(
        "sys.argv",
        ["model_training", "--data-path", tmp_file_path, "--model-dir", str(tmp_path / "models")],
    )

    mt.main()

    # No model artifacts should have been written
    model_dir = str(tmp_path / "models")
    for name in MODEL_REGISTRY:
        assert not os.path.exists(os.path.join(model_dir, f"{name}.joblib"))

    os.unlink(tmp_file_path)


# ---------------------------------------------------------------------------
# Grand 2 (issue #671, Task C): `python -m detection.model_training` must
# never overwrite an already-promoted production model directory.
# ---------------------------------------------------------------------------


def test_main_refuses_to_overwrite_already_promoted_model_dir(tmp_path, monkeypatch):
    import tempfile

    import detection.model_training as mt

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    original_metrics = {"random_forest": {"artifact_sha256": "a" * 64}}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(original_metrics, f)
    with open(os.path.join(model_dir, "random_forest.joblib"), "wb") as f:
        f.write(b"already-promoted-production-bytes")

    monkeypatch.setattr(mt.config, "MODEL_DIR", model_dir)

    df = generate_synthetic_dataset(n_wallets=30, seed=11)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        df.to_parquet(tmp_file.name)
        tmp_file_path = tmp_file.name

    monkeypatch.setattr(
        "sys.argv", ["model_training", "--data-path", tmp_file_path, "--model-dir", model_dir]
    )

    try:
        with pytest.raises(SystemExit) as excinfo:
            mt.main()
        assert excinfo.value.code == 1
    finally:
        os.unlink(tmp_file_path)

    # The pre-existing "promoted production" artifact must be untouched.
    with open(os.path.join(model_dir, "metrics.json")) as f:
        assert json.load(f) == original_metrics
    with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
        assert f.read() == b"already-promoted-production-bytes"


def test_save_models_and_save_training_artifacts_call_the_production_write_guard(
    tmp_path, trained_output, monkeypatch
):
    """Regression guard: if a future refactor removes the
    guard_production_write() call from either function, this must fail
    loudly rather than silently reopening the ungated write path."""
    output, _ = trained_output
    calls = []
    monkeypatch.setattr(
        "detection.production_write_guard.guard_production_write", lambda d: calls.append(d)
    )

    model_dir = str(tmp_path)
    save_models(output["results"], model_dir)
    save_training_artifacts(output, "data/synthetic.parquet", model_dir)

    assert calls == [model_dir, model_dir]
