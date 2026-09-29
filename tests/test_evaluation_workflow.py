"""Tests for evaluation.quality_drift_workflow (combined quality + drift gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.quality_drift_workflow import (
    DriftGate,
    EvaluationResult,
    GateFailure,
    QualityGate,
    _build_reference_distribution,
    _numeric_feature_columns,
    run_evaluation_workflow,
)


def _make_dataset(path, n=200, feature_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    feature_a = rng.normal(loc=0.0 + feature_shift, scale=1.0, size=n)
    feature_b = rng.uniform(0, 1, size=n)
    label = (feature_a > 0.5).astype(int)
    df = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "label": label,
            "asset_pair": ["XLM:native/USDC:GISSUER"] * n,
        }
    )
    df.to_parquet(path)
    return df


def _perfect_predict_fn(row):
    return 1.0 if row["feature_a"] > 0.5 else 0.0


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------


def test_quality_gate_rejects_unknown_metric():
    with pytest.raises(ValueError):
        QualityGate(metric="not_a_real_metric", min_value=0.5)


def test_quality_gate_requires_a_bound():
    with pytest.raises(ValueError):
        QualityGate(metric="roc_auc")


def test_quality_gate_passes_when_within_bound():
    gate = QualityGate(metric="roc_auc", min_value=0.5)
    assert gate.evaluate({"roc_auc": 0.9}) is None


def test_quality_gate_fails_below_min():
    gate = QualityGate(metric="roc_auc", min_value=0.9)
    failure = gate.evaluate({"roc_auc": 0.5})
    assert isinstance(failure, GateFailure)
    assert "below minimum" in failure.message


def test_quality_gate_fails_above_max():
    gate = QualityGate(metric="recall", max_value=0.2)
    failure = gate.evaluate({"recall": 0.9})
    assert failure is not None
    assert "above maximum" in failure.message


def test_quality_gate_fails_on_missing_metric():
    gate = QualityGate(metric="f1", min_value=0.5)
    failure = gate.evaluate({})
    assert failure is not None
    assert failure.actual == "missing"


# ---------------------------------------------------------------------------
# DriftGate
# ---------------------------------------------------------------------------


def test_drift_gate_flags_features_above_threshold():
    gate = DriftGate(max_psi=0.25)
    drift_report = {
        "features": [
            {"feature": "feature_a", "psi": 0.5, "drift_flag": True},
            {"feature": "feature_b", "psi": 0.05, "drift_flag": False},
        ]
    }
    failures = gate.evaluate(drift_report)
    assert len(failures) == 1
    assert failures[0].gate_name == "drift:feature_a"


def test_drift_gate_restricted_to_single_feature():
    gate = DriftGate(max_psi=0.25, feature="feature_b")
    drift_report = {
        "features": [
            {"feature": "feature_a", "psi": 0.9, "drift_flag": True},
            {"feature": "feature_b", "psi": 0.9, "drift_flag": True},
        ]
    }
    failures = gate.evaluate(drift_report)
    assert len(failures) == 1
    assert failures[0].gate_name == "drift:feature_b"


# ---------------------------------------------------------------------------
# _build_reference_distribution
# ---------------------------------------------------------------------------


def test_build_reference_distribution_shape():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=100)})
    dist = _build_reference_distribution(df, ["x"], n_bins=5)
    assert "x" in dist
    assert len(dist["x"]["bin_edges"]) == len(dist["x"]["expected_proportions"]) + 1
    assert pytest.approx(sum(dist["x"]["expected_proportions"]), abs=1e-6) == 1.0


def test_build_reference_distribution_skips_insufficient_samples():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    dist = _build_reference_distribution(df, ["x"], n_bins=10)
    assert "x" not in dist


def test_numeric_feature_columns_excludes_label_and_non_numeric():
    df = pd.DataFrame({"a": [1, 2], "label": [0, 1], "asset_pair": ["x", "y"]})
    cols = _numeric_feature_columns(df, exclude={"label"})
    assert cols == ["a"]


# ---------------------------------------------------------------------------
# run_evaluation_workflow — end to end
# ---------------------------------------------------------------------------


def test_run_evaluation_workflow_passes_with_no_drift_and_good_quality(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    _make_dataset(reference_path, seed=1)
    _make_dataset(current_path, seed=2)  # same distribution, different draw

    result = run_evaluation_workflow(
        str(reference_path),
        str(current_path),
        {"predict_fn": _perfect_predict_fn},
        str(tmp_path / "out"),
        quality_gates=(QualityGate(metric="roc_auc", min_value=0.9),),
        drift_gates=(DriftGate(max_psi=0.5),),
    )

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.failures == []
    assert (tmp_path / "out" / "evaluation_report.json").exists()


def test_run_evaluation_workflow_fails_on_quality_gate(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    _make_dataset(reference_path, seed=1)
    _make_dataset(current_path, seed=2)

    def _bad_predict_fn(row):
        return 0.0  # always predicts negative — poor discrimination

    result = run_evaluation_workflow(
        str(reference_path),
        str(current_path),
        {"predict_fn": _bad_predict_fn},
        str(tmp_path / "out"),
        quality_gates=(QualityGate(metric="roc_auc", min_value=0.9),),
        drift_gates=(DriftGate(max_psi=0.5),),
    )

    assert result.passed is False
    assert any(f.gate_name == "quality:roc_auc" for f in result.failures)


def test_run_evaluation_workflow_fails_on_drift_gate(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    _make_dataset(reference_path, seed=1, feature_shift=0.0)
    _make_dataset(current_path, seed=2, feature_shift=5.0)  # heavily shifted

    result = run_evaluation_workflow(
        str(reference_path),
        str(current_path),
        {"predict_fn": _perfect_predict_fn},
        str(tmp_path / "out"),
        quality_gates=(QualityGate(metric="roc_auc", min_value=0.0),),
        drift_gates=(DriftGate(max_psi=0.1),),
    )

    assert result.passed is False
    assert any(f.gate_name.startswith("drift:") for f in result.failures)


def test_run_evaluation_workflow_writes_report_even_when_failing(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    _make_dataset(reference_path, seed=1)
    _make_dataset(current_path, seed=2, feature_shift=10.0)

    out_dir = tmp_path / "out"
    result = run_evaluation_workflow(
        str(reference_path),
        str(current_path),
        {"predict_fn": _perfect_predict_fn},
        str(out_dir),
        drift_gates=(DriftGate(max_psi=0.01),),
    )

    assert result.passed is False
    report_path = out_dir / "evaluation_report.json"
    assert report_path.exists()
