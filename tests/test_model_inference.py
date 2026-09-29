"""Tests for detection/model_inference.py — BFT voting and RiskScorer."""

import os

import numpy as np
import pytest

from detection.model_inference import (
    RiskScorer,
    _has_consensus,
    bft_trimmed_mean,
)
from detection.model_training import train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from tests.conftest import build_signed_model_dir


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=60, seed=2)
    output = train_models(df, test_size=0.3, random_state=2)
    model_dir = str(tmp_path_factory.mktemp("models"))
    public_key, transparency_log = build_signed_model_dir(output, model_dir)
    return output, model_dir, df, public_key, transparency_log


# ---------------------------------------------------------------------------
# BFT trimmed mean
# ---------------------------------------------------------------------------


def test_bft_trimmed_mean_median_of_three():
    score, diverged = bft_trimmed_mean([20.0, 50.0, 80.0])
    assert score == 50.0
    assert diverged is True  # |80-20| = 60 > default threshold 30


def test_bft_trimmed_mean_no_divergence():
    score, diverged = bft_trimmed_mean([40.0, 45.0, 50.0])
    # span = 10 < 30; median = 45
    assert score == 45.0
    assert diverged is False


def test_bft_divergence_flag_raised_when_span_exceeds_threshold():
    _, diverged = bft_trimmed_mean([0.0, 50.0, 100.0])
    assert diverged is True


def test_bft_trimmed_mean_single_value():
    score, diverged = bft_trimmed_mean([77.0])
    assert score == 77.0
    assert diverged is False


def test_bft_trimmed_mean_all_identical_scores():
    # A unanimous ensemble: no divergence, returns the common score.
    assert bft_trimmed_mean([30.0, 30.0, 30.0]) == (30.0, False)


def test_bft_trimmed_mean_trims_outliers_for_many_models():
    # Span (99) far exceeds the default threshold (30): the min and max are
    # trimmed and the remaining scores are averaged.
    score, diverged = bft_trimmed_mean([0.0, 50.0, 52.0, 54.0, 99.0])
    assert diverged is True
    assert score == pytest.approx(52.0)  # (50 + 52 + 54) / 3


def test_bft_trimmed_mean_plain_mean_without_divergence():
    # No outlier, so for >3 models the result is the plain (non-trimmed) mean.
    score, diverged = bft_trimmed_mean([40.0, 42.0, 44.0, 46.0])
    assert diverged is False
    assert score == pytest.approx(43.0)


def test_bft_divergence_flag_false_at_threshold_boundary():
    # The default threshold is 30; a span of exactly 30 does NOT flag divergence
    # because the comparison is strict (`>`), so the median is still used.
    score, diverged = bft_trimmed_mean([10.0, 30.0, 40.0])
    assert diverged is False
    assert score == 30.0


# ---------------------------------------------------------------------------
# Consensus check
# ---------------------------------------------------------------------------


def test_consensus_failure_when_no_two_models_agree():
    # Scores spread 40 points apart — no two within 10
    assert _has_consensus([0.0, 40.0, 80.0]) is False


def test_consensus_passes_when_two_models_agree():
    assert _has_consensus([45.0, 50.0, 90.0]) is True


# ---------------------------------------------------------------------------
# RiskScorer integration
# ---------------------------------------------------------------------------


def test_risk_scorer_score_returns_contract_shape(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    assert "score" in result
    assert "benford_flag" in result
    assert "ml_flag" in result
    assert "confidence" in result
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100


def test_risk_scorer_score_matrix(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )
    features = df.drop(columns=["label"])
    scored = scorer.score_matrix(features)

    assert "wallet" in scored.columns
    assert "score" in scored.columns
    assert len(scored) == len(features)


def test_risk_scorer_raises_without_models(tmp_path):
    scorer = RiskScorer(model_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        scorer.score(
            generate_synthetic_dataset(n_wallets=2, seed=3).drop(columns=["label"]).iloc[0]
        )


def test_bft_divergence_key_present_when_flagged(trained_models, monkeypatch):
    """Patch model outputs to force divergence and verify bft_divergence=True."""
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    # Monkey-patch models to return known divergent probabilities
    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return np.array([[1 - self.prob, self.prob]])

    scorer.models = {
        "random_forest": FakeModel(0.1),  # score=10
        "xgboost": FakeModel(0.5),  # score=50
        "lightgbm": FakeModel(0.9),  # score=90  — span=80>30
    }

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert result.get("bft_divergence") is True


def test_bft_prometheus_counter_incremented_on_divergence(trained_models, monkeypatch):
    import detection.model_inference as mi

    counter_calls = []

    monkeypatch.setattr(mi, "_increment_bft_counter", lambda: counter_calls.append(1))

    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return np.array([[1 - self.prob, self.prob]])

    scorer.models = {
        "random_forest": FakeModel(0.1),
        "xgboost": FakeModel(0.5),
        "lightgbm": FakeModel(0.9),
    }

    row = df.drop(columns=["label"]).iloc[0]
    scorer.score(row)
    assert len(counter_calls) == 1


def test_risk_scorer_default_weights_none_preserves_bft_behavior(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )
    assert scorer.weights is None

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert "calibrated" not in result


def test_risk_scorer_weighted_mode_returns_calibrated_score(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir,
        weights={"random_forest": 0.5, "xgboost": 0.3, "lightgbm": 0.2},
        public_key=public_key,
        transparency_log=transparency_log,
    )
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    assert result["calibrated"] is True
    assert 0 <= result["score"] <= 100
    assert "consensus_failure" not in result


def test_risk_scorer_weights_must_sum_to_one(trained_models):
    with pytest.raises(ValueError):
        RiskScorer(weights={"random_forest": 0.5, "xgboost": 0.5, "lightgbm": 0.5})


def test_risk_scorer_weighted_mode_rejects_unknown_model_names(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir,
        weights={"random_forest": 1.0},
        public_key=public_key,
        transparency_log=transparency_log,
    )
    scorer.weights = {"not_a_real_model": 1.0}

    row = df.drop(columns=["label"]).iloc[0]
    with pytest.raises(ValueError):
        scorer.score(row)


def test_consensus_failure_score(trained_models, monkeypatch):
    """When no two models agree, score must be 100 and consensus_failure=True."""
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return np.array([[1 - self.prob, self.prob]])

    scorer.models = {
        "random_forest": FakeModel(0.0),  # 0
        "xgboost": FakeModel(0.4),  # 40
        "lightgbm": FakeModel(0.85),  # 85
    }

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert result["consensus_failure"] is True
    assert result["score"] == 100
    assert result["confidence"] == 0


# ---------------------------------------------------------------------------
# Grand 2 (issue #671) — hard-block trust chain in RiskScorer._load_models
# ---------------------------------------------------------------------------


class TestTrustChainHardBlock:
    def test_tampered_artifact_raises_and_never_appears_in_models(self, trained_models):
        """Acceptance criterion: a single byte flipped post-signing must
        raise and the model must never appear in the active `models` dict —
        not be silently skipped, not logged-and-loaded-anyway."""
        from detection.persistence import ModelIntegrityError

        _, model_dir, df, public_key, transparency_log = trained_models
        artifact_path = os.path.join(model_dir, "random_forest.joblib")
        with open(artifact_path, "rb") as f:
            original = bytearray(f.read())
        tampered = bytearray(original)
        tampered[0] ^= 0xFF
        with open(artifact_path, "wb") as f:
            f.write(bytes(tampered))

        try:
            with pytest.raises(ModelIntegrityError):
                RiskScorer(
                    model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
                )
        finally:
            # Restore for any other test sharing this module-scoped fixture.
            with open(artifact_path, "wb") as f:
                f.write(bytes(original))

    def test_unsigned_artifact_raises(self, tmp_path):
        """No metrics.json/signature at all — must hard-block, not warn."""
        from detection.model_training import save_models, train_models
        from detection.persistence import ModelIntegrityError
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset

        df = generate_synthetic_dataset(n_wallets=20, seed=5)
        output = train_models(df, test_size=0.3, random_state=5)
        model_dir = str(tmp_path)
        save_models(output["results"], model_dir)  # no metrics.json written

        with pytest.raises(ModelIntegrityError):
            RiskScorer(model_dir=model_dir)

    def test_untrusted_transparency_log_raises(self, trained_models):
        """Validly signed, but the hash was never published to the
        transparency log this RiskScorer trusts — must hard-block."""
        from detection.persistence import (
            ModelIntegrityError,
            TransparencyLog,
            get_engine,
            get_session_factory,
        )

        _, model_dir, df, public_key, _unused_log = trained_models
        empty_log = TransparencyLog(get_session_factory(get_engine("sqlite:///:memory:")))

        with pytest.raises(ModelIntegrityError, match="not in the transparency log"):
            RiskScorer(model_dir=model_dir, public_key=public_key, transparency_log=empty_log)

    def test_incompatible_schema_raises_distinct_typed_error(self, tmp_path):
        """A validly-signed artifact with an incompatible feature schema
        must be rejected with a typed error distinct from ModelIntegrityError."""
        from detection.artifact_compatibility import ArtifactCompatibilityError
        from detection.model_training import train_models
        from detection.persistence import ModelIntegrityError
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset
        from tests.conftest import build_signed_model_dir

        df = generate_synthetic_dataset(n_wallets=20, seed=6)
        output = train_models(df, test_size=0.3, random_state=6)
        model_dir = str(tmp_path)
        public_key, transparency_log = build_signed_model_dir(output, model_dir)

        # Corrupt the per-model manifest's feature schema hash so it no
        # longer matches model_metadata.json's — a real schema mismatch,
        # independent of signature validity.
        manifest_path = os.path.join(model_dir, "random_forest__artifact_manifest.json")
        import json

        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["feature_schema_hash"] = "sha256:" + "0" * 64
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ArtifactCompatibilityError):
            RiskScorer(
                model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
            )

        # And it must be a genuinely different exception class than the
        # integrity/signature failure path — callers can tell them apart.
        assert not issubclass(ArtifactCompatibilityError, ModelIntegrityError)
        assert not issubclass(ModelIntegrityError, ArtifactCompatibilityError)

    def test_integrity_override_actor_skips_failing_model_but_loads_others(
        self, trained_models, tmp_path, monkeypatch
    ):
        """The documented emergency-override path: a configured
        integrity_override_actor lets construction succeed by skipping the
        one failing model, rather than either (a) hard-blocking construction
        entirely or (b) loading the unverified model anyway."""
        from config import config as cfg

        monkeypatch.setattr(cfg, "RISK_SCORE_DB_URL", f"sqlite:///{tmp_path}/audit.db")

        _, model_dir, df, public_key, transparency_log = trained_models
        artifact_path = os.path.join(model_dir, "random_forest.joblib")
        with open(artifact_path, "rb") as f:
            original = bytearray(f.read())
        tampered = bytearray(original)
        tampered[0] ^= 0xFF
        with open(artifact_path, "wb") as f:
            f.write(bytes(tampered))

        try:
            scorer = RiskScorer(
                model_dir=model_dir,
                public_key=public_key,
                transparency_log=transparency_log,
                integrity_override_actor="oncall-engineer",
            )
            assert "random_forest" not in scorer.models
            assert "xgboost" in scorer.models
            assert "lightgbm" in scorer.models

            # The override itself must be audited.
            from detection.persistence import PromotionAuditLog, get_engine, get_session_factory

            audit_log = PromotionAuditLog(get_session_factory(get_engine()))
            rows = audit_log.recent()
            assert any(
                r.actor == "oncall-engineer"
                and r.action == "integrity_override"
                and r.model_name == "random_forest"
                for r in rows
            )
        finally:
            with open(artifact_path, "wb") as f:
                f.write(bytes(original))

    def test_integrity_override_still_raises_if_all_models_fail(self, tmp_path, monkeypatch):
        """Override must not turn a total trust failure into a silently
        empty, "successfully constructed" RiskScorer."""
        from detection.model_training import save_models, train_models
        from detection.persistence import ModelIntegrityError
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset

        df = generate_synthetic_dataset(n_wallets=20, seed=9)
        output = train_models(df, test_size=0.3, random_state=9)
        model_dir = str(tmp_path)
        save_models(output["results"], model_dir)  # unsigned — every model fails verification

        with pytest.raises(ModelIntegrityError):
            RiskScorer(model_dir=model_dir, integrity_override_actor="oncall-engineer")
