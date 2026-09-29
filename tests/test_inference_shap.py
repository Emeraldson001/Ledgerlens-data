import json
import os

import numpy as np
import pandas as pd
import pytest

from detection.model_inference import RiskScorer
from detection.model_training import save_models, train_models
from detection.shap_explainer import ShapExplainer
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from tests.conftest import build_signed_model_dir, sign_and_trust_models, write_minimal_metrics


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=60, seed=2)
    output = train_models(df, test_size=0.3, random_state=2)
    model_dir = str(tmp_path_factory.mktemp("models"))
    public_key, transparency_log = build_signed_model_dir(output, model_dir)
    return output, model_dir, df, public_key, transparency_log


def test_risk_scorer_score_returns_contract_shape(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    required = {"score", "benford_flag", "ml_flag", "confidence"}
    assert required.issubset(set(result))
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100
    assert isinstance(result["benford_flag"], bool)
    assert isinstance(result["ml_flag"], bool)


def test_risk_scorer_score_matrix(trained_models):
    _, model_dir, df, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    features = df.drop(columns=["label"])
    scored = scorer.score_matrix(features)

    assert "wallet" in scored.columns
    assert {"score", "benford_flag", "ml_flag", "confidence"}.issubset(set(scored.columns))
    assert len(scored) == len(features)


def test_risk_scorer_raises_without_models(tmp_path):
    scorer = RiskScorer(model_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        scorer.score(
            generate_synthetic_dataset(n_wallets=2, seed=3).drop(columns=["label"]).iloc[0]
        )


def test_risk_scorer_exposes_metadata(trained_models):
    output, model_dir, _, public_key, transparency_log = trained_models
    scorer = RiskScorer(
        model_dir=model_dir, public_key=public_key, transparency_log=transparency_log
    )

    assert scorer.metadata is not None
    assert isinstance(scorer.metadata["trained_at"], str)
    assert len(scorer.metadata["trained_at"]) > 0
    assert scorer.metadata["feature_columns"] == output["feature_columns"]


def test_risk_scorer_raises_on_schema_mismatch(trained_models):
    _, model_dir, _df, public_key, transparency_log = trained_models

    # Manually corrupt the metadata hash
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["feature_schema_hash"] = "sha256:wronghash"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # A corrupted feature_schema_hash in model_metadata.json is now caught by
    # the compatibility gate at construction time (Grand 2 / issue #671),
    # rather than surfacing only later at .score() time as a bare
    # RuntimeError — fail fast, with a typed error distinguishing this from
    # an integrity (signature/tamper) failure.
    from detection.artifact_compatibility import ArtifactCompatibilityError

    with pytest.raises(ArtifactCompatibilityError, match="[Ss]chema"):
        RiskScorer(model_dir=model_dir, public_key=public_key, transparency_log=transparency_log)


def test_risk_scorer_metadata_none_without_metadata_file(trained_models, tmp_path):
    output, _, df, _, _ = trained_models
    # Copy models to a new dir without model_metadata.json — but still
    # trust-chain-signed, since that is a separate concern from metadata.
    new_dir = str(tmp_path)
    save_models(output["results"], new_dir)
    write_minimal_metrics(new_dir, list(output["results"]))
    public_key, transparency_log = sign_and_trust_models(new_dir)

    scorer = RiskScorer(model_dir=new_dir, public_key=public_key, transparency_log=transparency_log)
    assert scorer.metadata is None

    # Scoring should still work
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert "score" in result


def test_metadata_backward_compat_no_raise_without_file(trained_models, tmp_path):
    output, _, df, _, _ = trained_models
    new_dir = str(tmp_path)
    save_models(output["results"], new_dir)
    write_minimal_metrics(new_dir, list(output["results"]))
    public_key, transparency_log = sign_and_trust_models(new_dir)

    # Should not raise during init or score
    scorer = RiskScorer(model_dir=new_dir, public_key=public_key, transparency_log=transparency_log)
    row = df.drop(columns=["label"]).iloc[0]
    scorer.score(row)


def test_shap_explainer_explain(trained_models):
    output, _, df, _, _ = trained_models
    results = output["results"]
    model = results["random_forest"]["model"]
    explainer = ShapExplainer(model)

    row = df.drop(columns=["label"]).iloc[0]
    explanation = explainer.explain(row, top_n=3)

    assert len(explanation) == 3
    for entry in explanation:
        assert {"feature", "contribution", "value"}.issubset(set(entry))


def test_shap_explainer_explain_ensemble(trained_models):
    output, _, df, _, _ = trained_models
    results = output["results"]
    models = {name: result["model"] for name, result in results.items()}
    explainer = ShapExplainer()

    row = df.drop(columns=["label"]).iloc[0]
    explanation = explainer.explain_ensemble(row, models, top_n=3)

    assert len(explanation) == 3
    for entry in explanation:
        assert {"feature", "contribution", "value"}.issubset(set(entry))


# ---------------------------------------------------------------------------
# SHAP interaction value tests (Issue #267)
# ---------------------------------------------------------------------------


def _make_single_feature_model():
    """Train a single depth-1 decision tree that only splits on f0.

    With max_depth=1 and n_estimators=1, the tree makes exactly one split on the
    most informative feature (f0). f1 and f2 are never used, so all pairwise
    SHAP interaction values involving them are identically 0.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, 3))
    y = (X[:, 0] > 0.5).astype(int)  # label depends only on f0
    clf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
    clf.fit(X, y)
    return clf, pd.DataFrame(X[:20], columns=["f0", "f1", "f2"])


def test_interaction_values_zero_for_non_informative_pairs(monkeypatch):
    """For a model linear in f0 only, interactions for (f1,f2) must be < 0.001."""
    import config as cfg_module

    monkeypatch.setattr(cfg_module.config, "SHAP_INTERACTIONS_ENABLED", True)

    model, X = _make_single_feature_model()
    explainer = ShapExplainer(model)
    interactions = explainer.compute_interaction_values(model, X, top_n=3)

    # Find the (f1, f2) interaction — it must be near zero
    f1_f2 = next(
        (ix for ix in interactions if set([ix["feature_a"], ix["feature_b"]]) == {"f1", "f2"}),
        None,
    )
    # If it's not in top_n, it's even smaller — that also passes
    if f1_f2 is not None:
        assert (
            f1_f2["interaction"] < 0.001
        ), f"Expected (f1, f2) interaction < 0.001, got {f1_f2['interaction']}"


def test_format_top_interactions_produces_five_strings():
    """format_top_interactions must return exactly 5 correctly formatted strings."""
    import re

    from detection.shap_explainer import format_top_interactions

    raw = [
        {"feature_a": f"feat_{i}", "feature_b": f"feat_{i+1}", "interaction": float(i) * 0.1}
        for i in range(5)
    ]
    result = format_top_interactions(raw)

    assert len(result) == 5
    pattern = re.compile(r"^.+ x .+ contributes [0-9]+\.[0-9]+ points to the score$")
    for s in result:
        assert pattern.match(s), f"String {s!r} does not match expected format"
