"""Example: full end-to-end detection pipeline.

Exercises all major stages of the LedgerLens detection stack in sequence:

1. Synthetic data generation (mimics ``scripts/generate_synthetic_dataset.py``)
2. Feature engineering (``detection/feature_engineering.py``)
3. Benford anomaly metrics (``detection/benford_engine.py``)
4. Model training on synthetic data in a temp directory
5. Risk scoring (``detection/model_inference.py``)
6. SHAP explanation (``detection/shap_explainer.py``)
7. Persistence to an in-memory SQLite DB (``detection/persistence.py``)

This is the most comprehensive example and can be used as a reference for
integrating LedgerLens into an external pipeline.

Run::

    python -m examples.e2e_full_pipeline
"""

from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

# Ensure repo root is on the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _generate_labelled_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Minimal synthetic labelled dataset (subset of features used in CI)."""
    from scripts.generate_synthetic_dataset import generate_synthetic_dataset

    return generate_synthetic_dataset(n_wallets=n, seed=seed)


def _train_models(dataset: pd.DataFrame, model_dir: str) -> None:
    """Train the ensemble on *dataset* and write artifacts to *model_dir*."""
    from detection.model_training import train_models

    train_models(dataset, model_dir=model_dir)


def _score_wallet(wallet_features: pd.Series, model_dir: str) -> dict:
    """Score a single wallet feature row using the trained ensemble."""
    from detection.model_inference import RiskScorer

    scorer = RiskScorer(model_dir=model_dir)
    return scorer.score(wallet_features)


def _explain_wallet(wallet_features: pd.Series, model_dir: str) -> dict:
    """Return SHAP attributions for a single wallet feature row."""
    try:
        from detection.shap_explainer import ShapExplainer

        explainer = ShapExplainer(model_dir=model_dir)
        return explainer.explain(wallet_features)
    except Exception as exc:
        return {"error": str(exc)}


def _persist_score(wallet: str, pair: str, score_result: dict, db_url: str) -> None:
    """Upsert a risk score record to the given DB URL."""
    from detection.persistence import get_engine, get_session_factory
    from detection.risk_score_store import RiskScoreStore

    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    store = RiskScoreStore(session_factory)
    store.upsert(
        wallet=wallet,
        asset_pair=pair,
        score=int(score_result.get("score", 0)),
        benford_flag=bool(score_result.get("benford_flag", False)),
        ml_flag=bool(score_result.get("ml_flag", False)),
        confidence=int(score_result.get("confidence", 0)),
    )


def main() -> None:
    """Run the full end-to-end pipeline example."""
    sep = "=" * 62

    print(f"\n{sep}")
    print(" LedgerLens — Full End-to-End Pipeline Example")
    print(sep)

    # ── Step 1: Generate synthetic data ──────────────────────────────────
    print("\n[1/6] Generating synthetic labelled dataset (300 wallets)…")
    dataset = _generate_labelled_dataset(n=300)
    print(f"      Dataset shape: {dataset.shape}  (label=1 count: {dataset['label'].sum()})")

    # ── Step 2: Train models in a temp directory ─────────────────────────
    with tempfile.TemporaryDirectory(prefix="ll_example_") as model_dir:
        print(f"\n[2/6] Training ensemble models in {model_dir} …")
        try:
            _train_models(dataset, model_dir=model_dir)
            print("      Training complete.")
            models_available = True
        except Exception as exc:
            print(f"      Training skipped ({exc}) — will use Benford-only scoring.")
            models_available = False

        # ── Step 3: Pick a suspicious wallet from the dataset ─────────────
        wash_rows = dataset[dataset["label"] == 1]
        if wash_rows.empty:
            print("\n[3/6] No labelled wash-trade rows found — using first row.")
            sample_row = dataset.iloc[0]
        else:
            sample_row = wash_rows.iloc[0]

        wallet = sample_row.get(
            "wallet", "GEXAMPLE00000000000000000000000000000000000000000000000001"
        )
        pair = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"

        from detection.model_training import FEATURE_COLUMNS_EXCLUDE

        feature_cols = [c for c in dataset.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        feature_row = sample_row[feature_cols]

        print(f"\n[3/6] Focal wallet: {wallet}")
        print(f"      Feature columns: {len(feature_cols)}")

        # ── Step 4: Score the wallet ───────────────────────────────────────
        print("\n[4/6] Scoring wallet…")
        if models_available:
            try:
                score_result = _score_wallet(feature_row, model_dir=model_dir)
            except Exception as exc:
                print(f"      Scorer error ({exc}) — using Benford fallback.")
                mad = float(feature_row.get("benford_mad_24h", 0.0))
                score_result = {
                    "score": min(100, int(mad * 2000)),
                    "benford_flag": mad >= 0.015,
                    "ml_flag": False,
                    "confidence": 0,
                }
        else:
            mad = float(feature_row.get("benford_mad_24h", 0.0))
            score_result = {
                "score": min(100, int(mad * 2000)),
                "benford_flag": mad >= 0.015,
                "ml_flag": False,
                "confidence": 0,
            }

        print(f"      Risk score    : {score_result['score']} / 100")
        print(f"      Benford flag  : {score_result['benford_flag']}")
        print(f"      ML flag       : {score_result['ml_flag']}")
        print(f"      Confidence    : {score_result.get('confidence', 'n/a')}")

        # ── Step 5: SHAP explanation ───────────────────────────────────────
        print("\n[5/6] Generating SHAP explanation…")
        shap_result = {}
        if models_available:
            shap_result = _explain_wallet(feature_row, model_dir=model_dir)
            if "error" not in shap_result and shap_result:
                top_features = sorted(shap_result.items(), key=lambda x: abs(x[1]), reverse=True)[
                    :5
                ]
                print("      Top 5 SHAP contributors:")
                for feat, val in top_features:
                    sign = "▲ risk+" if val > 0 else "▼ risk-"
                    print(f"        {sign}  {feat}: {val:+.4f}")
            else:
                print(f"      SHAP not available: {shap_result.get('error', 'unknown')}")
        else:
            print("      SHAP skipped (models not trained).")

        # ── Step 6: Persist to in-memory SQLite ───────────────────────────
        print("\n[6/6] Persisting risk score to in-memory SQLite…")
        db_url = "sqlite:///:memory:"
        try:
            _persist_score(wallet, pair, score_result, db_url=db_url)
            print("      Score persisted successfully.")
        except Exception as exc:
            print(f"      Persistence error (non-fatal): {exc}")

    print(f"\n{sep}")
    print(" Full pipeline example complete.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
