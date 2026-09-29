"""Shared helpers used by the end-to-end detection examples.

Provides ``build_trades``, ``run_detection``, and ``print_result`` so each
example module stays focused on its scenario rather than boilerplate.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Make imports work when running from the repo root with ``python -m examples.*``
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import config
from detection.benford_engine import BenfordEngine
from detection.feature_engineering import build_feature_matrix
from utils.logging import get_logger

logger = get_logger(__name__)

PAIR_ID = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
EXAMPLE_WALLET = "GEXAMPLEWALLET000000000000000000000000000000000000000000001"


# ---------------------------------------------------------------------------
# Trade-row builder
# ---------------------------------------------------------------------------


def make_trade(
    *,
    wallet: str = EXAMPLE_WALLET,
    counterparty: str = "GCOUNTERPARTY0000000000000000000000000000000000000000000001",
    amount: float,
    timestamp: datetime | None = None,
    pair_id: str = PAIR_ID,
) -> dict[str, Any]:
    ts = timestamp or datetime.now(UTC)
    return {
        "wallet": wallet,
        "counterparty": counterparty,
        "amount": amount,
        "pair_id": pair_id,
        "timestamp": ts,
        "trade_type": "buy",
        "base_asset_code": "USDC",
        "counter_asset_code": "XLM",
    }


def build_trades_df(trade_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of trade dicts to the DataFrame shape expected by the
    detection pipeline."""
    df = pd.DataFrame(trade_dicts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["amount"] = df["amount"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Detection pipeline runner
# ---------------------------------------------------------------------------


def run_detection(
    trades_df: pd.DataFrame,
    *,
    wallet: str = EXAMPLE_WALLET,
    pair_id: str = PAIR_ID,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Run the full detection stack on *trades_df* and return a result dict.

    Steps:
    1. Compute Benford metrics.
    2. Build the ML feature matrix.
    3. Score with the trained ensemble (falls back gracefully if no models are
       present by using the Benford signal only).
    4. Return a dict with ``score``, ``benford_flag``, ``features``, and
       ``shap_values`` (empty dict when models not present).
    """
    benford = BenfordEngine()
    amounts = trades_df["amount"].dropna().tolist()

    if len(amounts) < 5:
        logger.warning("Only %d trade amounts — Benford metrics will be unreliable", len(amounts))

    benford_result = benford.compute_all(amounts)
    mad = benford_result.get("mad", 0.0)
    benford_flag = mad >= 0.015

    # Build feature matrix
    features: dict[str, float] = {}
    try:
        feature_df = build_feature_matrix(trades_df)
        wallet_row = (
            feature_df[feature_df["wallet"] == wallet]
            if "wallet" in feature_df.columns
            else feature_df
        )
        if not wallet_row.empty:
            features = wallet_row.iloc[0].to_dict()
    except Exception as exc:
        logger.debug("Feature matrix build error (non-fatal): %s", exc)

    # Model inference (best-effort — models may not be trained locally)
    score: float = 0.0
    shap_values: dict[str, float] = {}
    ml_flag = False

    try:
        from detection.model_inference import RiskScorer

        scorer = RiskScorer()
        if features:
            feature_row = pd.Series(features)
            result = scorer.score(feature_row)
            score = float(result.get("score", 0.0))
            ml_flag = result.get("ml_flag", False)
        else:
            # Synthesise a minimal feature row from Benford output
            feature_row = _benford_to_feature_row(benford_result)
            result = scorer.score(feature_row)
            score = float(result.get("score", 0.0))
            ml_flag = result.get("ml_flag", False)
    except Exception as exc:
        logger.debug("RiskScorer unavailable (%s) — using Benford-only score", exc)
        # Fallback: derive a rough score from Benford MAD
        score = min(100.0, mad * 2000.0)
        ml_flag = False

    output = {
        "wallet": wallet,
        "pair_id": pair_id,
        "score": score,
        "benford_flag": benford_flag,
        "ml_flag": ml_flag,
        "benford": benford_result,
        "features": features,
        "shap_values": shap_values,
        "n_trades": len(trades_df),
    }

    if print_summary:
        print_result(output)

    return output


def _benford_to_feature_row(benford_result: dict[str, Any]) -> pd.Series:
    """Build a minimal feature Series from Benford output for scoring when
    a full feature matrix cannot be computed."""
    windows = ["1h", "4h", "24h", "168h", "720h"]
    row: dict[str, float] = {}
    for w in windows:
        row[f"benford_chi_square_{w}"] = float(benford_result.get("chi_square", 0.0))
        row[f"benford_mad_{w}"] = float(benford_result.get("mad", 0.0))
        row[f"benford_z_max_{w}"] = float(benford_result.get("z_max", 0.0))

    # Zero-fill all other expected features so the model doesn't error

    try:
        import joblib

        rf_path = os.path.join(config.MODEL_DIR, "random_forest.joblib")
        if os.path.exists(rf_path):
            rf = joblib.load(rf_path)
            for fname in rf.feature_names_in_:
                if fname not in row:
                    row[fname] = 0.0
    except Exception:
        pass

    return pd.Series(row)


# ---------------------------------------------------------------------------
# Result printer
# ---------------------------------------------------------------------------


def print_result(result: dict[str, Any], *, label: str = "") -> None:
    sep = "─" * 60
    header = f" LedgerLens Detection Result {'─ ' + label if label else ''}".rstrip()
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(f"  Wallet  : {result['wallet']}")
    print(f"  Pair    : {result['pair_id']}")
    print(f"  Trades  : {result['n_trades']}")
    print(f"  Score   : {result['score']:.1f} / 100")
    print(f"  Benford : {'⚠ ANOMALY' if result['benford_flag'] else '✓ Normal'}")
    print(f"  ML flag : {'⚠ FLAGGED' if result['ml_flag'] else '✓ Clean'}")
    b = result.get("benford", {})
    if b:
        print(f"  MAD     : {b.get('mad', 0):.4f}  (threshold: 0.015)")
        print(f"  χ²      : {b.get('chi_square', 0):.2f}")
    if result.get("shap_values"):
        print("  Top SHAP contributions:")
        top = sorted(result["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in top:
            sign = "▲" if val > 0 else "▼"
            print(f"    {sign} {feat}: {val:+.4f}")
    print(sep + "\n")
