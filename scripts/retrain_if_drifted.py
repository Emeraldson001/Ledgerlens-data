"""CLI script that detects feature drift and triggers automated retraining.

Exit codes:
    0  — No drift detected, no retraining needed.
    2  — Drift detected, retrained (full or incremental) and promoted.
    3  — Drift detected, retrained but NOT promoted (metric regression).
    1  — Fatal error.

Usage:
    python -m scripts.retrain_if_drifted --lookback-days 30
    python -m scripts.retrain_if_drifted --lookback-days 30 --retrain-data-path data/synthetic_dataset.parquet

Incremental training mode (``--incremental``):
    When the flag is passed, the script attempts to run
    ``incremental_train_lightgbm`` instead of a full ensemble retrain.  A
    full retrain is forced if the staleness detector reports that
    ``MAX_INCREMENTAL_ROUNDS`` consecutive incremental passes have already
    occurred, or if the buffer does not yet contain enough samples.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import joblib
import pandas as pd

from config import config
from detection import model_governance
from detection.drift_monitor import DriftMonitor
from detection.feature_cache import RecentDataBuffer
from detection.feature_engineering import build_feature_matrix
from detection.model_compatibility import (
    FeatureContractError,
    validate_feature_compatibility,
)
from detection.model_training import (
    MODEL_REGISTRY,
    IncrementalTrainingStalenessDetector,
    compute_feature_schema_hash,
    incremental_train_lightgbm,
    load_training_data,
    save_models,
    save_training_artifacts,
    train_models,
)
from detection.persistence import PromotionAuditLog, get_engine, get_session_factory
from utils.logging import get_logger

logger = get_logger(__name__)

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "archive"
)
REPORTS_DIR = "reports"
PROMOTION_TOLERANCE = 0.01
_SHADOW_STATE_FILE = "shadow_deployment_state.json"


def get_feature_data(lookback_days: int) -> pd.DataFrame:
    """Load recent trades from Horizon and build a feature matrix.

    This is the "current distribution" used for drift detection.
    In production this hits the Horizon API; in tests it is mocked.
    """
    from ingestion.historical_loader import load_watched_pairs_to_dataframe

    since = datetime.now(UTC) - timedelta(days=lookback_days)
    logger.info("Loading trades since %s", since.isoformat())
    trades_df = load_watched_pairs_to_dataframe(start_time=since)
    logger.info("Loaded %d trades", len(trades_df))

    if trades_df.empty:
        logger.warning("No trades loaded; returning empty feature matrix")
        return pd.DataFrame()

    logger.info("Building feature matrix for drift detection")
    feature_matrix = build_feature_matrix(trades_df)
    logger.info("Built features for %d wallets", len(feature_matrix))
    return feature_matrix


def load_model_metadata(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(path):
        logger.error("model_metadata.json not found in %s", model_dir)
        return None
    with open(path) as f:
        return cast(dict[Any, Any], json.load(f))


def load_metrics(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        logger.error("metrics.json not found in %s", model_dir)
        return None
    with open(path) as f:
        return cast(dict[Any, Any], json.load(f))


def archive_current_models(model_dir: str) -> str:
    """Archive the current production models to models/archive/{timestamp}/.

    Returns the archive path.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, timestamp)
    os.makedirs(archive_path, exist_ok=True)

    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, os.path.join(archive_path, item))

    os.chmod(archive_path, 0o750)
    logger.info("Archived models to %s", archive_path)
    return archive_path


def evaluate_new_model(model_dir: str) -> dict | None:
    """Evaluate the newly trained model's metrics.

    Loads the metrics.json that was written by save_training_artifacts.
    """
    return load_metrics(model_dir)


def should_promote(
    old_metrics: dict[str, dict],
    new_metrics: dict[str, dict],
    tolerance: float = PROMOTION_TOLERANCE,
) -> tuple[bool, str]:
    """Check if the new model should be promoted.

    Requires AUC-ROC >= old_auc - tolerance AND F1 >= old_f1 - tolerance for
    every model in the ensemble. This is now a thin wrapper around
    ``detection.model_governance.evaluate_regression_gate`` — the same
    regression gate ``promote_candidate`` enforces — kept as a
    module-level function here so existing callers/tests that check the
    offline metric comparison in isolation (without exercising the full
    signature/authorization pipeline) keep working.

    Returns (promote: bool, reason: str). Unlike the original
    implementation, a model family missing from *old_metrics* is treated as
    "nothing to regress against" (approved) rather than a hard block —
    matching ``evaluate_regression_gate``'s handling of brand-new model
    families; a family present in *old_metrics* but missing from
    *new_metrics* is still blocked.
    """
    decision = model_governance.evaluate_regression_gate(
        old_metrics, new_metrics, model_names=list(MODEL_REGISTRY), tolerance=tolerance
    )
    return decision.approved, decision.reason


def write_retrain_report(
    drift_report: dict,
    old_metrics: dict | None,
    new_metrics: dict | None,
    promotion_decision: bool,
    reason: str,
    archive_path: str | None,
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"retrain_report_{timestamp}.json")

    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "drift_report": drift_report,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "promotion_decision": promotion_decision,
        "reason": reason,
        "archive_path": archive_path,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote retrain report to %s", path)
    return path


def _shadow_state_path(model_dir: str) -> str:
    return os.path.join(model_dir, _SHADOW_STATE_FILE)


def load_shadow_state(model_dir: str) -> dict | None:
    """Load persisted shadow deployment state from model_dir."""
    path = _shadow_state_path(model_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return cast(dict[Any, Any], json.load(f))


def save_shadow_state(model_dir: str, state: dict) -> None:
    path = _shadow_state_path(model_dir)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Shadow state saved to %s", path)


def clear_shadow_state(model_dir: str) -> None:
    path = _shadow_state_path(model_dir)
    if os.path.exists(path):
        os.remove(path)


def compute_false_positive_rate(model_dir: str, test_data_path: str) -> float | None:
    """Estimate FP rate for the model in model_dir using the labelled test set.

    Returns the FP rate (false positives / (false positives + true negatives)),
    or None if the model cannot be loaded or the test set is missing.
    """
    try:

        from detection.model_inference import RiskScorer

        if not os.path.exists(test_data_path):
            logger.warning("Test data not found at %s — cannot compute FP rate", test_data_path)
            return None

        from detection.model_training import load_training_data

        df = load_training_data(test_data_path)
        if df.empty or "label" not in df.columns:
            return None

        negatives = df[df["label"] == 0]
        if negatives.empty:
            return None

        scorer = RiskScorer(model_dir=model_dir)
        from detection.model_training import FEATURE_COLUMNS_EXCLUDE

        feature_cols = [c for c in negatives.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        fps = 0
        for _, row in negatives[feature_cols].iterrows():
            try:
                result = scorer.score(pd.Series(row))
                if result["ml_flag"]:
                    fps += 1
            except Exception:
                continue

        return fps / len(negatives)
    except Exception as exc:
        logger.warning("FP rate computation failed: %s", exc)
        return None


def evaluate_shadow_candidate(
    shadow_state: dict,
    model_dir: str,
    retrain_data_path: str | None,
    session_factory=None,
) -> int:
    """Check if the shadow candidate is ready for promotion or needs rollback.

    Returns the appropriate exit code (5 = promoted, 6 = rolled back, 4 = still waiting).

    Promotion (once the shadow period has elapsed and drift/FP-rate checks
    pass) is delegated to ``detection.model_governance.promote_candidate`` —
    the single gated path that re-runs the offline AUC/F1 regression gate
    plus the Ed25519 + transparency-log trust chain before anything is
    copied into ``model_dir``. A shadow candidate that clears the *live*
    drift/FP checks below can still be rejected there (e.g. it also
    regressed on the offline test set, or fails signing).
    """
    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    audit_log = PromotionAuditLog(sf)

    candidate_dir = shadow_state.get("candidate_dir")
    shadow_start_str = shadow_state.get("shadow_start")
    version_id = shadow_state.get("version_id", "unknown")

    if not candidate_dir or not os.path.exists(candidate_dir):
        logger.error("Shadow candidate dir %s not found — clearing shadow state", candidate_dir)
        clear_shadow_state(model_dir)
        return 1

    shadow_start = (
        datetime.fromisoformat(shadow_start_str) if shadow_start_str else datetime.now(UTC)
    )
    elapsed_hours = (datetime.now(UTC) - shadow_start).total_seconds() / 3600

    if elapsed_hours < config.SHADOW_PERIOD_HOURS:
        remaining = config.SHADOW_PERIOD_HOURS - elapsed_hours
        logger.info(
            "Shadow period for %s still running — %.1f h remaining (started %s)",
            version_id,
            remaining,
            shadow_start_str,
        )
        return 4

    # Shadow period complete — read accumulated drift stats
    drift_rate = shadow_state.get("drift_rate", 0.0)
    drift_stats = {
        "drift_rate": drift_rate,
        "drift_events": shadow_state.get("drift_events", 0),
        "total_shadow_requests": shadow_state.get("total_shadow_requests", 0),
    }
    logger.info(
        "Shadow period complete for %s: drift_rate=%.2f%% (threshold %.2f%%)",
        version_id,
        drift_rate * 100,
        config.SHADOW_DRIFT_MAX_RATE * 100,
    )

    if drift_rate >= config.SHADOW_DRIFT_MAX_RATE:
        reason = (
            f"Shadow drift rate {drift_rate:.2%} exceeds threshold "
            f"{config.SHADOW_DRIFT_MAX_RATE:.2%}"
        )
        logger.warning("%s — blocking promotion of %s", reason, version_id)
        model_governance.record_shadow_outcome(
            version_id,
            status="rolled_back",
            reason=reason,
            drift_stats=drift_stats,
            session_factory=sf,
        )
        clear_shadow_state(model_dir)
        shutil.rmtree(candidate_dir, ignore_errors=True)
        return 6

    # Check FP rate regression against production
    if retrain_data_path:
        prod_fp = compute_false_positive_rate(model_dir, retrain_data_path)
        cand_fp = compute_false_positive_rate(candidate_dir, retrain_data_path)
        if prod_fp is not None and cand_fp is not None:
            excess = cand_fp - prod_fp
            if excess > config.SHADOW_FP_RATE_MAX_EXCESS:
                reason = (
                    f"Candidate FP rate {cand_fp:.2%} exceeds production {prod_fp:.2%} by "
                    f"{excess:.2%} (max {config.SHADOW_FP_RATE_MAX_EXCESS:.2%})"
                )
                logger.warning("%s — rolling back %s", reason, version_id)
                model_governance.record_shadow_outcome(
                    version_id,
                    status="rolled_back",
                    reason=reason,
                    drift_stats=drift_stats,
                    session_factory=sf,
                )
                clear_shadow_state(model_dir)
                shutil.rmtree(candidate_dir, ignore_errors=True)
                logger.warning(
                    "Automatic rollback triggered for %s — production model retained. "
                    "See docs/model_rollback_runbook.md for manual intervention steps.",
                    version_id,
                )
                return 6

    # Live checks passed — attempt gated promotion (offline regression gate +
    # trust chain still apply here).
    old_metrics = load_metrics(model_dir)
    new_metrics = load_metrics(candidate_dir)
    actor, credential = model_governance.system_actor_credential()
    try:
        model_governance.promote_candidate(
            candidate_dir=candidate_dir,
            model_dir=model_dir,
            actor=actor,
            credential=credential,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            reason=f"shadow deployment complete (shadow_version_id={version_id})",
            session_factory=sf,
            audit_log=audit_log,
        )
    except model_governance.PromotionError as exc:
        logger.warning("Gated promotion of shadow candidate %s failed: %s", version_id, exc)
        model_governance.record_shadow_outcome(
            version_id,
            status="rolled_back",
            reason=str(exc),
            drift_stats=drift_stats,
            session_factory=sf,
        )
        clear_shadow_state(model_dir)
        shutil.rmtree(candidate_dir, ignore_errors=True)
        return 6

    logger.info("Candidate %s promoted to %s", version_id, model_dir)
    model_governance.record_shadow_outcome(
        version_id,
        status="archived",
        reason="promoted",
        drift_stats=drift_stats,
        session_factory=sf,
    )
    clear_shadow_state(model_dir)
    shutil.rmtree(candidate_dir, ignore_errors=True)
    return 5


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (split from parse_args() so
    tests/test_docs_cli_consistency.py can introspect the real flag set
    without parsing argv — the historical bug this guards against is
    docs/CLI drift: --check-shadow/--no-shadow were documented and read
    from `args` but never actually registered here)."""
    parser = argparse.ArgumentParser(description="Detect feature drift and trigger retraining")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to look back for current feature distribution (default: 30)",
    )
    parser.add_argument(
        "--retrain-data-path",
        default=None,
        help="Path to labelled parquet dataset for retraining (required if drift detected)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory (default: config.MODEL_DIR)",
    )
    parser.add_argument(
        "--feature-data-path",
        default=None,
        help="Path to pre-computed feature matrix parquet for drift detection (bypasses Horizon API)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split ratio for retraining"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for train/test split"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help=(
            "Attempt incremental LightGBM training instead of a full ensemble retrain. "
            "A full retrain is forced when the staleness cap (MAX_INCREMENTAL_ROUNDS) "
            "is reached or the buffer has too few samples."
        ),
    )
    parser.add_argument(
        "--incremental-buffer-path",
        default=None,
        help=(
            "Path to a labelled parquet file used as the RecentDataBuffer contents "
            "for incremental training (bypasses the live streaming buffer).  "
            "Only used when --incremental is set."
        ),
    )
    parser.add_argument(
        "--n-new-trees",
        type=int,
        default=None,
        help=(
            "Number of new trees to append per incremental pass "
            "(default: config.INCREMENTAL_N_NEW_TREES = 100)."
        ),
    )
    parser.add_argument(
        "--staleness-state-path",
        default=None,
        help="Override path for the staleness detector state JSON file.",
    )
    parser.add_argument(
        "--check-shadow",
        action="store_true",
        default=False,
        help=(
            "Evaluate a previously-registered shadow candidate (see "
            "shadow_deployment_state.json in --model-dir) for promotion or rollback "
            "and exit, without running drift detection. Exit codes: 0 (no shadow "
            "candidate registered), 4 (shadow period still running), 5 (promoted), "
            "6 (rolled back)."
        ),
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        default=False,
        help=(
            "Skip shadow deployment: attempt gated promotion of a freshly retrained "
            "candidate immediately via detection.model_governance.promote_candidate "
            "(still subject to the regression gate and the Ed25519/transparency-log "
            "trust chain — this only skips the live shadow-traffic evaluation period)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Incremental training helpers
# ---------------------------------------------------------------------------


def run_incremental_training(
    model_dir: str,
    new_data: pd.DataFrame,
    n_new_trees: int,
    staleness_detector: IncrementalTrainingStalenessDetector,
    candidate_dir: str,
) -> tuple[bool, str]:
    """Attempt an incremental LightGBM update.

    Appends *n_new_trees* to the existing LightGBM model using *new_data* and
    writes the result into *candidate_dir* (never *model_dir* directly — see
    ``detection.model_governance.guard_production_write``): every other file
    currently in *model_dir* is copied into *candidate_dir* unchanged so the
    candidate is a complete, self-contained bundle that
    ``model_governance.promote_candidate`` can gate and publish atomically.
    Increments the staleness counter on success.

    Returns:
        ``(success: bool, reason: str)``
    """
    lgbm_path = os.path.join(model_dir, "lightgbm.joblib")
    if not os.path.exists(lgbm_path):
        return False, f"LightGBM artifact not found at {lgbm_path}"

    # Load metadata to get the authoritative feature column list
    metadata = load_model_metadata(model_dir)
    if metadata is None:
        return False, "Cannot load model_metadata.json — skipping incremental training"

    reference_feature_columns: list[str] = metadata.get("feature_columns", [])
    if not reference_feature_columns:
        return False, "model_metadata.json has no feature_columns — cannot validate schema"

    existing_lgbm = joblib.load(lgbm_path)

    try:
        updated_lgbm = incremental_train_lightgbm(
            existing_model=existing_lgbm,
            new_data=new_data,
            n_new_trees=n_new_trees,
            reference_feature_columns=reference_feature_columns,
        )
    except Exception as exc:
        return False, f"incremental_train_lightgbm failed: {exc}"

    os.makedirs(candidate_dir, exist_ok=True)
    for fname in os.listdir(model_dir):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(candidate_dir, fname))

    lgbm_candidate_path = os.path.join(candidate_dir, "lightgbm.joblib")
    # UNGUARDED-OK: writes only to candidate_dir (a staging bundle), never to
    # model_dir directly — publishing candidate_dir to production goes
    # through detection.model_governance.promote_candidate, called by this
    # script's main() after this returns.
    joblib.dump(updated_lgbm, lgbm_candidate_path)

    # Refresh the recorded hash for the updated artifact so the candidate's
    # metrics.json — copied from production above — reflects the new bytes
    # before promote_candidate signs it.
    metrics_path = os.path.join(candidate_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        sha = hashlib.sha256()
        with open(lgbm_candidate_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        metrics.setdefault("lightgbm", {})["artifact_sha256"] = sha.hexdigest()
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    stale = staleness_detector.increment()
    msg = (
        f"Incremental update applied ({n_new_trees} new trees). "
        f"Staleness round {staleness_detector.rounds}/{staleness_detector.max_rounds}."
    )
    if stale:
        msg += " Staleness cap reached — full retrain required next cycle."

    logger.info(msg)
    return True, msg


def _evaluate_incremental_model(
    model_dir: str,
    new_data: pd.DataFrame,
    reference_feature_columns: list[str],
) -> dict | None:
    """Compute AUC-ROC for the updated LightGBM model on *new_data*.

    Returns a metrics dict ``{"lightgbm": {"auc_roc": float}}`` or ``None``
    on failure.
    """
    from sklearn.metrics import roc_auc_score

    lgbm_path = os.path.join(model_dir, "lightgbm.joblib")
    if not os.path.exists(lgbm_path):
        return None

    try:
        from detection.model_training import validate_incremental_samples

        X_val = validate_incremental_samples(new_data, reference_feature_columns)
        y_val = new_data.loc[X_val.index, "label"]

        if len(y_val.unique()) < 2:
            logger.warning(
                "_evaluate_incremental_model: only one class in validation — skipping AUC"
            )
            return None

        lgbm = joblib.load(lgbm_path)
        probs = lgbm.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, probs))
        return {"lightgbm": {"auc_roc": auc}}
    except Exception as exc:
        logger.warning("_evaluate_incremental_model failed: %s", exc)
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = args.model_dir or config.MODEL_DIR
    session_factory = get_session_factory(get_engine(config.RISK_SCORE_DB_URL))

    # ------------------------------------------------------------------
    # Shadow evaluation mode: check a previously registered candidate
    # ------------------------------------------------------------------
    if args.check_shadow:
        shadow_state = load_shadow_state(model_dir)
        if shadow_state is None:
            logger.info("No shadow candidate registered in %s", model_dir)
            return 0
        return evaluate_shadow_candidate(
            shadow_state, model_dir, args.retrain_data_path, session_factory=session_factory
        )

    # Check for pending shadow candidate before running drift detection
    shadow_state = load_shadow_state(model_dir)
    if shadow_state is not None and not args.no_shadow:
        logger.info(
            "Pending shadow candidate detected (version %s) — evaluating before drift check",
            shadow_state.get("version_id"),
        )
        rc = evaluate_shadow_candidate(
            shadow_state, model_dir, args.retrain_data_path, session_factory=session_factory
        )
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    logger.info("Loading model metadata from %s", model_dir)
    metadata = load_model_metadata(model_dir)
    if metadata is None:
        logger.error("Cannot proceed without model_metadata.json")
        return 1

    feature_distributions = metadata.get("feature_distributions")
    if feature_distributions is None:
        logger.error(
            "model_metadata.json has no feature_distributions (re-train with updated model_training.py)"
        )
        return 1

    logger.info("Loading current feature distribution")
    if args.feature_data_path:
        current_data = pd.read_parquet(args.feature_data_path)
    else:
        current_data = get_feature_data(args.lookback_days)

    if current_data.empty:
        logger.warning("Current feature matrix is empty — cannot compute drift")
        return 0

    logger.info("Computing feature drift")
    monitor = DriftMonitor(feature_distributions)
    feature_cols = [c for c in current_data.columns if c not in {"wallet", "label"}]
    drift_report = monitor.compute(current_data[feature_cols])

    drift_dict = drift_report.to_dict()
    logger.info(
        "Drift check complete: %d/%d features drifted",
        drift_dict["n_features_drifted"],
        drift_dict["n_features_checked"],
    )

    if not drift_report.any_drift_detected:
        logger.info("No significant drift detected — no retraining needed")
        write_retrain_report(drift_dict, None, None, False, "No drift detected", None)
        return 0

    logger.info("Drift detected — starting retraining pipeline")

    # -----------------------------------------------------------------------
    # Incremental training path
    # -----------------------------------------------------------------------
    if args.incremental:
        n_new_trees = args.n_new_trees or getattr(config, "INCREMENTAL_N_NEW_TREES", 100)
        staleness_detector = IncrementalTrainingStalenessDetector(
            state_path=args.staleness_state_path,
        )

        if staleness_detector.is_stale():
            logger.info(
                "Staleness cap reached (%d/%d rounds) — forcing full retrain",
                staleness_detector.rounds,
                staleness_detector.max_rounds,
            )
            # Fall through to full-retrain path below; reset counter after
            # successful promotion.
            args.incremental = False  # trigger full-retrain branch
        else:
            # Load incremental buffer data
            if args.incremental_buffer_path:
                new_data = pd.read_parquet(args.incremental_buffer_path)
            elif args.retrain_data_path:
                # Fallback: use the labelled dataset as the incremental buffer
                new_data = load_training_data(args.retrain_data_path)
            else:
                logger.error(
                    "Incremental training requested but neither --incremental-buffer-path "
                    "nor --retrain-data-path provided"
                )
                return 1

            min_samples = getattr(config, "INCREMENTAL_BUFFER_SIZE", 10_000) // 10
            buffer = RecentDataBuffer(
                max_size=getattr(config, "INCREMENTAL_BUFFER_SIZE", 10_000),
                min_samples=min_samples,
            )
            buffer.add(new_data)

            if not buffer.is_ready(force=True):
                logger.warning(
                    "Incremental buffer has too few samples (%d < %d min) — "
                    "skipping incremental training",
                    len(buffer),
                    min_samples,
                )
                write_retrain_report(
                    drift_dict, None, None, False, "Buffer too small for incremental training", None
                )
                return 0

            buffered_data = buffer.flush()

            reference_feature_columns: list[str] = metadata.get("feature_columns", [])
            old_metrics = load_metrics(model_dir)
            incremental_candidate_dir = model_dir.rstrip("/\\") + "_incremental_new"

            success, reason = run_incremental_training(
                model_dir=model_dir,
                new_data=buffered_data,
                n_new_trees=n_new_trees,
                staleness_detector=staleness_detector,
                candidate_dir=incremental_candidate_dir,
            )

            if not success:
                logger.error("Incremental training failed: %s", reason)
                write_retrain_report(drift_dict, old_metrics, None, False, reason, None)
                shutil.rmtree(incremental_candidate_dir, ignore_errors=True)
                return 1

            # Evaluate the updated LightGBM candidate on the buffered data
            new_metrics = _evaluate_incremental_model(
                incremental_candidate_dir, buffered_data, reference_feature_columns
            )

            # Gated promotion: only `lightgbm` changed, so only its metrics
            # are evaluated by the regression gate, but the full trust-chain
            # (signature + transparency log) and compatibility checks still
            # run before anything overwrites production.
            actor, credential = model_governance.system_actor_credential()
            try:
                model_governance.promote_candidate(
                    candidate_dir=incremental_candidate_dir,
                    model_dir=model_dir,
                    actor=actor,
                    credential=credential,
                    old_metrics=old_metrics,
                    new_metrics=new_metrics,
                    reason="drift-triggered incremental LightGBM retrain",
                    model_names=["lightgbm"],
                    session_factory=session_factory,
                )
            except model_governance.PromotionError as exc:
                reason = str(exc)
                logger.warning("Incremental candidate not promoted: %s", reason)
                write_retrain_report(drift_dict, old_metrics, new_metrics, False, reason, None)
                shutil.rmtree(incremental_candidate_dir, ignore_errors=True)
                return 3

            write_retrain_report(
                drift_dict, old_metrics, new_metrics, True, "Incremental update promoted", None
            )
            shutil.rmtree(incremental_candidate_dir, ignore_errors=True)

            if staleness_detector.is_stale():
                logger.warning(
                    "Staleness cap now reached — next drift event will trigger a full retrain"
                )

            return 2

    # -----------------------------------------------------------------------
    # Full-retrain path (original behaviour, also used after staleness reset)
    # -----------------------------------------------------------------------
    if not args.retrain_data_path:
        logger.error("Drift detected but --retrain-data-path not provided")
        return 1

    old_metrics = load_metrics(model_dir)

    logger.info("Loading training data from %s", args.retrain_data_path)
    df = load_training_data(args.retrain_data_path)
    logger.info("Loaded %d labelled rows", len(df))

    logger.info("Training new ensemble models")
    training_output = train_models(df, test_size=args.test_size, random_state=args.random_state)
    results = training_output["results"]
    feature_cols = training_output.get("feature_columns", [])
    feature_hash = compute_feature_schema_hash(feature_cols) if feature_cols else None

    temp_model_dir = model_dir + "_new"
    os.makedirs(temp_model_dir, exist_ok=True)
    save_models(
        results, temp_model_dir, feature_columns=feature_cols, feature_schema_hash=feature_hash
    )
    save_training_artifacts(training_output, args.retrain_data_path, temp_model_dir)

    new_metrics = evaluate_new_model(temp_model_dir)
    if new_metrics is None:
        logger.error("Failed to evaluate new model metrics")
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        return 1

    # Feature-contract compatibility is checked up front regardless of
    # --no-shadow — a schema-incompatible candidate is rejected outright, it
    # never even reaches shadow deployment.
    candidate_metadata = load_model_metadata(temp_model_dir)
    compatible, compat_reason = True, ""
    if "feature_columns" in metadata:
        if candidate_metadata is None:
            compatible, compat_reason = False, "candidate metadata is missing"
        else:
            try:
                compatibility = validate_feature_compatibility(
                    metadata,
                    candidate_metadata,
                    reference_source=os.path.join(model_dir, "model_metadata.json"),
                    candidate_source=os.path.join(temp_model_dir, "model_metadata.json"),
                )
                if not compatibility.compatible:
                    compatible = False
                    compat_reason = " ".join(compatibility.diagnostics)
            except FeatureContractError as exc:
                compatible, compat_reason = False, str(exc)

    if not compatible:
        reason = f"Feature compatibility gate failed: {compat_reason}"
        logger.warning("%s — archiving candidate, not promoted", reason)
        archive_path = archive_current_models(model_dir)
        write_retrain_report(drift_dict, old_metrics, new_metrics, False, reason, archive_path)
        for fname in os.listdir(temp_model_dir):
            shutil.copy2(os.path.join(temp_model_dir, fname), os.path.join(archive_path, fname))
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        return 3

    # ------------------------------------------------------------------
    # --no-shadow: attempt gated promotion immediately. Otherwise: always
    # start a shadow deployment — the offline regression gate and trust
    # chain are (re-)enforced by promote_candidate at --check-shadow time,
    # once live drift/FP-rate checks have also passed.
    # ------------------------------------------------------------------
    if args.no_shadow:
        actor, credential = model_governance.system_actor_credential()
        try:
            model_governance.promote_candidate(
                candidate_dir=temp_model_dir,
                model_dir=model_dir,
                actor=actor,
                credential=credential,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                reason="drift-triggered full retrain (--no-shadow)",
                session_factory=session_factory,
            )
        except model_governance.PromotionError as exc:
            reason = str(exc)
            logger.warning("Full-retrain candidate not promoted: %s", reason)
            archive_path = archive_current_models(model_dir)
            write_retrain_report(drift_dict, old_metrics, new_metrics, False, reason, archive_path)
            for fname in os.listdir(temp_model_dir):
                shutil.copy2(os.path.join(temp_model_dir, fname), os.path.join(archive_path, fname))
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            return 3

        logger.info("New models promoted immediately (--no-shadow) to %s", model_dir)
        write_retrain_report(
            drift_dict, old_metrics, new_metrics, True, "Promoted immediately (--no-shadow)", None
        )
        shutil.rmtree(temp_model_dir, ignore_errors=True)

        # Reset staleness counter — a full retrain resets the incremental clock
        try:
            staleness_detector = IncrementalTrainingStalenessDetector(
                state_path=args.staleness_state_path
            )
            staleness_detector.reset()
            logger.info("Staleness counter reset after full retrain")
        except Exception as exc:
            logger.warning("Could not reset staleness counter: %s", exc)

        return 2

    version_id = str(uuid.uuid4())
    shadow_state = {
        "version_id": version_id,
        "candidate_dir": temp_model_dir,
        "shadow_start": datetime.now(UTC).isoformat(),
        "drift_rate": 0.0,
        "drift_events": 0,
        "total_shadow_requests": 0,
    }
    save_shadow_state(model_dir, shadow_state)
    model_governance.record_shadow_start(
        version_id, temp_model_dir, metrics=new_metrics, session_factory=session_factory
    )
    logger.info(
        "Shadow deployment started for version %s — candidate in %s. "
        "Run with --check-shadow after %d hours to evaluate promotion.",
        version_id,
        temp_model_dir,
        config.SHADOW_PERIOD_HOURS,
    )
    write_retrain_report(
        drift_dict, old_metrics, new_metrics, False, "Shadow deployment started", None
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())
# TODO(#732): Reject --lookback-days <= 0 at argparse time to prevent silent empty date ranges and false no-drift results.
