"""Full LedgerLens detection pipeline entry point.

Usage:
    python run_pipeline.py --since 2024-01-01

Pipeline stages:
    1. Load historical trades per asset pair into a dict keyed by pair_id
    2. For each pair:
       a. Load order-book events (optional, skipped with ``--no-orderbook``)
       b. Load account activity and build the wallet funding graph (optional,
          skipped with ``--no-graph``)
       c. Build the per-wallet feature matrix
       d. Score each wallet with the trained ensemble (model_inference)
       e. Persist one RiskScore record per (wallet, pair_id) and optionally
          submit flagged wallets on-chain

Stage 2d requires trained models in `config.MODEL_DIR` — run
`detection/model_training.py` against a labelled dataset first. Until
models are trained, this script falls back to reporting Benford-only flags
(and persistence is skipped, since the `RiskScore` shape isn't available).

Checkpoint / resume
--------------------
Each asset pair is an independent unit of work (its own Horizon fetch,
feature build, and scoring pass). Pass ``--checkpoint-file`` to make a run
resumable: pairs already recorded as complete are skipped on the next
invocation, and a pair that raises is recorded as failed — logged and
retried on the next run — instead of aborting every other pair. Without
``--checkpoint-file`` the pipeline behaves exactly as before: a failure in
any pair propagates and aborts the run. See docs/checkpointing.md.
"""

import argparse
from datetime import UTC, datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset

from config import config
from config.contracts import validate_mode
from detection.feature_engineering import build_feature_matrix
from detection.risk_score_store import RiskScoreStore
from detection.wallet_graph import build_funding_graph, detect_wash_trading_rings, ring_id_map
from ingestion.account_activity_loader import load_accounts_activity
from ingestion.historical_loader import (
    load_pair_to_dataframe,
)
from ingestion.orderbook_loader import load_accounts_orderbook_events
from utils.correlation import PipelineStage, correlation_context, generate_correlation_id
from utils.logging import get_logger

logger = get_logger(__name__)


def _iso_date(value: str) -> datetime:
    """Argparse ``type`` for ``--since``: parse an ISO 8601 date/datetime string.

    Raises ``argparse.ArgumentTypeError`` (caught by argparse at parse time,
    before any network calls) with a clear, actionable message on bad input.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --since value {value!r}: expected an ISO 8601 date or "
            "datetime, e.g. 2024-01-01 or 2024-01-01T00:00:00"
        ) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LedgerLens detection pipeline")
    parser.add_argument(
        "--since",
        type=_iso_date,
        default=None,
        help="ISO date to start loading historical trades from (default: all available)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing scored wallets to RISK_SCORE_DB_URL",
    )
    parser.add_argument(
        "--no-orderbook",
        action="store_true",
        help="Skip loading order-book events (faster, but order_cancellation_rate stays 0)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip loading account activity and building the funding graph (faster, but "
        "funding_source_similarity and network_centrality stay 0)",
    )
    parser.add_argument(
        "--submit-onchain",
        action="store_true",
        help="Submit flagged wallets' RiskScore to the ledgerlens-score contract",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all pipeline stages but skip all writes (DB persist and on-chain submission).",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Path to a JSON checkpoint file enabling resumable per-pair processing. "
        "When set, pairs already recorded as complete are skipped and a pair that "
        "raises is recorded as failed (retried on the next --checkpoint-file run) "
        "instead of aborting the whole pipeline. Ignored with --dry-run.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing --checkpoint-file and start over. No effect without "
        "--checkpoint-file.",
    )
    return parser.parse_args()


def _build_pair_specs(xlm: SdkAsset) -> list[tuple[str, SdkAsset]]:
    """Return the ordered ``(pair_id, base_asset)`` units of work for this run."""
    specs = []
    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = xlm if issuer == "native" else SdkAsset(code, issuer)
        if asset == xlm:
            continue
        specs.append((f"{code}:{issuer}/XLM:native", asset))
    return specs


def _process_pair(
    pair_id: str,
    asset: SdkAsset,
    xlm: SdkAsset,
    args: argparse.Namespace,
    store: RiskScoreStore | None,
) -> pd.DataFrame:
    """Run stages 1–2e for a single asset pair and return its flagged-wallet rows.

    Raised exceptions propagate to the caller, which decides (based on whether
    checkpointing is enabled) whether to abort the whole run or record this
    pair as failed and continue with the rest.
    """
    logger.info("[pair=%s] Loading trades", pair_id)
    trades_df = load_pair_to_dataframe(asset, xlm, start_time=args.since)
    logger.info("[pair=%s] Loaded %d trades", pair_id, len(trades_df))

    if trades_df.empty:
        logger.info("[pair=%s] No trades — skipping", pair_id)
        return pd.DataFrame()

    wallets = list(pd.unique(trades_df[["base_account", "counter_account"]].values.ravel()))

    # --- Stage 2a: order-book events ---
    orderbook_events = None
    if not args.no_orderbook:
        logger.info("[pair=%s] Loading order-book events", pair_id)
        orderbook_events = load_accounts_orderbook_events(wallets)
        logger.info("[pair=%s] Loaded %d order-book events", pair_id, len(orderbook_events))

    # --- Stage 2b: funding graph ---
    funding_graph = None
    if not args.no_graph:
        logger.info("[pair=%s] Loading account activity and building funding graph", pair_id)
        activities = load_accounts_activity(wallets)
        funding_graph = build_funding_graph(activities)
        logger.info(
            "[pair=%s] Funding graph: %d nodes, %d edges",
            pair_id,
            funding_graph.number_of_nodes(),
            funding_graph.number_of_edges(),
        )

    # --- Stage 2c: feature matrix ---
    logger.info("[pair=%s] Building feature matrix", pair_id)
    feature_matrix = build_feature_matrix(
        trades_df, orderbook_events=orderbook_events, funding_graph=funding_graph
    )
    logger.info("[pair=%s] Built features for %d wallets", pair_id, len(feature_matrix))

    # --- Stage 2d: scoring ---
    logger.info("[pair=%s] Scoring wallets", pair_id)
    try:
        from detection.model_inference import RiskScorer

        scorer = RiskScorer()
        scored = scorer.score_matrix(feature_matrix)
    except (RuntimeError, ImportError) as exc:
        logger.warning("[pair=%s] Skipping ML scoring: %s", pair_id, exc)
        logger.warning("[pair=%s] Falling back to Benford-only flags", pair_id)
        mad_cols = [c for c in feature_matrix.columns if c.startswith("benford_mad_")]
        scored = feature_matrix[["wallet"] + mad_cols].copy()
        scored["benford_flag"] = (scored[mad_cols] > 0.015).any(axis=1)

    # --- Stage 2d.5: wash-trading ring id per wallet ---
    if funding_graph is not None and funding_graph.number_of_nodes() > 0:
        community_map = detect_wash_trading_rings(funding_graph)
        rid_map = ring_id_map(community_map, funding_graph)
    else:
        rid_map = {}
    scored["ring_id"] = [rid_map.get(w) for w in scored["wallet"]]

    # --- Stage 2e: persist + flag ---
    if "score" in scored:
        flagged = scored[scored["score"] >= config.RISK_SCORE_FLAG_THRESHOLD]

        if store is not None:
            for _, row in scored.iterrows():
                store.upsert(
                    wallet=row["wallet"],
                    asset_pair=pair_id,
                    risk_score={
                        "score": row["score"],
                        "benford_flag": row["benford_flag"],
                        "ml_flag": row["ml_flag"],
                        "confidence": row["confidence"],
                        "ring_id": row.get("ring_id"),
                    },
                    finality="final",
                )
            logger.info("[pair=%s] Persisted %d scored wallets", pair_id, len(scored))
    else:
        flagged = scored[scored["benford_flag"]]

    logger.info("[pair=%s] Flagged wallets (%d):\n%s", pair_id, len(flagged), flagged)

    if args.submit_onchain:
        if args.dry_run:
            logger.warning("[pair=%s] [DRY RUN] Skipping on-chain submission", pair_id)
        elif "score" not in scored:
            logger.warning(
                "[pair=%s] Skipping on-chain submission: no ML scores available", pair_id
            )
        else:
            submit_flagged_onchain(flagged, pair_id)

    return flagged


def main() -> None:
    args = parse_args()
    # Validated *after* parsing --submit-onchain: calling this before parsing
    # args always validated the non-onchain contract, silently skipping the
    # LEDGERLENS_CONTRACT_ID / LEDGERLENS_SUBMITTER_SECRET checks even when
    # --submit-onchain was passed.
    validate_mode("pipeline_onchain" if args.submit_onchain else "pipeline")

    pipeline_run_id = generate_correlation_id()
    with correlation_context(
        pipeline_run_id,
        stage=PipelineStage.DETECTION,
        extra={"pipeline": "batch", "dry_run": args.dry_run},
    ):
        if args.dry_run:
            logger.info("[DRY RUN] No data will be written.")

        xlm = SdkAsset.native()

        # --- Stage 1: load trades per pair ---
        logger.info("[ingestion] Loading trades per pair: %s", config.WATCHED_ASSET_PAIRS)
        pairs_df: dict[str, pd.DataFrame] = {}
        for code, issuer in config.WATCHED_ASSET_PAIRS:
            asset = xlm if issuer == "native" else SdkAsset(code, issuer)
            if asset == xlm:
                continue
            pair_id = f"{code}:{issuer}/XLM:native"
            with correlation_context(
                pipeline_run_id,
                stage=PipelineStage.INGESTION,
                pair_id=pair_id,
            ):
                logger.info("Loading pair %s", pair_id)
                pairs_df[pair_id] = load_pair_to_dataframe(asset, xlm, start_time=args.since)
                logger.info("Loaded %d trades for %s", len(pairs_df[pair_id]), pair_id)

        if not pairs_df:
            logger.warning("No pairs configured or all pairs skipped.")
            return

        all_flagged: list[pd.DataFrame] = []
        store = RiskScoreStore() if not args.no_persist and not args.dry_run else None

        for pair_id, trades_df in pairs_df.items():
            with correlation_context(
                pipeline_run_id,
                stage=PipelineStage.DETECTION,
                pair_id=pair_id,
            ):
                logger.info("Processing pair %s", pair_id)

                if trades_df.empty:
                    logger.info("No trades — skipping")
                    continue

                wallets = list(
                    pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())
                )

                # --- Stage 2a: order-book events ---
                orderbook_events = None
                if not args.no_orderbook:
                    logger.info("Loading order-book events")
                    orderbook_events = load_accounts_orderbook_events(wallets)
                    logger.info("Loaded %d order-book events", len(orderbook_events))

                # --- Stage 2b: funding graph ---
                funding_graph = None
                if not args.no_graph:
                    logger.info("Loading account activity and building funding graph")
                    activities = load_accounts_activity(wallets)
                    funding_graph = build_funding_graph(activities)
                    logger.info(
                        "Funding graph: %d nodes, %d edges",
                        funding_graph.number_of_nodes(),
                        funding_graph.number_of_edges(),
                    )

                # --- Stage 2c: feature matrix ---
                logger.info("Building feature matrix")
                feature_matrix = build_feature_matrix(
                    trades_df, orderbook_events=orderbook_events, funding_graph=funding_graph
                )
                logger.info("Built features for %d wallets", len(feature_matrix))

                # --- Stage 2d: scoring ---
                logger.info("Scoring wallets")
                try:
                    from detection.model_inference import RiskScorer

                    scorer = RiskScorer()
                    scored = scorer.score_matrix(feature_matrix)
                except (RuntimeError, ImportError) as exc:
                    logger.warning("Skipping ML scoring: %s", exc)
                    logger.warning("Falling back to Benford-only flags")
                    mad_cols = [c for c in feature_matrix.columns if c.startswith("benford_mad_")]
                    scored = feature_matrix[["wallet"] + mad_cols].copy()
                    scored["benford_flag"] = (scored[mad_cols] > 0.015).any(axis=1)

                # --- Stage 2d.5: wash-trading ring id per wallet ---
                if funding_graph is not None and funding_graph.number_of_nodes() > 0:
                    community_map = detect_wash_trading_rings(funding_graph)
                    rid_map = ring_id_map(community_map, funding_graph)
                else:
                    rid_map = {}
                scored["ring_id"] = [rid_map.get(w) for w in scored["wallet"]]

                # --- Stage 2e: persist + flag ---
                if "score" in scored:
                    flagged = scored[scored["score"] >= config.RISK_SCORE_FLAG_THRESHOLD]

                    if store is not None:
                        with correlation_context(
                            pipeline_run_id,
                            stage=PipelineStage.PERSISTENCE,
                            pair_id=pair_id,
                        ):
                            for _, row in scored.iterrows():
                                store.upsert(
                                    wallet=row["wallet"],
                                    asset_pair=pair_id,
                                    risk_score={
                                        "score": row["score"],
                                        "benford_flag": row["benford_flag"],
                                        "ml_flag": row["ml_flag"],
                                        "confidence": row["confidence"],
                                        "ring_id": row.get("ring_id"),
                                    },
                                    finality="final",
                                )
                            logger.info("Persisted %d scored wallets", len(scored))
                else:
                    flagged = scored[scored["benford_flag"]]

                logger.info("Flagged wallets (%d):\n%s", len(flagged), flagged)
                all_flagged.append(flagged)

                if args.submit_onchain:
                    if args.dry_run:
                        logger.warning("[DRY RUN] Skipping on-chain submission")
                    elif "score" not in scored:
                        logger.warning("Skipping on-chain submission: no ML scores available")
                    else:
                        with correlation_context(
                            pipeline_run_id,
                            stage=PipelineStage.ONCHAIN,
                            pair_id=pair_id,
                        ):
                            submit_flagged_onchain(flagged, pair_id)

        combined_flagged = (
            pd.concat(all_flagged, ignore_index=True) if all_flagged else pd.DataFrame()
        )
        logger.info("Total flagged wallets across all pairs: %d", len(combined_flagged))


def submit_flagged_onchain(flagged: pd.DataFrame, pair_id: str) -> None:
    """Submit each flagged wallet's `RiskScore` to the `ledgerlens-score` contract."""
    from integrations.contract_client import LedgerLensContractClient

    client = LedgerLensContractClient()
    timestamp = int(datetime.now(UTC).timestamp())

    for _, row in flagged.iterrows():
        risk_score = {
            "score": int(row["score"]),
            "benford_flag": bool(row["benford_flag"]),
            "ml_flag": bool(row["ml_flag"]),
            "timestamp": timestamp,
            "confidence": int(row["confidence"]),
        }
        client.submit_score(wallet=row["wallet"], asset_pair=pair_id, risk_score=risk_score)

    logger.info("      Submitted %d RiskScores on-chain for %s", len(flagged), pair_id)


if __name__ == "__main__":
    main()
