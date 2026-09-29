"""Scenario Replay Tooling — Issue #536.

Replays a stored historical anomaly scenario through the current detection
pipeline so engineers can:
  - Validate that a known wash-trade case is still detected after model updates
  - Reproduce false-positive / false-negative cases for debugging
  - Benchmark scoring latency on a fixed, reproducible input

Scenarios are managed by ``detection.scenario_store.ScenarioStore``.  Each
scenario is a frozen snapshot of the trades, computed features, and original
risk score for a specific wallet/pair anomaly.

Usage::

    # Save the current detection result for a wallet as a scenario
    python -m scripts.replay_scenario save \\
        --wallet GABC... \\
        --pair "USDC:GA5Z.../XLM:native" \\
        --label 1 \\
        --campaign-id campaign_001 \\
        --notes "Known wash-trade ring, January 2024"

    # Replay a stored scenario through the current pipeline
    python -m scripts.replay_scenario replay \\
        --scenario-id sc_3f2a1b8c9e4d5f6a

    # List all stored scenarios
    python -m scripts.replay_scenario list

    # Replay all scenarios and produce a regression summary
    python -m scripts.replay_scenario regression \\
        --output reports/scenario_regression.json

Exit codes:
    0 — all replayed scenarios produced the expected classification
    1 — one or more scenario regressions detected
    2 — fatal error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from detection.scenario_store import ScenarioStore
from utils.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_scorer() -> Any:
    """Lazy-import RiskScorer so the module is importable without trained models."""
    from detection.model_inference import RiskScorer

    return RiskScorer()


def _score_features(scorer: Any, features: dict[str, Any]) -> dict[str, Any]:
    """Run features through the scorer and return a risk score dict."""
    import pandas as pd

    row = pd.DataFrame([features])
    return scorer.score(row)


def _build_features_from_trades(
    wallet: str,
    pair_id: str,
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute features from raw trades (best-effort; returns empty dict if unavailable)."""
    try:
        import pandas as pd

        from detection.feature_engineering import build_feature_matrix

        df = pd.DataFrame(trades)
        if df.empty:
            return {}
        feat_df = build_feature_matrix(df)
        if feat_df.empty:
            return {}
        # Return features for the requested wallet/pair if present
        mask = pd.Series([True] * len(feat_df))
        if "wallet" in feat_df.columns:
            mask &= feat_df["wallet"] == wallet
        if "pair_id" in feat_df.columns:
            mask &= feat_df["pair_id"] == pair_id
        row = feat_df[mask]
        if row.empty:
            row = feat_df.iloc[:1]
        return row.iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature computation from trades failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_save(args: argparse.Namespace, store: ScenarioStore) -> int:
    """Score a wallet/pair NOW and save the result as a scenario."""
    try:
        scorer = _load_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot load RiskScorer: %s", exc)
        return 2

    # Optionally load trades from a JSON file
    trades: list[dict[str, Any]] = []
    if args.trades_file:
        try:
            trades = json.loads(Path(args.trades_file).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Cannot read trades file %s: %s", args.trades_file, exc)
            return 2

    # Compute features (either from trades or from a features file)
    features: dict[str, Any] = {}
    if args.features_file:
        try:
            features = json.loads(Path(args.features_file).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Cannot read features file %s: %s", args.features_file, exc)
            return 2
    elif trades:
        features = _build_features_from_trades(args.wallet, args.pair, trades)

    if not features:
        logger.error("No features available — provide --trades-file or --features-file")
        return 2

    # Score
    try:
        risk_score = _score_features(scorer, features)
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc)
        return 2

    sid = store.save(
        wallet=args.wallet,
        pair_id=args.pair,
        trades=trades,
        features=features,
        risk_score=risk_score,
        label=args.label,
        campaign_id=args.campaign_id,
        notes=args.notes or "",
        scenario_id=args.scenario_id,
    )
    print(f"[replay] Scenario saved: {sid}")
    return 0


def cmd_replay(args: argparse.Namespace, store: ScenarioStore) -> int:
    """Replay a stored scenario and compare the new score to the original."""
    try:
        scenario = store.load(args.scenario_id)
    except FileNotFoundError:
        logger.error("Scenario not found: %s", args.scenario_id)
        return 2

    try:
        scorer = _load_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot load RiskScorer: %s", exc)
        return 2

    features = scenario.get("features", {})
    if not features:
        logger.error("Scenario %s has no stored features", args.scenario_id)
        return 2

    t0 = time.perf_counter()
    try:
        new_score = _score_features(scorer, features)
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc)
        return 2
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    orig_score = scenario.get("risk_score", {})
    label = scenario.get("label")

    # Classification match check
    orig_flagged = orig_score.get("score", 0) >= 70
    new_flagged = new_score.get("score", 0) >= 70
    label_match = (label is None) or (bool(new_flagged) == bool(label))
    regression = (orig_flagged != new_flagged) or not label_match

    result = {
        "scenario_id": args.scenario_id,
        "wallet": scenario.get("wallet"),
        "pair_id": scenario.get("pair_id"),
        "campaign_id": scenario.get("campaign_id"),
        "label": label,
        "original_score": orig_score.get("score"),
        "new_score": new_score.get("score"),
        "originally_flagged": orig_flagged,
        "now_flagged": new_flagged,
        "regression": regression,
        "latency_ms": latency_ms,
        "replayed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    status = "REGRESSION" if regression else "OK"
    print(
        f"[replay] {args.scenario_id} | orig={orig_score.get('score')} "
        f"new={new_score.get('score')} | {status} | {latency_ms}ms"
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))

    return 1 if regression else 0


def cmd_list(args: argparse.Namespace, store: ScenarioStore) -> int:
    """List all stored scenarios."""
    scenarios = store.list_scenarios(
        campaign_id=args.campaign_id if hasattr(args, "campaign_id") else None,
        label=args.label if hasattr(args, "label") else None,
    )
    if not scenarios:
        print("[replay] No scenarios found.")
        return 0

    print(f"[replay] {len(scenarios)} scenario(s) found:\n")
    print(f"{'ID':<22} {'Wallet':<20} {'Pair':<28} {'Label':>5} {'Score':>6} {'Campaign'}")
    print("-" * 100)
    for s in scenarios:
        wallet_short = (s.get("wallet") or "")[:18]
        pair_short = (s.get("pair_id") or "")[:26]
        print(
            f"{s['scenario_id']:<22} {wallet_short:<20} {pair_short:<28} "
            f"{str(s.get('label') or '-'):>5} {str(s.get('score') or '-'):>6} "
            f"{s.get('campaign_id') or '-'}"
        )
    return 0


def cmd_regression(args: argparse.Namespace, store: ScenarioStore) -> int:
    """Replay ALL stored scenarios and produce a regression summary report."""
    scenarios = store.list_scenarios()
    if not scenarios:
        print("[replay] No scenarios to replay.")
        return 0

    try:
        scorer = _load_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot load RiskScorer: %s", exc)
        return 2

    results: list[dict[str, Any]] = []
    regression_count = 0

    for meta in scenarios:
        sid = meta["scenario_id"]
        try:
            scenario = store.load(sid)
        except FileNotFoundError:
            logger.warning("Scenario file missing: %s", sid)
            continue

        features = scenario.get("features", {})
        if not features:
            logger.warning("No features for scenario %s — skipping", sid)
            continue

        t0 = time.perf_counter()
        try:
            new_score = _score_features(scorer, features)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scoring failed for %s: %s", sid, exc)
            results.append({"scenario_id": sid, "error": str(exc)})
            continue
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        orig_score = scenario.get("risk_score", {})
        label = scenario.get("label")
        orig_flagged = (orig_score.get("score") or 0) >= 70
        new_flagged = (new_score.get("score") or 0) >= 70
        label_match = (label is None) or (bool(new_flagged) == bool(label))
        regression = (orig_flagged != new_flagged) or not label_match

        if regression:
            regression_count += 1

        results.append(
            {
                "scenario_id": sid,
                "wallet": scenario.get("wallet"),
                "campaign_id": scenario.get("campaign_id"),
                "label": label,
                "original_score": orig_score.get("score"),
                "new_score": new_score.get("score"),
                "originally_flagged": orig_flagged,
                "now_flagged": new_flagged,
                "regression": regression,
                "latency_ms": latency_ms,
            }
        )

        status = "REGRESSION" if regression else "ok"
        print(
            f"  {sid} | orig={orig_score.get('score')} new={new_score.get('score')} " f"| {status}"
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "total_scenarios": len(results),
        "regressions": regression_count,
        "pass_rate": round((len(results) - regression_count) / len(results), 4) if results else 1.0,
        "results": results,
    }

    out_path = Path(args.output) if args.output else REPORTS_DIR / "scenario_regression.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(
        f"\n[replay] Regression summary: {regression_count}/{len(results)} regressions "
        f"| report → {out_path}"
    )

    return 1 if regression_count > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scenario replay tooling for LedgerLens historical anomaly cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--scenarios-dir",
        default="data/scenarios",
        help="Directory where scenarios are stored (default: data/scenarios).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- save ---
    s = sub.add_parser("save", help="Score a wallet/pair and save the result as a scenario.")
    s.add_argument("--wallet", required=True, help="Stellar wallet address.")
    s.add_argument("--pair", required=True, help="Asset pair identifier (CODE:ISSUER/CODE:ISSUER).")
    s.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        default=None,
        help="Ground-truth label (1=wash-trade, 0=clean).",
    )
    s.add_argument("--campaign-id", default=None, help="Campaign or event identifier.")
    s.add_argument("--notes", default="", help="Free-text notes.")
    s.add_argument(
        "--trades-file", default=None, help="Path to a JSON file containing the trade records."
    )
    s.add_argument(
        "--features-file",
        default=None,
        help="Path to a JSON file containing pre-computed features.",
    )
    s.add_argument("--scenario-id", default=None, help="Override the auto-generated scenario ID.")

    # --- replay ---
    r = sub.add_parser("replay", help="Replay a stored scenario through the current pipeline.")
    r.add_argument("--scenario-id", required=True, help="Scenario ID to replay.")
    r.add_argument("--output", default=None, help="Write the replay result to this JSON file.")

    # --- list ---
    ll = sub.add_parser("list", help="List all stored scenarios.")
    ll.add_argument("--campaign-id", default=None, help="Filter by campaign ID.")
    ll.add_argument("--label", type=int, choices=[0, 1], default=None, help="Filter by label.")

    # --- regression ---
    rg = sub.add_parser("regression", help="Replay ALL scenarios and produce a regression report.")
    rg.add_argument(
        "--output",
        default=None,
        help="Path for the regression JSON report (default: reports/scenario_regression.json).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    store = ScenarioStore(scenarios_dir=args.scenarios_dir)

    if args.command == "save":
        return cmd_save(args, store)
    elif args.command == "replay":
        return cmd_replay(args, store)
    elif args.command == "list":
        return cmd_list(args, store)
    elif args.command == "regression":
        return cmd_regression(args, store)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
