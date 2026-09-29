"""CLI for querying the current differential privacy budget status.

Usage::

    python scripts/query_privacy_budget.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from detection.privacy.budget_tracker import DPBudgetTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query LedgerLens DP privacy budget status")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument(
        "--total-epsilon",
        type=float,
        default=None,
        help="Override total epsilon budget (default: from config/env)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help="Override alert threshold epsilon (default: from config/env)",
    )
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to start filtering budget events from (default: all available)",
    )
    parser.add_argument(
        "--until",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to stop filtering budget events at (default: all available)",
    )
    args = parser.parse_args(argv)

    tracker = DPBudgetTracker(
        total_epsilon=args.total_epsilon,
        alert_threshold_epsilon=args.alert_threshold,
    )
    status = tracker.status()

    # Filter events by date range if --since/--until provided
    if args.since or args.until:
        filtered_events = []
        for event in status["events"]:
            if event["created_at"] is None:
                continue
            event_dt = datetime.fromisoformat(event["created_at"])
            if args.since and event_dt < args.since:
                continue
            if args.until and event_dt > args.until:
                continue
            filtered_events.append(event)
        status["events"] = filtered_events

    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    print("DP Budget Status")
    print(f"  Total epsilon:       {status['total_epsilon']:.4f}")
    print(f"  Cumulative epsilon:  {status['cumulative_epsilon']:.4f}")
    print(f"  Remaining epsilon:   {status['remaining_epsilon']:.4f}")
    print(f"  Alert threshold:     {status['alert_threshold']:.4f}")
    print(f"  Budget exhausted:    {status['budget_exhausted']}")
    print(f"  Events recorded:     {len(status['events'])}")

    if status["events"]:
        print()
        print(f"  {'ID':<5} {'Kind':<10} {'Epsilon':<10} {'Cumulative':<12} {'Version/Type'}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*12} {'-'*20}")
        for e in status["events"][-20:]:
            label = e.get("model_version") or e.get("query_type") or ""
            print(
                f"  {e['id']:<5} {e['kind']:<10} {e['epsilon']:<10.4f} "
                f"{e['cumulative_epsilon']:<12.4f} {label}"
            )

    if status["budget_exhausted"]:
        print(
            "\nWARNING: Privacy budget is exhausted — training/inference will violate guarantees."
        )
        return 2
    if status["remaining_epsilon"] < status["alert_threshold"]:
        print(
            f"\nWARNING: Remaining epsilon ({status['remaining_epsilon']:.4f}) is below alert threshold."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
