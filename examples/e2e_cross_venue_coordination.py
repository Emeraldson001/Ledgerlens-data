"""Example: cross-venue coordination detection workflow.

Simulates a wallet that simultaneously submits identical-sized trades on
multiple asset pairs within a 30-second synchrony window — a strong
wash-trading signal detected by the cross-pair trade synchrony feature.

Run::

    python -m examples.e2e_cross_venue_coordination
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from examples._helpers import (
    EXAMPLE_WALLET,
    build_trades_df,
    make_trade,
    print_result,
)

PAIRS = [
    "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native",
    "AQUA:GBNZILSTVQZ4R7IKQDGHYGY2QXL5QOFJYQMXPKWRRM5PAV7Y4M67AQUA/XLM:native",
    "yXLM:GARDNV3Q7YGT4AKSDF25LT32YSCCW4EV22Y2TV3I2PU2MMXJTEDL5T55/XLM:native",
]


def _build_coordinated_trades(n_bursts: int = 40) -> pd.DataFrame:
    """Generate trade rows where the focal wallet fires on all pairs within a
    tight time window on every burst — high cross-pair synchrony."""
    rng = np.random.default_rng(17)
    now = datetime.now(UTC)
    trades = []
    counterparty = f"GSOCKPUPPET{'A' * 45}"[:56]

    for burst in range(n_bursts):
        base_time = now - timedelta(hours=burst * 0.5)
        amount = float(rng.choice([500.0, 1000.0, 2500.0]))
        for pair in PAIRS:
            # Trades on different pairs within 10 seconds of each other
            jitter_secs = rng.integers(0, 10)
            trades.append(
                make_trade(
                    wallet=EXAMPLE_WALLET,
                    counterparty=counterparty,
                    amount=amount,
                    timestamp=base_time + timedelta(seconds=int(jitter_secs)),
                    pair_id=pair,
                )
            )

    return build_trades_df(trades)


def main() -> dict:
    """Run the cross-venue coordination end-to-end example."""
    df = _build_coordinated_trades(n_bursts=40)

    # Compute cross-pair trade synchrony manually so the example is self-contained
    synchrony_window_secs = 30
    pairs_by_time: dict[str, list[datetime]] = {}
    focal = df[df["wallet"] == EXAMPLE_WALLET]
    for _, row in focal.iterrows():
        p = row["pair_id"]
        pairs_by_time.setdefault(p, []).append(row["timestamp"])

    synced_count = 0
    total_count = 0
    for _, row in focal.iterrows():
        ts = row["timestamp"]
        other_pairs = [p for p in PAIRS if p != row["pair_id"]]
        total_count += 1
        for other_pair in other_pairs:
            other_times = pairs_by_time.get(other_pair, [])
            window_start = ts - timedelta(seconds=synchrony_window_secs)
            window_end = ts + timedelta(seconds=synchrony_window_secs)
            if any(window_start <= t <= window_end for t in other_times):
                synced_count += 1
                break

    synchrony_score = synced_count / total_count if total_count > 0 else 0.0

    print("\n=== Cross-Venue Coordination Example ===")
    print(f"  Pairs traded        : {len(PAIRS)}")
    print("  Total trade bursts  : 40")
    print(f"  Synchrony window    : {synchrony_window_secs}s")
    print(f"  Cross-pair synchrony: {synchrony_score:.3f}  (> 0.7 → suspicious)")

    # Run detection on the first pair only (single-pair feature matrix)
    pair_df = df[df["pair_id"] == PAIRS[0]]
    try:
        from examples._helpers import run_detection

        result = run_detection(pair_df, wallet=EXAMPLE_WALLET, pair_id=PAIRS[0], print_summary=True)
    except Exception:
        # Graceful fallback when models not trained
        result = {
            "wallet": EXAMPLE_WALLET,
            "pair_id": PAIRS[0],
            "score": min(100.0, synchrony_score * 100.0),
            "benford_flag": False,
            "ml_flag": synchrony_score > 0.7,
            "benford": {},
            "features": {"cross_pair_trade_synchrony": synchrony_score},
            "shap_values": {},
            "n_trades": len(pair_df),
        }
        print_result(result, label="cross-venue (fallback)")

    print(f"[cross-venue] Synchrony={synchrony_score:.3f}  Expected high score")
    return result


if __name__ == "__main__":
    main()
