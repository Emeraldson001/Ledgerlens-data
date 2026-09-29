"""Example: wash-trading ring detection workflow.

Simulates a 4-wallet wash-trading ring where all wallets trade fixed round
amounts with each other in a cycle.  Fixed lot sizes violate Benford's Law
(leading-digit distribution becomes degenerate) and the round-trip pattern
triggers ML features.

Run::

    python -m examples.e2e_wash_trading_ring
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from examples._helpers import (
    PAIR_ID,
    build_trades_df,
    make_trade,
    run_detection,
)

RING_WALLETS = [f"GRING{str(i).zfill(51)}"[:56] for i in range(1, 5)]
# Fixed lot sizes — non-Benford by design
_LOT_SIZES = [1000.0, 2000.0, 5000.0, 10000.0]


def main() -> dict:
    """Run the wash-trading-ring end-to-end example and return the result dict."""
    rng = np.random.default_rng(7)
    now = datetime.now(UTC)
    trades = []

    # 50 round-trips per wallet pair in the ring
    for cycle in range(50):
        for idx, wallet in enumerate(RING_WALLETS):
            counterparty = RING_WALLETS[(idx + 1) % len(RING_WALLETS)]
            amount = float(rng.choice(_LOT_SIZES))
            # Small jitter to avoid identical timestamps
            offset_minutes = cycle * 5 + idx
            trades.append(
                make_trade(
                    wallet=wallet,
                    counterparty=counterparty,
                    amount=amount,
                    timestamp=now - timedelta(minutes=offset_minutes),
                    pair_id=PAIR_ID,
                )
            )
            # Immediate round-trip back
            trades.append(
                make_trade(
                    wallet=counterparty,
                    counterparty=wallet,
                    amount=amount,
                    timestamp=now - timedelta(minutes=offset_minutes, seconds=30),
                    pair_id=PAIR_ID,
                )
            )

    df = build_trades_df(trades)

    # Analyse the first ring member
    focal_wallet = RING_WALLETS[0]
    wallet_df = df[df["wallet"] == focal_wallet]
    result = run_detection(wallet_df, wallet=focal_wallet, pair_id=PAIR_ID, print_summary=True)
    print(f"[wash-ring] Expected: score > 60  Got: {result['score']:.1f}")
    return result


if __name__ == "__main__":
    main()
