"""Example: clean (legitimate) trading workflow.

Generates 200 synthetic trade amounts drawn from a log-normal distribution
that closely follows Benford's Law — mimicking genuine market-maker activity.
Runs the full detection pipeline and shows the resulting (expected low) risk
score.

Run::

    python -m examples.e2e_clean_trading
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from examples._helpers import (
    EXAMPLE_WALLET,
    PAIR_ID,
    build_trades_df,
    make_trade,
    run_detection,
)


def main() -> dict:
    """Run the clean-trading end-to-end example and return the result dict."""
    rng = np.random.default_rng(42)

    # Log-normal amounts → naturally conforms to Benford's Law
    raw_amounts = rng.lognormal(mean=3.5, sigma=1.8, size=200).tolist()

    now = datetime.now(UTC)
    trades = []
    for i, amt in enumerate(raw_amounts):
        counterparty = f"GC{str(i % 20).zfill(4)}{'A' * 50}"[:56]
        trades.append(
            make_trade(
                wallet=EXAMPLE_WALLET,
                counterparty=counterparty,
                amount=amt,
                timestamp=now - timedelta(minutes=i * 3),
                pair_id=PAIR_ID,
            )
        )

    df = build_trades_df(trades)
    result = run_detection(df, wallet=EXAMPLE_WALLET, pair_id=PAIR_ID, print_summary=True)
    print(f"[clean-trading] Expected: score < 40  Got: {result['score']:.1f}")
    return result


if __name__ == "__main__":
    main()
