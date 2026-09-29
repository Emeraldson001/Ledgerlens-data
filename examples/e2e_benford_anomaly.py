"""Example: Benford's Law anomaly detection workflow.

Demonstrates how the BenfordEngine flags a wallet whose trade amounts are
drawn from a uniform distribution (all leading digits equally likely) versus
the expected Benford distribution (leading digit 1 should appear ~30% of the
time).

Run::

    python -m examples.e2e_benford_anomaly
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from detection.benford_engine import BenfordEngine
from examples._helpers import (
    EXAMPLE_WALLET,
    PAIR_ID,
    build_trades_df,
    make_trade,
    run_detection,
)


def _leading_digit_distribution(amounts: list[float]) -> dict[int, float]:
    """Return the observed frequency of each leading digit 1–9."""
    counts: dict[int, int] = {d: 0 for d in range(1, 10)}
    for a in amounts:
        if a <= 0:
            continue
        s = str(abs(a)).lstrip("0").replace(".", "")
        if s:
            counts[int(s[0])] += 1
    total = sum(counts.values())
    if total == 0:
        return {d: 0.0 for d in range(1, 10)}
    return {d: counts[d] / total for d, c in counts.items()}


def main() -> dict:
    """Run the Benford anomaly end-to-end example and return the result dict."""
    rng = np.random.default_rng(99)
    now = datetime.now(UTC)

    # --- Scenario A: Benford-conforming (log-normal) ---
    clean_amounts = rng.lognormal(mean=2.0, sigma=2.0, size=300).tolist()
    clean_trades = [
        make_trade(amount=a, timestamp=now - timedelta(minutes=i))
        for i, a in enumerate(clean_amounts)
    ]
    build_trades_df(clean_trades)

    engine = BenfordEngine()
    clean_benford = engine.compute_all(clean_amounts)

    print("\n=== Scenario A: Legitimate trading (log-normal amounts) ===")
    print(f"  MAD: {clean_benford['mad']:.4f}  (< 0.015 = conforming)")
    dist = _leading_digit_distribution(clean_amounts)
    print(f"  Leading-digit freq: { {d: f'{v:.2f}' for d, v in dist.items()} }")

    # --- Scenario B: Non-conforming (uniform random amounts → flat distribution) ---
    # Uniform amounts on [100, 999] make every leading digit (1–9) equally likely.
    suspicious_amounts = (rng.uniform(100, 999, size=300)).tolist()
    suspicious_trades = [
        make_trade(amount=a, timestamp=now - timedelta(minutes=i))
        for i, a in enumerate(suspicious_amounts)
    ]
    suspicious_df = build_trades_df(suspicious_trades)

    suspicious_benford = engine.compute_all(suspicious_amounts)

    print("\n=== Scenario B: Suspicious trading (uniform-random amounts) ===")
    print(f"  MAD: {suspicious_benford['mad']:.4f}  (>= 0.015 = anomalous)")
    dist_s = _leading_digit_distribution(suspicious_amounts)
    print(f"  Leading-digit freq: { {d: f'{v:.2f}' for d, v in dist_s.items()} }")

    # Run full detection on scenario B
    result = run_detection(
        suspicious_df,
        wallet=EXAMPLE_WALLET,
        pair_id=PAIR_ID,
        print_summary=True,
    )
    print(
        f"[benford-anomaly] Benford flag = {result['benford_flag']}  MAD = {suspicious_benford['mad']:.4f}"
    )
    return result


if __name__ == "__main__":
    main()
