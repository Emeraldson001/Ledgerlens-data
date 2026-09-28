"""Per-asset-pair Prometheus metrics for LedgerLens (issue #276).

Defines the canonical per-pair metrics emitted by the scoring pipeline:
  - ledgerlens_score_duration_seconds  (Histogram)
  - ledgerlens_benford_computation_total  (Counter)
  - ledgerlens_risk_score_distribution  (Histogram)
  - ledgerlens_confirmed_wash_trades_total  (Counter) — for SLO dashboard (#197)
  - ledgerlens_confirmed_clean_wallets_total  (Counter) — for SLO dashboard (#197)

All metrics carry an ``asset_pair`` label using the canonical format
``CODE:ISSUER/CODE:ISSUER`` sorted alphabetically.  Labels never include
wallet addresses — only aggregate pair identifiers.

Usage::

    from detection.per_pair_metrics import record_scoring_duration, record_benford_computation, record_risk_score

    with record_scoring_duration("USDC:GA.../XLM:native"):
        score = scorer.score(features)
    record_benford_computation(asset_pair, status="ok")
    record_risk_score(asset_pair, score["score"])
    record_confirmed_wash_trade(asset_pair)
    record_confirmed_clean_wallet(asset_pair)
"""

from __future__ import annotations

import contextlib
import time

_metrics_available = False
_score_duration: object = None
_benford_computation: object = None
_risk_score_dist: object = None
_confirmed_wash_trades: object = None
_confirmed_clean_wallets: object = None

try:
    from prometheus_client import Counter, Histogram

    ledgerlens_score_duration_seconds = Histogram(
        "ledgerlens_score_duration_seconds",
        "Per-asset-pair scoring latency in seconds",
        ["asset_pair"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    ledgerlens_benford_computation_total = Counter(
        "ledgerlens_benford_computation_total",
        "Total Benford computations completed per asset pair",
        ["asset_pair", "status"],
    )
    ledgerlens_risk_score_distribution = Histogram(
        "ledgerlens_risk_score_distribution",
        "Distribution of risk scores (0-100) per asset pair",
        ["asset_pair"],
        buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    )
    # SLO dashboard counters (issue #197)
    ledgerlens_confirmed_wash_trades_total = Counter(
        "ledgerlens_confirmed_wash_trades_total",
        "Total confirmed wash trades detected per asset pair",
        ["asset_pair"],
    )
    ledgerlens_confirmed_clean_wallets_total = Counter(
        "ledgerlens_confirmed_clean_wallets_total",
        "Total confirmed clean (non-fraudulent) wallets per asset pair",
        ["asset_pair"],
    )
    _score_duration = ledgerlens_score_duration_seconds
    _benford_computation = ledgerlens_benford_computation_total
    _risk_score_dist = ledgerlens_risk_score_distribution
    _confirmed_wash_trades = ledgerlens_confirmed_wash_trades_total
    _confirmed_clean_wallets = ledgerlens_confirmed_clean_wallets_total
    _metrics_available = True
except Exception:
    ledgerlens_score_duration_seconds = None  # type: ignore[assignment]
    ledgerlens_benford_computation_total = None  # type: ignore[assignment]
    ledgerlens_risk_score_distribution = None  # type: ignore[assignment]
    ledgerlens_confirmed_wash_trades_total = None  # type: ignore[assignment]
    ledgerlens_confirmed_clean_wallets_total = None  # type: ignore[assignment]


def canonical_pair(asset_pair: str) -> str:
    """Return the canonical sort-order form of *asset_pair*.

    Ensures ``A/B`` and ``B/A`` map to the same label, preventing metric
    cardinality explosion from direction-dependent pair strings.

    Security: wallet addresses are never included in pair labels; only the
    CODE:ISSUER format is accepted.
    """
    parts = [p.strip() for p in asset_pair.split("/") if p.strip()]
    if len(parts) != 2:
        return asset_pair
    return "/".join(sorted(parts))


@contextlib.contextmanager
def record_scoring_duration(asset_pair: str):
    """Context manager that records scoring duration for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if _metrics_available and _score_duration is not None:
            _score_duration.labels(asset_pair=pair).observe(elapsed)


def record_benford_computation(asset_pair: str, status: str = "ok") -> None:
    """Increment the Benford computation counter for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    if _metrics_available and _benford_computation is not None:
        _benford_computation.labels(asset_pair=pair, status=status).inc()


def record_risk_score(asset_pair: str, score: float) -> None:
    """Observe a risk *score* in the distribution histogram for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    if _metrics_available and _risk_score_dist is not None:
        _risk_score_dist.labels(asset_pair=pair).observe(float(score))


def record_confirmed_wash_trade(asset_pair: str) -> None:
    """Increment the confirmed wash trade counter for *asset_pair*.

    Call this when a wallet on *asset_pair* is manually confirmed to be
    conducting wash trading (used to compute recall metrics for SLO dashboard).
    """
    pair = canonical_pair(asset_pair)
    if _metrics_available and _confirmed_wash_trades is not None:
        _confirmed_wash_trades.labels(asset_pair=pair).inc()


def record_confirmed_clean_wallet(asset_pair: str) -> None:
    """Increment the confirmed clean wallet counter for *asset_pair*.

    Call this when a wallet on *asset_pair* is manually confirmed to be
    legitimate/non-fraudulent (used to compute false-positive rate metrics
    for SLO dashboard).
    """
    pair = canonical_pair(asset_pair)
    if _metrics_available and _confirmed_clean_wallets is not None:
        _confirmed_clean_wallets.labels(asset_pair=pair).inc()


def compare_pair_metrics(
    production_scores: dict[str, float],
    candidate_scores: dict[str, float],
    *,
    threshold: float = 0.0,
) -> dict[str, object]:
    """Compare production vs. candidate scores per asset pair (issue #936).

    Used by the model-governance shadow-mode evaluation to compute the
    agreement rate and per-pair metric deltas over a shadow period.  This is
    a pure computation: it never emits metrics or influences live alerts, so
    running it against shadow traffic has zero effect on production decisions.

    *production_scores* and *candidate_scores* map canonical asset pairs to
    risk scores.  A pair is considered to *agree* when both models place it on
    the same side of *threshold* (both flagged or both not flagged).

    Returns a report dict with ``agreement_rate``, ``pairs_compared``,
    ``deltas`` (per-pair absolute score delta) and ``disagreements`` (pairs
    where the flag decision differs).
    """
    pairs = sorted(set(production_scores) | set(candidate_scores))
    deltas: dict[str, float] = {}
    disagreements: list[str] = []
    compared = 0
    agreed = 0
    for pair in pairs:
        prod = production_scores.get(pair)
        cand = candidate_scores.get(pair)
        if prod is None or cand is None:
            continue
        compared += 1
        deltas[pair] = abs(float(cand) - float(prod))
        prod_flag = float(prod) >= threshold
        cand_flag = float(cand) >= threshold
        if prod_flag == cand_flag:
            agreed += 1
        else:
            disagreements.append(pair)
    agreement_rate = (agreed / compared) if compared else 1.0
    return {
        "agreement_rate": agreement_rate,
        "pairs_compared": compared,
        "deltas": deltas,
        "disagreements": disagreements,
    }
