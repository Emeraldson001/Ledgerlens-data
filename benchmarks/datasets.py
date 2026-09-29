"""
benchmarks/datasets.py — Reproducible benchmark dataset factory.

Each dataset covers a distinct detector-performance scenario:

  benford_baseline        Clean Benford-conforming trades  (all legitimate)
  benford_wash_trades     Mixed legitimate + wash trades   (wash ≈ 25%)
  round_number_wash       Round-amount wash-trade pattern
  high_frequency_ring     High-frequency ring activity (5 sock-puppet wallets)
  sparse_low_volume       Thin-market, low-volume pair     (stress-test cold-start)
  cross_pair_coordination Multi-pair coordinated wash trades

All datasets are generated deterministically via a fixed random seed so CI
results are reproducible across machines and Python versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import ClassVar

import numpy as np
import pandas as pd

from benchmarks.contracts import BenchmarkDataset

logger = logging.getLogger(__name__)

# Master seed — change this only when deliberately regenerating all datasets
_MASTER_SEED: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(extra_seed: int = 0) -> np.random.Generator:
    """Return a seeded Generator for reproducible draws."""
    return np.random.default_rng(_MASTER_SEED + extra_seed)


def _make_timestamps(n: int, rng: np.random.Generator) -> pd.Series:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    offsets = np.cumsum(rng.exponential(scale=30.0, size=n))  # seconds between trades
    return pd.Series([base + pd.Timedelta(seconds=float(s)) for s in offsets])


def _benford_amounts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Amounts that approximate Benford's Law (lognormal draws)."""
    return rng.lognormal(mean=3.0, sigma=2.5, size=n)


def _round_amounts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Round-number amounts — violates Benford's Law."""
    base = rng.choice([10, 50, 100, 500, 1000, 5000], size=n)
    return base.astype(float)


def _wallet_ids(n: int, prefix: str = "G", pool_size: int = 50) -> list[str]:
    return [f"{prefix}{i % pool_size:04d}" for i in range(n)]


def _build_base_frame(
    n: int,
    amounts: np.ndarray,
    rng: np.random.Generator,
    wallet_pool: int = 50,
    pair: str = "USDC:GA001/XLM:native",
) -> pd.DataFrame:
    wallets = [f"G{i % wallet_pool:04d}" for i in range(n)]
    return pd.DataFrame(
        {
            "wallet_id": wallets,
            "asset_pair": pair,
            "amount": amounts,
            "timestamp": _make_timestamps(n, rng),
        }
    )


# ---------------------------------------------------------------------------
# Individual dataset builders
# ---------------------------------------------------------------------------


def _build_benford_baseline() -> BenchmarkDataset:
    """All-legitimate trades conforming to Benford's Law.  No wash trades."""
    rng = _rng(0)
    n = 500
    amounts = _benford_amounts(n, rng)
    df = _build_base_frame(n, amounts, rng)
    labels = pd.Series([False] * n, dtype=bool)
    return BenchmarkDataset(
        name="benford_baseline",
        description=(
            "500 legitimate trades with lognormal amounts that conform to Benford's Law. "
            "A well-calibrated detector should flag ≈ 0 wallets."
        ),
        trades=df,
        labels=labels,
        metadata={"expected_positive_rate": 0.0, "difficulty": "easy"},
    )


def _build_benford_wash_trades() -> BenchmarkDataset:
    """25% wash trades injected into a Benford-conforming baseline."""
    rng = _rng(1)
    n_clean = 375
    n_wash = 125  # 25%

    clean_amounts = _benford_amounts(n_clean, rng)
    wash_amounts = _round_amounts(n_wash, rng)

    clean_df = _build_base_frame(n_clean, clean_amounts, rng, wallet_pool=50)
    wash_df = _build_base_frame(
        n_wash,
        wash_amounts,
        rng,
        wallet_pool=5,  # few wallets → high counterparty concentration
        pair="USDC:GA001/XLM:native",
    )
    wash_df["wallet_id"] = [f"WASH{i % 5:03d}" for i in range(n_wash)]

    df = pd.concat([clean_df, wash_df], ignore_index=True)
    labels = pd.Series([False] * n_clean + [True] * n_wash, dtype=bool)
    return BenchmarkDataset(
        name="benford_wash_trades",
        description=(
            "375 legitimate trades mixed with 125 round-amount wash trades (25% positive). "
            "Tests basic Benford + counterparty concentration signals."
        ),
        trades=df,
        labels=labels,
        metadata={"expected_positive_rate": 0.25, "difficulty": "medium"},
    )


def _build_round_number_wash() -> BenchmarkDataset:
    """Pure round-number wash trades — extreme Benford deviation."""
    rng = _rng(2)
    n = 300
    amounts = _round_amounts(n, rng)
    df = _build_base_frame(n, amounts, rng, wallet_pool=3)
    df["wallet_id"] = [f"ROUND{i % 3:02d}" for i in range(n)]
    labels = pd.Series([True] * n, dtype=bool)
    return BenchmarkDataset(
        name="round_number_wash",
        description=(
            "300 wash trades all using round amounts (10/50/100/500/1000/5000). "
            "Strong Benford violation; tests recall under a maximally easy pattern."
        ),
        trades=df,
        labels=labels,
        metadata={"expected_positive_rate": 1.0, "difficulty": "easy"},
    )


def _build_high_frequency_ring() -> BenchmarkDataset:
    """5-wallet ring executing high-frequency small trades to obscure wash patterns."""
    rng = _rng(3)
    n_ring = 400
    n_noise = 100

    # Ring trades: small amounts, tightly-timed, few wallets
    ring_amounts = rng.uniform(0.1, 2.0, size=n_ring)
    ring_wallets = [f"RING{i % 5:02d}" for i in range(n_ring)]
    base_ts = pd.Timestamp("2024-03-01", tz="UTC")
    ring_timestamps = pd.Series(
        [base_ts + pd.Timedelta(seconds=float(s)) for s in np.arange(n_ring) * 0.5]
    )

    ring_df = pd.DataFrame(
        {
            "wallet_id": ring_wallets,
            "asset_pair": "XLM:native/AQUA:GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
            "amount": ring_amounts,
            "timestamp": ring_timestamps,
        }
    )

    # Noise trades: Benford-conforming, different pair
    noise_amounts = _benford_amounts(n_noise, rng)
    noise_df = _build_base_frame(
        n_noise, noise_amounts, rng, wallet_pool=30, pair="USDC:GA001/XLM:native"
    )

    df = pd.concat([ring_df, noise_df], ignore_index=True)
    labels = pd.Series([True] * n_ring + [False] * n_noise, dtype=bool)
    return BenchmarkDataset(
        name="high_frequency_ring",
        description=(
            "400 high-frequency ring trades from 5 sock-puppet wallets mixed with "
            "100 legitimate background trades. Tests ring detection and timing features."
        ),
        trades=df,
        labels=labels,
        metadata={"expected_positive_rate": 0.80, "difficulty": "hard", "ring_size": 5},
    )


def _build_sparse_low_volume() -> BenchmarkDataset:
    """Thin market with very few trades per wallet — stresses cold-start handling."""
    rng = _rng(4)
    n = 80  # intentionally small
    amounts = _benford_amounts(n, rng)
    df = _build_base_frame(n, amounts, rng, wallet_pool=40)
    labels = pd.Series([False] * n, dtype=bool)
    return BenchmarkDataset(
        name="sparse_low_volume",
        description=(
            "80 trades across 40 wallets (2 trades per wallet on average). "
            "Tests graceful degradation when per-wallet sample sizes are very small."
        ),
        trades=df,
        labels=labels,
        metadata={"expected_positive_rate": 0.0, "difficulty": "edge_case"},
    )


def _build_cross_pair_coordination() -> BenchmarkDataset:
    """Two wash-trade pairs coordinated by the same set of wallets."""
    rng = _rng(5)
    n_per_pair = 150

    frames = []
    for pair_idx, pair in enumerate(["USDC:GA001/XLM:native", "AQUA:GBBD/XLM:native"]):
        wash_amounts = _round_amounts(n_per_pair, rng)
        wallets = [f"COORD{i % 4:02d}" for i in range(n_per_pair)]
        # Synchronise timestamps — same wallets trading both pairs within seconds
        base_ts = pd.Timestamp("2024-06-01", tz="UTC")
        timestamps = pd.Series(
            [
                base_ts + pd.Timedelta(seconds=float(s) * 10 + pair_idx * 2)
                for s in range(n_per_pair)
            ]
        )
        frames.append(
            pd.DataFrame(
                {
                    "wallet_id": wallets,
                    "asset_pair": pair,
                    "amount": wash_amounts,
                    "timestamp": timestamps,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    labels = pd.Series([True] * (n_per_pair * 2), dtype=bool)
    return BenchmarkDataset(
        name="cross_pair_coordination",
        description=(
            "300 coordinated wash trades across 2 pairs by the same 4 wallets, "
            "timed within 2 seconds of each other. Tests cross-pair synchrony features."
        ),
        trades=df,
        labels=labels,
        metadata={
            "expected_positive_rate": 1.0,
            "difficulty": "hard",
            "pairs": 2,
            "coordination_window_seconds": 10,
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BenchmarkRegistry:
    """Central registry of all named benchmark datasets.

    Datasets are built lazily and cached so repeated calls don't re-generate
    the same data.

    Usage::

        registry = BenchmarkRegistry()
        datasets = registry.all()
        single   = registry.get("benford_baseline")
    """

    _BUILDERS: ClassVar[dict[str, object]] = {
        "benford_baseline": _build_benford_baseline,
        "benford_wash_trades": _build_benford_wash_trades,
        "round_number_wash": _build_round_number_wash,
        "high_frequency_ring": _build_high_frequency_ring,
        "sparse_low_volume": _build_sparse_low_volume,
        "cross_pair_coordination": _build_cross_pair_coordination,
    }

    def __init__(self) -> None:
        self._cache: dict[str, BenchmarkDataset] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return all registered dataset names."""
        return list(self._BUILDERS.keys())

    def get(self, name: str) -> BenchmarkDataset:
        """Return (and cache) the named dataset.

        Raises ``KeyError`` with a clear diagnostic if the name is unknown.
        """
        if name not in self._BUILDERS:
            available = ", ".join(sorted(self._BUILDERS))
            raise KeyError(
                f"Unknown benchmark dataset '{name}'. " f"Available datasets: {available}"
            )
        if name not in self._cache:
            builder = self._BUILDERS[name]
            logger.debug("Building benchmark dataset '%s'…", name)
            self._cache[name] = builder()  # type: ignore[operator]
        return self._cache[name]

    def all(self) -> list[BenchmarkDataset]:
        """Return all datasets (building and caching any that haven't been built)."""
        return [self.get(name) for name in self.names()]

    def checksum(self, name: str) -> str:
        """SHA-256 of the dataset's trade amounts, for reproducibility auditing."""
        ds = self.get(name)
        raw = ds.trades["amount"].round(8).to_csv(index=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def manifest(self) -> list[dict[str, object]]:
        """Return a JSON-serialisable manifest of all datasets and their checksums."""
        return [
            {
                "name": name,
                "n_trades": len(self.get(name).trades),
                "n_positives": int(self.get(name).labels.sum()),
                "positive_rate": round(float(self.get(name).labels.mean()), 4),
                "checksum_sha256": self.checksum(name),
                "metadata": self.get(name).metadata,
            }
            for name in self.names()
        ]


def build_benchmark_datasets(
    names: list[str] | None = None,
) -> list[BenchmarkDataset]:
    """Build and return a list of benchmark datasets.

    Args:
        names: If given, only build the named datasets.  Defaults to all.

    Returns:
        List of :class:`BenchmarkDataset` objects.

    Example::

        from benchmarks import build_benchmark_datasets
        datasets = build_benchmark_datasets()          # all 6 datasets
        subset   = build_benchmark_datasets(["benford_baseline"])
    """
    registry = BenchmarkRegistry()
    if names is None:
        return registry.all()
    return [registry.get(n) for n in names]


# ---------------------------------------------------------------------------
# CLI entry point  (python -m benchmarks.datasets)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    registry = BenchmarkRegistry()
    manifest = registry.manifest()
    print(json.dumps(manifest, indent=2, default=str))
    print(f"\n{len(manifest)} benchmark datasets ready.", file=sys.stderr)
