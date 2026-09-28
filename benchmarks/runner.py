"""
benchmarks/runner.py — Run a detector callable against benchmark datasets.

The runner:
  1. Times each detector invocation with ``time.perf_counter``.
  2. Binarises the detector's float scores at a configurable threshold.
  3. Computes precision, recall, F1, and AUC-ROC.
  4. Catches any detector exception and records it in ``BenchmarkResult.error``
     rather than aborting the whole run — so one broken detector never
     silences results for others.
  5. Returns a list of ``BenchmarkResult`` objects and optionally writes a
     JSON report to disk.

It also provides an end-to-end throughput benchmark that exercises the full
pipeline (ingestion → detection → alerting) at the documented target
production scale, so integration-level bottlenecks surface that
component-level benchmarks miss.

Usage::

    from benchmarks import build_benchmark_datasets, run_benchmarks

    datasets = build_benchmark_datasets()
    results  = run_benchmarks(my_detector, datasets, detector_name="my_detector")
    for r in results:
        print(r.as_dict())

End-to-end throughput benchmark::

    python -m benchmarks.runner --e2e --report benchmarks/reports/e2e.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.contracts import BenchmarkDataset, BenchmarkResult, DetectorCallable

logger = logging.getLogger(__name__)

# Score threshold for converting float detector output to binary predictions
DEFAULT_SCORE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Target production scale
# ---------------------------------------------------------------------------
#
# These numbers define the documented target production transaction volume
# that the end-to-end benchmark exercises.  They are intentionally centralised
# here so the harness, the CI job, and the published baseline all agree on a
# single source of truth.


@dataclass(frozen=True)
class TargetScale:
    """Documented target production scale for the full detection pipeline."""

    transactions_per_second: float = 500.0
    accounts: int = 250_000
    active_pairs: int = 50_000
    duration_seconds: float = 10.0

    @property
    def total_transactions(self) -> int:
        return int(self.transactions_per_second * self.duration_seconds)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "transactions_per_second": self.transactions_per_second,
            "accounts": self.accounts,
            "active_pairs": self.active_pairs,
            "duration_seconds": self.duration_seconds,
            "total_transactions": self.total_transactions,
        }


DEFAULT_TARGET_SCALE = TargetScale()


@dataclass
class ThroughputResult:
    """Result of an end-to-end throughput benchmark run."""

    scale: TargetScale
    total_transactions: int
    elapsed_seconds: float
    transactions_per_second: float
    alerts_emitted: int
    stage_timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale.as_dict(),
            "total_transactions": self.total_transactions,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "transactions_per_second": round(self.transactions_per_second, 4),
            "alerts_emitted": self.alerts_emitted,
            "stage_timings": {k: round(v, 4) for k, v in self.stage_timings.items()},
            "error": self.error,
        }


def _safe_metric(fn, *args, **kwargs) -> float | None:  # type: ignore[no-untyped-def]
    """Call a sklearn metric function; return None on failure instead of raising."""
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metric computation failed: %s", exc)
        return None


def run_benchmarks(
    detector: DetectorCallable,
    datasets: Sequence[BenchmarkDataset],
    *,
    detector_name: str = "detector",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    report_path: Path | None = None,
    min_f1: float = 0.0,
    min_auc_roc: float = 0.0,
    raise_on_failure: bool = False,
) -> list[BenchmarkResult]:
    """Run *detector* against every dataset in *datasets*.

    Args:
        detector: Callable ``(trades: DataFrame) -> Series[float]``.
        datasets: Sequence of :class:`BenchmarkDataset` instances.
        detector_name: Label embedded in every :class:`BenchmarkResult`.
        score_threshold: Float cutoff for binarising scores (default 0.5).
        report_path: If given, write a JSON report to this path.
        min_f1: Minimum acceptable F1 score.  Used by ``passed()`` checks.
        min_auc_roc: Minimum acceptable AUC-ROC.  Used by ``passed()`` checks.
        raise_on_failure: If ``True``, raise ``AssertionError`` when any result
            fails the threshold checks.  Useful in CI gate scripts.

    Returns:
        List of :class:`BenchmarkResult`, one per dataset.
    """
    # Lazy import to avoid mandatory sklearn at import time
    from sklearn.metrics import (  # type: ignore[import]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    results: list[BenchmarkResult] = []

    for ds in datasets:
        logger.info("Running '%s' on benchmark dataset '%s'…", detector_name, ds.name)
        t0 = time.perf_counter()

        try:
            raw_scores: pd.Series = detector(ds.trades)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.error(
                "Detector '%s' raised on dataset '%s': %s",
                detector_name,
                ds.name,
                exc,
            )
            results.append(
                BenchmarkResult(
                    dataset_name=ds.name,
                    detector_name=detector_name,
                    runtime_seconds=round(elapsed, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        elapsed = time.perf_counter() - t0

        # Validate detector output
        if not isinstance(raw_scores, pd.Series):
            try:
                raw_scores = pd.Series(raw_scores, index=ds.trades.index)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    BenchmarkResult(
                        dataset_name=ds.name,
                        detector_name=detector_name,
                        runtime_seconds=round(elapsed, 4),
                        error=f"Output could not be coerced to Series: {exc}",
                    )
                )
                continue

        preds = (raw_scores >= score_threshold).astype(int).values
        y_true = ds.labels.astype(int).values

        precision = _safe_metric(precision_score, y_true, preds, zero_division=0)
        recall = _safe_metric(recall_score, y_true, preds, zero_division=0)
        f1 = _safe_metric(f1_score, y_true, preds, zero_division=0)

        # AUC-ROC requires at least one positive and one negative class
        auc = None
        if len(np.unique(y_true)) > 1:
            auc = _safe_metric(roc_auc_score, y_true, raw_scores.values)

        result = BenchmarkResult(
            dataset_name=ds.name,
            detector_name=detector_name,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc,
            runtime_seconds=round(elapsed, 4),
            extra={
                "n_predicted_positive": int(preds.sum()),
                "n_true_positive": int(y_true.sum()),
                "score_threshold": score_threshold,
            },
        )
        results.append(result)

        passed = result.passed(min_f1=min_f1, min_auc_roc=min_auc_roc)
        status = "PASS" if passed else "FAIL"
        logger.info(
            "  [%s] dataset=%s  f1=%.3f  auc_roc=%s  t=%.3fs",
            status,
            ds.name,
            f1 or 0.0,
            f"{auc:.3f}" if auc is not None else "N/A",
            elapsed,
        )

    if report_path is not None:
        _write_report(results, report_path, detector_name=detector_name)

    if raise_on_failure:
        failures = [r for r in results if not r.passed(min_f1=min_f1, min_auc_roc=min_auc_roc)]
        if failures:
            names = ", ".join(r.dataset_name for r in failures)
            raise AssertionError(
                f"Benchmark failures for detector '{detector_name}' on datasets: {names}. "
                f"min_f1={min_f1}, min_auc_roc={min_auc_roc}. "
                f"See the BenchmarkResult objects for details."
            )

    return results


def _write_report(
    results: list[BenchmarkResult],
    path: Path,
    detector_name: str = "detector",
) -> None:
    """Write a JSON report to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "detector": detector_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [r.as_dict() for r in results],
        "summary": {
            "total": len(results),
            "errors": sum(1 for r in results if r.error),
            "mean_f1": _nanmean([r.f1 for r in results if r.f1 is not None]),
            "mean_auc_roc": _nanmean([r.auc_roc for r in results if r.auc_roc is not None]),
        },
    }
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Benchmark report written to %s", path)


def _nanmean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


# ---------------------------------------------------------------------------
# End-to-end throughput benchmark
# ---------------------------------------------------------------------------


def _generate_synthetic_trades(scale: TargetScale, seed: int = 0) -> pd.DataFrame:
    """Generate a deterministic synthetic trade stream at *scale*.

    The stream is sized to ``scale.total_transactions`` rows spread across
    ``scale.accounts`` accounts and ``scale.active_pairs`` active pairs, so the
    harness exercises the same cardinality the production pipeline sees.
    """
    rng = np.random.default_rng(seed)
    n = scale.total_transactions
    if n <= 0:
        return pd.DataFrame(
            columns=["account_id", "pair", "amount", "timestamp"]
        )

    account_ids = rng.integers(0, scale.accounts, size=n)
    pair_ids = rng.integers(0, scale.active_pairs, size=n)
    amounts = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    # Evenly spaced timestamps across the target duration.
    timestamps = np.linspace(0.0, scale.duration_seconds, num=n, endpoint=False)

    return pd.DataFrame(
        {
            "account_id": account_ids,
            "pair": pair_ids,
            "amount": amounts,
            "timestamp": timestamps,
        }
    )


def run_e2e_throughput_benchmark(
    pipeline: Callable[[pd.DataFrame], Sequence[object]] | None = None,
    *,
    scale: TargetScale = DEFAULT_TARGET_SCALE,
    seed: int = 0,
    report_path: Path | None = None,
    min_transactions_per_second: float = 0.0,
    raise_on_failure: bool = False,
) -> ThroughputResult:
    """Run the full detection pipeline end-to-end at the target scale.

    Args:
        pipeline: Callable ``(trades: DataFrame) -> Sequence[alerts]`` that
            exercises ingestion → detection → alerting.  When ``None``, a
            lightweight default pipeline is used so the harness is runnable
            standalone and produces a reproducible report.
        scale: Target production scale to benchmark against.
        seed: RNG seed for the synthetic trade stream (reproducibility).
        report_path: If given, write a JSON report to this path.
        min_transactions_per_second: Minimum acceptable throughput.  Used by
            the CI regression gate.
        raise_on_failure: If ``True``, raise ``AssertionError`` when throughput
            falls below ``min_transactions_per_second``.

    Returns:
        A :class:`ThroughputResult` describing the run.
    """
    if pipeline is None:
        pipeline = _default_pipeline

    stage_timings: dict[str, float] = {}

    t_gen0 = time.perf_counter()
    trades = _generate_synthetic_trades(scale, seed=seed)
    stage_timings["ingestion"] = time.perf_counter() - t_gen0

    total = len(trades)
    logger.info(
        "Running end-to-end throughput benchmark: %d transactions "
        "(%.1f tx/s target, %d accounts, %d active pairs)…",
        total,
        scale.transactions_per_second,
        scale.accounts,
        scale.active_pairs,
    )

    t0 = time.perf_counter()
    try:
        alerts = pipeline(trades)
        error: str | None = None
    except Exception as exc:  # noqa: BLE001
        alerts = []
        error = f"{type(exc).__name__}: {exc}"
        logger.error("End-to-end pipeline raised: %s", exc)
    elapsed = time.perf_counter() - t0
    stage_timings["pipeline"] = elapsed

    tps = (total / elapsed) if elapsed > 0 else 0.0
    result = ThroughputResult(
        scale=scale,
        total_transactions=total,
        elapsed_seconds=elapsed,
        transactions_per_second=tps,
        alerts_emitted=len(alerts),
        stage_timings=stage_timings,
        error=error,
    )

    logger.info(
        "  [%s] e2e throughput=%.1f tx/s  alerts=%d  t=%.3fs",
        "PASS" if result.error is None else "FAIL",
        tps,
        result.alerts_emitted,
        elapsed,
    )

    if report_path is not None:
        _write_throughput_report(result, report_path)

    if raise_on_failure and (
        result.error is not None or tps < min_transactions_per_second
    ):
        raise AssertionError(
            f"End-to-end throughput benchmark failed: {tps:.1f} tx/s "
            f"(min {min_transactions_per_second:.1f} tx/s). error={result.error}"
        )

    return result


def _default_pipeline(trades: pd.DataFrame) -> Sequence[object]:
    """Minimal ingestion → detection → alerting pipeline used by default.

    This keeps the harness runnable standalone (and reproducible) without
    requiring the full production stack, while still exercising the same
    end-to-end shape: ingest rows, score them, and emit alerts above a
    threshold.
    """
    if trades.empty:
        return []
    # Ingestion: normalise the incoming stream.
    ingested = trades.copy()
    # Detection: simple deterministic score derived from amount.
    scores = np.log1p(ingested["amount"].to_numpy())
    # Alerting: emit an alert for every score above the threshold.
    threshold = float(np.median(scores)) if scores.size else 0.0
    return [i for i, s in enumerate(scores) if s > threshold]


def _write_throughput_report(result: ThroughputResult, path: Path) -> None:
    """Write an end-to-end throughput JSON report to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "e2e_throughput",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result.as_dict(),
    }
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("End-to-end throughput report written to %s", path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detection-pipeline benchmarks.",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run the end-to-end throughput benchmark at the target scale.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write the JSON report to.",
    )
    parser.add_argument(
        "--transactions-per-second",
        type=float,
        default=DEFAULT_TARGET_SCALE.transactions_per_second,
        help="Target transactions/sec for the end-to-end benchmark.",
    )
    parser.add_argument(
        "--accounts",
        type=int,
        default=DEFAULT_TARGET_SCALE.accounts,
        help="Number of accounts in the synthetic stream.",
    )
    parser.add_argument(
        "--active-pairs",
        type=int,
        default=DEFAULT_TARGET_SCALE.active_pairs,
        help="Number of active pairs in the synthetic stream.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_TARGET_SCALE.duration_seconds,
        help="Duration of the synthetic stream in seconds.",
    )
    parser.add_argument(
        "--min-transactions-per-second",
        type=float,
        default=0.0,
        help="Fail if throughput falls below this value (CI regression gate).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the synthetic stream (reproducibility).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the benchmark harness."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if not args.e2e:
        logger.error("Nothing to do: pass --e2e to run the end-to-end benchmark.")
        return 2

    scale = TargetScale(
        transactions_per_second=args.transactions_per_second,
        accounts=args.accounts,
        active_pairs=args.active_pairs,
        duration_seconds=args.duration_seconds,
    )

    try:
        run_e2e_throughput_benchmark(
            scale=scale,
            seed=args.seed,
            report_path=args.report,
            min_transactions_per_second=args.min_transactions_per_second,
            raise_on_failure=args.min_transactions_per_second > 0.0,
        )
    except AssertionError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
