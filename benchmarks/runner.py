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

Usage::

    from benchmarks import build_benchmark_datasets, run_benchmarks

    datasets = build_benchmark_datasets()
    results  = run_benchmarks(my_detector, datasets, detector_name="my_detector")
    for r in results:
        print(r.as_dict())
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.contracts import BenchmarkDataset, BenchmarkResult, DetectorCallable

logger = logging.getLogger(__name__)

# Score threshold for converting float detector output to binary predictions
DEFAULT_SCORE_THRESHOLD: float = 0.5


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
