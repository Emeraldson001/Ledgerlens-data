"""
benchmarks/ — Detector performance benchmark datasets and contracts.

Provides:
  - BenchmarkDataset: typed dataclass describing a benchmark scenario
  - BenchmarkResult: typed dataclass capturing per-detector run metrics
  - BenchmarkRegistry: central registry of all named benchmark datasets
  - build_benchmark_datasets: factory that generates reproducible dataset fixtures
  - run_benchmarks: run a detector callable against all registered datasets
    and return a list of BenchmarkResult objects

Usage::

    from benchmarks import build_benchmark_datasets, run_benchmarks, BenchmarkRegistry

    datasets = build_benchmark_datasets()
    results  = run_benchmarks(my_detector_fn, datasets)

See benchmarks/datasets.py for dataset definitions.
See benchmarks/runner.py for the runner contract.
"""

from benchmarks.contracts import BenchmarkDataset, BenchmarkResult, DetectorCallable
from benchmarks.datasets import BenchmarkRegistry, build_benchmark_datasets
from benchmarks.runner import run_benchmarks

__all__ = [
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "DetectorCallable",
    "build_benchmark_datasets",
    "run_benchmarks",
]
