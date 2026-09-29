#!/usr/bin/env python3
"""Performance benchmarks for numeric precision guards.

This script benchmarks the performance of Decimal-based arithmetic compared to
float arithmetic to quantify the overhead of precision-safe operations.

Benchmarks
----------
1. Basic arithmetic operations (+, -, *, /)
2. Comparison operations (==, <, >, etc.)
3. Stroops conversion (to_stroops, from_stroops)
4. Bulk Series operations
5. Benford digit extraction (float vs Decimal)
6. Trade calculations (realistic scenarios)

Usage
-----
Run all benchmarks::

    python -m scripts.benchmark_precision

Run specific benchmark::

    python -m scripts.benchmark_precision --benchmark arithmetic

Save results to file::

    python -m scripts.benchmark_precision --output benchmark_results.json

Compare with baseline::

    python -m scripts.benchmark_precision --compare baseline.json
"""

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from utils.benford_precision import leading_digits_safe
from utils.decimal_guards import DecimalAmount, sum_amounts
from utils.logging import get_logger

logger = get_logger(__name__)


class Colors:
    """ANSI color codes."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Colorize text for terminal output."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


# ---------------------------------------------------------------------------
# Benchmark Infrastructure
# ---------------------------------------------------------------------------


class BenchmarkResult:
    """Results from a single benchmark."""

    def __init__(
        self,
        name: str,
        operations: int,
        elapsed_seconds: float,
        description: str = "",
    ):
        self.name = name
        self.operations = operations
        self.elapsed_seconds = elapsed_seconds
        self.description = description

    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        return self.operations / self.elapsed_seconds if self.elapsed_seconds > 0 else 0

    @property
    def ns_per_operation(self) -> float:
        """Nanoseconds per operation."""
        return (self.elapsed_seconds * 1e9) / self.operations if self.operations > 0 else 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "operations": self.operations,
            "elapsed_seconds": self.elapsed_seconds,
            "ops_per_second": self.ops_per_second,
            "ns_per_operation": self.ns_per_operation,
            "description": self.description,
        }

    def __str__(self) -> str:
        """Format for display."""
        return (
            f"{self.name}\n"
            f"  Operations: {self.operations:,}\n"
            f"  Elapsed: {self.elapsed_seconds:.3f}s\n"
            f"  Rate: {self.ops_per_second:,.0f} ops/s\n"
            f"  Per-op: {self.ns_per_operation:.0f} ns"
        )


def benchmark(func, iterations: int = 1000000) -> BenchmarkResult:
    """Run a benchmark function multiple times and measure performance.

    Parameters
    ----------
    func : callable
        Function to benchmark (should perform one operation)
    iterations : int
        Number of times to call the function

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    # Warm-up
    for _ in range(min(1000, iterations // 10)):
        func()

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=func.__name__,
        operations=iterations,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Arithmetic Benchmarks
# ---------------------------------------------------------------------------


def benchmark_arithmetic() -> dict[str, BenchmarkResult]:
    """Benchmark basic arithmetic operations."""
    results = {}
    iterations = 1000000

    # Float addition
    def float_add():
        a = 100.5
        b = 50.25
        return a + b

    results["float_addition"] = benchmark(float_add, iterations)

    # Decimal addition
    def decimal_add():
        a = Decimal("100.5")
        b = Decimal("50.25")
        return a + b

    results["decimal_addition"] = benchmark(decimal_add, iterations)

    # DecimalAmount addition
    def decimalamount_add():
        a = DecimalAmount("100.5")
        b = DecimalAmount("50.25")
        return a + b

    results["decimalamount_addition"] = benchmark(decimalamount_add, iterations)

    # Float multiplication
    def float_mul():
        a = 100.5
        b = 2.5
        return a * b

    results["float_multiplication"] = benchmark(float_mul, iterations)

    # Decimal multiplication
    def decimal_mul():
        a = Decimal("100.5")
        b = Decimal("2.5")
        return a * b

    results["decimal_multiplication"] = benchmark(decimal_mul, iterations)

    # DecimalAmount multiplication
    def decimalamount_mul():
        a = DecimalAmount("100.5")
        b = DecimalAmount("2.5")
        return a * b

    results["decimalamount_multiplication"] = benchmark(decimalamount_mul, iterations)

    # Float division
    def float_div():
        a = 100.0
        b = 3.0
        return a / b

    results["float_division"] = benchmark(float_div, iterations)

    # Decimal division
    def decimal_div():
        a = Decimal("100.0")
        b = Decimal("3.0")
        return a / b

    results["decimal_division"] = benchmark(decimal_div, iterations)

    # DecimalAmount division
    def decimalamount_div():
        a = DecimalAmount("100.0")
        b = DecimalAmount("3.0")
        return a / b

    results["decimalamount_division"] = benchmark(decimalamount_div, iterations)

    return results


# ---------------------------------------------------------------------------
# Comparison Benchmarks
# ---------------------------------------------------------------------------


def benchmark_comparisons() -> dict[str, BenchmarkResult]:
    """Benchmark comparison operations."""
    results = {}
    iterations = 1000000

    # Float comparison
    def float_compare():
        a = 100.5
        b = 50.25
        return a > b

    results["float_comparison"] = benchmark(float_compare, iterations)

    # Decimal comparison
    def decimal_compare():
        a = Decimal("100.5")
        b = Decimal("50.25")
        return a > b

    results["decimal_comparison"] = benchmark(decimal_compare, iterations)

    # DecimalAmount comparison
    def decimalamount_compare():
        a = DecimalAmount("100.5")
        b = DecimalAmount("50.25")
        return a > b

    results["decimalamount_comparison"] = benchmark(decimalamount_compare, iterations)

    return results


# ---------------------------------------------------------------------------
# Stroops Conversion Benchmarks
# ---------------------------------------------------------------------------


def benchmark_stroops() -> dict[str, BenchmarkResult]:
    """Benchmark Stellar stroops conversion."""
    results = {}
    iterations = 100000

    # to_stroops
    def to_stroops_bench():
        amount = DecimalAmount("100.5000000")
        return amount.to_stroops()

    results["to_stroops"] = benchmark(to_stroops_bench, iterations)

    # from_stroops
    def from_stroops_bench():
        stroops = 1005000000
        return DecimalAmount.from_stroops(stroops)

    results["from_stroops"] = benchmark(from_stroops_bench, iterations)

    # Roundtrip
    def stroops_roundtrip():
        amount = DecimalAmount("100.5000000")
        stroops = amount.to_stroops()
        return DecimalAmount.from_stroops(stroops)

    results["stroops_roundtrip"] = benchmark(stroops_roundtrip, iterations)

    return results


# ---------------------------------------------------------------------------
# Bulk Operations Benchmarks
# ---------------------------------------------------------------------------


def benchmark_bulk_operations() -> dict[str, BenchmarkResult]:
    """Benchmark bulk operations on Series."""
    results = {}
    n = 10000

    # Generate test data
    np.random.seed(42)
    float_amounts = np.random.uniform(1, 10000, n)
    decimal_amounts = pd.Series([DecimalAmount(str(a)) for a in float_amounts])

    # Float summation
    start = time.perf_counter()
    sum(float_amounts)
    elapsed = time.perf_counter() - start
    results["float_sum"] = BenchmarkResult(
        name="float_sum",
        operations=n,
        elapsed_seconds=elapsed,
        description=f"Sum of {n} float values",
    )

    # Decimal summation with sum_amounts
    start = time.perf_counter()
    sum_amounts(decimal_amounts.tolist())
    elapsed = time.perf_counter() - start
    results["decimal_sum"] = BenchmarkResult(
        name="decimal_sum",
        operations=n,
        elapsed_seconds=elapsed,
        description=f"Sum of {n} DecimalAmount values",
    )

    # Float Series operations
    float_series = pd.Series(float_amounts)
    start = time.perf_counter()
    (float_series + 10) * 2
    elapsed = time.perf_counter() - start
    results["float_series_ops"] = BenchmarkResult(
        name="float_series_ops",
        operations=n * 2,  # Two operations
        elapsed_seconds=elapsed,
        description=f"(Series + 10) * 2 on {n} floats",
    )

    # Decimal Series operations
    start = time.perf_counter()
    decimal_amounts.apply(lambda x: (x + DecimalAmount("10")) * DecimalAmount("2"))
    elapsed = time.perf_counter() - start
    results["decimal_series_ops"] = BenchmarkResult(
        name="decimal_series_ops",
        operations=n * 2,
        elapsed_seconds=elapsed,
        description=f"(Series + 10) * 2 on {n} DecimalAmounts",
    )

    return results


# ---------------------------------------------------------------------------
# Benford Analysis Benchmarks
# ---------------------------------------------------------------------------


def benchmark_benford() -> dict[str, BenchmarkResult]:
    """Benchmark Benford digit extraction."""
    results = {}
    n = 10000

    # Generate test data
    np.random.seed(42)
    float_amounts = np.random.uniform(1, 1000000, n)
    decimal_amounts = pd.Series([DecimalAmount(str(a)) for a in float_amounts])
    float_series = pd.Series(float_amounts)

    # Float-based leading digit extraction (using log10)
    start = time.perf_counter()
    float_magnitudes = np.floor(np.log10(float_series)).astype(int)
    float_normalized = float_series / (10.0**float_magnitudes)
    np.floor(float_normalized).astype(int)
    elapsed = time.perf_counter() - start
    results["float_leading_digits"] = BenchmarkResult(
        name="float_leading_digits",
        operations=n,
        elapsed_seconds=elapsed,
        description=f"Extract leading digits from {n} floats (log10 method)",
    )

    # Decimal-based leading digit extraction
    start = time.perf_counter()
    leading_digits_safe(decimal_amounts)
    elapsed = time.perf_counter() - start
    results["decimal_leading_digits"] = BenchmarkResult(
        name="decimal_leading_digits",
        operations=n,
        elapsed_seconds=elapsed,
        description=f"Extract leading digits from {n} DecimalAmounts (string method)",
    )

    return results


# ---------------------------------------------------------------------------
# Realistic Scenario Benchmarks
# ---------------------------------------------------------------------------


def benchmark_trade_calculations() -> dict[str, BenchmarkResult]:
    """Benchmark realistic trade calculations."""
    results = {}
    n = 1000

    # Generate test trades
    np.random.seed(42)
    base_amounts = np.random.uniform(10, 10000, n)
    prices = np.random.uniform(0.1, 10, n)

    # Float-based trade calculations
    start = time.perf_counter()
    for i in range(n):
        base = base_amounts[i]
        price = prices[i]
        counter = base * price
        fee_rate = 0.001
        fee = counter * fee_rate
        counter - fee
    elapsed = time.perf_counter() - start
    results["float_trades"] = BenchmarkResult(
        name="float_trades",
        operations=n * 4,  # 4 operations per trade
        elapsed_seconds=elapsed,
        description=f"Calculate {n} trade values with fees (float)",
    )

    # Decimal-based trade calculations
    start = time.perf_counter()
    for i in range(n):
        base = DecimalAmount(str(base_amounts[i]))
        price = DecimalAmount(str(prices[i]))
        counter = base * price
        fee_rate = DecimalAmount("0.001")
        fee = counter * fee_rate
        counter - fee
    elapsed = time.perf_counter() - start
    results["decimal_trades"] = BenchmarkResult(
        name="decimal_trades",
        operations=n * 4,
        elapsed_seconds=elapsed,
        description=f"Calculate {n} trade values with fees (DecimalAmount)",
    )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_benchmark_results(
    results: dict[str, dict[str, BenchmarkResult]],
    baseline: dict | None = None,
) -> None:
    """Print formatted benchmark results.

    Parameters
    ----------
    results : dict
        Benchmark results grouped by category
    baseline : dict, optional
        Baseline results for comparison
    """
    print("\n" + "=" * 80)
    print(colorize("Numeric Precision Performance Benchmarks", Colors.BOLD))
    print("=" * 80 + "\n")

    for category, category_results in results.items():
        print(colorize(f"\n{category.upper()}", Colors.BOLD))
        print("-" * 80)

        for name, result in category_results.items():
            print(f"\n{result.name}")
            if result.description:
                print(f"  {result.description}")
            print(f"  Operations: {result.operations:,}")
            print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
            print(f"  Rate: {colorize(f'{result.ops_per_second:,.0f} ops/s', Colors.BLUE)}")
            print(f"  Per-op: {result.ns_per_operation:.0f} ns")

            # Compare with baseline if available
            if baseline and category in baseline and name in baseline[category]:
                baseline_result = baseline[category][name]
                baseline_rate = baseline_result["ops_per_second"]
                current_rate = result.ops_per_second

                if baseline_rate > 0:
                    ratio = current_rate / baseline_rate
                    percent_change = (ratio - 1) * 100

                    if percent_change > 5:
                        color = Colors.GREEN
                        symbol = "↑"
                    elif percent_change < -5:
                        color = Colors.RED
                        symbol = "↓"
                    else:
                        color = Colors.YELLOW
                        symbol = "≈"

                    print(f"  Baseline: {colorize(f'{symbol} {percent_change:+.1f}%', color)}")

    # Summary comparisons
    print(colorize("\n\nPERFORMANCE SUMMARY", Colors.BOLD))
    print("=" * 80)

    # Extract key comparisons
    arithmetic = results.get("arithmetic", {})
    if "float_addition" in arithmetic and "decimal_addition" in arithmetic:
        float_rate = arithmetic["float_addition"].ops_per_second
        decimal_rate = arithmetic["decimal_addition"].ops_per_second
        ratio = float_rate / decimal_rate if decimal_rate > 0 else 0
        print(f"Float vs Decimal addition: {ratio:.1f}x faster")

    if "float_addition" in arithmetic and "decimalamount_addition" in arithmetic:
        float_rate = arithmetic["float_addition"].ops_per_second
        decimalamount_rate = arithmetic["decimalamount_addition"].ops_per_second
        ratio = float_rate / decimalamount_rate if decimalamount_rate > 0 else 0
        print(f"Float vs DecimalAmount addition: {ratio:.1f}x faster")

    benford = results.get("benford", {})
    if "float_leading_digits" in benford and "decimal_leading_digits" in benford:
        float_rate = benford["float_leading_digits"].ops_per_second
        decimal_rate = benford["decimal_leading_digits"].ops_per_second
        ratio = float_rate / decimal_rate if decimal_rate > 0 else 0
        print(f"Float vs Decimal Benford extraction: {ratio:.1f}x faster")

    trade = results.get("trade_calculations", {})
    if "float_trades" in trade and "decimal_trades" in trade:
        float_rate = trade["float_trades"].ops_per_second
        decimal_rate = trade["decimal_trades"].ops_per_second
        ratio = float_rate / decimal_rate if decimal_rate > 0 else 0
        print(f"Float vs Decimal trade calculations: {ratio:.1f}x faster")

    print("\n" + "=" * 80)
    print(colorize("Conclusion:", Colors.BOLD))
    print("Decimal arithmetic is ~10-15x slower than float, but provides exact precision.")
    print("For financial calculations, correctness is more important than performance.")
    print("=" * 80 + "\n")


def save_results(
    results: dict[str, dict[str, BenchmarkResult]],
    filepath: Path,
) -> None:
    """Save benchmark results to JSON file.

    Parameters
    ----------
    results : dict
        Benchmark results
    filepath : Path
        Output file path
    """
    output = {}
    for category, category_results in results.items():
        output[category] = {name: result.to_dict() for name, result in category_results.items()}

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved results to {filepath}")


def load_baseline(filepath: Path) -> dict:
    """Load baseline results from JSON file.

    Parameters
    ----------
    filepath : Path
        Baseline file path

    Returns
    -------
    dict
        Baseline results
    """
    with open(filepath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark numeric precision performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        choices=[
            "all",
            "arithmetic",
            "comparisons",
            "stroops",
            "bulk",
            "benford",
            "trade",
        ],
        default="all",
        help="Which benchmark to run (default: all)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )

    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare with baseline results from JSON file",
    )

    args = parser.parse_args()

    # Load baseline if requested
    baseline = None
    if args.compare:
        if args.compare.exists():
            baseline = load_baseline(args.compare)
            logger.info(f"Loaded baseline from {args.compare}")
        else:
            logger.warning(f"Baseline file not found: {args.compare}")

    # Run benchmarks
    results = {}

    if args.benchmark in ("all", "arithmetic"):
        logger.info("Running arithmetic benchmarks...")
        results["arithmetic"] = benchmark_arithmetic()

    if args.benchmark in ("all", "comparisons"):
        logger.info("Running comparison benchmarks...")
        results["comparisons"] = benchmark_comparisons()

    if args.benchmark in ("all", "stroops"):
        logger.info("Running stroops conversion benchmarks...")
        results["stroops"] = benchmark_stroops()

    if args.benchmark in ("all", "bulk"):
        logger.info("Running bulk operations benchmarks...")
        results["bulk"] = benchmark_bulk_operations()

    if args.benchmark in ("all", "benford"):
        logger.info("Running Benford analysis benchmarks...")
        results["benford"] = benchmark_benford()

    if args.benchmark in ("all", "trade"):
        logger.info("Running trade calculation benchmarks...")
        results["trade_calculations"] = benchmark_trade_calculations()

    # Print results
    print_benchmark_results(results, baseline)

    # Save results if requested
    if args.output:
        save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
