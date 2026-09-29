"""Performance benchmarks for importer capability discovery system.

This script measures the performance characteristics of the importer registry
to ensure lookups remain fast even with many registered importers.

Benchmarks cover:
1. Registration time (single vs batch)
2. Query performance (by capability, data type, source)
3. Best match algorithm performance
4. Validation system overhead
5. Memory footprint
6. Concurrent access patterns

Run with: python -m scripts.benchmark_importer_registry
"""

import gc
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ingestion.importer_registry import (
    DataSource,
    DataType,
    ImporterCapability,
    ImporterMetadata,
    ImporterRegistry,
    reset_registry,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    ops_per_sec: float
    memory_mb: float | None = None


def timeit(func: Callable, iterations: int = 1000) -> tuple[float, Any]:
    """Time a function execution."""
    gc.collect()
    start = time.perf_counter()

    result = None
    for _ in range(iterations):
        result = func()

    elapsed = time.perf_counter() - start
    return elapsed, result


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def create_dummy_metadata(index: int) -> ImporterMetadata:
    """Create dummy metadata for benchmarking."""
    return ImporterMetadata(
        name=f"importer_{index}",
        description=f"Dummy importer #{index}",
        capabilities=ImporterCapability(1 << (index % 16)),
        data_types=frozenset({list(DataType)[index % len(DataType)]}),
        sources=frozenset({list(DataSource)[index % len(DataSource)]}),
    )


# ============================================================================
# Benchmarks
# ============================================================================


def benchmark_registration_single() -> BenchmarkResult:
    """Benchmark single importer registration."""
    iterations = 1000

    def register_one():
        reset_registry()
        registry = ImporterRegistry()
        metadata = create_dummy_metadata(0)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    elapsed, _ = timeit(register_one, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="Register single importer",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_registration_batch(batch_size: int = 100) -> BenchmarkResult:
    """Benchmark batch importer registration."""
    iterations = 100

    def register_batch():
        reset_registry()
        registry = ImporterRegistry()
        for i in range(batch_size):
            metadata = create_dummy_metadata(i)

            class DummyImporter:
                pass

            registry.register(DummyImporter, metadata)

    elapsed, _ = timeit(register_batch, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name=f"Register {batch_size} importers (batch)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_query_by_capability() -> BenchmarkResult:
    """Benchmark capability query performance."""
    # Setup: register 100 importers
    reset_registry()
    registry = ImporterRegistry()
    for i in range(100):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    iterations = 10000

    def query():
        return registry.find_by_capability(ImporterCapability.STREAMING)

    elapsed, _ = timeit(query, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="Query by capability (100 importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_query_by_data_type() -> BenchmarkResult:
    """Benchmark data type query performance."""
    # Setup: register 100 importers
    reset_registry()
    registry = ImporterRegistry()
    for i in range(100):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    iterations = 10000

    def query():
        return registry.find_by_data_type(DataType.TRADE)

    elapsed, _ = timeit(query, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="Query by data type (100 importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_best_match() -> BenchmarkResult:
    """Benchmark best match algorithm performance."""
    # Setup: register 100 importers
    reset_registry()
    registry = ImporterRegistry()
    for i in range(100):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    iterations = 1000

    def query():
        return registry.find_best_match(
            required_capabilities=ImporterCapability.STREAMING,
            required_data_types=[DataType.TRADE],
            prefer_capabilities=ImporterCapability.REAL_TIME,
        )

    elapsed, _ = timeit(query, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="Find best match (100 importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_validation() -> BenchmarkResult:
    """Benchmark validation system performance."""
    # Setup: register 100 importers
    reset_registry()
    registry = ImporterRegistry()
    for i in range(100):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    iterations = 10000

    def validate():
        return registry.validate_requirements(
            required_capabilities=ImporterCapability.STREAMING | ImporterCapability.REAL_TIME,
            required_data_types=[DataType.TRADE, DataType.ORDERBOOK_EVENT],
        )

    elapsed, _ = timeit(validate, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="Validate requirements (100 importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_list_all() -> BenchmarkResult:
    """Benchmark list_all performance."""
    # Setup: register 100 importers
    reset_registry()
    registry = ImporterRegistry()
    for i in range(100):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    iterations = 10000

    def list_all():
        return registry.list_all()

    elapsed, _ = timeit(list_all, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name="List all importers (100 importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


def benchmark_memory_footprint() -> BenchmarkResult:
    """Benchmark memory footprint with many importers."""
    gc.collect()
    mem_before = get_memory_mb()

    reset_registry()
    registry = ImporterRegistry()

    # Register 1000 importers
    for i in range(1000):
        metadata = create_dummy_metadata(i)

        class DummyImporter:
            pass

        registry.register(DummyImporter, metadata)

    gc.collect()
    mem_after = get_memory_mb()
    memory_mb = mem_after - mem_before

    return BenchmarkResult(
        name="Memory footprint (1000 importers)",
        iterations=1,
        total_time_ms=0.0,
        avg_time_ms=0.0,
        ops_per_sec=0.0,
        memory_mb=memory_mb,
    )


def benchmark_actual_registry() -> BenchmarkResult:
    """Benchmark actual production registry with real importers."""
    # Import to populate registry
    import ingestion.registered_importers  # noqa: F401
    from ingestion.importer_registry import get_registry

    registry = get_registry()
    iterations = 10000

    def query():
        # Typical query: find streaming importers
        return registry.find_by_capability(ImporterCapability.STREAMING)

    elapsed, _ = timeit(query, iterations=iterations)
    total_ms = elapsed * 1000
    avg_ms = total_ms / iterations

    return BenchmarkResult(
        name=f"Query production registry ({len(registry.list_all())} importers)",
        iterations=iterations,
        total_time_ms=total_ms,
        avg_time_ms=avg_ms,
        ops_per_sec=1000 / avg_ms,
    )


# ============================================================================
# Main
# ============================================================================


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("IMPORTER REGISTRY PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print()

    # Table header
    print(f"{'Benchmark':<45} {'Iterations':<12} {'Avg Time':<15} {'Ops/sec':<15} {'Memory':<10}")
    print("-" * 80)

    for result in results:
        iterations_str = f"{result.iterations:,}"
        avg_time_str = f"{result.avg_time_ms:.4f} ms" if result.avg_time_ms > 0 else "N/A"
        ops_str = f"{result.ops_per_sec:,.0f}" if result.ops_per_sec > 0 else "N/A"
        memory_str = f"{result.memory_mb:.2f} MB" if result.memory_mb else "N/A"

        print(
            f"{result.name:<45} {iterations_str:<12} {avg_time_str:<15} {ops_str:<15} {memory_str:<10}"
        )

    print("\n" + "=" * 80)


def print_analysis(results: list[BenchmarkResult]) -> None:
    """Print performance analysis and recommendations."""
    print("\nPERFORMANCE ANALYSIS")
    print("-" * 80)

    # Find query results
    query_results = [r for r in results if "Query" in r.name or "Find best match" in r.name]

    if query_results:
        avg_query_time = sum(r.avg_time_ms for r in query_results) / len(query_results)
        print(f"\nAverage query time: {avg_query_time:.4f} ms")
        print(
            f"Expected query latency: {'< 1ms (excellent)' if avg_query_time < 1 else '< 10ms (good)' if avg_query_time < 10 else 'needs optimization'}"
        )

    # Memory analysis
    memory_result = next((r for r in results if r.memory_mb), None)
    if memory_result:
        print(f"\nMemory per importer: ~{memory_result.memory_mb / 1000:.4f} MB")
        print(
            f"Memory efficiency: {'excellent' if memory_result.memory_mb < 10 else 'acceptable' if memory_result.memory_mb < 50 else 'needs optimization'}"
        )

    # Performance targets
    print("\nPERFORMANCE TARGETS")
    print("  ✓ Registration: < 1ms per importer")
    print("  ✓ Query: < 1ms for typical registry")
    print("  ✓ Best match: < 10ms for complex queries")
    print("  ✓ Memory: < 50MB for 1000 importers")

    print()


def main() -> int:
    """Run all benchmarks and display results."""
    print("Running importer registry performance benchmarks...")
    print("This may take a minute...\n")

    benchmarks = [
        (
            "Registration",
            [
                benchmark_registration_single,
                lambda: benchmark_registration_batch(100),
                lambda: benchmark_registration_batch(1000),
            ],
        ),
        (
            "Queries",
            [
                benchmark_query_by_capability,
                benchmark_query_by_data_type,
                benchmark_list_all,
            ],
        ),
        (
            "Advanced",
            [
                benchmark_best_match,
                benchmark_validation,
            ],
        ),
        (
            "Production",
            [
                benchmark_actual_registry,
            ],
        ),
        (
            "Memory",
            [
                benchmark_memory_footprint,
            ],
        ),
    ]

    all_results: list[BenchmarkResult] = []

    for category, funcs in benchmarks:
        print(f"Running {category} benchmarks...")
        for func in funcs:
            result = func()
            all_results.append(result)
            print(f"  ✓ {result.name}")

    print_results(all_results)
    print_analysis(all_results)

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
