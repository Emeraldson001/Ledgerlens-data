"""Parallel ingestion executor with per-category resource isolation.

Each task category (e.g. a source type such as "horizon", "amm", "file")
gets its own bounded `ThreadPoolExecutor`, so a slow or misbehaving source
can only exhaust its own workers and never starves other categories.
Per-category metrics (queue depth, in-flight, completed, durations) help
identify noisy neighbors. See docs/parallel_executor.md for defaults.
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

DEFAULT_CATEGORY_LIMIT = 4


class CategoryMetrics:
    def __init__(self):
        self.queued = 0
        self.in_flight = 0
        self.completed = 0
        self.failed = 0
        self.durations: list[float] = []

    @property
    def queue_depth(self) -> int:
        return self.queued

    def snapshot(self) -> dict:
        d = self.durations
        return {
            "queue_depth": self.queued,
            "in_flight": self.in_flight,
            "completed": self.completed,
            "failed": self.failed,
            "avg_duration_s": sum(d) / len(d) if d else 0.0,
            "max_duration_s": max(d) if d else 0.0,
        }


class PartitionedExecutor:
    def __init__(
        self,
        category_limits: dict[str, int] | None = None,
        default_limit: int = DEFAULT_CATEGORY_LIMIT,
    ):
        self.category_limits = dict(category_limits or {})
        self.default_limit = default_limit
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._metrics: dict[str, CategoryMetrics] = defaultdict(CategoryMetrics)
        self._lock = threading.Lock()

    def _pool(self, category: str) -> ThreadPoolExecutor:
        with self._lock:
            if category not in self._pools:
                self._pools[category] = ThreadPoolExecutor(
                    max_workers=self.category_limits.get(category, self.default_limit),
                    thread_name_prefix=f"ingest-{category}",
                )
            return self._pools[category]

    def submit(self, category: str, fn, *args, **kwargs) -> Future:
        m = self._metrics[category]
        with self._lock:
            m.queued += 1

        def run():
            with self._lock:
                m.queued -= 1
                m.in_flight += 1
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            except Exception:
                ok = False
                raise
            finally:
                with self._lock:
                    m.in_flight -= 1
                    m.durations.append(time.perf_counter() - start)
                    if ok:
                        m.completed += 1
                    else:
                        m.failed += 1

        return self._pool(category).submit(run)

    def metrics(self) -> dict[str, dict]:
        with self._lock:
            return {c: m.snapshot() for c, m in self._metrics.items()}

    def shutdown(self, wait: bool = True) -> None:
        for pool in self._pools.values():
            pool.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
