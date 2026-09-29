"""Reusable concurrency validation utilities for streaming workers.

Provides helpers to write deterministic, repeatable concurrent tests
that surface race conditions, deadlocks, and data corruption in the
streaming pipeline's threaded components.

Usage
-----
    from tests.concurrent_validators import StressRunner, assert_eventually

    errors = StressRunner(target=my_worker.do_something).run(n_threads=8, n_iters=100)
    assert not errors

    assert_eventually(lambda: queue.qsize() == 0, timeout=5.0)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class ThreadError:
    """Captures the full context of an exception raised in a worker thread."""

    thread_id: int
    iteration: int
    exception: Exception
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.monotonic()


# ---------------------------------------------------------------------------
# Stress runner
# ---------------------------------------------------------------------------


class StressRunner:
    """Run a target function concurrently across *n_threads* for *n_iters* iterations.

    Every exception from every thread is captured with full context (thread id,
    iteration, traceback) so the caller can assert that no errors occurred and
    can inspect individual failures when they do.

    Parameters
    ----------
    target:
        Called as ``target(thread_id, iteration)`` from each worker thread.
    setup:
        Optional callable invoked as ``setup(thread_id)`` once per thread
        before the iteration loop.
    teardown:
        Optional callable invoked as ``teardown(thread_id)`` after each
        thread's iteration loop completes (or on exception).
    """

    def __init__(
        self,
        target: Callable[[int, int], Any],
        setup: Callable[[int], Any] | None = None,
        teardown: Callable[[int], Any] | None = None,
    ) -> None:
        self._target = target
        self._setup = setup
        self._teardown = teardown

    def run(
        self,
        n_threads: int,
        n_iters: int,
        timeout: float = 30.0,
    ) -> list[ThreadError]:
        """Execute *target* across *n_threads* threads, each doing *n_iters* iterations.

        Returns a list of :class:`ThreadError` — empty when all threads
        completed without exception.
        """
        errors: list[ThreadError] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(n_threads, timeout=timeout)

        def _worker(thread_id: int) -> None:
            try:
                if self._setup:
                    self._setup(thread_id)
                barrier.wait()
                for iteration in range(n_iters):
                    try:
                        self._target(thread_id, iteration)
                    except Exception as exc:
                        with errors_lock:
                            errors.append(
                                ThreadError(
                                    thread_id=thread_id,
                                    iteration=iteration,
                                    exception=exc,
                                )
                            )
            except Exception as exc:
                with errors_lock:
                    errors.append(
                        ThreadError(
                            thread_id=thread_id,
                            iteration=-1,
                            exception=exc,
                        )
                    )
            finally:
                if self._teardown:
                    try:
                        self._teardown(thread_id)
                    except Exception:
                        pass

        threads = [
            threading.Thread(target=_worker, args=(tid,), daemon=True) for tid in range(n_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        return errors


# ---------------------------------------------------------------------------
# Eventually-consistent assertion
# ---------------------------------------------------------------------------


def assert_eventually(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.05,
    msg: str = "",
) -> None:
    """Assert that *predicate* evaluates to ``True`` within *timeout* seconds.

    Polls every *interval* seconds.  Raises ``AssertionError`` with a
    descriptive message on timeout.

    Parameters
    ----------
    predicate:
        Zero-argument callable returning a bool.
    timeout:
        Maximum seconds to wait for the predicate to become true.
    interval:
        Seconds between predicate evaluations.
    msg:
        Optional message prefix appended to the assertion error.

    Raises
    ------
    AssertionError
        If the predicate does not become true within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(
        f"{msg.strip() + ': ' if msg else ''}"
        f"Predicate did not become True within {timeout}s "
        f"(polled every {interval}s)"
    )


# ---------------------------------------------------------------------------
# Concurrent read/write invariant checker
# ---------------------------------------------------------------------------


class ConcurrentReadWriteValidator:
    """Validate a data structure under a concurrent read/write workload.

    Combines reader threads and writer threads that operate on the same
    shared object.  Readers assert an invariants check function never fails;
    writers mutate the object and increment a shared counter that can be
    verified after the test.
    """

    def __init__(
        self,
        read_fn: Callable[[], Any],
        write_fn: Callable[[int], Any],
        invariant: Callable[[], bool] | None = None,
    ) -> None:
        self._read_fn = read_fn
        self._write_fn = write_fn
        self._invariant = invariant

    def run(
        self,
        n_readers: int = 4,
        n_writers: int = 4,
        n_ops: int = 50,
        timeout: float = 30.0,
    ) -> list[ThreadError]:
        """Run concurrent readers and writers, returning any thread errors."""
        errors: list[ThreadError] = []
        errors_lock = threading.Lock()

        writer_barrier = threading.Barrier(n_writers + n_readers, timeout=timeout)

        def _reader(thread_id: int) -> None:
            writer_barrier.wait()
            for iteration in range(n_ops):
                try:
                    self._read_fn()
                    if self._invariant:
                        assert (
                            self._invariant()
                        ), f"Invariant failed at reader {thread_id} iter {iteration}"
                except Exception as exc:
                    with errors_lock:
                        errors.append(
                            ThreadError(thread_id=thread_id, iteration=iteration, exception=exc)
                        )

        def _writer(thread_id: int) -> None:
            writer_barrier.wait()
            for iteration in range(n_ops):
                try:
                    self._write_fn(iteration)
                except Exception as exc:
                    with errors_lock:
                        errors.append(
                            ThreadError(thread_id=thread_id, iteration=iteration, exception=exc)
                        )

        threads: list[threading.Thread] = []
        for tid in range(n_readers):
            t = threading.Thread(target=_reader, args=(tid,), daemon=True)
            threads.append(t)
        for tid in range(n_writers):
            t = threading.Thread(target=_writer, args=(n_readers + tid,), daemon=True)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        return errors


# ---------------------------------------------------------------------------
# Thread-safe counter for verifying concurrent op counts
# ---------------------------------------------------------------------------


class AtomicCounter:
    """Simple thread-safe counter backed by a Lock."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        with self._lock:
            self._value += delta
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> None:
        with self._lock:
            self._value = value
