"""Streaming analysis primitives for high-volume ledger feeds.

Provides composable, thread-safe building blocks for real-time analysis
of trade and account-activity streams.  Each primitive is designed to be
efficient at high throughput and to integrate with the contract-based
ingestion framework.

Primitives
----------
- ``SlidingWindowAggregator`` — O(1) sliding-window statistics
- ``RateCounter`` — per-second / per-minute rate tracking
- ``ThroughputMeter`` — bandwidth monitoring in trades/sec
- ``BoundedOrderedBuffer`` — bounded, ordered buffer with watermark tracking
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, TypeVar

T = TypeVar("T")


class SlidingWindowAggregator:
    """O(1) sliding-window statistics over a stream of numeric values.

    Maintains a deque of (timestamp, value) pairs, evicting entries
    outside the window on each update.  Thread-safe.

    Usage::

        agg = SlidingWindowAggregator(window_seconds=60.0)
        agg.add(10.0)
        agg.add(20.0)
        agg.mean()   # 15.0
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._lock = threading.Lock()
        self._values: deque[tuple[float, float]] = deque()
        self._sum = 0.0
        self._count = 0

    def add(self, value: float, timestamp: float | None = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._values.append((now, value))
            self._sum += value
            self._count += 1
            self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._values and self._values[0][0] < cutoff:
            _, val = self._values.popleft()
            self._sum -= val
            self._count -= 1

    def mean(self) -> float:
        with self._lock:
            self._evict(time.time())
            return self._sum / self._count if self._count else 0.0

    def count(self) -> int:
        with self._lock:
            self._evict(time.time())
            return self._count

    def max(self) -> float:
        with self._lock:
            self._evict(time.time())
            return max(v for _, v in self._values) if self._values else 0.0

    def min(self) -> float:
        with self._lock:
            self._evict(time.time())
            return min(v for _, v in self._values) if self._values else 0.0

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._sum = 0.0
            self._count = 0


class RateCounter:
    """Tracks per-second and per-minute rates of events.

    Thread-safe.  Each call to ``tick()`` increments the counter;
    ``rate_per_sec()`` and ``rate_per_min()`` return the current rate
    based on a sliding window.
    """

    def __init__(self, window_seconds: float = 10.0) -> None:
        self._agg = SlidingWindowAggregator(window_seconds=window_seconds)
        self._lock = threading.Lock()

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self._agg.add(1.0)

    def rate_per_sec(self) -> float:
        with self._lock:
            return self._agg.mean()

    def rate_per_min(self) -> float:
        with self._lock:
            return self._agg.mean() * 60.0

    def total(self) -> int:
        return self._agg.count()

    def reset(self) -> None:
        self._agg.reset()


class ThroughputMeter:
    """Measures data throughput in items (or bytes) per second.

    Usage::

        meter = ThroughputMeter()
        meter.record(100)  # 100 items processed
        meter.record(150)
        meter.items_per_sec()  # average over window
    """

    def __init__(self, window_seconds: float = 30.0) -> None:
        self._window = window_seconds
        self._lock = threading.Lock()
        self._records: deque[tuple[float, float]] = deque()

    def record(self, count: float, timestamp: float | None = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._records.append((now, count))
            self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._records and self._records[0][0] < cutoff:
            self._records.popleft()

    def items_per_sec(self) -> float:
        now = time.time()
        with self._lock:
            self._evict(now)
            if not self._records:
                return 0.0
            total = sum(c for _, c in self._records)
            elapsed = now - self._records[0][0]
            return total / elapsed if elapsed > 0 else total


class BoundedOrderedBuffer:
    """Bounded buffer that preserves insertion order and supports watermark tracking.

    Useful for maintaining a fixed-size window of recent events where
    order matters (e.g. trade sequences for feature computation).

    Usage::

        buf = BoundedOrderedBuffer(maxsize=1000)
        buf.push(item)
        buf.push(item2)
        buf.items()  # [item, item2] up to maxsize
        buf.watermark  # highest timestamp seen
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._buffer: deque[Any] = deque(maxlen=maxsize)
        self._watermark: float = 0.0

    def push(self, item: Any, timestamp: float | None = None) -> None:
        with self._lock:
            self._buffer.append(item)
            if timestamp is not None and timestamp > self._watermark:
                self._watermark = timestamp

    def items(self) -> list:
        with self._lock:
            return list(self._buffer)

    @property
    def watermark(self) -> float:
        return self._watermark

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._watermark = 0.0
