import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from detection.model_inference import _score_one


class AdaptiveBatchTuner:
    """Dynamically tunes batch size from observed latency and queue depth.

    The tuner watches the recent per-item latency and the pending queue depth
    and nudges the batch size toward a near-optimal tradeoff between throughput
    and latency. Bounds (min/max) prevent pathological oscillation.

    Knobs (all overridable via ``config``):
      * ``BATCH_SCORER_MIN_BATCH``  - lower bound on batch size.
      * ``BATCH_SCORER_MAX_BATCH``  - upper bound on batch size.
      * ``BATCH_SCORER_TARGET_LATENCY`` - latency (seconds) we aim to stay under.
      * ``BATCH_SCORER_STEP``       - multiplicative step per adjustment.
    """

    def __init__(
        self,
        min_batch: int = None,
        max_batch: int = None,
        target_latency: float = None,
        step: float = None,
    ):
        self.min_batch = max(1, min_batch if min_batch is not None else getattr(config, "BATCH_SCORER_MIN_BATCH", 1))
        self.max_batch = max(
            self.min_batch,
            max_batch if max_batch is not None else getattr(config, "BATCH_SCORER_MAX_BATCH", 64),
        )
        self.target_latency = (
            target_latency if target_latency is not None else getattr(config, "BATCH_SCORER_TARGET_LATENCY", 0.5)
        )
        self.step = step if step is not None else getattr(config, "BATCH_SCORER_STEP", 0.25)
        self._batch_size = self.min_batch
        self._lock = threading.Lock()

    @property
    def batch_size(self) -> int:
        with self._lock:
            return self._batch_size

    def observe(self, latency: float, queue_depth: int) -> int:
        """Adjust the batch size given the latest latency and queue depth.

        * If latency is above target, shrink the batch (latency-bound).
        * If latency is comfortably below target and the queue is backing up,
          grow the batch (throughput-bound).
        * Otherwise hold steady to avoid oscillation.
        """
        with self._lock:
            current = self._batch_size
            if latency > self.target_latency:
                new_size = int(current * (1.0 - self.step))
            elif latency < self.target_latency * 0.5 and queue_depth > current:
                new_size = int(current * (1.0 + self.step)) + 1
            else:
                new_size = current
            self._batch_size = max(self.min_batch, min(self.max_batch, new_size))
            return self._batch_size


def score_batch(
    wallets: list[str],
    max_workers: int = config.BATCH_SCORER_WORKERS,
    tuner: AdaptiveBatchTuner = None,
) -> list[dict]:
    """Score a batch of wallets with adaptive batch-size tuning.

    When a ``tuner`` is supplied the batch is processed in adaptive chunks and
    the tuner is fed the observed latency and remaining queue depth after each
    chunk, so the batch size converges on a near-optimal value for the model's
    latency/throughput profile.
    """
    if tuner is None:
        tuner = AdaptiveBatchTuner()

    results = []
    remaining = list(wallets)
    while remaining:
        chunk_size = tuner.batch_size
        chunk, remaining = remaining[:chunk_size], remaining[chunk_size:]
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score_one, w): w for w in chunk}
            for future in as_completed(futures):
                wallet = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"wallet": wallet, "error": str(exc)})
        latency = time.monotonic() - start
        tuner.observe(latency, len(remaining))
    return results
