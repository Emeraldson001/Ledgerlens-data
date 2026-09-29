"""Feature-store contracts for the features / streaming boundary.

Defines the ``FeatureStore`` protocol that both the batch feature
pipeline and the streaming feature cache satisfy, allowing them to
be swapped without changing callers.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class FeatureStore(Protocol):
    """Interface for a key-value feature store (Redis-backed, in-memory, …).

    Usage::

        store: FeatureStore = RedisFeatureStore(...)
        features = store.get_or_compute(
            wallet="GABCD...",
            pair="USDC_native",
            compute_fn=lambda: {"volume_1h": 42.0},
        )
    """

    def get_or_compute(
        self,
        wallet: str,
        pair: str,
        compute_fn: Callable[[], dict[str, typing.Any]],
    ) -> dict[str, typing.Any]:
        """Return cached features or compute and cache them."""
