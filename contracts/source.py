"""Trade-source contracts for the ingestion layer boundary.

Defines the ``TradeSource`` protocol that every trade data source
(Horizon SSE, Kafka, historical loader, simulator, …) satisfies.
This decouples pipeline orchestration from specific source
implementations.
"""

from __future__ import annotations

import typing
from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class TradeSource(Protocol):
    """Protocol for anything that produces a stream of trades.

    Implementations must provide a ``stream_trades`` method that
    yields ``dict`` objects conforming to the trade schema.

    Usage::

        source: TradeSource = HorizonSSESource(...)
        for trade in source.stream_trades("USDC_native"):
            process(trade)
    """

    def stream_trades(
        self,
        asset_pair: str,
        since: float | None = None,
    ) -> Iterator[dict[str, typing.Any]]:
        """Yield trade dicts for *asset_pair*, optionally from *since* (unix ts)."""
