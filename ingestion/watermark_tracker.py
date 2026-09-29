"""Watermark tracking for incremental ledger ingestion.

Issue #526 — Stellar Wave advanced build.

Provides a durable, per-pair ``WatermarkTracker`` that records the last
successfully-ingested position for each asset pair as:

- ``paging_token`` — the Horizon paging token of the last consumed trade
  (used as the Horizon cursor for the next page fetch).
- ``ledger_close_time`` — the ``datetime`` of that trade (used for
  time-based filtering and lag metrics).
- ``trade_count`` — running count of trades ingested for this pair.
- ``updated_at`` — wall-clock timestamp of the last update (ISO-8601).

Watermarks are persisted atomically to a JSON file (write-to-tmp + rename)
so a crash mid-write never corrupts the store.  The file is keyed by the
canonical pair identifier string (e.g. ``"USDC:GA5Z.../XLM:native"``).

Configuration:

.. code-block:: bash

    WATERMARK_STORE_PATH=data/watermarks.json   # default location
    WATERMARK_FLUSH_EVERY_N=1                   # write to disk every N updates (0 = manual only)

Usage::

    from ingestion.watermark_tracker import WatermarkTracker

    tracker = WatermarkTracker()

    # Resume: get the last cursor for a pair
    cursor = tracker.get_cursor("USDC:GA5Z.../XLM:native")
    # cursor is None on first run → load from the beginning

    # After successfully processing a page, advance the watermark
    tracker.advance(
        pair_id="USDC:GA5Z.../XLM:native",
        paging_token=last_record["paging_token"],
        ledger_close_time=trade.ledger_close_time,
    )

    # Persist state to disk (called automatically every FLUSH_EVERY_N updates)
    tracker.flush()

    # Reset a specific pair (e.g. after a full re-ingestion)
    tracker.reset("USDC:GA5Z.../XLM:native")

    # Export all watermarks as a plain dict (for monitoring / dashboards)
    state = tracker.to_dict()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_STORE_PATH: str = getattr(config, "WATERMARK_STORE_PATH", "data/watermarks.json")
_FLUSH_EVERY_N: int = int(getattr(config, "WATERMARK_FLUSH_EVERY_N", 1))

# ---------------------------------------------------------------------------
# Watermark data type
# ---------------------------------------------------------------------------


class Watermark:
    """Tracks the last-ingested position for a single asset pair.

    Attributes
    ----------
    pair_id:
        Canonical pair identifier, e.g. ``"USDC:GA5Z.../XLM:native"``.
    paging_token:
        Horizon paging token of the last consumed trade record.  ``None``
        means no trades have been ingested for this pair yet.
    ledger_close_time:
        ``datetime`` (UTC) of the last trade.  ``None`` until the first trade.
    trade_count:
        Running total of trades ingested for this pair.
    updated_at:
        Wall-clock timestamp (UTC) of the most recent :meth:`advance` call.
    """

    __slots__ = ("pair_id", "paging_token", "ledger_close_time", "trade_count", "updated_at")

    def __init__(
        self,
        pair_id: str,
        paging_token: str | None = None,
        ledger_close_time: datetime | None = None,
        trade_count: int = 0,
        updated_at: datetime | None = None,
    ) -> None:
        self.pair_id = pair_id
        self.paging_token = paging_token
        self.ledger_close_time = ledger_close_time
        self.trade_count = trade_count
        self.updated_at = updated_at or datetime.now(UTC)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "pair_id": self.pair_id,
            "paging_token": self.paging_token,
            "ledger_close_time": (
                self.ledger_close_time.isoformat() if self.ledger_close_time else None
            ),
            "trade_count": self.trade_count,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Watermark:
        """Deserialise from a JSON dictionary."""
        lct_raw = data.get("ledger_close_time")
        lct = datetime.fromisoformat(lct_raw) if lct_raw else None

        updated_raw = data.get("updated_at")
        updated = datetime.fromisoformat(updated_raw) if updated_raw else datetime.now(UTC)

        return cls(
            pair_id=data["pair_id"],
            paging_token=data.get("paging_token"),
            ledger_close_time=lct,
            trade_count=int(data.get("trade_count", 0)),
            updated_at=updated,
        )

    def __repr__(self) -> str:
        return (
            f"Watermark(pair_id={self.pair_id!r}, "
            f"paging_token={self.paging_token!r}, "
            f"trade_count={self.trade_count}, "
            f"ledger_close_time={self.ledger_close_time!r})"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class WatermarkTracker:
    """Thread-safe, durable watermark store for incremental ledger ingestion.

    Watermarks are kept in memory and flushed atomically to a JSON file.
    Concurrent access from multiple ingestion threads is safe (one lock
    per tracker instance).  Multiple *processes* each reading/writing the
    same file is safe because writes are atomic (tempfile + rename) and
    re-reads happen on ``__init__``.  For multi-process safety across
    separate deployments, point each process at a different file path.

    Parameters
    ----------
    store_path:
        Path to the JSON watermark store.  Created if absent.
    flush_every_n:
        Automatically flush to disk after every N ``advance`` calls.
        ``0`` disables auto-flush (caller must call :meth:`flush` manually).
    """

    def __init__(
        self,
        store_path: str | None = None,
        *,
        flush_every_n: int | None = None,
    ) -> None:
        self._path = Path(store_path or _STORE_PATH)
        self._flush_every_n: int = flush_every_n if flush_every_n is not None else _FLUSH_EVERY_N
        self._lock = threading.Lock()
        self._watermarks: dict[str, Watermark] = {}
        self._update_counter: int = 0

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state from disk
        self._load()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, pair_id: str) -> Watermark | None:
        """Return the :class:`Watermark` for *pair_id*, or ``None`` if
        no trades have been ingested for this pair yet."""
        with self._lock:
            return self._watermarks.get(pair_id)

    def get_cursor(self, pair_id: str) -> str | None:
        """Return the Horizon paging token for the last-ingested trade of
        *pair_id*, or ``None`` if no trades have been ingested yet.

        Pass the returned value as the ``cursor`` parameter to the Horizon
        paginated trades endpoint to resume from where ingestion left off.
        """
        wm = self.get(pair_id)
        return wm.paging_token if wm else None

    def get_since(self, pair_id: str) -> datetime | None:
        """Return the ``ledger_close_time`` of the last-ingested trade for
        *pair_id*, or ``None`` if no trades have been ingested yet.

        Useful for time-based filtering when Horizon does not accept a
        paging token cursor (e.g. AMM pool endpoints).
        """
        wm = self.get(pair_id)
        return wm.ledger_close_time if wm else None

    def all_pair_ids(self) -> list[str]:
        """Return the list of pair IDs that have at least one watermark."""
        with self._lock:
            return list(self._watermarks.keys())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Export all watermarks as a ``{pair_id: watermark_dict}`` mapping."""
        with self._lock:
            return {pid: wm.to_dict() for pid, wm in self._watermarks.items()}

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def advance(
        self,
        pair_id: str,
        paging_token: str,
        ledger_close_time: datetime,
        *,
        increment: int = 1,
    ) -> Watermark:
        """Advance the watermark for *pair_id* to the given position.

        Parameters
        ----------
        pair_id:
            Canonical pair identifier.
        paging_token:
            Horizon paging token of the last trade consumed in this batch.
        ledger_close_time:
            Ledger close time of that trade.
        increment:
            Number of trades to add to the running ``trade_count``.  For
            page-level updates pass the page size; for record-level updates
            pass ``1`` (default).

        Returns
        -------
        Watermark
            The updated watermark (for convenience).
        """
        now = datetime.now(UTC)
        with self._lock:
            existing = self._watermarks.get(pair_id)
            if existing is None:
                wm = Watermark(
                    pair_id=pair_id,
                    paging_token=paging_token,
                    ledger_close_time=ledger_close_time,
                    trade_count=increment,
                    updated_at=now,
                )
            else:
                if (
                    existing.ledger_close_time is not None
                    and ledger_close_time < existing.ledger_close_time
                ):
                    logger.warning(
                        "Skipping out-of-order watermark update for pair %s: incoming %s is older than current %s",
                        pair_id,
                        ledger_close_time.isoformat(),
                        existing.ledger_close_time.isoformat(),
                    )
                    return existing

                existing.paging_token = paging_token
                existing.ledger_close_time = ledger_close_time
                existing.trade_count += increment
                existing.updated_at = now
                wm = existing
            self._watermarks[pair_id] = wm
            self._update_counter += 1

            if self._flush_every_n > 0 and self._update_counter % self._flush_every_n == 0:
                self._flush_unlocked()

        logger.debug(
            "Watermark advanced: %s paging_token=%s count=%d",
            pair_id,
            paging_token,
            wm.trade_count,
        )
        return wm

    def reset(self, pair_id: str) -> None:
        """Remove the watermark for *pair_id* so the next ingestion run starts
        from the beginning (full re-ingestion of this pair)."""
        with self._lock:
            if pair_id in self._watermarks:
                del self._watermarks[pair_id]
                logger.info("Watermark reset for pair: %s", pair_id)
            self._flush_unlocked()

    def reset_all(self) -> None:
        """Remove **all** watermarks from the store.

        This forces a full re-ingestion on the next run.  Use with care.
        """
        with self._lock:
            self._watermarks.clear()
            self._flush_unlocked()
        logger.warning("All watermarks reset — next ingestion run will start from the beginning")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush the current in-memory state to disk atomically."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Internal flush — caller must hold ``self._lock``."""
        data = {pid: wm.to_dict() for pid, wm in self._watermarks.items()}
        # Write atomically: tmp file in same directory, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp", prefix=".watermarks_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.debug("Watermarks flushed to %s (%d pairs)", self._path, len(data))

    def _load(self) -> None:
        """Load watermarks from disk.  Called once at construction."""
        if not self._path.exists():
            logger.debug("WatermarkTracker: no store at %s, starting fresh", self._path)
            return
        try:
            with self._path.open(encoding="utf-8") as f:
                raw: dict[str, dict] = json.load(f)
            for pid, entry in raw.items():
                entry.setdefault("pair_id", pid)
                self._watermarks[pid] = Watermark.from_dict(entry)
            logger.info(
                "WatermarkTracker loaded %d watermark(s) from %s",
                len(self._watermarks),
                self._path,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(
                "WatermarkTracker: failed to load %s: %s — starting with empty state",
                self._path,
                exc,
            )
            self._watermarks = {}

    # ------------------------------------------------------------------
    # Lag / monitoring helpers
    # ------------------------------------------------------------------

    def lag_seconds(self, pair_id: str) -> float | None:
        """Return the number of seconds since the last trade was ingested for
        *pair_id*, or ``None`` if no trades have been recorded.

        Useful for monitoring pipelines to detect stalled ingestion.
        """
        wm = self.get(pair_id)
        if wm is None or wm.ledger_close_time is None:
            return None
        now = datetime.now(UTC)
        lct = (
            wm.ledger_close_time
            if wm.ledger_close_time.tzinfo is not None
            else wm.ledger_close_time.replace(tzinfo=UTC)
        )
        return (now - lct).total_seconds()

    def summary(self) -> dict[str, Any]:
        """Return a human-readable monitoring summary of all watermarks."""
        with self._lock:
            pairs = []
            for pid, wm in sorted(self._watermarks.items()):
                lag = self.lag_seconds(pid)
                pairs.append(
                    {
                        "pair_id": pid,
                        "paging_token": wm.paging_token,
                        "ledger_close_time": (
                            wm.ledger_close_time.isoformat() if wm.ledger_close_time else None
                        ),
                        "trade_count": wm.trade_count,
                        "lag_seconds": lag,
                    }
                )
        return {
            "store_path": str(self._path),
            "total_pairs": len(pairs),
            "watermarks": pairs,
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> WatermarkTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()

    def __repr__(self) -> str:
        return (
            f"WatermarkTracker(store_path={str(self._path)!r}, "
            f"pairs={len(self._watermarks)}, "
            f"flush_every_n={self._flush_every_n})"
        )
