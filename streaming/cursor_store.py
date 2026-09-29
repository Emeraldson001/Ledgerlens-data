"""Resumable streaming cursor store for event processing (#525).

Provides durable, atomic cursor persistence across streaming reconnections, worker restarts,
and system crash recoveries.

Supported Store Types:
    - ``FileCursorStore``: Atomic JSON file persistence with lock guards.
    - ``SQLiteCursorStore``: High-throughput SQLite table persistence with WAL mode.
    - ``InMemoryCursorStore``: Volatile store for unit testing or ephemeral streams.

Usage:
    store = get_cursor_store()  # uses config.CURSOR_STORE_TYPE / CURSOR_STORE_PATH
    cursor = store.get_cursor("trades:USDC:XLM", default="now")

    # In event loop:
    store.save_cursor("trades:USDC:XLM", cursor="12345678", metadata={"trade_id": "999"})
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


class BaseCursorStore(ABC):
    """Abstract base class for event stream cursor stores."""

    @abstractmethod
    def get_cursor(self, stream_id: str, default: str = "now") -> str:
        """Retrieve the last persisted cursor for a stream.

        Parameters
        ----------
        stream_id : str
            Unique identifier for the event stream (e.g. ``"trades:USDC:XLM"``).
        default : str
            Fallback cursor if no prior checkpoint exists (default ``"now"``).
        """
        ...

    @abstractmethod
    def save_cursor(
        self,
        stream_id: str,
        cursor: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist cursor position and optional event metadata.

        Parameters
        ----------
        stream_id : str
            Unique stream identifier.
        cursor : str
            Horizon or stream paging token / cursor value.
        metadata : dict[str, Any] | None
            Contextual details (e.g. timestamp, event count, sequence id).
        """
        ...

    @abstractmethod
    def delete_cursor(self, stream_id: str) -> None:
        """Remove saved cursor state for a stream."""
        ...

    @abstractmethod
    def list_cursors(self) -> dict[str, dict[str, Any]]:
        """Return all persisted cursor records."""
        ...

    def close(self) -> None:  # noqa: B027 - optional no-op hook for in-memory stores
        """Release underlying storage resources."""
        pass


class FileCursorStore(BaseCursorStore):
    """Atomic JSON file-based cursor store.

    Uses temporary files and atomic file replacement (``os.replace``) to guarantee
    corruption-free persistence even if a crash occurs mid-write.
    """

    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = os.path.abspath(
            file_path if file_path is not None else config.CURSOR_STORE_PATH
        )
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception as exc:
            logger.error("Failed to load cursor store from %s: %s", self.file_path, exc)
            self._data = {}

    def _flush(self) -> None:
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".cursors_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(temp_path, self.file_path)
        except Exception as exc:
            logger.error("Failed to flush cursor store to %s: %s", self.file_path, exc)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def get_cursor(self, stream_id: str, default: str = "now") -> str:
        with self._lock:
            record = self._data.get(stream_id)
            if record and "cursor" in record:
                return str(record["cursor"])
            return default

    def save_cursor(
        self,
        stream_id: str,
        cursor: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._data[stream_id] = {
                "cursor": str(cursor),
                "updated_at": time.time(),
                "metadata": metadata or {},
            }
            self._flush()

    def delete_cursor(self, stream_id: str) -> None:
        with self._lock:
            if stream_id in self._data:
                del self._data[stream_id]
                self._flush()

    def list_cursors(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._data)


class SQLiteCursorStore(BaseCursorStore):
    """SQLite database-backed cursor store with WAL mode for high-throughput concurrency."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path if db_path is not None else config.CURSOR_STORE_PATH
        if path.endswith(".json"):
            path = path[:-5] + ".db"
        self.db_path = os.path.abspath(path)
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS streaming_cursors (
                        stream_id TEXT PRIMARY KEY,
                        cursor TEXT NOT NULL,
                        metadata TEXT,
                        updated_at REAL NOT NULL
                    )
                    """)
                conn.commit()

    def get_cursor(self, stream_id: str, default: str = "now") -> str:
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT cursor FROM streaming_cursors WHERE stream_id = ?", (stream_id,)
                )
                row = cur.fetchone()
                return str(row[0]) if row else default

    def save_cursor(
        self,
        stream_id: str,
        cursor: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO streaming_cursors (stream_id, cursor, metadata, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(stream_id) DO UPDATE SET
                        cursor=excluded.cursor,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at
                    """,
                    (stream_id, str(cursor), meta_json, now),
                )
                conn.commit()

    def delete_cursor(self, stream_id: str) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM streaming_cursors WHERE stream_id = ?", (stream_id,))
                conn.commit()

    def list_cursors(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT stream_id, cursor, metadata, updated_at FROM streaming_cursors"
                )
                result = {}
                for row in cur.fetchall():
                    s_id, cursor_val, meta_str, updated_at = row
                    meta = {}
                    if meta_str:
                        try:
                            meta = json.loads(meta_str)
                        except Exception:
                            pass
                    result[s_id] = {
                        "cursor": cursor_val,
                        "updated_at": updated_at,
                        "metadata": meta,
                    }
                return result


class InMemoryCursorStore(BaseCursorStore):
    """In-memory cursor store for transient operations or testing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursors: dict[str, dict[str, Any]] = {}

    def get_cursor(self, stream_id: str, default: str = "now") -> str:
        with self._lock:
            record = self._cursors.get(stream_id)
            return record["cursor"] if record else default

    def save_cursor(
        self,
        stream_id: str,
        cursor: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._cursors[stream_id] = {
                "cursor": str(cursor),
                "updated_at": time.time(),
                "metadata": metadata or {},
            }

    def delete_cursor(self, stream_id: str) -> None:
        with self._lock:
            self._cursors.pop(stream_id, None)

    def list_cursors(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._cursors)


class ResumableCursorManager:
    """Helper manager that wraps cursor persistence with batching and auto-checkpointing logic."""

    def __init__(
        self,
        stream_id: str,
        cursor_store: BaseCursorStore | None = None,
        flush_interval_events: int | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.cursor_store = cursor_store or get_cursor_store()
        self.flush_interval_events = (
            flush_interval_events
            if flush_interval_events is not None
            else config.CURSOR_STORE_FLUSH_INTERVAL_EVENTS
        )
        self._event_count = 0
        self._current_cursor = self.cursor_store.get_cursor(self.stream_id, default="now")

    @property
    def current_cursor(self) -> str:
        """Return active cursor position."""
        return self._current_cursor

    def checkpoint(
        self,
        cursor: str,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """Record cursor and flush to store if event threshold is met."""
        self._current_cursor = str(cursor)
        self._event_count += 1

        if force or (self._event_count % max(1, self.flush_interval_events) == 0):
            meta = metadata or {}
            meta["event_count"] = self._event_count
            self.cursor_store.save_cursor(self.stream_id, self._current_cursor, metadata=meta)


_default_store: BaseCursorStore | None = None
_store_lock = threading.Lock()


def get_cursor_store(
    store_type: str | None = None,
    store_path: str | None = None,
) -> BaseCursorStore:
    """Factory function for acquiring a configured cursor store instance."""
    global _default_store
    with _store_lock:
        if store_type is not None or store_path is not None:
            stype = (store_type or config.CURSOR_STORE_TYPE).lower()
            spath = store_path or config.CURSOR_STORE_PATH
            if stype == "sqlite":
                return SQLiteCursorStore(spath)
            elif stype == "memory":
                return InMemoryCursorStore()
            else:
                return FileCursorStore(spath)

        if _default_store is None:
            stype = config.CURSOR_STORE_TYPE.lower()
            if stype == "sqlite":
                _default_store = SQLiteCursorStore(config.CURSOR_STORE_PATH)
            elif stype == "memory":
                _default_store = InMemoryCursorStore()
            else:
                _default_store = FileCursorStore(config.CURSOR_STORE_PATH)

        return _default_store
