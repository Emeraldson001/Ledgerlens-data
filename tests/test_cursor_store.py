"""Tests for streaming.cursor_store — crash recovery and cold-start resume.

Covers the crash-recovery contract (issue #683):
- A cursor persisted before a simulated crash is picked up by a new store
  instance (fresh reader) so processing resumes without skipping trades.
- The cold-start case (no cursor yet) resumes from the configured default.

Both tests use the store backends directly (in-memory and file-backed) —
no external stream/broker is touched.
"""

from __future__ import annotations

from streaming.cursor_store import FileCursorStore, InMemoryCursorStore

STREAM_ID = "trades:USDC:XLM"


def test_resume_from_persisted_cursor_after_crash(tmp_path):
    """A new store instance picks up the cursor persisted before the 'crash'."""
    store_path = str(tmp_path / "cursors.json")

    # Before the crash: a producer checkpoints its position.
    crashed_store = FileCursorStore(store_path)
    crashed_store.save_cursor(STREAM_ID, cursor="12345678", metadata={"trade_id": "999"})

    # Simulate the crash: a brand-new instance reads the same durable file.
    fresh_store = FileCursorStore(store_path)
    assert fresh_store.get_cursor(STREAM_ID, default="now") == "12345678"


def test_cold_start_without_a_cursor_uses_the_default():
    """With no persisted cursor, a fresh reader resumes from the configured default."""
    store = InMemoryCursorStore()
    assert store.get_cursor(STREAM_ID, default="now") == "now"
    assert store.get_cursor(STREAM_ID, default="0") == "0"
