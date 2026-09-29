"""
ci_metrics/store.py — Append-only JSON-lines store for CI run records.

Records are written as newline-delimited JSON (one object per line) to a
configurable file path.  Reads load all lines and parse them back to
:class:`CIRunRecord` objects.

Thread-safety: writes use ``fcntl.flock`` (POSIX) with a fallback no-op on
Windows so the store is safe for concurrent CI runners appending to a shared
artifact store.

Usage::

    store = MetricsStore(Path("ci_metrics/history.jsonl"))
    store.append(record)
    recent = store.recent(n=10)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ci_metrics.contracts import CIRunRecord

logger = logging.getLogger(__name__)

# Default path — resolved relative to repo root at import time
DEFAULT_STORE_PATH = Path("ci_metrics") / "history.jsonl"


class MetricsStore:
    """Append-only JSON-lines store for :class:`CIRunRecord` objects.

    Args:
        path: Path to the ``.jsonl`` file.  Created on first ``append``.
    """

    def __init__(self, path: Path | str = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, record: CIRunRecord) -> None:
        """Append *record* to the store.  Creates the file if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.as_dict(), separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as fh:
            _lock(fh)
            try:
                fh.write(line + "\n")
            finally:
                _unlock(fh)
        logger.debug("Appended run_id=%s to %s", record.run_id, self.path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all(self) -> list[CIRunRecord]:
        """Return all stored records in insertion order."""
        if not self.path.exists():
            return []
        records = []
        for lineno, line in enumerate(self.path.read_text("utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CIRunRecord.from_dict(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed record at %s line %d: %s", self.path, lineno, exc
                )
        return records

    def recent(self, n: int = 20) -> list[CIRunRecord]:
        """Return the *n* most recent records."""
        return self.all()[-n:]

    def for_branch(self, branch: str) -> list[CIRunRecord]:
        """Return all records for a specific branch."""
        return [r for r in self.all() if r.branch == branch]

    def metric_series(self, metric_name: str, n: int = 50) -> list[tuple[str, float]]:
        """Return ``[(run_id, value), …]`` for *metric_name* over the last *n* runs.

        Runs that did not record this metric are skipped.
        """
        result = []
        for record in self.recent(n):
            for m in record.metrics:
                if m.name == metric_name:
                    result.append((record.run_id, m.value))
                    break
        return result

    def __len__(self) -> int:
        return len(self.all())


# ---------------------------------------------------------------------------
# File-locking helpers
# ---------------------------------------------------------------------------


def _lock(fh) -> None:  # type: ignore[no-untyped-def]
    """Acquire an exclusive advisory lock on *fh* (POSIX only; no-op on Windows)."""
    if sys.platform != "win32":
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:
            pass


def _unlock(fh) -> None:  # type: ignore[no-untyped-def]
    """Release the advisory lock on *fh* (POSIX only; no-op on Windows)."""
    if sys.platform != "win32":
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_UN)
        except ImportError:
            pass
