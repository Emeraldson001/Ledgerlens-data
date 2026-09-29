"""Checkpointing support for long-running data workflows.

Long ingestion backfills, feature backfills, and batch scoring jobs can run
for hours and are frequently interrupted (OOM kill, deploy, network blip).
Without durable progress tracking, a restart means reprocessing everything
from scratch — wasted compute and, for anything with side effects (writes,
alerts, external API calls), risk of duplicate work.

``CheckpointStore`` persists workflow progress to disk as versioned,
checksummed JSON with atomic writes, so a checkpoint file is never observed
half-written. ``CheckpointedWorkflow`` builds step-level and item-level
resumption on top of the store: skip steps that already finished, and within
a step, skip individual items that were already processed before the crash.

Usage:
    store = CheckpointStore(".checkpoints")
    workflow = CheckpointedWorkflow(store, workflow_id="backfill_amm_2026")

    for step_name in workflow.remaining_steps(["fetch", "transform", "load"]):
        with workflow.step(step_name):
            do_work(step_name)

    # Item-level resumption within a single step (e.g. batch scoring):
    with workflow.step("score_wallets") as ckpt:
        for wallet_id in ckpt.remaining_items(all_wallet_ids):
            score(wallet_id)
            ckpt.mark_done(wallet_id)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

#: Default retention window (in days) for stale-checkpoint cleanup. Checkpoint
#: files accumulate per ``workflow_id`` and are never pruned automatically, so
#: long-lived deployments should call ``CheckpointStore.cleanup_stale`` to purge
#: checkpoints older than this window (see docs/checkpointing.md).
CHECKPOINT_RETENTION_DAYS = 30


class CheckpointError(Exception):
    """Base class for checkpoint failures. Carries diagnostics for triage."""

    def __init__(self, message: str, *, workflow_id: str, path: str | None = None):
        self.workflow_id = workflow_id
        self.path = path
        super().__init__(
            f"{message} (workflow_id={workflow_id!r}" + (f", path={path!r}" if path else "") + ")"
        )


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint file exists but fails integrity validation.

    This means the on-disk state cannot be trusted for resumption. Callers
    should treat this as a hard failure requiring operator attention rather
    than silently restarting from scratch, since silent restart can mask
    duplicate side effects from a partially-applied previous run.
    """


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint was written by an incompatible schema version."""


def _checksum(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory + os.replace.

    Guarantees a reader never observes a partially-written checkpoint, even
    if the process is killed mid-write — the rename is atomic on POSIX and
    on Windows (as of Python's os.replace implementation).
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@dataclass
class CheckpointRecord:
    workflow_id: str
    schema_version: int
    updated_at: float
    completed_steps: list[str] = field(default_factory=list)
    step_progress: dict[str, list[str]] = field(default_factory=dict)


class CheckpointStore:
    """Durable, checksummed checkpoint persistence for a directory of workflows."""

    def __init__(self, directory: str = ".checkpoints"):
        self.directory = directory

    def _path(self, workflow_id: str) -> str:
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in workflow_id)
        return os.path.join(self.directory, f"{safe_id}.json")

    def exists(self, workflow_id: str) -> bool:
        return os.path.exists(self._path(workflow_id))

    def load(self, workflow_id: str) -> CheckpointRecord | None:
        path = self._path(workflow_id)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            raw = json.load(f)

        stored_checksum = raw.pop("checksum", None)
        expected_checksum = _checksum(raw)
        if stored_checksum != expected_checksum:
            raise CheckpointCorruptionError(
                "checksum mismatch — checkpoint file was truncated, edited, "
                "or written by a non-atomic process",
                workflow_id=workflow_id,
                path=path,
            )

        if raw.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointVersionError(
                f"checkpoint schema_version={raw.get('schema_version')} is incompatible "
                f"with the running schema_version={SCHEMA_VERSION}; delete the checkpoint "
                "or run a migration before resuming",
                workflow_id=workflow_id,
                path=path,
            )

        return CheckpointRecord(
            workflow_id=raw["workflow_id"],
            schema_version=raw["schema_version"],
            updated_at=raw["updated_at"],
            completed_steps=list(raw.get("completed_steps", [])),
            step_progress={k: list(v) for k, v in raw.get("step_progress", {}).items()},
        )

    def save(self, record: CheckpointRecord) -> None:
        body = {
            "workflow_id": record.workflow_id,
            "schema_version": record.schema_version,
            "updated_at": record.updated_at,
            "completed_steps": record.completed_steps,
            "step_progress": record.step_progress,
        }
        body["checksum"] = _checksum(body)
        _atomic_write_json(self._path(record.workflow_id), body)

    def clear(self, workflow_id: str) -> None:
        path = self._path(workflow_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cleared checkpoint for workflow_id=%s", workflow_id)

    def cleanup_stale(self, retention_days: int = CHECKPOINT_RETENTION_DAYS) -> list[str]:
        """Remove checkpoint files older than *retention_days*.

        Unbounded checkpoint accumulation is a real disk-usage risk for
        long-lived deployments, so stale checkpoints are pruned based on each
        file's mtime. Returns the list of removed checkpoint file paths.
        """
        removed: list[str] = []
        cutoff = time.time() - retention_days * 86400
        if not os.path.isdir(self.directory):
            return removed
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    logger.info(
                        "Removed stale checkpoint %s (older than %d days)",
                        path,
                        retention_days,
                    )
                    removed.append(path)
            except OSError:  # pragma: no cover - file vanished between scan and stat
                continue
        return removed


class StepCheckpoint:
    """Tracks item-level progress within a single workflow step."""

    def __init__(self, workflow: CheckpointedWorkflow, step_name: str):
        self._workflow = workflow
        self.step_name = step_name
        self._done: set[str] = set(workflow._record.step_progress.get(step_name, []))

    def remaining_items(self, items: Iterable[Any], *, key: Any = str) -> Iterator[Any]:
        """Yield only items not yet marked done, in original order."""
        for item in items:
            item_key = key(item)
            if item_key in self._done:
                continue
            yield item

    def mark_done(self, item: Any, *, key: Any = str, flush: bool = True) -> None:
        item_key = key(item)
        self._done.add(item_key)
        self._workflow._record.step_progress[self.step_name] = sorted(self._done)
        if flush:
            self._workflow._flush()

    def is_done(self, item: Any, *, key: Any = str) -> bool:
        return key(item) in self._done


class CheckpointedWorkflow:
    """Resumable multi-step workflow backed by a ``CheckpointStore``.

    A workflow is identified by ``workflow_id``. Progress (which steps
    finished, and which items within a step finished) survives process
    restarts. Call ``clear()`` to intentionally discard progress and start
    over — e.g. when the underlying source data changed.
    """

    def __init__(self, store: CheckpointStore, workflow_id: str):
        self.store = store
        self.workflow_id = workflow_id
        loaded = store.load(workflow_id)
        self._record = loaded or CheckpointRecord(
            workflow_id=workflow_id,
            schema_version=SCHEMA_VERSION,
            updated_at=time.time(),
            completed_steps=[],
            step_progress={},
        )

    def _flush(self) -> None:
        self._record.updated_at = time.time()
        self.store.save(self._record)

    def is_step_done(self, step_name: str) -> bool:
        return step_name in self._record.completed_steps

    def remaining_steps(self, steps: Iterable[str]) -> Iterator[str]:
        """Yield steps not yet completed, skipping finished ones and logging why."""
        for step_name in steps:
            if self.is_step_done(step_name):
                logger.info(
                    "Skipping already-completed step %r for workflow_id=%s",
                    step_name,
                    self.workflow_id,
                )
                continue
            yield step_name

    class _StepContext:
        def __init__(self, workflow: CheckpointedWorkflow, step_name: str):
            self.workflow = workflow
            self.step_name = step_name
            self.checkpoint = StepCheckpoint(workflow, step_name)

        def __enter__(self) -> StepCheckpoint:
            return self.checkpoint

        def __exit__(self, exc_type, exc, tb) -> bool:
            if exc_type is None:
                if self.step_name not in self.workflow._record.completed_steps:
                    self.workflow._record.completed_steps.append(self.step_name)
                self.workflow._flush()
            else:
                logger.warning(
                    "Step %r failed for workflow_id=%s: %s — progress up to this point "
                    "is saved, step will be retried on next run",
                    self.step_name,
                    self.workflow.workflow_id,
                    exc,
                )
                self.workflow._flush()
            return False

    def step(self, step_name: str) -> CheckpointedWorkflow._StepContext:
        return CheckpointedWorkflow._StepContext(self, step_name)

    def clear(self) -> None:
        self.store.clear(self.workflow_id)
        self._record = CheckpointRecord(
            workflow_id=self.workflow_id,
            schema_version=SCHEMA_VERSION,
            updated_at=time.time(),
            completed_steps=[],
            step_progress={},
        )
