"""Durable checkpoint/resume primitive for long-running batch pipelines.

Batch entry points such as ``run_pipeline.py`` and
``scripts/backfill_amm_trades.py`` process a list of independent *units of
work* (asset pairs, AMM pools, backtest windows, ...). Each unit typically
costs one or more paginated Horizon API calls plus feature computation —
expensive and rate-limited (see ``ingestion/rate_limiter.py``). Without a
checkpoint, a crash or Ctrl-C partway through means starting over from the
first unit.

``PipelineCheckpoint`` gives any such pipeline the same contract::

    ckpt = PipelineCheckpoint.load_or_create(
        path=args.checkpoint_file,
        pipeline="run_pipeline",
        fingerprint_inputs={"since": str(args.since), "pairs": sorted(pair_ids)},
        fresh=args.fresh,
    )
    for unit_id in ckpt.pending(all_unit_ids):
        try:
            result = do_work(unit_id)
        except Exception as exc:
            ckpt.record_failure(unit_id, exc)
            continue
        ckpt.record_success(unit_id, metadata={"rows": len(result)})

Design contract
----------------
* State is a single JSON file, written atomically (temp file + ``os.replace``)
  after every unit completes — a crash never corrupts the file and never
  loses progress beyond the in-flight unit.
* ``fingerprint_inputs`` must capture every argument that changes what
  "done" means for a unit (date ranges, watched pairs, feature flags, ...).
  Resuming a checkpoint with different inputs raises
  :class:`CheckpointMismatchError`, which reports exactly which inputs
  changed instead of silently mixing incompatible runs.
* Failed units are recorded, not silently dropped, so a resumed run retries
  them automatically, and :meth:`PipelineCheckpoint.summary` gives
  maintainers an at-a-glance diagnostic of what succeeded, failed, and
  remains pending.
* An optional ``artifact_path`` per completed unit lets a pipeline cache
  expensive per-unit intermediate output (e.g. a Parquet file of fetched
  trades) and skip recomputation entirely on resume, not just skip
  re-marking the unit done.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

#: Bump when the on-disk JSON shape changes so old checkpoints fail loudly
#: instead of being silently misinterpreted.
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(Exception):
    """Base class for checkpoint-related failures."""


class CheckpointMismatchError(CheckpointError):
    """Raised when an existing checkpoint belongs to a different run."""


class CheckpointCorruptError(CheckpointError):
    """Raised when the checkpoint file on disk cannot be parsed or trusted."""


def compute_fingerprint(inputs: dict[str, Any]) -> str:
    """Stable short hash of *inputs* — anything that changes what "done" means."""
    payload = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _describe_input_diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    keys = sorted(set(old) | set(new))
    lines = []
    for key in keys:
        old_val, new_val = old.get(key, "<unset>"), new.get(key, "<unset>")
        if old_val != new_val:
            lines.append(f"  {key}: {old_val!r} -> {new_val!r}")
    if not lines:
        return "  (no field-level diff — hash collision or non-JSON input)"
    return "\n".join(lines)


@dataclass
class PipelineCheckpoint:
    """Tracks unit-of-work completion for one pipeline run, persisted to disk."""

    path: Path
    pipeline: str
    fingerprint: str
    fingerprint_inputs: dict[str, Any]
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: str = field(default_factory=_now_iso)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        pipeline: str,
        fingerprint_inputs: dict[str, Any],
        *,
        fresh: bool = False,
    ) -> PipelineCheckpoint:
        """Load the checkpoint at *path* if it matches, else start a fresh one.

        Args:
            path: JSON file used to persist progress.
            pipeline: Logical pipeline name (e.g. ``"run_pipeline"``). Guards
                against pointing two different pipelines at the same file.
            fingerprint_inputs: JSON-serialisable dict of every run parameter
                that affects what "done" means for a unit of work.
            fresh: Discard any existing checkpoint at *path* and start over.

        Raises:
            CheckpointMismatchError: an existing checkpoint belongs to a
                different pipeline, or was created with different
                ``fingerprint_inputs``, and ``fresh`` is False.
            CheckpointCorruptError: the file exists but is not valid JSON or
                does not match the expected schema version.
        """
        path = Path(path)
        fingerprint = compute_fingerprint(fingerprint_inputs)

        if fresh and path.exists():
            logger.info("[checkpoint] --fresh requested — discarding %s", path)
            path.unlink()

        if not path.exists():
            ckpt = cls(
                path=path,
                pipeline=pipeline,
                fingerprint=fingerprint,
                fingerprint_inputs=fingerprint_inputs,
            )
            ckpt._save()
            logger.info("[checkpoint] Starting fresh run for %s at %s", pipeline, path)
            return ckpt

        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError(
                f"Checkpoint file {path} exists but could not be parsed ({exc}). "
                "Delete it manually or re-run with --fresh to start over."
            ) from exc

        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptError(
                f"Checkpoint file {path} has schema_version={raw.get('schema_version')!r}, "
                f"expected {CHECKPOINT_SCHEMA_VERSION}. Re-run with --fresh to start over."
            )

        if raw.get("pipeline") != pipeline:
            raise CheckpointMismatchError(
                f"Checkpoint file {path} belongs to pipeline {raw.get('pipeline')!r}, "
                f"not {pipeline!r}. Point --checkpoint-file at a different path, or use "
                "--fresh if this is intentional."
            )

        if raw.get("fingerprint") != fingerprint:
            diff = _describe_input_diff(raw.get("fingerprint_inputs", {}), fingerprint_inputs)
            raise CheckpointMismatchError(
                f"Checkpoint file {path} was created for a different run configuration.\n"
                f"Changed inputs:\n{diff}\n"
                "Re-run with --fresh to discard the stale checkpoint and start over, or "
                "point --checkpoint-file at a new path for this configuration."
            )

        ckpt = cls(
            path=path,
            pipeline=pipeline,
            fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            completed=raw.get("completed", {}),
            failed=raw.get("failed", {}),
            started_at=raw.get("started_at", _now_iso()),
        )
        logger.info(
            "[checkpoint] Resuming %s from %s (%d done, %d previously failed)",
            pipeline,
            path,
            len(ckpt.completed),
            len(ckpt.failed),
        )
        return ckpt

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self.completed

    def pending(self, unit_ids: list[str]) -> list[str]:
        """Return *unit_ids* minus those already completed, preserving order."""
        skipped = [u for u in unit_ids if self.is_done(u)]
        if skipped:
            preview = ", ".join(skipped[:10]) + (", ..." if len(skipped) > 10 else "")
            logger.info(
                "[checkpoint] Skipping %d already-completed unit(s): %s", len(skipped), preview
            )
        return [u for u in unit_ids if not self.is_done(u)]

    def artifact_path(self, unit_id: str) -> str | None:
        """Path to a cached artifact for a completed unit, if one was recorded."""
        entry = self.completed.get(unit_id)
        return entry.get("artifact_path") if entry else None

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------

    def record_success(
        self,
        unit_id: str,
        *,
        artifact_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark *unit_id* done and persist immediately."""
        self.completed[unit_id] = {
            "completed_at": _now_iso(),
            "artifact_path": artifact_path,
            "metadata": metadata or {},
        }
        self.failed.pop(unit_id, None)
        self._save()

    def record_failure(self, unit_id: str, error: BaseException | str) -> None:
        """Record that *unit_id* failed this attempt; it stays pending for the next resume."""
        prev_attempts = self.failed.get(unit_id, {}).get("attempts", 0)
        self.failed[unit_id] = {
            "failed_at": _now_iso(),
            "error": str(error),
            "attempts": prev_attempts + 1,
        }
        self._save()
        logger.error(
            "[checkpoint] Unit %r failed (attempt %d): %s — will retry on the next --resume",
            unit_id,
            prev_attempts + 1,
            error,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Snapshot for end-of-run logging: counts plus which unit IDs failed."""
        return {
            "pipeline": self.pipeline,
            "path": str(self.path),
            "completed": len(self.completed),
            "failed": sorted(self.failed),
            "started_at": self.started_at,
        }

    def clear(self) -> None:
        """Delete the checkpoint file (e.g. once a caller confirms a fully clean run)."""
        if self.path.exists():
            self.path.unlink()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "pipeline": self.pipeline,
            "fingerprint": self.fingerprint,
            "fingerprint_inputs": self.fingerprint_inputs,
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            "completed": self.completed,
            "failed": self.failed,
        }
        fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
