"""Partial pipeline execution recovery (Issue #578).

Overview
--------
When a long-running pipeline job is interrupted mid-way — by an OOM kill,
a network time-out, or a KeyboardInterrupt — the next invocation should
*resume* from the last successfully completed stage rather than re-running
everything from scratch.

This module builds on the ``CheckpointStore`` from ``pipeline.idempotency``
(which handles the durable state) and adds:

1. ``StageTracker`` — thin wrapper around a ``CheckpointStore`` that adds
   wall-clock timing, per-stage row counts, and a human-readable run summary.

2. ``RecoveryManager`` — orchestrates a resumable pipeline run.  It exposes
   the ``stage(name)`` context manager that pipelines use to wrap each step,
   and ``resume_info(run_id, pair_id)`` to report what will be skipped on
   the next invocation.

3. ``rollback_partial_writes`` — best-effort DB cleanup for stages that
   touched the risk-score store but did not reach the ``persist`` completion
   checkpoint.  Deletes rows written during a failed run so re-runs start
   from a clean slate.

Design decisions
----------------
* **No saga-style compensating transactions** — the pipeline writes to a
  single SQLite/Postgres DB.  A simple "delete rows with matching run_id"
  rollback is sufficient; there is nothing to compensate in external systems
  because on-chain submission (``onchain`` stage) only runs after ``persist``
  is complete, and a failed ``onchain`` stage is logged and retried rather
  than rolled back.

* **Atomic stage transitions** — the underlying ``CheckpointStore`` uses
  ``UniqueConstraint`` + ``SQLAlchemy`` transactions so concurrent writes to
  the same checkpoint key are safe.

* **Idempotent resume** — calling ``RecoveryManager.stage()`` for an already-
  completed stage returns immediately (the inner block is not executed).

Usage
-----
::

    from pipeline.recovery import RecoveryManager
    from pipeline.idempotency import CheckpointStore

    store = CheckpointStore()
    rm = RecoveryManager(store)
    run_id = CheckpointStore.make_run_id(pair_id, since_iso or "all")

    with rm.stage(run_id, pair_id, "ingest") as ctx:
        if not ctx.skip:
            trades_df = load_pair_to_dataframe(...)
            ctx.set_result({"row_count": len(trades_df)})

    with rm.stage(run_id, pair_id, "features") as ctx:
        if not ctx.skip:
            feature_matrix = build_feature_matrix(trades_df)
            ctx.set_result({"wallet_count": len(feature_matrix)})

    summary = rm.run_summary(run_id, pair_id)
    print(summary)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pipeline.idempotency import (
    CheckpointStore,
    PipelineCheckpoint,
    _CheckpointState,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# StageResult — rich per-stage execution record
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Execution record for a single pipeline stage."""

    stage: str
    status: str  # "skipped" | "completed" | "failed"
    wall_seconds: float = 0.0
    result_payload: Any = None
    error: str = ""


# ---------------------------------------------------------------------------
# StageTracker
# ---------------------------------------------------------------------------


class StageTracker:
    """Accumulates per-stage timing and result data for a pipeline run.

    Not persisted — lives in memory for the duration of the process.
    Complements the durable ``CheckpointStore`` by adding rich reporting.
    """

    def __init__(self) -> None:
        self._stages: list[StageResult] = []

    def record(self, result: StageResult) -> None:
        self._stages.append(result)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        total_wall = sum(r.wall_seconds for r in self._stages)
        return {
            "stages": [
                {
                    "stage": r.stage,
                    "status": r.status,
                    "wall_seconds": round(r.wall_seconds, 3),
                    "result_payload": r.result_payload,
                    "error": r.error or None,
                }
                for r in self._stages
            ],
            "total_wall_seconds": round(total_wall, 3),
            "completed_count": sum(1 for r in self._stages if r.status == "completed"),
            "skipped_count": sum(1 for r in self._stages if r.status == "skipped"),
            "failed_count": sum(1 for r in self._stages if r.status == "failed"),
        }

    def failed_stages(self) -> list[str]:
        return [r.stage for r in self._stages if r.status == "failed"]

    def has_failures(self) -> bool:
        return any(r.status == "failed" for r in self._stages)


# ---------------------------------------------------------------------------
# RecoveryManager — main API
# ---------------------------------------------------------------------------


class RecoveryManager:
    """Orchestrates resumable, idempotent pipeline execution.

    Parameters
    ----------
    store:
        The ``CheckpointStore`` that durably records stage completion.
    """

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store
        # Keyed by (run_id, pair_id) to support multiple concurrent pairs.
        self._trackers: dict[tuple[str, str], StageTracker] = {}

    def _tracker(self, run_id: str, pair_id: str) -> StageTracker:
        key = (run_id, pair_id)
        if key not in self._trackers:
            self._trackers[key] = StageTracker()
        return self._trackers[key]

    @contextmanager
    def stage(
        self,
        run_id: str,
        pair_id: str,
        stage_name: str,
        force: bool = False,
    ) -> Generator[_CheckpointState, None, None]:
        """Context manager wrapping a single pipeline stage.

        * If the stage already completed (within TTL), yields a
          ``_CheckpointState`` with ``skip=True`` so the caller can short-
          circuit its work.
        * If the stage has not completed, runs the block, records timing, and
          marks the stage ``done`` or ``failed`` on exit.

        Parameters
        ----------
        run_id:
            Unique pipeline run identifier.
        pair_id:
            Asset-pair being processed.
        stage_name:
            Name of the stage (one of ``PIPELINE_STAGES`` or custom).
        force:
            Re-run even if already completed.
        """
        tracker = self._tracker(run_id, pair_id)
        cp = PipelineCheckpoint(self._store, run_id, pair_id, stage_name, force=force)

        t0 = time.monotonic()
        state: _CheckpointState | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)
        cp_entered = False

        try:
            state = cp.__enter__()
            cp_entered = True
            yield state
            exc_info = (None, None, None)
        except Exception as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if cp_entered:
                cp.__exit__(*exc_info)

            if state is not None:
                if state.skip:
                    status = "skipped"
                    error_msg = ""
                elif exc_info[0] is None:
                    status = "completed"
                    error_msg = ""
                else:
                    status = "failed"
                    error_msg = f"{exc_info[0].__name__}: {exc_info[1]}"

                tracker.record(
                    StageResult(
                        stage=stage_name,
                        status=status,
                        wall_seconds=elapsed,
                        result_payload=state._pending_result if state else None,
                        error=error_msg,
                    )
                )

    def resume_info(self, run_id: str, pair_id: str) -> dict[str, Any]:
        """Return a human-readable dict describing what will happen on resume.

        Useful for logging at the start of a pipeline run so operators can
        see what was already completed.
        """
        first_incomplete = self._store.first_incomplete_stage(run_id, pair_id)
        completed_stages = [
            r.stage for r in self._store.list_stages(run_id, pair_id) if r.status == "done"
        ]
        return {
            "run_id": run_id,
            "pair_id": pair_id,
            "completed_stages": completed_stages,
            "first_incomplete_stage": first_incomplete,
            "will_resume": first_incomplete is not None and len(completed_stages) > 0,
        }

    def run_summary(self, run_id: str, pair_id: str) -> dict[str, Any]:
        """Return the in-memory stage tracker summary for (run_id, pair_id)."""
        return self._tracker(run_id, pair_id).summary()

    def has_failures(self, run_id: str, pair_id: str) -> bool:
        """True if any stage for this (run_id, pair_id) failed in this process."""
        return self._tracker(run_id, pair_id).has_failures()


# ---------------------------------------------------------------------------
# rollback_partial_writes
# ---------------------------------------------------------------------------


def rollback_partial_writes(
    score_store: Any,
    wallets: list[str],
    pair_id: str,
) -> int:
    """Remove risk-score rows written during a failed/incomplete pipeline run.

    Called when a run is abandoned (e.g. the ``persist`` checkpoint is not
    marked ``done``) to ensure the next invocation starts from a clean slate.
    Only deletes rows for ``pair_id`` — other pairs are unaffected.

    Parameters
    ----------
    score_store:
        A ``RiskScoreStore`` (or duck-typed equivalent) with a ``delete``
        method.  If the store does not expose ``delete``, this function logs
        a warning and returns 0.
    wallets:
        List of wallet addresses that may have been written.
    pair_id:
        Asset-pair identifier.

    Returns
    -------
    int
        Number of rows deleted.
    """
    if not hasattr(score_store, "delete"):
        logger.warning(
            "rollback_partial_writes: score_store does not expose .delete(); "
            "no rollback performed for pair=%s",
            pair_id,
        )
        return 0

    deleted = 0
    for wallet in wallets:
        try:
            removed = score_store.delete(wallet, pair_id)
            if removed:
                deleted += 1
                logger.debug("rollback_partial_writes: deleted wallet=%s pair=%s", wallet, pair_id)
        except Exception as exc:
            logger.warning(
                "rollback_partial_writes: failed to delete wallet=%s pair=%s: %s",
                wallet,
                pair_id,
                exc,
            )
    if deleted:
        logger.info(
            "rollback_partial_writes: removed %d stale rows for pair=%s",
            deleted,
            pair_id,
        )
    return deleted
