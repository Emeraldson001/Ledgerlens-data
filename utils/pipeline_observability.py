"""Structured observability for pipeline execution.

`run_pipeline.py` (and any future batch pipeline) runs several stages per
invocation with no way to correlate log lines from a single run, no
stage-level timing, and failures surface as a bare traceback with no
indication of which stage — or which asset pair — was in flight.
`PipelineRun` fixes this:

- Every stage started under a `PipelineRun` shares a `run_id`, attached to
  every log line via `extra=` (surfaced as a real field when
  `LOG_FORMAT=json`) so a single run's stages can be filtered/grepped
  together.
- Each stage is wrapped in an OTel span (`utils.tracing.get_tracer`) with
  `run_id` / `stage` / caller-supplied attributes / `duration_ms`, so
  failures show up in distributed traces even when sampled out of logs.
- On stage failure, the exception is logged with full stage context (stage
  name, run_id, elapsed_ms, and any caller attributes such as `pair_id`)
  before re-raising — actionable diagnostics instead of a bare traceback.
- `PipelineRun.summary()` returns a structured dict of stage timings and
  outcomes for a final run-level log line.

Usage:
    run = PipelineRun("detection_pipeline")
    with run.stage("load_trades", pair_id=pair_id):
        ...
    logger.info("Pipeline run complete", extra=run.summary())
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from utils.logging import get_logger
from utils.tracing import get_tracer

logger = get_logger(__name__)


class PipelineRun:
    """Tracks structured, correlated observability for one pipeline invocation."""

    def __init__(self, name: str):
        self.name = name
        self.run_id = uuid.uuid4().hex[:16]
        self._tracer = get_tracer(f"pipeline.{name}")
        self._start = time.monotonic()
        self.stages: list[dict[str, Any]] = []
        logger.info("Pipeline run started", extra={"run_id": self.run_id, "pipeline": self.name})

    @contextmanager
    def stage(self, stage_name: str, **attributes: Any) -> Iterator[None]:
        """Wrap one pipeline stage with a span plus structured start/end/failure logs."""
        start = time.monotonic()
        log_ctx = {
            "run_id": self.run_id,
            "pipeline": self.name,
            "stage": stage_name,
            **attributes,
        }
        logger.info("Stage started: %s", stage_name, extra=log_ctx)

        with self._tracer.start_as_current_span(stage_name) as span:
            span.set_attribute("run_id", self.run_id)
            for key, value in attributes.items():
                span.set_attribute(key, value)
            try:
                yield
            except Exception as exc:
                elapsed_ms = round((time.monotonic() - start) * 1000, 1)
                span.record_exception(exc)
                record = {**log_ctx, "duration_ms": elapsed_ms, "outcome": "failed"}
                logger.error(
                    "Stage failed: %s after %.1fms (run_id=%s): %s",
                    stage_name,
                    elapsed_ms,
                    self.run_id,
                    exc,
                    extra=record,
                )
                self.stages.append(record)
                raise
            else:
                elapsed_ms = round((time.monotonic() - start) * 1000, 1)
                span.set_attribute("duration_ms", elapsed_ms)
                record = {**log_ctx, "duration_ms": elapsed_ms, "outcome": "ok"}
                logger.info("Stage completed: %s in %.1fms", stage_name, elapsed_ms, extra=record)
                self.stages.append(record)

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of this run's stages, for a final log line."""
        total_ms = round((time.monotonic() - self._start) * 1000, 1)
        failed_stages = [s["stage"] for s in self.stages if s["outcome"] == "failed"]
        return {
            "run_id": self.run_id,
            "pipeline": self.name,
            "total_duration_ms": total_ms,
            "stage_count": len(self.stages),
            "failed_stages": failed_stages,
            "stages": self.stages,
        }
