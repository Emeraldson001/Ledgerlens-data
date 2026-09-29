import json

import pytest

from ingestion.batch_processor import BatchProcessor


def test_chunks_all_items_and_reports_totals():
    processor = BatchProcessor(chunk_size=10, max_retries=1)
    seen = []

    def process(chunk):
        seen.append(list(chunk))

    records = list(range(25))
    summary = processor.run(records, process, job_id="job-a", resume=False)

    assert summary.total_items == 25
    assert summary.succeeded_items == 25
    assert summary.failed_items == 0
    assert summary.chunks_processed == 3
    assert [len(c) for c in seen] == [10, 10, 5]
    assert summary.ok


def test_retries_transient_failures_then_succeeds():
    processor = BatchProcessor(chunk_size=5, max_retries=3, backoff_seconds=0)
    attempts = {"count": 0}

    def flaky(chunk):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient db timeout")

    summary = processor.run(list(range(5)), flaky, job_id="job-b", resume=False)

    assert summary.ok
    assert attempts["count"] == 2
    assert summary.failed_items == 0


def test_chunk_failure_recorded_but_does_not_abort_job():
    processor = BatchProcessor(chunk_size=2, max_retries=2, backoff_seconds=0)

    def always_fail_second_chunk(chunk):
        if chunk[0] == 2:
            raise ValueError("bad row")

    summary = processor.run(
        [0, 1, 2, 3, 4, 5], always_fail_second_chunk, job_id="job-c", resume=False
    )

    assert not summary.ok
    assert summary.chunks_failed == 1
    assert summary.chunks_processed == 3
    assert summary.failed_items == 2
    assert summary.succeeded_items == 4
    assert "bad row" in summary.failures[0].error


def test_resume_skips_already_completed_chunks(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.json")
    calls = []

    def record_and_fail_on_chunk_1(chunk):
        calls.append(list(chunk))
        if chunk == [2, 3]:
            raise RuntimeError("boom")

    processor = BatchProcessor(chunk_size=2, max_retries=1, checkpoint_path=checkpoint_path)
    first_summary = processor.run([0, 1, 2, 3, 4, 5], record_and_fail_on_chunk_1, job_id="job-d")
    assert not first_summary.ok
    assert calls == [[0, 1], [2, 3], [4, 5]]

    calls.clear()

    def succeed(chunk):
        calls.append(list(chunk))

    second_summary = processor.run([0, 1, 2, 3, 4, 5], succeed, job_id="job-d")
    # chunk 0 ([0,1]) and chunk 2 ([4,5]) were already checkpointed as done;
    # only the previously-failed chunk 1 ([2,3]) should be re-processed.
    assert calls == [[2, 3]]
    assert second_summary.chunks_skipped_resume == 2
    assert second_summary.ok


def test_successful_job_clears_checkpoint(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.json")
    processor = BatchProcessor(chunk_size=2, checkpoint_path=checkpoint_path)
    processor.run([0, 1, 2, 3], lambda chunk: None, job_id="job-e")

    with open(checkpoint_path) as f:
        data = json.load(f)
    assert data.get("job-e", []) == []


def test_on_progress_callback_invoked_per_chunk():
    outcomes = []
    processor = BatchProcessor(chunk_size=1, on_progress=outcomes.append)
    processor.run([1, 2, 3], lambda chunk: None, job_id="job-f", resume=False)
    assert len(outcomes) == 3
    assert all(o.ok for o in outcomes)


def test_invalid_chunk_size_raises():
    processor = BatchProcessor(chunk_size=0)
    with pytest.raises(ValueError):
        processor.run([1, 2, 3], lambda chunk: None, job_id="job-g", resume=False)
