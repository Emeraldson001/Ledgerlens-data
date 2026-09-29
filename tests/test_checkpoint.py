"""Tests for utils/checkpoint.py — the pipeline checkpoint/resume primitive."""

import json

import pytest

from utils.checkpoint import (
    CheckpointCorruptError,
    CheckpointMismatchError,
    PipelineCheckpoint,
    compute_fingerprint,
)


def _ckpt(tmp_path, **kwargs):
    return PipelineCheckpoint.load_or_create(
        path=tmp_path / "ckpt.json",
        pipeline="test_pipeline",
        fingerprint_inputs=kwargs.pop("fingerprint_inputs", {"since": "2024-01-01"}),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fresh creation / basic round trip
# ---------------------------------------------------------------------------


def test_fresh_checkpoint_has_no_completed_units(tmp_path):
    ckpt = _ckpt(tmp_path)
    assert ckpt.pending(["a", "b"]) == ["a", "b"]
    assert not ckpt.is_done("a")


def test_creating_fresh_checkpoint_writes_file_immediately(tmp_path):
    path = tmp_path / "ckpt.json"
    PipelineCheckpoint.load_or_create(path, "test_pipeline", {"x": 1})
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["pipeline"] == "test_pipeline"
    assert data["completed"] == {}


def test_record_success_marks_unit_done_and_persists(tmp_path):
    path = tmp_path / "ckpt.json"
    ckpt = _ckpt(tmp_path)
    ckpt.record_success("pair-a", metadata={"rows": 5})

    assert ckpt.is_done("pair-a")
    assert ckpt.pending(["pair-a", "pair-b"]) == ["pair-b"]

    on_disk = json.loads(path.read_text())
    assert on_disk["completed"]["pair-a"]["metadata"] == {"rows": 5}


def test_record_success_with_artifact_path_is_retrievable(tmp_path):
    ckpt = _ckpt(tmp_path)
    ckpt.record_success("pool-1", artifact_path="/tmp/pool-1.parquet")
    assert ckpt.artifact_path("pool-1") == "/tmp/pool-1.parquet"
    assert ckpt.artifact_path("unknown-unit") is None


def test_record_failure_keeps_unit_pending_and_tracks_attempts(tmp_path):
    ckpt = _ckpt(tmp_path)
    ckpt.record_failure("pair-a", RuntimeError("boom"))

    assert not ckpt.is_done("pair-a")
    assert ckpt.pending(["pair-a"]) == ["pair-a"]
    assert ckpt.failed["pair-a"]["attempts"] == 1
    assert "boom" in ckpt.failed["pair-a"]["error"]

    ckpt.record_failure("pair-a", RuntimeError("boom again"))
    assert ckpt.failed["pair-a"]["attempts"] == 2


def test_record_success_clears_prior_failure(tmp_path):
    ckpt = _ckpt(tmp_path)
    ckpt.record_failure("pair-a", "transient error")
    assert "pair-a" in ckpt.failed

    ckpt.record_success("pair-a")
    assert "pair-a" not in ckpt.failed
    assert ckpt.is_done("pair-a")


# ---------------------------------------------------------------------------
# Resume across process boundaries (re-load from disk)
# ---------------------------------------------------------------------------


def test_resuming_loads_prior_progress_from_disk(tmp_path):
    ckpt1 = _ckpt(tmp_path)
    ckpt1.record_success("pair-a")
    ckpt1.record_failure("pair-b", "timeout")

    ckpt2 = _ckpt(tmp_path)
    assert ckpt2.is_done("pair-a")
    assert not ckpt2.is_done("pair-b")
    assert ckpt2.pending(["pair-a", "pair-b", "pair-c"]) == ["pair-b", "pair-c"]
    assert ckpt2.failed["pair-b"]["attempts"] == 1


def test_summary_reports_counts_and_failed_unit_ids(tmp_path):
    ckpt = _ckpt(tmp_path)
    ckpt.record_success("pair-a")
    ckpt.record_failure("pair-b", "timeout")

    summary = ckpt.summary()
    assert summary["completed"] == 1
    assert summary["failed"] == ["pair-b"]
    assert summary["pipeline"] == "test_pipeline"


# ---------------------------------------------------------------------------
# Fingerprint mismatch / diagnostics
# ---------------------------------------------------------------------------


def test_mismatched_fingerprint_inputs_raise_with_actionable_diff(tmp_path):
    _ckpt(tmp_path, fingerprint_inputs={"since": "2024-01-01", "pairs": ["USDC"]})

    with pytest.raises(CheckpointMismatchError) as exc_info:
        _ckpt(tmp_path, fingerprint_inputs={"since": "2024-06-01", "pairs": ["USDC"]})

    message = str(exc_info.value)
    assert "since" in message
    assert "2024-01-01" in message
    assert "2024-06-01" in message
    assert "--fresh" in message


def test_fresh_flag_discards_mismatched_checkpoint(tmp_path):
    ckpt1 = _ckpt(tmp_path, fingerprint_inputs={"since": "2024-01-01"})
    ckpt1.record_success("pair-a")

    ckpt2 = _ckpt(tmp_path, fingerprint_inputs={"since": "2024-06-01"}, fresh=True)
    assert ckpt2.pending(["pair-a"]) == ["pair-a"]


def test_different_pipeline_name_at_same_path_raises_mismatch(tmp_path):
    path = tmp_path / "ckpt.json"
    PipelineCheckpoint.load_or_create(path, "pipeline_a", {"x": 1})

    with pytest.raises(CheckpointMismatchError, match="pipeline_a"):
        PipelineCheckpoint.load_or_create(path, "pipeline_b", {"x": 1})


def test_matching_fingerprint_inputs_resume_without_error(tmp_path):
    ckpt1 = _ckpt(tmp_path, fingerprint_inputs={"since": "2024-01-01", "pairs": ["USDC"]})
    ckpt1.record_success("pair-a")

    ckpt2 = _ckpt(tmp_path, fingerprint_inputs={"since": "2024-01-01", "pairs": ["USDC"]})
    assert ckpt2.is_done("pair-a")


# ---------------------------------------------------------------------------
# Corruption handling
# ---------------------------------------------------------------------------


def test_corrupt_json_raises_actionable_error(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text("{not valid json")

    with pytest.raises(CheckpointCorruptError, match="--fresh"):
        PipelineCheckpoint.load_or_create(path, "test_pipeline", {"x": 1})


def test_unknown_schema_version_raises_actionable_error(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text(json.dumps({"schema_version": 999, "pipeline": "test_pipeline"}))

    with pytest.raises(CheckpointCorruptError, match="schema_version"):
        PipelineCheckpoint.load_or_create(path, "test_pipeline", {"x": 1})


def test_fresh_flag_recovers_from_corrupt_checkpoint(tmp_path):
    path = tmp_path / "ckpt.json"
    path.write_text("{not valid json")

    ckpt = PipelineCheckpoint.load_or_create(path, "test_pipeline", {"x": 1}, fresh=True)
    assert ckpt.pending(["a"]) == ["a"]


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_removes_checkpoint_file(tmp_path):
    ckpt = _ckpt(tmp_path)
    ckpt.record_success("pair-a")
    assert ckpt.path.exists()

    ckpt.clear()
    assert not ckpt.path.exists()


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


def test_compute_fingerprint_is_order_independent():
    fp1 = compute_fingerprint({"a": 1, "b": 2})
    fp2 = compute_fingerprint({"b": 2, "a": 1})
    assert fp1 == fp2


def test_compute_fingerprint_differs_on_value_change():
    fp1 = compute_fingerprint({"a": 1})
    fp2 = compute_fingerprint({"a": 2})
    assert fp1 != fp2
