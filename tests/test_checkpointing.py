"""Tests for utils/checkpointing.py — durable resumable workflow progress."""

import json
import os
import time

import pytest

from utils.checkpointing import (
    CheckpointCorruptionError,
    CheckpointedWorkflow,
    CheckpointStore,
    CheckpointVersionError,
)


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(str(tmp_path / "checkpoints"))


def test_new_workflow_has_no_completed_steps(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    assert list(workflow.remaining_steps(["fetch", "transform", "load"])) == [
        "fetch",
        "transform",
        "load",
    ]


def test_completed_step_persists_across_reload(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with workflow.step("fetch"):
        pass

    # Simulate a process restart: build a fresh workflow object from the store.
    resumed = CheckpointedWorkflow(store, workflow_id="wf1")
    assert resumed.is_step_done("fetch")
    assert list(resumed.remaining_steps(["fetch", "transform", "load"])) == [
        "transform",
        "load",
    ]


def test_failed_step_is_not_marked_complete(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with pytest.raises(RuntimeError):
        with workflow.step("fetch"):
            raise RuntimeError("boom")

    resumed = CheckpointedWorkflow(store, workflow_id="wf1")
    assert not resumed.is_step_done("fetch")


def test_item_level_resumption_within_a_step(store):
    items = ["a", "b", "c", "d"]
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")

    with workflow.step("score") as ckpt:
        for item in ckpt.remaining_items(items):
            if item == "c":
                break
            ckpt.mark_done(item)

    resumed = CheckpointedWorkflow(store, workflow_id="wf1")
    with resumed.step("score") as ckpt:
        remaining = list(ckpt.remaining_items(items))

    assert remaining == ["c", "d"]


def test_clear_discards_progress(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with workflow.step("fetch"):
        pass
    workflow.clear()

    resumed = CheckpointedWorkflow(store, workflow_id="wf1")
    assert not resumed.is_step_done("fetch")
    assert not store.exists("wf1")


def test_corrupted_checkpoint_raises_with_diagnostics(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with workflow.step("fetch"):
        pass

    path = store._path("wf1")
    with open(path) as f:
        data = json.load(f)
    data["completed_steps"].append("tampered")
    with open(path, "w") as f:
        json.dump(data, f)

    with pytest.raises(CheckpointCorruptionError) as exc_info:
        CheckpointedWorkflow(store, workflow_id="wf1")

    assert "wf1" in str(exc_info.value)
    assert path in str(exc_info.value)


def test_incompatible_schema_version_raises(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with workflow.step("fetch"):
        pass

    path = store._path("wf1")
    with open(path) as f:
        data = json.load(f)
    data["schema_version"] = 999
    data.pop("checksum", None)
    from utils.checkpointing import _checksum

    data["checksum"] = _checksum(data)
    with open(path, "w") as f:
        json.dump(data, f)

    with pytest.raises(CheckpointVersionError):
        CheckpointedWorkflow(store, workflow_id="wf1")


def test_atomic_write_leaves_no_temp_files_behind(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf1")
    with workflow.step("fetch"):
        pass

    directory = store.directory
    leftovers = [f for f in os.listdir(directory) if f.endswith(".tmp")]
    assert leftovers == []


def test_workflow_ids_are_sanitised_for_filesystem_safety(store):
    workflow = CheckpointedWorkflow(store, workflow_id="wf/with:weird chars")
    with workflow.step("fetch"):
        pass
    assert store.exists("wf/with:weird chars")


def test_cleanup_stale_removes_only_expired_checkpoints(store):
    """Checkpoints older than the retention window are purged; newer ones are kept (Issue #698)."""
    for workflow_id in ["old_a", "old_b", "new_a", "new_b"]:
        workflow = CheckpointedWorkflow(store, workflow_id=workflow_id)
        with workflow.step("fetch"):
            pass

    # Backdate two checkpoints past the default 30-day retention window.
    now = time.time()
    stale_mtime = now - 31 * 86400
    for workflow_id in ["old_a", "old_b"]:
        os.utime(store._path(workflow_id), (stale_mtime, stale_mtime))

    removed = store.cleanup_stale(retention_days=30)

    assert len(removed) == 2
    assert not store.exists("old_a")
    assert not store.exists("old_b")
    assert store.exists("new_a")
    assert store.exists("new_b")
