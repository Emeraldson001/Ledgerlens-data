"""Tests for data/reproducibility.py — Issue #533.

Covers:
* DatasetSnapshot.freeze: creates snapshot dir, copies file, writes manifest
* DatasetSnapshot.verify: passes on intact snapshot, raises IntegrityError on tamper
* DatasetSnapshot.restore: reconstructs the original file at a target path
* DatasetSnapshot.list_snapshots and delete
* SnapshotManifest round-trip serialisation
* SnapshotRegistry list, find, and prune helpers
* Freeze with missing source raises FileNotFoundError
* Verify with missing snapshot raises FileNotFoundError
* Verify with missing data file raises IntegrityError
* Parquet-specific row_count and column list capture
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from data.reproducibility import (
    DatasetSnapshot,
    IntegrityError,
    SnapshotManifest,
    SnapshotRegistry,
    _sha256_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet(path: Path, n: int = 20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"a": range(n), "b": [f"wallet_{i}" for i in range(n)]})
    df.to_parquet(path, index=False)
    return path


def _make_text(path: Path, content: str = "hello") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


class TestSha256File:
    def test_known_hash(self, tmp_path):
        f = _make_text(tmp_path / "test.txt", "hello world")
        h = _sha256_file(f)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert h == expected

    def test_different_files_different_hashes(self, tmp_path):
        f1 = _make_text(tmp_path / "a.txt", "aaa")
        f2 = _make_text(tmp_path / "b.txt", "bbb")
        assert _sha256_file(f1) != _sha256_file(f2)


# ---------------------------------------------------------------------------
# SnapshotManifest
# ---------------------------------------------------------------------------


class TestSnapshotManifest:
    def _make_manifest(self):
        return SnapshotManifest(
            snapshot_id="snap_20240601T000000_test_abc123",
            source_filename="dataset.parquet",
            sha256="abc" * 21 + "ab",
            row_count=100,
            columns=["a", "b"],
            created_at="2024-06-01T00:00:00Z",
            label="test",
            ledgerlens_version="0.2.0",
        )

    def test_to_dict_round_trip(self):
        m = self._make_manifest()
        d = m.to_dict()
        m2 = SnapshotManifest.from_dict(d)
        assert m2.snapshot_id == m.snapshot_id
        assert m2.sha256 == m.sha256
        assert m2.row_count == 100

    def test_save_and_load(self, tmp_path):
        m = self._make_manifest()
        m.save(tmp_path)
        assert (tmp_path / "snapshot.json").exists()
        loaded = SnapshotManifest.load(tmp_path)
        assert loaded.snapshot_id == m.snapshot_id

    def test_canonical_json_excludes_signature(self):
        m = self._make_manifest()
        m.signature = "deadbeef"
        canon = m.canonical_json()
        assert "signature" not in json.loads(canon)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SnapshotManifest.load(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# DatasetSnapshot — freeze
# ---------------------------------------------------------------------------


class TestDatasetSnapshotFreeze:
    def test_freeze_creates_snapshot_dir(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source, label="test")
        assert (tmp_path / "snaps" / manifest.snapshot_id).is_dir()

    def test_freeze_copies_file(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        snap_file = tmp_path / "snaps" / manifest.snapshot_id / source.name
        assert snap_file.exists()

    def test_freeze_records_sha256(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        expected_hash = _sha256_file(source)
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        assert manifest.sha256 == expected_hash

    def test_freeze_records_row_count(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet", n=42)
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        assert manifest.row_count == 42

    def test_freeze_records_columns(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        assert set(manifest.columns) == {"a", "b"}

    def test_freeze_label_in_snapshot_id(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source, label="my-run")
        assert "my-run" in manifest.snapshot_id

    def test_freeze_stores_metadata(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source, metadata={"run_id": "xyz"})
        assert manifest.metadata["run_id"] == "xyz"

    def test_freeze_missing_source_raises(self, tmp_path):
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        with pytest.raises(FileNotFoundError):
            snap.freeze(tmp_path / "nonexistent.parquet")

    def test_freeze_text_file(self, tmp_path):
        source = _make_text(tmp_path / "readme.txt", "hello")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        assert manifest.source_filename == "readme.txt"
        assert manifest.row_count == 0  # not a parquet file

    def test_freeze_manifest_saved_to_disk(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        snap_dir = tmp_path / "snaps" / manifest.snapshot_id
        assert (snap_dir / "snapshot.json").exists()

    def test_multiple_freezes_create_different_ids(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        m1 = snap.freeze(source, label="v1")
        m2 = snap.freeze(source, label="v2")
        assert m1.snapshot_id != m2.snapshot_id


# ---------------------------------------------------------------------------
# DatasetSnapshot — verify
# ---------------------------------------------------------------------------


class TestDatasetSnapshotVerify:
    def test_verify_intact_snapshot_passes(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        assert snap.verify(manifest.snapshot_id) is True

    def test_verify_tampered_file_raises(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        # Tamper with the stored file
        snap_file = tmp_path / "snaps" / manifest.snapshot_id / source.name
        snap_file.write_bytes(b"tampered content")
        with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
            snap.verify(manifest.snapshot_id)

    def test_verify_missing_data_file_raises(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        snap_file = tmp_path / "snaps" / manifest.snapshot_id / source.name
        snap_file.unlink()
        with pytest.raises(IntegrityError, match="missing"):
            snap.verify(manifest.snapshot_id)

    def test_verify_nonexistent_snapshot_raises(self, tmp_path):
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        with pytest.raises(FileNotFoundError):
            snap.verify("snap_nonexistent")


# ---------------------------------------------------------------------------
# DatasetSnapshot — restore
# ---------------------------------------------------------------------------


class TestDatasetSnapshotRestore:
    def test_restore_creates_target(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        target = tmp_path / "restored" / "dataset.parquet"
        snap.restore(manifest.snapshot_id, target=target)
        assert target.exists()

    def test_restore_content_matches(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        target = tmp_path / "restored.parquet"
        snap.restore(manifest.snapshot_id, target=target)
        original_df = pd.read_parquet(source)
        restored_df = pd.read_parquet(target)
        pd.testing.assert_frame_equal(original_df, restored_df)

    def test_restore_without_verify(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        target = tmp_path / "restored.parquet"
        snap.restore(manifest.snapshot_id, target=target, verify=False)
        assert target.exists()

    def test_restore_tampered_raises_with_verify(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        snap_file = tmp_path / "snaps" / manifest.snapshot_id / source.name
        snap_file.write_bytes(b"bad data")
        with pytest.raises(IntegrityError):
            snap.restore(manifest.snapshot_id, target=tmp_path / "out.parquet")


# ---------------------------------------------------------------------------
# DatasetSnapshot — list and delete
# ---------------------------------------------------------------------------


class TestDatasetSnapshotListDelete:
    def test_list_snapshots_empty(self, tmp_path):
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        assert snap.list_snapshots() == []

    def test_list_snapshots_after_freeze(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        snap.freeze(source, label="v1")
        snap.freeze(source, label="v2")
        ids = snap.list_snapshots()
        assert len(ids) == 2

    def test_delete_snapshot(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        manifest = snap.freeze(source)
        snap.delete(manifest.snapshot_id)
        assert snap.list_snapshots() == []

    def test_delete_nonexistent_raises(self, tmp_path):
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        with pytest.raises(FileNotFoundError):
            snap.delete("snap_nonexistent")

    def test_get_manifest(self, tmp_path):
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        m1 = snap.freeze(source, label="check")
        m2 = snap.get_manifest(m1.snapshot_id)
        assert m2.label == "check"
        assert m2.sha256 == m1.sha256


# ---------------------------------------------------------------------------
# SnapshotRegistry
# ---------------------------------------------------------------------------


class TestSnapshotRegistry:
    def _populate(self, tmp_path: Path, n: int = 3) -> DatasetSnapshot:
        source = _make_parquet(tmp_path / "src" / "dataset.parquet")
        snap = DatasetSnapshot(snapshot_root=tmp_path / "snaps")
        for i in range(n):
            snap.freeze(source, label=f"run_{i}")
        return snap

    def test_list_all_returns_manifests(self, tmp_path):
        self._populate(tmp_path, n=3)
        reg = SnapshotRegistry(tmp_path / "snaps")
        manifests = reg.list_all()
        assert len(manifests) == 3

    def test_find_by_label(self, tmp_path):
        self._populate(tmp_path, n=3)
        reg = SnapshotRegistry(tmp_path / "snaps")
        found = reg.find_by_label("run_1")
        assert len(found) == 1
        assert found[0].label == "run_1"

    def test_find_by_sha256(self, tmp_path):
        self._populate(tmp_path, n=2)
        reg = SnapshotRegistry(tmp_path / "snaps")
        all_manifests = reg.list_all()
        sha = all_manifests[0].sha256
        found = reg.find_by_sha256(sha[:8])
        assert len(found) >= 1

    def test_prune_old_keeps_latest(self, tmp_path):
        self._populate(tmp_path, n=5)
        reg = SnapshotRegistry(tmp_path / "snaps")
        deleted = reg.prune_old(keep_latest_n=2)
        assert len(deleted) == 3
        assert len(reg.list_all()) == 2

    def test_to_dict(self, tmp_path):
        self._populate(tmp_path, n=2)
        reg = SnapshotRegistry(tmp_path / "snaps")
        d = reg.to_dict()
        assert isinstance(d, list)
        assert len(d) == 2

    def test_empty_registry(self, tmp_path):
        reg = SnapshotRegistry(tmp_path / "empty_snaps")
        assert reg.list_all() == []
