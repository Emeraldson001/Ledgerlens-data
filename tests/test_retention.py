"""Tests for data/retention.py — Issue #530.

Covers:
* RetentionPolicy construction and validation
* FileRetentionManager dry-run mode
* FileRetentionManager live purge
* Audit log write, read, and HMAC verification
* RetentionReport summary and to_dict
* Default policies list
* Disabled policy skipping
* File-not-found / stat-error handling
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from data.retention import (
    DEFAULT_POLICIES,
    FileRetentionManager,
    RetentionAuditLog,
    RetentionDecision,
    RetentionPolicy,
    RetentionReport,
    verify_audit_entry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str = "data", age_days: float = 0.0) -> Path:
    """Create *path* with optional backdated mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if age_days > 0:
        old_mtime = time.time() - age_days * 86400
        os.utime(path, (old_mtime, old_mtime))
    return path


# ---------------------------------------------------------------------------
# RetentionPolicy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_valid_policy(self):
        p = RetentionPolicy(name="test", pattern="data/*.parquet", max_age_days=7)
        assert p.enabled is True
        assert p.max_age_days == 7

    def test_non_positive_max_age_raises(self):
        with pytest.raises(ValueError, match="max_age_days must be positive"):
            RetentionPolicy(name="bad", pattern="*.parquet", max_age_days=0)

    def test_negative_max_age_raises(self):
        with pytest.raises(ValueError, match="max_age_days must be positive"):
            RetentionPolicy(name="bad", pattern="*.parquet", max_age_days=-5)

    def test_disabled_policy(self):
        p = RetentionPolicy(name="skip", pattern="*.parquet", max_age_days=7, enabled=False)
        assert p.enabled is False


# ---------------------------------------------------------------------------
# FileRetentionManager — dry-run
# ---------------------------------------------------------------------------


class TestFileRetentionManagerDryRun:
    def _manager(self, tmp_path: Path, policies=None, audit=None):
        return FileRetentionManager(
            policies=policies or [],
            base_dir=tmp_path,
            audit_log_path=audit,
        )

    def test_dry_run_no_deletion(self, tmp_path):
        old_file = _write_file(tmp_path / "features_old.parquet", age_days=10)
        manager = self._manager(
            tmp_path,
            [RetentionPolicy(name="test", pattern="features_*.parquet", max_age_days=5)],
        )
        report = manager.run(dry_run=True)
        assert len(report.purged) == 1
        assert report.purged[0].action == "dry_run_purge"
        assert old_file.exists()  # not deleted

    def test_dry_run_keeps_recent_files(self, tmp_path):
        _write_file(tmp_path / "features_new.parquet", age_days=0)
        manager = self._manager(
            tmp_path,
            [RetentionPolicy(name="test", pattern="features_*.parquet", max_age_days=5)],
        )
        report = manager.run(dry_run=True)
        assert len(report.kept) == 1
        assert len(report.purged) == 0

    def test_scan_returns_policy_keyed_dict(self, tmp_path):
        _write_file(tmp_path / "features_old.parquet", age_days=10)
        manager = self._manager(
            tmp_path,
            [RetentionPolicy(name="old_features", pattern="features_*.parquet", max_age_days=5)],
        )
        result = manager.scan()
        assert "old_features" in result
        assert len(result["old_features"]) == 1

    def test_disabled_policy_skipped(self, tmp_path):
        _write_file(tmp_path / "features_old.parquet", age_days=10)
        manager = self._manager(
            tmp_path,
            [
                RetentionPolicy(
                    name="disabled",
                    pattern="features_*.parquet",
                    max_age_days=5,
                    enabled=False,
                )
            ],
        )
        report = manager.run(dry_run=True)
        assert len(report.decisions) == 0

    def test_multiple_policies_evaluated(self, tmp_path):
        _write_file(tmp_path / "feat_old.parquet", age_days=10)
        _write_file(tmp_path / "cache_old.joblib", age_days=10)
        manager = self._manager(
            tmp_path,
            [
                RetentionPolicy(name="feats", pattern="feat_*.parquet", max_age_days=5),
                RetentionPolicy(name="cache", pattern="cache_*.joblib", max_age_days=5),
            ],
        )
        report = manager.run(dry_run=True)
        assert len(report.decisions) == 2


# ---------------------------------------------------------------------------
# FileRetentionManager — live purge
# ---------------------------------------------------------------------------


class TestFileRetentionManagerPurge:
    def test_purge_deletes_old_file(self, tmp_path):
        old_file = _write_file(tmp_path / "features_old.parquet", age_days=10)
        manager = FileRetentionManager(
            policies=[RetentionPolicy(name="test", pattern="features_*.parquet", max_age_days=5)],
            base_dir=tmp_path,
        )
        report = manager.run(dry_run=False)
        assert len(report.purged) == 1
        assert report.purged[0].action == "purged"
        assert not old_file.exists()

    def test_purge_leaves_recent_file(self, tmp_path):
        new_file = _write_file(tmp_path / "features_new.parquet", age_days=0)
        manager = FileRetentionManager(
            policies=[RetentionPolicy(name="test", pattern="features_*.parquet", max_age_days=5)],
            base_dir=tmp_path,
        )
        report = manager.run(dry_run=False)
        assert len(report.kept) == 1
        assert new_file.exists()

    def test_purge_writes_audit_log(self, tmp_path):
        _write_file(tmp_path / "features_old.parquet", age_days=10)
        audit_path = tmp_path / "audit.ndjson"
        manager = FileRetentionManager(
            policies=[RetentionPolicy(name="test", pattern="features_*.parquet", max_age_days=5)],
            base_dir=tmp_path,
            audit_log_path=audit_path,
        )
        manager.run(dry_run=False)
        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["action"] in {"purged", "kept"}

    def test_mixed_old_and_new(self, tmp_path):
        old_file = _write_file(tmp_path / "feat_old.parquet", age_days=10)
        new_file = _write_file(tmp_path / "feat_new.parquet", age_days=0)
        manager = FileRetentionManager(
            policies=[RetentionPolicy(name="test", pattern="feat_*.parquet", max_age_days=5)],
            base_dir=tmp_path,
        )
        report = manager.run(dry_run=False)
        assert len(report.purged) == 1
        assert len(report.kept) == 1
        assert not old_file.exists()
        assert new_file.exists()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestRetentionAuditLog:
    def test_append_creates_file(self, tmp_path):
        log = RetentionAuditLog(tmp_path / "audit.ndjson")
        decision = RetentionDecision(
            policy_name="test",
            path="/tmp/file.parquet",
            age_days=5.0,
            action="purged",
        )
        log.append(decision)
        assert (tmp_path / "audit.ndjson").exists()

    def test_append_and_read_all(self, tmp_path):
        log = RetentionAuditLog(tmp_path / "audit.ndjson")
        for i in range(3):
            log.append(
                RetentionDecision(
                    policy_name=f"policy_{i}",
                    path=f"/tmp/file_{i}.parquet",
                    age_days=float(i),
                    action="purged",
                )
            )
        entries = log.read_all()
        assert len(entries) == 3

    def test_verify_all_valid(self, tmp_path):
        log = RetentionAuditLog(tmp_path / "audit.ndjson")
        for i in range(3):
            log.append(
                RetentionDecision(
                    policy_name="p",
                    path=f"/tmp/f{i}.parquet",
                    age_days=float(i),
                    action="kept",
                )
            )
        valid, invalid = log.verify_all()
        assert valid == 3
        assert invalid == 0

    def test_tampered_entry_fails_verify(self, tmp_path):
        audit_path = tmp_path / "audit.ndjson"
        log = RetentionAuditLog(audit_path)
        log.append(
            RetentionDecision(
                policy_name="p",
                path="/tmp/file.parquet",
                age_days=5.0,
                action="purged",
            )
        )
        # Tamper with the file
        content = audit_path.read_text()
        tampered = content.replace('"purged"', '"kept"')
        audit_path.write_text(tampered)

        valid, invalid = log.verify_all()
        assert invalid >= 1

    def test_read_empty_log(self, tmp_path):
        log = RetentionAuditLog(tmp_path / "empty.ndjson")
        assert log.read_all() == []

    def test_verify_entry_standalone(self):
        decision = RetentionDecision(
            policy_name="p",
            path="/tmp/file.parquet",
            age_days=5.0,
            action="purged",
            timestamp="2024-06-01T00:00:00Z",
        )
        entry = decision.to_dict()
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        from data.retention import _sign_entry

        entry["hmac"] = _sign_entry(payload)
        assert verify_audit_entry(dict(entry))

    def test_verify_entry_missing_hmac(self):
        decision = RetentionDecision(
            policy_name="p",
            path="/tmp/file.parquet",
            age_days=5.0,
            action="purged",
        )
        assert not verify_audit_entry(decision.to_dict())


# ---------------------------------------------------------------------------
# RetentionReport
# ---------------------------------------------------------------------------


class TestRetentionReport:
    def _make_report(self, n_purged=2, n_kept=3, n_errors=1, dry_run=False):
        report = RetentionReport(dry_run=dry_run)
        for i in range(n_purged):
            report.decisions.append(
                RetentionDecision(
                    policy_name="p",
                    path=f"/tmp/old_{i}.parquet",
                    age_days=10.0,
                    action="dry_run_purge" if dry_run else "purged",
                )
            )
        for i in range(n_kept):
            report.decisions.append(
                RetentionDecision(
                    policy_name="p",
                    path=f"/tmp/new_{i}.parquet",
                    age_days=1.0,
                    action="kept",
                )
            )
        for i in range(n_errors):
            report.decisions.append(
                RetentionDecision(
                    policy_name="p",
                    path=f"/tmp/err_{i}.parquet",
                    age_days=0.0,
                    action="error",
                    reason="stat failed",
                )
            )
        return report

    def test_purged_property(self):
        report = self._make_report(n_purged=2)
        assert len(report.purged) == 2

    def test_kept_property(self):
        report = self._make_report(n_kept=3)
        assert len(report.kept) == 3

    def test_errors_property(self):
        report = self._make_report(n_errors=1)
        assert len(report.errors) == 1

    def test_summary_contains_counts(self):
        report = self._make_report(n_purged=2, n_kept=3, n_errors=1)
        s = report.summary()
        assert "purged" in s
        assert "kept" in s
        assert "errors" in s

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        assert "run_at" in d
        assert "total_purged" in d
        assert "decisions" in d

    def test_dry_run_report(self):
        report = self._make_report(n_purged=2, dry_run=True)
        s = report.summary()
        assert "dry_run=True" in s


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------


class TestDefaultPolicies:
    def test_default_policies_are_list(self):
        assert isinstance(DEFAULT_POLICIES, list)
        assert len(DEFAULT_POLICIES) >= 3

    def test_all_default_policies_enabled(self):
        for p in DEFAULT_POLICIES:
            assert p.enabled is True

    def test_all_default_policies_have_positive_max_age(self):
        for p in DEFAULT_POLICIES:
            assert p.max_age_days > 0

    def test_default_policy_names_unique(self):
        names = [p.name for p in DEFAULT_POLICIES]
        assert len(names) == len(set(names))
