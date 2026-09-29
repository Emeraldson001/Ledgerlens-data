"""Data retention controls for sensitive intermediate files — Issue #530.

Provides a policy-driven, auditable framework for purging intermediate files
produced by the LedgerLens pipeline (raw feature matrices, intermediate Parquet
shards, SHAP explanation caches, drift snapshots, etc.) that contain sensitive
on-chain data and should not be retained indefinitely.

Components
----------
``RetentionPolicy``
    Dataclass encoding a single policy: which files it covers (via glob pattern),
    how long to keep them (``max_age_days``), and optional severity metadata.

``FileRetentionManager``
    Applies a collection of ``RetentionPolicy`` objects against the filesystem,
    writes tamper-evident audit-log entries, and supports a dry-run mode for
    safe inspection.

``RetentionAuditLog``
    Append-only NDJSON audit-log writer with an HMAC-SHA256 line signature so
    purge records cannot be silently altered.

Usage example
-------------
>>> manager = FileRetentionManager(
...     policies=[
...         RetentionPolicy(
...             name="raw_features",
...             pattern="data/intermediate/features_*.parquet",
...             max_age_days=7,
...         ),
...         RetentionPolicy(
...             name="shap_cache",
...             pattern="models/shap_cache_*.joblib",
...             max_age_days=30,
...         ),
...     ],
...     audit_log_path=Path("reports/retention_audit.ndjson"),
... )
>>> report = manager.run(dry_run=False)
>>> print(report.summary())

Design notes
------------
* Age is computed from the file's ``mtime`` (modification time), not creation
  time, because writes are the meaningful event for pipeline output files.
* Purge is always preceded by an audit log entry written and fsynced *before*
  deletion, so a crash during deletion still leaves a record.
* The audit log is HMAC-signed using ``RETENTION_HMAC_SECRET`` from the
  environment (falls back to a repo-internal default in tests).  The secret is
  never stored in the log itself.
* A ``RetentionReport`` collects every decision (kept / purged / skipped) and
  provides a ``summary()`` method for human-readable output and CI assertions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RetentionPolicy",
    "RetentionDecision",
    "RetentionReport",
    "RetentionAuditLog",
    "FileRetentionManager",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HMAC secret — read from environment; used to sign audit-log entries.
# ---------------------------------------------------------------------------

_DEFAULT_HMAC_SECRET = "ledgerlens-retention-dev-secret-not-for-production"
_HMAC_SECRET: str = os.environ.get("RETENTION_HMAC_SECRET", _DEFAULT_HMAC_SECRET)


def _sign_entry(payload: str) -> str:
    """Return a hex HMAC-SHA256 over *payload* using the configured secret."""
    return hmac.new(
        _HMAC_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_audit_entry(entry: dict[str, Any]) -> bool:
    """Return ``True`` iff the ``hmac`` field in *entry* is valid."""
    sig = entry.pop("hmac", None)
    if sig is None:
        return False
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    expected = _sign_entry(payload)
    entry["hmac"] = sig  # restore
    return hmac.compare_digest(expected, sig)


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------


@dataclass
class RetentionPolicy:
    """A single retention rule.

    Parameters
    ----------
    name:
        Human-readable policy name; appears in audit log and reports.
    pattern:
        A glob pattern (relative to ``FileRetentionManager.base_dir``) that
        selects the files this policy governs.
    max_age_days:
        Files older than this many days (by ``mtime``) are candidates for
        deletion.
    description:
        Optional free-text description of why this policy exists.
    enabled:
        If ``False``, the policy is listed in the report but no files are
        evaluated.
    """

    name: str
    pattern: str
    max_age_days: float
    description: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_age_days <= 0:
            raise ValueError(
                f"RetentionPolicy {self.name!r}: max_age_days must be positive, "
                f"got {self.max_age_days}"
            )


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class RetentionDecision:
    """Records the retention outcome for a single file."""

    policy_name: str
    path: str
    age_days: float
    action: str  # "purged" | "kept" | "dry_run_purge" | "skipped" | "error"
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "path": self.path,
            "age_days": round(self.age_days, 4),
            "action": self.action,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class RetentionReport:
    """Aggregates all :class:`RetentionDecision` objects from one manager run."""

    decisions: list[RetentionDecision] = field(default_factory=list)
    run_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    dry_run: bool = False

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    @property
    def purged(self) -> list[RetentionDecision]:
        return [d for d in self.decisions if d.action in {"purged", "dry_run_purge"}]

    @property
    def kept(self) -> list[RetentionDecision]:
        return [d for d in self.decisions if d.action == "kept"]

    @property
    def errors(self) -> list[RetentionDecision]:
        return [d for d in self.decisions if d.action == "error"]

    def summary(self) -> str:
        lines = [
            f"RetentionReport (dry_run={self.dry_run}, run_at={self.run_at})",
            f"  evaluated : {len(self.decisions)}",
            f"  purged    : {len(self.purged)}",
            f"  kept      : {len(self.kept)}",
            f"  errors    : {len(self.errors)}",
        ]
        if self.purged:
            lines.append("  purged files:")
            for d in self.purged:
                lines.append(f"    [{d.action}] {d.path} (age={d.age_days:.1f}d)")
        if self.errors:
            lines.append("  errors:")
            for d in self.errors:
                lines.append(f"    [error] {d.path}: {d.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at,
            "dry_run": self.dry_run,
            "total_evaluated": len(self.decisions),
            "total_purged": len(self.purged),
            "total_kept": len(self.kept),
            "total_errors": len(self.errors),
            "decisions": [d.to_dict() for d in self.decisions],
        }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class RetentionAuditLog:
    """Append-only NDJSON audit log for retention decisions.

    Each line is a JSON object with an ``"hmac"`` field whose value is an
    HMAC-SHA256 of the rest of the record (sorted keys, compact JSON).  This
    allows downstream validators to detect tampering.

    Parameters
    ----------
    path:
        Path to the ``.ndjson`` audit-log file.  Parent directories are
        created on first write.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, decision: RetentionDecision) -> None:
        """Append *decision* to the log, fsyncing before returning."""
        entry = decision.to_dict()
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        entry["hmac"] = _sign_entry(payload)
        line = json.dumps(entry, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        """Read all entries from the log and return as a list of dicts."""
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.error("Corrupt audit log line %d: %s", lineno, exc)
        return entries

    def verify_all(self) -> tuple[int, int]:
        """Verify HMAC signatures on every entry.

        Returns ``(valid_count, invalid_count)``.
        """
        entries = self.read_all()
        valid = invalid = 0
        for entry in entries:
            if verify_audit_entry(dict(entry)):
                valid += 1
            else:
                invalid += 1
        return valid, invalid


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FileRetentionManager:
    """Apply retention policies to the filesystem.

    Parameters
    ----------
    policies:
        List of :class:`RetentionPolicy` objects to evaluate.
    base_dir:
        Base directory from which glob patterns are resolved.  Defaults to
        the repository root (``Path(".")``).
    audit_log_path:
        Where to write the NDJSON audit log.  If ``None``, audit log entries
        are only written to Python's ``logging`` at INFO level.
    """

    def __init__(
        self,
        policies: list[RetentionPolicy],
        base_dir: Path | str = Path("."),
        audit_log_path: Path | str | None = None,
    ) -> None:
        self.policies = policies
        self.base_dir = Path(base_dir)
        self.audit_log: RetentionAuditLog | None = (
            RetentionAuditLog(audit_log_path) if audit_log_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> RetentionReport:
        """Evaluate all policies and optionally purge matched files.

        Parameters
        ----------
        dry_run:
            If ``True``, no files are deleted; decisions are recorded with
            action ``"dry_run_purge"`` for candidates that would be removed.

        Returns
        -------
        RetentionReport
        """
        report = RetentionReport(dry_run=dry_run)

        for policy in self.policies:
            if not policy.enabled:
                logger.info("Skipping disabled policy %r", policy.name)
                continue

            candidates = self._collect_candidates(policy)
            logger.info(
                "Policy %r: evaluating %d candidate(s) (max_age=%sd)",
                policy.name,
                len(candidates),
                policy.max_age_days,
            )

            for path in candidates:
                decision = self._evaluate_file(policy, path, dry_run)
                report.decisions.append(decision)
                self._record(decision)

        return report

    def scan(self) -> dict[str, list[dict[str, Any]]]:
        """Return a scan report without deleting anything (alias for dry-run).

        Useful for monitoring: returns a dict mapping policy names to lists of
        file info dicts for candidates that *would* be purged.
        """
        report = self.run(dry_run=True)
        result: dict[str, list[dict[str, Any]]] = {}
        for d in report.purged:
            result.setdefault(d.policy_name, []).append(d.to_dict())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_candidates(self, policy: RetentionPolicy) -> list[Path]:
        """Expand *policy.pattern* relative to *self.base_dir*."""
        return sorted(self.base_dir.glob(policy.pattern))

    def _file_age_days(self, path: Path) -> float:
        """Return file age in days based on ``mtime``."""
        mtime = path.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 86400.0

    def _evaluate_file(
        self, policy: RetentionPolicy, path: Path, dry_run: bool
    ) -> RetentionDecision:
        try:
            age_days = self._file_age_days(path)
        except OSError as exc:
            return RetentionDecision(
                policy_name=policy.name,
                path=str(path),
                age_days=0.0,
                action="error",
                reason=f"stat failed: {exc}",
            )

        if age_days < policy.max_age_days:
            return RetentionDecision(
                policy_name=policy.name,
                path=str(path),
                age_days=age_days,
                action="kept",
                reason=f"age {age_days:.2f}d < max {policy.max_age_days}d",
            )

        # Candidate for purge
        if dry_run:
            return RetentionDecision(
                policy_name=policy.name,
                path=str(path),
                age_days=age_days,
                action="dry_run_purge",
                reason=f"age {age_days:.2f}d >= max {policy.max_age_days}d",
            )

        # Write audit log entry *before* deleting so a crash leaves a record
        decision = RetentionDecision(
            policy_name=policy.name,
            path=str(path),
            age_days=age_days,
            action="purged",
            reason=f"age {age_days:.2f}d >= max {policy.max_age_days}d",
        )
        self._record(decision)

        try:
            path.unlink()
            logger.info("Purged %s (policy=%r, age=%.1fd)", path, policy.name, age_days)
        except OSError as exc:
            decision.action = "error"
            decision.reason = f"unlink failed: {exc}"
            logger.error("Failed to purge %s: %s", path, exc)

        return decision

    def _record(self, decision: RetentionDecision) -> None:
        """Write *decision* to the audit log and the Python logger."""
        logger.info(
            "[retention] %s %s (policy=%r, age=%.2fd)",
            decision.action,
            decision.path,
            decision.policy_name,
            decision.age_days,
        )
        if self.audit_log is not None:
            self.audit_log.append(decision)


# ---------------------------------------------------------------------------
# Default policy set — opinionated defaults for the LedgerLens pipeline
# ---------------------------------------------------------------------------

#: Ready-to-use policies for the standard LedgerLens pipeline layout.
#: Callers may extend or override this list via ``FileRetentionManager``.
DEFAULT_POLICIES: list[RetentionPolicy] = [
    RetentionPolicy(
        name="intermediate_features",
        pattern="data/intermediate/features_*.parquet",
        max_age_days=7,
        description="Intermediate feature matrices are regenerated on each pipeline run; "
        "retain for 7 days for debugging then purge.",
    ),
    RetentionPolicy(
        name="shap_cache",
        pattern="models/shap_cache_*.joblib",
        max_age_days=30,
        description="SHAP explanation caches are large and regenerable; purge after 30 days.",
    ),
    RetentionPolicy(
        name="drift_reports",
        pattern="reports/drift_report_*.json",
        max_age_days=90,
        description="PSI drift reports; keep 3 months for trend analysis.",
    ),
    RetentionPolicy(
        name="forensic_reports",
        pattern="reports/forensic/*.json",
        max_age_days=365,
        description="Forensic reports may be required for compliance; retain 1 year.",
    ),
    RetentionPolicy(
        name="retrain_reports",
        pattern="reports/retrain_*.json",
        max_age_days=180,
        description="Retrain trigger reports; keep 6 months.",
    ),
]
