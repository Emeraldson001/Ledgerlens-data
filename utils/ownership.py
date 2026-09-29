"""Ownership metadata for critical subsystems.

Provides programmatic access to subsystem ownership information, enabling
automated workflows to route reviews, alerts, and notifications to the
appropriate teams.

Usage::

    from utils.ownership import OwnershipRegistry

    registry = OwnershipRegistry.load()

    # Get owners for a specific file
    owners = registry.get_owners("detection/benford_engine.py")
    # -> ["@Ledger-Lenz/ml-team"]

    # Get all subsystems
    subsystems = registry.list_subsystems()
    # -> ["ingestion", "detection", "streaming", ...]

    # Get subsystem health metadata
    info = registry.get_subsystem_info("detection")
    # -> {"owners": [...], "critical_files": [...], ...}
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Subsystem definitions ────────────────────────────────────────────────────

SUBSYSTEMS: dict[str, dict[str, Any]] = {
    "ingestion": {
        "description": "Data ingestion pipeline: Horizon API, Kafka, rate limiting",
        "critical_files": [
            "ingestion/horizon_streamer.py",
            "ingestion/kafka_producer.py",
            "ingestion/rate_limiter.py",
            "ingestion/historical_loader.py",
            "ingestion/data_models.py",
        ],
        "owners": ["@Ledger-Lenz/data-team"],
        "review_required": True,
    },
    "detection": {
        "description": "Fraud detection engine: Benford's Law, ML models, feature engineering",
        "critical_files": [
            "detection/benford_engine.py",
            "detection/feature_engineering.py",
            "detection/model_training.py",
            "detection/model_inference.py",
            "detection/wallet_graph.py",
            "detection/gnn_encoder.py",
            "detection/shap_explainer.py",
        ],
        "owners": ["@Ledger-Lenz/ml-team"],
        "review_required": True,
    },
    "streaming": {
        "description": "Real-time streaming pipeline: WebSocket, Kafka workers, alert dispatch",
        "critical_files": [
            "streaming/pipeline.py",
            "streaming/kafka_worker.py",
            "streaming/ws_server.py",
            "streaming/alert_dispatcher.py",
            "streaming/feature_buffer.py",
        ],
        "owners": ["@Ledger-Lenz/streaming-team"],
        "review_required": True,
    },
    "integrations": {
        "description": "External integrations: Soroban contracts, ZK attestations",
        "critical_files": [
            "integrations/contract_client.py",
            "integrations/zk_attestor.py",
            "integrations/soroban_event_listener.py",
        ],
        "owners": ["@Ledger-Lenz/contract-team"],
        "review_required": True,
    },
    "monitoring": {
        "description": "Observability: Prometheus metrics, CUSUM detectors, alerting rules",
        "critical_files": [
            "monitoring/metrics_collector.py",
            "monitoring/cusum_detector.py",
            "monitoring/incident_responder.py",
            "monitoring/emergency_watchdog.py",
        ],
        "owners": ["@Ledger-Lenz/infra-team"],
        "review_required": True,
    },
    "privacy": {
        "description": "Privacy-preserving ML: differential privacy, DP-SGD, federated learning",
        "critical_files": [
            "detection/differential_privacy.py",
            "detection/privacy/",
        ],
        "owners": ["@Ledger-Lenz/ml-team", "@Ledger-Lenz/security-team"],
        "review_required": True,
    },
    "security": {
        "description": "Security-critical components: adversarial robustness, model integrity, encryption",
        "critical_files": [
            "detection/adversarial/",
            "detection/audit_trail.py",
            "utils/field_encryption.py",
        ],
        "owners": ["@Ledger-Lenz/security-team"],
        "review_required": True,
    },
    "config": {
        "description": "Central configuration and environment variable management",
        "critical_files": [
            "config.py",
        ],
        "owners": ["@Ledger-Lenz/core-maintainers"],
        "review_required": True,
    },
    "ci_cd": {
        "description": "CI/CD pipelines, build configuration, Docker",
        "critical_files": [
            ".github/workflows/",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
        ],
        "owners": ["@Ledger-Lenz/infra-team"],
        "review_required": True,
    },
    "reporting": {
        "description": "Forensic reporting: narrative generation, FATF compliance exports",
        "critical_files": [
            "reporting/fatf_exporter.py",
            "reporting/narrative_builder.py",
            "reporting/model_card_generator.py",
        ],
        "owners": ["@Ledger-Lenz/data-team"],
        "review_required": True,
    },
    "testing": {
        "description": "Test suites: unit, integration, fuzz, mutation testing",
        "critical_files": [
            "tests/",
            "tests/integration/",
            "tests/fuzz/",
        ],
        "owners": ["@Ledger-Lenz/core-maintainers"],
        "review_required": False,
    },
}

# ── CODEOWNERS parsing ───────────────────────────────────────────────────────

_CODEOWNERS_PATHS = [".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"]


@dataclass
class CodeOwnersEntry:
    """A single parsed CODEOWNERS rule."""

    pattern: str
    owners: list[str]
    line_number: int


@dataclass
class OwnershipRegistry:
    """Registry that combines CODEOWNERS rules with subsystem metadata.

    Provides a unified interface for querying ownership information across
    the LedgerLens repository.
    """

    codeowners_entries: list[CodeOwnersEntry] = field(default_factory=list)
    subsystems: dict[str, dict[str, Any]] = field(default_factory=dict)
    _repo_root: Path | None = None

    @classmethod
    def load(cls, repo_root: str | Path | None = None) -> OwnershipRegistry:
        """Load ownership metadata from CODEOWNERS and subsystem definitions.

        Args:
            repo_root: Path to the repository root. If None, attempts to
                       locate it by walking up from the current file.

        Returns:
            Populated OwnershipRegistry instance.
        """
        if repo_root is None:
            repo_root = cls._find_repo_root()
        else:
            repo_root = Path(repo_root)

        registry = cls(
            subsystems=SUBSYSTEMS.copy(),
            _repo_root=repo_root,
        )

        for codeowners_path in _CODEOWNERS_PATHS:
            full_path = repo_root / codeowners_path
            if full_path.is_file():
                registry.codeowners_entries = cls._parse_codeowners(full_path)
                break

        return registry

    @staticmethod
    def _find_repo_root() -> Path:
        """Walk up from this file's directory to find the repository root."""
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / ".git").is_dir():
                return current
            current = current.parent
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _parse_codeowners(path: Path) -> list[CodeOwnersEntry]:
        """Parse a CODEOWNERS file into structured entries.

        Handles comments, blank lines, patterns with leading '/', and
        multi-owner lines.
        """
        entries: list[CodeOwnersEntry] = []
        with open(path, encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pattern = parts[0]
                owners = [p for p in parts[1:] if p.startswith("@") or p.startswith("/")]
                if owners:
                    entries.append(
                        CodeOwnersEntry(
                            pattern=pattern,
                            owners=owners,
                            line_number=line_num,
                        )
                    )
        return entries

    def get_owners(self, file_path: str) -> list[str]:
        """Return the list of owners for a given file path.

        CODEOWNERS rules are evaluated in order; the last matching rule wins.
        If no CODEOWNERS rule matches, subsystem-based ownership is checked.

        Args:
            file_path: Relative path from the repository root.

        Returns:
            List of owner handles (e.g. ``["@Ledger-Lenz/ml-team"]``).
        """
        matched_owners: list[str] = []

        for entry in codeowners_entries(self.codeowners_entries):
            if _matches_pattern(file_path, entry.pattern):
                matched_owners = entry.owners

        if matched_owners:
            return matched_owners

        return self._get_subsystem_owners(file_path)

    def _get_subsystem_owners(self, file_path: str) -> list[str]:
        """Fallback: derive owners from subsystem critical_files metadata."""
        for _name, meta in self.subsystems.items():
            for critical_file in meta.get("critical_files", []):
                if file_path.startswith(critical_file.rstrip("/")):
                    return meta.get("owners", [])
        return []

    def list_subsystems(self) -> list[str]:
        """Return the names of all registered subsystems."""
        return sorted(self.subsystems.keys())

    def get_subsystem_info(self, subsystem: str) -> dict[str, Any] | None:
        """Return metadata for a specific subsystem.

        Args:
            subsystem: Subsystem name (e.g. ``"detection"``).

        Returns:
            Dict with keys ``description``, ``critical_files``, ``owners``,
            ``review_required``; or None if the subsystem is unknown.
        """
        return self.subsystems.get(subsystem)

    def get_review_required_files(self) -> list[str]:
        """Return all critical files that require code review."""
        files: list[str] = []
        for _name, meta in self.subsystems.items():
            if meta.get("review_required", False):
                files.extend(meta.get("critical_files", []))
        return sorted(set(files))

    def get_all_owners(self) -> list[str]:
        """Return a deduplicated list of all known owner handles."""
        owners: set[str] = set()
        for entry in self.codeowners_entries:
            owners.update(entry.owners)
        for meta in self.subsystems.values():
            owners.update(meta.get("owners", []))
        return sorted(owners)

    def validate(self) -> list[str]:
        """Validate ownership metadata for consistency.

        Returns:
            List of warning strings. Empty list means no issues found.
        """
        warnings: list[str] = []

        all_owners = self.get_all_owners()
        if not all_owners:
            warnings.append("No owners defined in CODEOWNERS or subsystem metadata.")

        for name, meta in self.subsystems.items():
            if not meta.get("owners"):
                warnings.append(f"Subsystem '{name}' has no owners defined.")
            if not meta.get("critical_files"):
                warnings.append(f"Subsystem '{name}' has no critical_files defined.")

        all_critical = self.get_review_required_files()
        for filepath in all_critical:
            owners = self.get_owners(filepath)
            if not owners:
                warnings.append(f"Critical file '{filepath}' has no owner.")

        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry to a JSON-compatible dictionary."""
        return {
            "subsystems": self.subsystems,
            "codeowners_rules": len(self.codeowners_entries),
            "all_owners": self.get_all_owners(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the registry to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def codeowners_entries(
    entries: list[CodeOwnersEntry],
) -> list[CodeOwnersEntry]:
    """Yield CODEOWNERS entries in order (identity for testing)."""
    yield from entries


def _matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if a file path matches a CODEOWNERS glob pattern.

    Handles leading '/' for root-anchored patterns and directory patterns
    with trailing '/'.
    """
    if pattern.startswith("/"):
        pattern = pattern[1:]

    if pattern.endswith("/"):
        return file_path.startswith(pattern) or fnmatch.fnmatch(file_path, pattern + "*")

    return fnmatch.fnmatch(file_path, pattern)


def check_ownership_compliance(
    changed_files: list[str],
    required_owners: set[str] | None = None,
    repo_root: str | Path | None = None,
) -> tuple[bool, list[str]]:
    """Check if changed files have owners defined in CODEOWNERS.

    Args:
        changed_files: List of file paths (relative to repo root) that were changed.
        required_owners: Set of owner handles that must be present for the check to pass.
                        If None, any owner is acceptable.
        repo_root: Repository root path. If None, walks up from utils/ownership.py.

    Returns:
        Tuple of (all_compliant: bool, violations: list[str]).
        Each violation string names the file and its matching CODEOWNERS pattern/owner.
    """
    registry = OwnershipRegistry.load(repo_root)
    violations: list[str] = []

    for file_path in changed_files:
        # Find the matching CODEOWNERS entry for this file
        matching_entry: CodeOwnersEntry | None = None
        for entry in registry.codeowners_entries:
            if _matches_pattern(file_path, entry.pattern):
                matching_entry = entry

        if matching_entry:
            owners = matching_entry.owners
            owner_list = ", ".join(owners)
            if required_owners and not any(owner in owners for owner in required_owners):
                violations.append(
                    f"  {file_path}: matches pattern '{matching_entry.pattern}' "
                    f"(owners: {owner_list}) — no reviewer from {required_owners} "
                    f"is assigned"
                )
        else:
            violations.append(
                f"  {file_path}: no CODEOWNERS pattern matches this file "
                f"— add a pattern to .github/CODEOWNERS"
            )

    return len(violations) == 0, violations
