"""Versioned model artifact lifecycle system.

This module provides a reusable, typed contract for tracking the lifecycle
of trained model artifacts (e.g. the ``.joblib`` files under ``models/``)
independently of the signing/transparency-log flow in
``scripts/publish_model_artifact.py``.

Where ``TransparencyLog`` (see ``detection/persistence.py``) is an
append-only audit trail of signed hashes, ``ModelArtifactRegistry`` is the
*operational* registry: it knows which version is currently active, which
versions are staged/validated/deprecated/rolled back, and enforces which
stage transitions are legal. It is backed by a single JSON manifest file so
it has no new infrastructure dependency and can be inspected/edited by hand
in an emergency.

Typical usage::

    registry = ModelArtifactRegistry(manifest_path="models/artifact_manifest.json")
    version = registry.register(
        name="rf",
        artifact_path="models/rf.joblib",
        metrics={"auc": 0.94},
    )
    registry.validate(name="rf", version=version)
    registry.promote(name="rf", version=version)
    active = registry.get_active("rf")

    # Bad rollout discovered in production:
    registry.rollback(name="rf", reason="AUC regression on canary traffic")
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ArtifactStage(StrEnum):
    """Legal lifecycle stages for a registered model artifact."""

    STAGED = "staged"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


# Adjacency list of legal forward/side transitions. Anything not listed here
# raises InvalidTransitionError, which is the main diagnostic surface for
# "why can't I do this" bug reports.
_LEGAL_TRANSITIONS: dict[ArtifactStage, set[ArtifactStage]] = {
    ArtifactStage.STAGED: {ArtifactStage.VALIDATED, ArtifactStage.DEPRECATED},
    ArtifactStage.VALIDATED: {ArtifactStage.PROMOTED, ArtifactStage.DEPRECATED},
    ArtifactStage.PROMOTED: {ArtifactStage.DEPRECATED, ArtifactStage.ROLLED_BACK},
    ArtifactStage.DEPRECATED: set(),
    ArtifactStage.ROLLED_BACK: {ArtifactStage.STAGED},
}


class ArtifactLifecycleError(Exception):
    """Base class for all artifact-lifecycle diagnostics."""


class ArtifactNotFoundError(ArtifactLifecycleError):
    def __init__(self, name: str, version: str | None = None):
        self.name = name
        self.version = version
        if version:
            super().__init__(
                f"No artifact registered for name={name!r} version={version!r}. "
                f"Check models/artifact_manifest.json or call registry.list_versions({name!r})."
            )
        else:
            super().__init__(f"No artifact family named {name!r} in the registry manifest.")


class InvalidTransitionError(ArtifactLifecycleError):
    def __init__(self, name: str, version: str, current: ArtifactStage, target: ArtifactStage):
        self.name = name
        self.version = version
        self.current = current
        self.target = target
        allowed = ", ".join(s.value for s in _LEGAL_TRANSITIONS[current]) or "(none)"
        super().__init__(
            f"Cannot move {name}:{version} from {current.value} -> {target.value}. "
            f"Allowed transitions from {current.value}: {allowed}."
        )


class TrustVerificationRequiredError(ArtifactLifecycleError):
    """Raised by ``promote()`` when the registry has no ``trust_verifier``.

    Grand 2 (issue #671) invariant: the trust chain and the promotion state
    machine must never disagree, which means ``PROMOTED`` can only be
    reached after a real cryptographic trust-chain check has run. A registry
    constructed without a ``trust_verifier`` cannot make that guarantee, so
    it refuses to promote anything rather than silently skipping the check.
    """

    def __init__(self, name: str, version: str):
        super().__init__(
            f"Cannot promote {name}:{version} — this ModelArtifactRegistry has no "
            "trust_verifier configured. Construct it with "
            "trust_verifier=detection.model_governance.make_trust_verifier(...) so "
            "PROMOTED can only be reached after the Ed25519/transparency-log trust "
            "chain succeeds."
        )


class IntegrityCheckError(ArtifactLifecycleError):
    def __init__(self, name: str, version: str, expected: str, actual: str):
        self.name = name
        self.version = version
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Integrity check failed for {name}:{version}: manifest sha256={expected} "
            f"but on-disk artifact sha256={actual}. The file may have been modified, "
            f"corrupted, or replaced after registration."
        )


@dataclass
class ArtifactRecord:
    """A single versioned artifact entry in the manifest."""

    name: str
    version: str
    artifact_path: str
    sha256: str
    stage: ArtifactStage
    created_at: float
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    parent_version: str | None = None
    rollback_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactRecord:
        d = dict(d)
        d["stage"] = ArtifactStage(d["stage"])
        return cls(**d)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelArtifactRegistry:
    """Durable, file-backed registry for versioned model artifacts.

    The manifest is a JSON object of the form::

        {"<name>": {"<version>": {...ArtifactRecord fields...}, ...}, ...}

    Writes are atomic (write-to-temp + os.replace) so a crash mid-write
    cannot leave a truncated/corrupt manifest.
    """

    def __init__(
        self,
        manifest_path: str = "models/artifact_manifest.json",
        trust_verifier: Callable[[ArtifactRecord], None] | None = None,
    ):
        """
        Args:
            manifest_path: JSON manifest file backing this registry.
            trust_verifier: Callable invoked with the :class:`ArtifactRecord`
                about to be promoted. Must raise (e.g. ``ModelIntegrityError``
                or ``ArtifactCompatibilityError``) if the artifact's
                cryptographic trust chain or compatibility contract does not
                hold; must return ``None`` on success. Required for
                ``promote()`` to succeed — see ``TrustVerificationRequiredError``.
                Typically built with
                ``detection.model_governance.make_trust_verifier(...)``.
        """
        self.manifest_path = manifest_path
        self._data: dict[str, dict[str, ArtifactRecord]] = {}
        self._trust_verifier = trust_verifier
        self._load()

    # -- persistence ------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.manifest_path):
            self._data = {}
            return
        with open(self.manifest_path) as f:
            raw = json.load(f)
        self._data = {
            name: {version: ArtifactRecord.from_dict(rec) for version, rec in versions.items()}
            for name, versions in raw.items()
        }

    def _save(self) -> None:
        serialisable = {
            name: {version: rec.to_dict() for version, rec in versions.items()}
            for name, versions in self._data.items()
        }
        directory = os.path.dirname(self.manifest_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".artifact_manifest_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(serialisable, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.manifest_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -- core API -----------------------------------------------------

    def register(
        self,
        name: str,
        artifact_path: str,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        version: str | None = None,
    ) -> str:
        """Register a new artifact version in STAGED stage. Returns the version id."""
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        version = version or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        sha = _sha256_file(artifact_path)
        parent = self.get_active(name, silent=True)

        record = ArtifactRecord(
            name=name,
            version=version,
            artifact_path=artifact_path,
            sha256=sha,
            stage=ArtifactStage.STAGED,
            created_at=time.time(),
            metrics=metrics or {},
            tags=tags or {},
            parent_version=parent.version if parent else None,
        )
        record.history.append({"stage": ArtifactStage.STAGED.value, "at": record.created_at})

        self._data.setdefault(name, {})[version] = record
        self._save()
        return version

    def _get(self, name: str, version: str) -> ArtifactRecord:
        try:
            return self._data[name][version]
        except KeyError as exc:
            raise ArtifactNotFoundError(name, version) from exc

    def _transition(
        self, name: str, version: str, target: ArtifactStage, **extra: Any
    ) -> ArtifactRecord:
        record = self._get(name, version)
        if target not in _LEGAL_TRANSITIONS[record.stage]:
            raise InvalidTransitionError(name, version, record.stage, target)
        record.stage = target
        record.history.append({"stage": target.value, "at": time.time(), **extra})
        self._save()
        return record

    def validate(self, name: str, version: str) -> ArtifactRecord:
        """Mark an artifact as passed offline/CI validation checks."""
        return self._transition(name, version, ArtifactStage.VALIDATED)

    def promote(self, name: str, version: str) -> ArtifactRecord:
        """Promote a validated artifact to active/production use.

        Before the state transition is applied, ``self._trust_verifier`` (if
        configured) is called with the target :class:`ArtifactRecord`. Any
        exception it raises propagates unchanged and the record's stage is
        left untouched — a failed trust check must never be recorded as a
        successful promotion, and a successful promotion can never happen
        without one. If no verifier is configured, ``promote()`` raises
        :class:`TrustVerificationRequiredError` rather than silently skipping
        the check (Grand 2 / issue #671, invariant 2: the trust chain and the
        promotion state machine must never be able to disagree).

        Any previously PROMOTED version of the same ``name`` is automatically
        moved to DEPRECATED so ``get_active`` always resolves to exactly one
        version.
        """
        record = self._get(name, version)
        if self._trust_verifier is None:
            raise TrustVerificationRequiredError(name, version)
        self._trust_verifier(record)  # raises on failure; nothing below runs

        for other_version, other in self._data.get(name, {}).items():
            if other.stage == ArtifactStage.PROMOTED and other_version != version:
                other.stage = ArtifactStage.DEPRECATED
                other.history.append(
                    {
                        "stage": ArtifactStage.DEPRECATED.value,
                        "at": time.time(),
                        "reason": "superseded",
                    }
                )
        return self._transition(name, version, ArtifactStage.PROMOTED)

    def deprecate(self, name: str, version: str, reason: str | None = None) -> ArtifactRecord:
        return self._transition(name, version, ArtifactStage.DEPRECATED, reason=reason)

    def rollback(
        self, name: str, version: str | None = None, reason: str | None = None
    ) -> ArtifactRecord:
        """Roll back the active (or given) version and re-activate its parent.

        Diagnostics: raises ArtifactNotFoundError if there is no promoted
        version to roll back, and InvalidTransitionError if the target
        version is not currently PROMOTED.
        """
        record = self._get(name, version) if version else self.get_active(name)
        rolled_back = self._transition(
            name, record.version, ArtifactStage.ROLLED_BACK, reason=reason
        )
        rolled_back.rollback_reason = reason

        if rolled_back.parent_version:
            parent = self._get(name, rolled_back.parent_version)
            if parent.stage in (ArtifactStage.DEPRECATED, ArtifactStage.VALIDATED):
                parent.stage = ArtifactStage.PROMOTED
                parent.history.append(
                    {
                        "stage": ArtifactStage.PROMOTED.value,
                        "at": time.time(),
                        "reason": "rollback_reactivation",
                    }
                )
        self._save()
        return rolled_back

    def verify_integrity(self, name: str, version: str) -> None:
        """Raise IntegrityCheckError if the on-disk artifact no longer matches
        the sha256 recorded at registration time."""
        record = self._get(name, version)
        if not os.path.exists(record.artifact_path):
            raise ArtifactNotFoundError(name, version)
        actual = _sha256_file(record.artifact_path)
        if actual != record.sha256:
            raise IntegrityCheckError(name, version, record.sha256, actual)

    # -- queries --------------------------------------------------------

    def get_active(self, name: str, silent: bool = False) -> ArtifactRecord | None:
        """Return the currently PROMOTED artifact for ``name``, or None/raise."""
        for record in self._data.get(name, {}).values():
            if record.stage == ArtifactStage.PROMOTED:
                return record
        if silent:
            return None
        raise ArtifactNotFoundError(name)

    def list_versions(self, name: str) -> list[ArtifactRecord]:
        return sorted(self._data.get(name, {}).values(), key=lambda r: r.created_at)

    def list_names(self) -> list[str]:
        return sorted(self._data.keys())
