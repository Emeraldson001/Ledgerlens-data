"""Reproducibility snapshots for training datasets — Issue #533.

Provides a deterministic, tamper-evident snapshot mechanism for LedgerLens
training datasets.  A snapshot captures:

* A copy of the dataset (Parquet or arbitrary file) under a content-addressed
  snapshot directory.
* A ``snapshot.json`` manifest containing the SHA-256 of the dataset, row count,
  column list, creation timestamp, pipeline version, and optional user-supplied
  metadata.
* An Ed25519 signature over the manifest (if a signing key is configured).

The snapshot can later be **restored** (copy back to a target path) or
**verified** (re-compute the hash and compare against the manifest) to confirm
that the dataset has not been modified since the snapshot was taken.

Components
----------
``DatasetSnapshot``
    The core class; call :meth:`freeze`, :meth:`restore`, :meth:`verify`.

``SnapshotManifest``
    Dataclass representation of ``snapshot.json``.

``SnapshotRegistry``
    Tracks all snapshots under a root directory; useful for listing and
    garbage-collecting old snapshots.

Usage example
-------------
>>> snap = DatasetSnapshot(snapshot_root=Path("data/snapshots"))
>>> manifest = snap.freeze(
...     source_path=Path("data/synthetic_dataset.parquet"),
...     label="pre-training-20240601",
...     metadata={"pipeline_run_id": "abc123"},
... )
>>> print(manifest.snapshot_id)   # "snap_20240601T120000_abc12345"
>>> ok = snap.verify(manifest.snapshot_id)   # True
>>> snap.restore(manifest.snapshot_id, target=Path("data/restored.parquet"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "DatasetSnapshot",
    "SnapshotManifest",
    "SnapshotRegistry",
    "IntegrityError",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IntegrityError(RuntimeError):
    """Raised when snapshot integrity verification fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path* (streaming, memory-efficient)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _parquet_shape(path: Path) -> tuple[int, list[str]]:
    """Return ``(row_count, column_list)`` for a Parquet file."""
    try:
        df = pd.read_parquet(path, columns=None)
        return len(df), sorted(df.columns.tolist())
    except Exception:
        return 0, []


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _snapshot_id(label: str | None = None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    if label:
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:32]
        return f"snap_{ts}_{safe_label}_{short}"
    return f"snap_{ts}_{short}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class SnapshotManifest:
    """Metadata record persisted alongside each snapshot file."""

    snapshot_id: str
    source_filename: str
    sha256: str
    row_count: int
    columns: list[str]
    created_at: str
    label: str
    ledgerlens_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None  # hex Ed25519 signature over canonical JSON

    FILENAME: str = "snapshot.json"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "source_filename": self.source_filename,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "columns": self.columns,
            "created_at": self.created_at,
            "label": self.label,
            "ledgerlens_version": self.ledgerlens_version,
            "metadata": self.metadata,
        }
        if self.signature is not None:
            d["signature"] = self.signature
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotManifest:
        return cls(
            snapshot_id=data["snapshot_id"],
            source_filename=data["source_filename"],
            sha256=data["sha256"],
            row_count=data.get("row_count", 0),
            columns=data.get("columns", []),
            created_at=data["created_at"],
            label=data.get("label", ""),
            ledgerlens_version=data.get("ledgerlens_version", "unknown"),
            metadata=data.get("metadata", {}),
            signature=data.get("signature"),
        )

    @classmethod
    def load(cls, snapshot_dir: Path) -> SnapshotManifest:
        """Load from *snapshot_dir/snapshot.json*.

        Raises
        ------
        FileNotFoundError
            If the manifest is absent.
        json.JSONDecodeError
            If the file is malformed.
        """
        path = snapshot_dir / cls.FILENAME
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, snapshot_dir: Path) -> None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / self.FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def canonical_json(self) -> str:
        """Return the canonical (sorted-key, no whitespace) JSON for signing."""
        d = {k: v for k, v in self.to_dict().items() if k != "signature"}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# DatasetSnapshot
# ---------------------------------------------------------------------------


class DatasetSnapshot:
    """Freeze, restore, and verify dataset snapshots.

    Parameters
    ----------
    snapshot_root:
        Directory under which all snapshot sub-directories are created.
        Defaults to ``"data/snapshots"``.
    ledgerlens_version:
        Version string recorded in every manifest.  Defaults to the value
        of ``config.ledgerlens_version`` if the config module is importable,
        otherwise ``"unknown"``.
    signing_key_path:
        If provided, the file is expected to contain a PEM-encoded Ed25519
        private key (``cryptography`` library).  The manifest is signed on
        freeze and the signature is verified on verify.  Signing is
        *optional*; absence does not prevent snapshot creation.
    """

    def __init__(
        self,
        snapshot_root: Path | str = Path("data/snapshots"),
        ledgerlens_version: str | None = None,
        signing_key_path: Path | str | None = None,
    ) -> None:
        self.snapshot_root = Path(snapshot_root)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        if ledgerlens_version is None:
            try:
                from utils.version_stamp import get_version

                ledgerlens_version = get_version()
            except Exception:
                try:
                    from config import config  # type: ignore

                    ledgerlens_version = getattr(config, "LEDGERLENS_VERSION", "unknown")
                except Exception:
                    ledgerlens_version = "unknown"
        self.ledgerlens_version = ledgerlens_version

        self._signing_key = None
        self._verifying_key = None
        if signing_key_path is not None:
            self._load_signing_key(Path(signing_key_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeze(
        self,
        source_path: Path | str,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotManifest:
        """Create an immutable snapshot of *source_path*.

        The file is copied into
        ``<snapshot_root>/<snapshot_id>/<source_filename>`` and a
        ``snapshot.json`` manifest is written alongside it.

        Parameters
        ----------
        source_path:
            Path to the file to snapshot.
        label:
            Human-readable label (e.g. ``"pre-training-2024-06"``).
        metadata:
            Arbitrary extra fields stored in the manifest.

        Returns
        -------
        SnapshotManifest

        Raises
        ------
        FileNotFoundError
            If *source_path* does not exist.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        sid = _snapshot_id(label)
        snap_dir = self.snapshot_root / sid

        logger.info("Freezing snapshot %r from %s", sid, source_path)

        # Compute hash before copying (cheaper: no extra I/O)
        sha256 = _sha256_file(source_path)
        row_count, columns = _parquet_shape(source_path)

        # Atomic copy: write to temp then rename into snapshot dir
        snap_dir.mkdir(parents=True, exist_ok=True)
        target = snap_dir / source_path.name
        with tempfile.NamedTemporaryFile(dir=snap_dir, suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copy2(source_path, tmp_path)
            tmp_path.rename(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        manifest = SnapshotManifest(
            snapshot_id=sid,
            source_filename=source_path.name,
            sha256=sha256,
            row_count=row_count,
            columns=columns,
            created_at=_now_iso(),
            label=label,
            ledgerlens_version=self.ledgerlens_version,
            metadata=metadata or {},
        )

        # Sign if a key is available
        if self._signing_key is not None:
            manifest.signature = self._sign(manifest.canonical_json())

        manifest.save(snap_dir)
        logger.info("Snapshot %r created: sha256=%s rows=%d", sid, sha256[:12], row_count)
        return manifest

    def verify(self, snapshot_id: str) -> bool:
        """Verify integrity of an existing snapshot.

        Re-computes the SHA-256 of the stored file and compares it to the
        manifest.  Optionally verifies the Ed25519 signature if a verifying
        key was loaded.

        Parameters
        ----------
        snapshot_id:
            The snapshot identifier as returned by :meth:`freeze`.

        Returns
        -------
        bool
            ``True`` if the snapshot passes all checks.

        Raises
        ------
        FileNotFoundError
            If the snapshot directory or manifest does not exist.
        IntegrityError
            If any integrity check fails (hash mismatch, signature invalid, or
            data file absent).
        """
        snap_dir = self.snapshot_root / snapshot_id
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

        manifest = SnapshotManifest.load(snap_dir)
        data_file = snap_dir / manifest.source_filename

        if not data_file.exists():
            raise IntegrityError(
                f"Snapshot {snapshot_id!r}: data file {manifest.source_filename!r} is missing"
            )

        actual_sha256 = _sha256_file(data_file)
        if actual_sha256 != manifest.sha256:
            raise IntegrityError(
                f"Snapshot {snapshot_id!r}: SHA-256 mismatch — "
                f"expected {manifest.sha256}, got {actual_sha256}"
            )

        if self._verifying_key is not None and manifest.signature is not None:
            if not self._verify_signature(manifest.canonical_json(), manifest.signature):
                raise IntegrityError(
                    f"Snapshot {snapshot_id!r}: Ed25519 signature verification failed"
                )

        logger.info(
            "Snapshot %r integrity verified (sha256=%s...)", snapshot_id, actual_sha256[:12]
        )
        return True

    def restore(
        self,
        snapshot_id: str,
        target: Path | str,
        verify: bool = True,
    ) -> Path:
        """Copy a snapshot's data file to *target*.

        Parameters
        ----------
        snapshot_id:
            Snapshot to restore.
        target:
            Destination path.  Parent directories are created automatically.
        verify:
            If ``True`` (default), integrity is verified before restoring.

        Returns
        -------
        Path
            Resolved path of the restored file.

        Raises
        ------
        IntegrityError
            If *verify* is ``True`` and integrity check fails.
        """
        if verify:
            self.verify(snapshot_id)

        snap_dir = self.snapshot_root / snapshot_id
        manifest = SnapshotManifest.load(snap_dir)
        data_file = snap_dir / manifest.source_filename

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copy2(data_file, tmp_path)
            tmp_path.rename(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info("Snapshot %r restored to %s", snapshot_id, target)
        return target

    def get_manifest(self, snapshot_id: str) -> SnapshotManifest:
        """Return the manifest for *snapshot_id*.

        Raises
        ------
        FileNotFoundError
        """
        snap_dir = self.snapshot_root / snapshot_id
        return SnapshotManifest.load(snap_dir)

    def list_snapshots(self) -> list[str]:
        """Return snapshot IDs sorted chronologically (newest first)."""
        ids = [
            p.name
            for p in sorted(self.snapshot_root.iterdir(), reverse=True)
            if p.is_dir() and (p / SnapshotManifest.FILENAME).exists()
        ]
        return ids

    def delete(self, snapshot_id: str) -> None:
        """Permanently delete a snapshot directory.

        Use with caution — this cannot be undone.
        """
        snap_dir = self.snapshot_root / snapshot_id
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id!r} not found")
        shutil.rmtree(snap_dir)
        logger.info("Deleted snapshot %r", snapshot_id)

    # ------------------------------------------------------------------
    # Signing helpers
    # ------------------------------------------------------------------

    def _load_signing_key(self, path: Path) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            with path.open("rb") as f:
                pem = f.read()
            key = serialization.load_pem_private_key(pem, password=None)
            if isinstance(key, Ed25519PrivateKey):
                self._signing_key = key
                self._verifying_key = key.public_key()
            else:
                logger.warning("Signing key at %s is not Ed25519; signing disabled", path)
        except ImportError:
            logger.warning("cryptography package not installed; Ed25519 signing disabled")
        except Exception as exc:
            logger.warning("Failed to load signing key %s: %s; signing disabled", path, exc)

    def _sign(self, payload: str) -> str:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        assert isinstance(self._signing_key, Ed25519PrivateKey)
        sig_bytes = self._signing_key.sign(payload.encode())
        return sig_bytes.hex()

    def _verify_signature(self, payload: str, signature_hex: str) -> bool:
        try:
            from cryptography.exceptions import InvalidSignature

            self._verifying_key.verify(bytes.fromhex(signature_hex), payload.encode())
            return True
        except InvalidSignature:
            return False
        except Exception as exc:
            logger.warning("Signature verification error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SnapshotRegistry:
    """High-level view over all snapshots under a root directory.

    Parameters
    ----------
    snapshot_root:
        Same root used by :class:`DatasetSnapshot`.
    """

    def __init__(self, snapshot_root: Path | str = Path("data/snapshots")) -> None:
        self.snapshot_root = Path(snapshot_root)

    def list_all(self) -> list[SnapshotManifest]:
        """Return all manifests, sorted chronologically (newest first)."""
        manifests: list[SnapshotManifest] = []
        if not self.snapshot_root.exists():
            return manifests
        for snap_dir in sorted(self.snapshot_root.iterdir(), reverse=True):
            if not snap_dir.is_dir():
                continue
            manifest_path = snap_dir / SnapshotManifest.FILENAME
            if not manifest_path.exists():
                continue
            try:
                manifests.append(SnapshotManifest.load(snap_dir))
            except Exception as exc:
                logger.warning("Could not load manifest from %s: %s", snap_dir, exc)
        return manifests

    def find_by_label(self, label: str) -> list[SnapshotManifest]:
        """Return all snapshots whose label matches *label* exactly."""
        return [m for m in self.list_all() if m.label == label]

    def find_by_sha256(self, sha256: str) -> list[SnapshotManifest]:
        """Return all snapshots whose dataset hash starts with *sha256*."""
        return [m for m in self.list_all() if m.sha256.startswith(sha256)]

    def prune_old(self, keep_latest_n: int) -> list[str]:
        """Delete all but the *keep_latest_n* most recent snapshots.

        Returns the list of deleted snapshot IDs.
        """
        manifests = self.list_all()
        to_delete = manifests[keep_latest_n:]
        deleted: list[str] = []
        for m in to_delete:
            snap_dir = self.snapshot_root / m.snapshot_id
            shutil.rmtree(snap_dir, ignore_errors=True)
            deleted.append(m.snapshot_id)
            logger.info("Registry pruned snapshot %r", m.snapshot_id)
        return deleted

    def to_dict(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.list_all()]
