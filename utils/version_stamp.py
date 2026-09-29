"""LedgerLens version stamping for generated outputs and artifacts.

Every trained model artifact, forensic report, scored output, and data
export should carry a consistent set of provenance fields so that:

1. A consumer can verify which version of the pipeline produced a given
   output.
2. Model compatibility checks can detect schema drift between model and
   scorer versions.
3. Audit logs are traceable back to a specific code commit and release.

Single source of truth
────────────────────────
The canonical version string is read (in priority order) from:

1. The ``LEDGERLENS_VERSION`` environment variable — set by CI/CD.
2. The ``version`` field in ``pyproject.toml`` — always present in the repo.
3. The installed package metadata (``importlib.metadata``) — available when
   the package is installed via ``pip install -e .`` or a release wheel.
4. The fallback string ``"0.0.0+unknown"`` — should never appear in a real
   deployment.

Public API
──────────
- :func:`get_version` — current version string.
- :func:`build_stamp` — full provenance dict to embed in artifacts.
- :func:`stamp_artifact` — in-place add ``_version_stamp`` to a dict.
- :func:`read_stamp` — extract the stamp from an artifact dict.
- :func:`verify_stamp` — check that an artifact stamp matches the running version.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from datetime import datetime, timezone
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "get_version",
    "build_stamp",
    "stamp_artifact",
    "read_stamp",
    "verify_stamp",
    "VersionMismatchError",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Key injected into every stamped artifact dict
STAMP_KEY = "_version_stamp"


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the canonical LedgerLens version string.

    Resolved in priority order:
    1. ``LEDGERLENS_VERSION`` environment variable.
    2. ``pyproject.toml`` ``[project] version`` field.
    3. ``importlib.metadata.version("ledgerlens-data")``.
    4. ``"0.0.0+unknown"`` fallback.
    """
    # 1. Env var override (CI, Docker builds)
    env_ver = os.getenv("LEDGERLENS_VERSION")
    if env_ver:
        return env_ver.strip()

    # 2. pyproject.toml (always present in the source tree)
    if _PYPROJECT.exists() and tomllib is not None:
        try:
            with open(_PYPROJECT, "rb") as f:
                data = tomllib.load(f)
            ver = data.get("project", {}).get("version")
            if ver:
                return str(ver)
        except Exception:
            pass

    # 3. Installed package metadata
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("ledgerlens-data")
    except Exception:
        pass

    return "0.0.0+unknown"


def get_git_commit() -> str | None:
    """Return the current HEAD commit hash (short), or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_branch() -> str | None:
    """Return the current branch name, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch and branch != "HEAD" else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Stamp construction
# ---------------------------------------------------------------------------


def build_stamp(
    *,
    include_git: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a version stamp dict to embed in any generated artifact.

    Returns a dict with the following fields:

    ============== =========================================================
    Field          Description
    ============== =========================================================
    version        LedgerLens version string (from :func:`get_version`)
    python_version Python ``sys.version`` string
    platform       ``platform.platform()`` (OS + arch)
    generated_at   ISO-8601 UTC timestamp of artifact generation
    git_commit     Short commit hash (``None`` if git not available)
    git_branch     Branch name (``None`` if unavailable / detached HEAD)
    ============== =========================================================

    Additional fields can be merged in via *extra*.
    """
    stamp: dict[str, Any] = {
        "version": get_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if include_git:
        stamp["git_commit"] = get_git_commit()
        stamp["git_branch"] = get_git_branch()
    if extra:
        stamp.update(extra)
    return stamp


def _content_hash(obj: Any) -> str:
    """Return a short SHA-256 hash of the JSON-serialised *obj*."""
    import json

    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stamp injection / extraction
# ---------------------------------------------------------------------------


def stamp_artifact(artifact: dict[str, Any], *, include_git: bool = True) -> dict[str, Any]:
    """Add a ``_version_stamp`` key to *artifact* in-place and return it.

    The stamp also includes a ``content_hash`` of the artifact's own keys
    (excluding the stamp itself) so downstream consumers can detect
    post-generation tampering.

    Parameters
    ----------
    artifact:
        The dict to stamp.  Modified in-place.
    include_git:
        When ``False``, omit git fields (useful in environments without git).

    Returns
    -------
    The same *artifact* dict with :data:`STAMP_KEY` added.
    """
    # Content hash of everything except the stamp key
    content_for_hash = {k: v for k, v in artifact.items() if k != STAMP_KEY}
    stamp = build_stamp(include_git=include_git)
    stamp["content_hash"] = _content_hash(content_for_hash)
    artifact[STAMP_KEY] = stamp
    return artifact


def read_stamp(artifact: dict[str, Any]) -> dict[str, Any] | None:
    """Return the :data:`STAMP_KEY` dict from *artifact*, or ``None`` if absent."""
    return artifact.get(STAMP_KEY)


# ---------------------------------------------------------------------------
# Stamp verification
# ---------------------------------------------------------------------------


class VersionMismatchError(RuntimeError):
    """Raised by :func:`verify_stamp` when an artifact's version is
    incompatible with the running version."""


def verify_stamp(
    artifact: dict[str, Any],
    *,
    strict: bool = False,
    verify_content_hash: bool = True,
) -> dict[str, Any]:
    """Verify the version stamp embedded in *artifact*.

    Parameters
    ----------
    artifact:
        Dict that must contain a :data:`STAMP_KEY` stamp.
    strict:
        When ``True``, raise :class:`VersionMismatchError` if the stamped
        version string differs from :func:`get_version`.  When ``False``
        (default), a version mismatch is returned in the result dict rather
        than raised.
    verify_content_hash:
        When ``True`` (default), re-compute the content hash and compare it
        against the stamped value.  A mismatch indicates the artifact was
        modified after stamping.

    Returns
    -------
    A dict with keys:

    - ``ok`` — ``True`` if all checks passed.
    - ``version_match`` — whether the artifact's version equals the current version.
    - ``artifact_version`` — the version recorded in the stamp.
    - ``current_version`` — the current running version.
    - ``content_hash_match`` — ``True`` / ``False`` / ``None`` (when verification
      was skipped).
    - ``errors`` — list of error strings (empty when ``ok`` is ``True``).
    """
    stamp = read_stamp(artifact)
    if stamp is None:
        if strict:
            raise VersionMismatchError(f"Artifact has no '{STAMP_KEY}' stamp.")
        return {
            "ok": False,
            "version_match": False,
            "artifact_version": None,
            "current_version": get_version(),
            "content_hash_match": None,
            "errors": [f"Artifact has no '{STAMP_KEY}' stamp."],
        }

    errors: list[str] = []
    artifact_version = stamp.get("version")
    current_version = get_version()
    version_match = artifact_version == current_version

    if not version_match:
        msg = (
            f"Version mismatch: artifact was produced by v{artifact_version}, "
            f"running v{current_version}."
        )
        errors.append(msg)
        if strict:
            raise VersionMismatchError(msg)

    content_hash_match: bool | None = None
    if verify_content_hash:
        stamped_hash = stamp.get("content_hash")
        if stamped_hash is not None:
            content_for_hash = {k: v for k, v in artifact.items() if k != STAMP_KEY}
            recomputed = _content_hash(content_for_hash)
            content_hash_match = recomputed == stamped_hash
            if not content_hash_match:
                errors.append(
                    f"Content hash mismatch: stamped={stamped_hash!r}, "
                    f"recomputed={recomputed!r}. "
                    "The artifact may have been modified after stamping."
                )

    return {
        "ok": len(errors) == 0,
        "version_match": version_match,
        "artifact_version": artifact_version,
        "current_version": current_version,
        "content_hash_match": content_hash_match,
        "errors": errors,
    }
