"""Pluggable pre-parse scanning hook for file-based historical loads.

Any file handed to `open_scanned` is passed to the configured `FileScanner`
*before* any bytes reach a parser. A scanner returns a `ScanResult`; a
non-clean result raises `FileRejectedError`.

Reference implementations:
  - `NoOpScanner` (default): passes every file through. It logs a warning on
    first use so operators know scanning is NOT active unless configured.
  - `HashListScanner`: SHA-256 allowlist/denylist check backed by
    `data/allowlist.json` / `data/denylist.json` (JSON lists of hex digests).

A ClamAV integration can be plugged in by implementing `FileScanner.scan`
(e.g. calling `clamd.ClamdUnixSocket().scan(path)`) and passing it to
`set_scanner`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils.logging import get_logger

logger = get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class ScanResult:
    clean: bool
    reason: str = ""


class FileRejectedError(Exception):
    """Raised when a scanner flags a file; no parsing has occurred."""


class FileScanner(Protocol):
    def scan(self, path: Path) -> ScanResult: ...


class NoOpScanner:
    """Pass-through scanner used when no scanner is configured."""

    _warned = False

    def scan(self, path: Path) -> ScanResult:
        if not NoOpScanner._warned:
            logger.warning(
                "File scanning is NOT active: NoOpScanner in use. "
                "Configure a scanner via secure_file_handler.set_scanner()."
            )
            NoOpScanner._warned = True
        return ScanResult(clean=True, reason="no-op")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_digests(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(d).lower() for d in json.loads(path.read_text() or "[]")}


class HashListScanner:
    """Reject files whose SHA-256 is denylisted (or, if strict, not allowlisted)."""

    def __init__(
        self,
        allowlist_path: Path = _DATA_DIR / "allowlist.json",
        denylist_path: Path = _DATA_DIR / "denylist.json",
        strict: bool = False,
    ):
        self.allow = _load_digests(Path(allowlist_path))
        self.deny = _load_digests(Path(denylist_path))
        self.strict = strict

    def scan(self, path: Path) -> ScanResult:
        digest = sha256_file(path)
        if digest in self.deny:
            return ScanResult(False, f"sha256 {digest} is denylisted")
        if self.strict and digest not in self.allow:
            return ScanResult(False, f"sha256 {digest} is not allowlisted")
        return ScanResult(True)


_scanner: FileScanner = NoOpScanner()


def set_scanner(scanner: FileScanner) -> None:
    global _scanner
    _scanner = scanner


def get_scanner() -> FileScanner:
    return _scanner


def check_file(path: str | Path, scanner: FileScanner | None = None) -> Path:
    """Scan `path`; raise `FileRejectedError` if flagged. Returns the Path."""
    path = Path(path)
    result = (scanner or _scanner).scan(path)
    if not result.clean:
        logger.error("Rejected file %s before parsing: %s", path, result.reason)
        raise FileRejectedError(f"{path}: {result.reason}")
    return path


def open_scanned(path: str | Path, mode: str = "rb", scanner: FileScanner | None = None):
    """Open a file for parsing only after it passes the scanner."""
    return open(check_file(path, scanner), mode)
