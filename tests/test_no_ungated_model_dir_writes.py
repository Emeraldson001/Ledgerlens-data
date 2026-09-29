"""Static check (Grand 2 / issue #671, Required verification): no code path
writes trained model artifacts to disk without going through
``detection.model_governance.guard_production_write`` — directly, or via a
caller that already calls it before invoking the write.

This is an allowlist-diff check, not a formal proof: it enumerates every
``joblib.dump(`` call site in production code (excluding tests and
``detection/model_governance.py`` itself, which IS the gate) and requires
each to be justified with an inline ``# GUARDED``/``# UNGUARDED-OK`` comment
explaining *how* it's protected. Adding a new raw model-artifact write
without one of those comments fails CI, forcing a reviewer to consciously
classify it rather than letting a new ungated write path slip in silently.

Comment vocabulary (checked verbatim, case-sensitive):
    # GUARDED: <reason>       — a guard_production_write() call actually
                                 protects this write (in this function or a
                                 caller in the same module).
    # UNGUARDED-OK: <reason>  — deliberately out of scope for the production
                                 trust/regression gate (e.g. an edge-deployment
                                 compressed variant with a distinct filename
                                 from the primary served artifact).
"""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories to scan for `joblib.dump(` call sites.
_SCAN_DIRS = ["detection", "scripts", "training"]

# detection/model_governance.py IS the gate — its own internal
# joblib-adjacent file operations (there are none today; it uses shutil.copy2
# to publish already-trained files) are exempt by construction.
_EXEMPT_FILES = {os.path.join(REPO_ROOT, "detection", "model_governance.py")}

_DUMP_RE = re.compile(r"joblib\.dump\(")
_GUARD_COMMENT_RE = re.compile(r"#\s*(GUARDED|UNGUARDED-OK)\s*:")


def _iter_python_files():
    for scan_dir in _SCAN_DIRS:
        base = os.path.join(REPO_ROOT, scan_dir)
        for root, _dirs, files in os.walk(base):
            for fname in files:
                if fname.endswith(".py"):
                    yield os.path.join(root, fname)


def _find_unjustified_dump_sites() -> list[str]:
    violations = []
    for path in _iter_python_files():
        if path in _EXEMPT_FILES:
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if not _DUMP_RE.search(line):
                continue
            # Look for a justifying comment on this line or the preceding
            # few lines (covers a comment placed just above the call).
            window = lines[max(0, i - 6) : i + 1]
            if any(_GUARD_COMMENT_RE.search(w) for w in window):
                continue
            violations.append(f"{os.path.relpath(path, REPO_ROOT)}:{i + 1}: {line.strip()}")
    return violations


def test_every_joblib_dump_site_is_classified_guarded_or_exempt():
    violations = _find_unjustified_dump_sites()
    assert not violations, (
        "Found joblib.dump(...) call site(s) writing model artifacts with no "
        "'# GUARDED: ...' or '# UNGUARDED-OK: ...' justification comment — "
        "every path that can write a model artifact to disk must go through "
        "detection.model_governance.guard_production_write (directly or via "
        "a caller), or be explicitly marked out of scope. Violations:\n" + "\n".join(violations)
    )
