"""Tests for utils/deprecation.py and scripts/check_deprecation_policy.py (Issue #511)."""

import textwrap
from pathlib import Path

import pytest

from scripts.check_deprecation_policy import check_file, check_repository
from utils.deprecation import deprecated, get_registered_deprecations


def test_deprecated_raises_without_reason():
    with pytest.raises(ValueError):
        deprecated(reason="", removal_version="0.4.0")


def test_deprecated_raises_without_removal_version():
    with pytest.raises(ValueError):
        deprecated(reason="No longer needed.", removal_version="")


def test_deprecated_wrapper_warns_and_delegates():
    @deprecated(reason="Superseded by new_fn.", removal_version="0.4.0", replacement="new_fn")
    def old_fn(x):
        return x * 2

    with pytest.warns(DeprecationWarning, match="old_fn is deprecated"):
        assert old_fn(3) == 6


def test_deprecated_wrapper_documents_removal_and_metadata():
    @deprecated(reason="Superseded.", removal_version="0.4.0", replacement="new_fn")
    def old_fn():
        """Original docstring."""

    assert ".. deprecated:: 0.4.0" in old_fn.__doc__
    assert old_fn.__deprecated__.removal_version == "0.4.0"
    assert old_fn.__deprecated__.replacement == "new_fn"


def test_registry_records_every_decorated_symbol():
    before = len(get_registered_deprecations())

    @deprecated(reason="r", removal_version="0.9.0")
    def tracked_fn():
        pass

    after = get_registered_deprecations()
    assert len(after) == before + 1
    assert after[-1].name == tracked_fn.__wrapped__.__qualname__
    assert after[-1].removal_version == "0.9.0"


# ---------------------------------------------------------------------------
# scripts/check_deprecation_policy.py
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, relative: str, content: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def test_past_due_removal_version_is_a_violation(tmp_path):
    path = _write(
        tmp_path,
        "old.py",
        """
        from utils.deprecation import deprecated

        @deprecated(reason="obsolete", removal_version="0.1.0")
        def old_symbol():
            pass
        """,
    )
    violations = check_file(path, current_version="0.2.0")
    assert any("has already passed" in v.message for v in violations)


def test_future_removal_version_is_not_a_violation(tmp_path):
    path = _write(
        tmp_path,
        "future.py",
        """
        from utils.deprecation import deprecated

        @deprecated(reason="obsolete", removal_version="9.0.0")
        def future_symbol():
            pass
        """,
    )
    violations = check_file(path, current_version="0.2.0")
    assert violations == []


def test_missing_removal_version_argument_is_a_violation(tmp_path):
    path = _write(
        tmp_path,
        "bad.py",
        """
        from utils.deprecation import deprecated

        @deprecated(reason="obsolete")
        def bad_symbol():
            pass
        """,
    )
    violations = check_file(path, current_version="0.2.0")
    assert any("missing a literal 'removal_version'" in v.message for v in violations)


def test_unstructured_deprecation_warning_without_docstring_is_a_violation(tmp_path):
    path = _write(
        tmp_path,
        "legacy.py",
        """
        import warnings

        def legacy_symbol():
            warnings.warn("old", DeprecationWarning, stacklevel=2)
        """,
    )
    violations = check_file(path, current_version="0.2.0")
    assert any("no '.. deprecated::' docstring note" in v.message for v in violations)


def test_unstructured_deprecation_warning_with_docstring_note_passes(tmp_path):
    path = _write(
        tmp_path,
        "legacy_documented.py",
        '''
        import warnings

        def legacy_symbol():
            """Does a thing.

            .. deprecated::
                Use new_symbol instead.
            """
            warnings.warn("old", DeprecationWarning, stacklevel=2)
        ''',
    )
    violations = check_file(path, current_version="0.2.0")
    assert violations == []


def test_check_repository_passes_on_real_repo():
    """Regression guard: the actual repo must always pass this policy check."""
    repo_root = Path(__file__).resolve().parent.parent
    violations = check_repository(repo_root)
    assert violations == [], [f"{v.file}:{v.line} {v.symbol}: {v.message}" for v in violations]
