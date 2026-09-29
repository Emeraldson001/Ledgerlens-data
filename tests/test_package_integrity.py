"""Tests for source package integrity checks (Issue #540)."""

from pathlib import Path

from utils.package_integrity import (
    DEFAULT_SOURCE_PACKAGES,
    check_source_package_integrity,
)


def test_repo_source_packages_pass_integrity_check():
    """Regression guard: the real repo tree must always be clean."""
    project_root = Path(__file__).resolve().parent.parent
    report = check_source_package_integrity(root=project_root)
    assert report.ok, report.render()
    assert report.files_checked > 0


def test_missing_init_py_is_flagged(tmp_path):
    pkg = tmp_path / "broken_pkg"
    pkg.mkdir()
    (pkg / "module.py").write_text("x = 1\n")

    report = check_source_package_integrity(root=tmp_path, packages=("broken_pkg",))

    assert not report.ok
    assert any(i.check == "missing-init" for i in report.issues)


def test_clean_package_passes(tmp_path):
    pkg = tmp_path / "clean_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "module.py").write_text("def f():\n    return 1\n")

    report = check_source_package_integrity(root=tmp_path, packages=("clean_pkg",))

    assert report.ok
    assert report.files_checked == 2


def test_merge_conflict_marker_is_flagged(tmp_path):
    pkg = tmp_path / "conflict_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "module.py").write_text("x = 1\n<<<<<<< HEAD\ny = 2\n=======\ny = 3\n>>>>>>> branch\n")

    report = check_source_package_integrity(root=tmp_path, packages=("conflict_pkg",))

    assert not report.ok
    markers_found = {
        i.message.split("'")[1] for i in report.issues if i.check == "merge-conflict-marker"
    }
    assert markers_found == {"<<<<<<<", "=======", ">>>>>>>"}


def test_syntax_error_is_flagged(tmp_path):
    pkg = tmp_path / "syntax_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "module.py").write_text("def f(:\n    return 1\n")

    report = check_source_package_integrity(root=tmp_path, packages=("syntax_pkg",))

    assert not report.ok
    assert any(i.check == "syntax-error" for i in report.issues)


def test_empty_non_init_module_is_flagged(tmp_path):
    pkg = tmp_path / "empty_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "module.py").write_text("")

    report = check_source_package_integrity(root=tmp_path, packages=("empty_pkg",))

    assert not report.ok
    assert any(i.check == "empty-module" for i in report.issues)


def test_missing_package_directory_is_flagged(tmp_path):
    report = check_source_package_integrity(root=tmp_path, packages=("does_not_exist",))

    assert not report.ok
    assert any(i.check == "missing-package" for i in report.issues)


def test_nested_directory_without_init_is_flagged(tmp_path):
    pkg = tmp_path / "nested_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "module.py").write_text("x = 1\n")

    report = check_source_package_integrity(root=tmp_path, packages=("nested_pkg",))

    assert not report.ok
    assert any(i.check == "missing-init" and "sub" in str(i.file) for i in report.issues)


def test_pycache_is_ignored(tmp_path):
    pkg = tmp_path / "cache_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc.py").write_text("garbage(((")

    report = check_source_package_integrity(root=tmp_path, packages=("cache_pkg",))

    assert report.ok


def test_default_source_packages_matches_repo_layout():
    project_root = Path(__file__).resolve().parent.parent
    for package in DEFAULT_SOURCE_PACKAGES:
        assert (project_root / package).is_dir(), f"{package} missing from repo root"
