"""Tests for dead-path detection reports (Issue #547)."""

from pathlib import Path

from analysis.dead_path_detector import detect_dead_paths


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_unreferenced_module_is_flagged_as_candidate(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "used.py", "def f():\n    return 1\n")
    _write(
        tmp_path / "widgets" / "caller.py",
        "from widgets.used import f\n\ndef g():\n    return f()\n",
    )
    _write(tmp_path / "widgets" / "orphan.py", "def unused():\n    return 42\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    candidate_modules = {m.module for m in report.candidates}
    assert candidate_modules == {"widgets.orphan"}


def test_module_imported_via_import_statement_is_not_flagged(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "used.py", "VALUE = 1\n")
    _write(tmp_path / "widgets" / "caller.py", "import widgets.used\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    used = next(m for m in report.modules if m.module == "widgets.used")
    assert used.python_refs > 0
    assert not used.is_dead_path_candidate


def test_module_referenced_only_from_tests_is_not_flagged(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "used.py", "VALUE = 1\n")
    _write(
        tmp_path / "tests" / "test_used.py",
        "from widgets.used import VALUE\n",
    )

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    used = next(m for m in report.modules if m.module == "widgets.used")
    assert not used.is_dead_path_candidate


def test_entrypoint_module_is_never_flagged(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(
        tmp_path / "widgets" / "cli.py",
        "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
    )

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    cli = next(m for m in report.modules if m.module == "widgets.cli")
    assert cli.is_entrypoint
    assert not cli.is_dead_path_candidate


def test_module_referenced_from_makefile_is_not_flagged(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "job.py", "def run():\n    pass\n")
    _write(tmp_path / "Makefile", "run-job:\n\tpython -m widgets.job\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    job = next(m for m in report.modules if m.module == "widgets.job")
    assert job.external_ref_hits == ["Makefile"]
    assert not job.is_dead_path_candidate


def test_ignorelisted_module_is_not_flagged_but_records_reason(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "plugin.py", "def register():\n    pass\n")
    _write(
        tmp_path / "analysis" / "dead_path_ignorelist.yaml",
        "ignored:\n  widgets.plugin: 'loaded dynamically by the plugin registry'\n",
    )

    report = detect_dead_paths(
        root=tmp_path,
        candidate_packages=("widgets",),
        ignorelist_path="analysis/dead_path_ignorelist.yaml",
    )

    plugin = next(m for m in report.modules if m.module == "widgets.plugin")
    assert plugin.ignored_reason == "loaded dynamically by the plugin registry"
    assert not plugin.is_dead_path_candidate


def test_relative_import_within_subpackage_counts_as_reference(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "sub" / "__init__.py")
    _write(tmp_path / "widgets" / "sub" / "helper.py", "VALUE = 1\n")
    _write(tmp_path / "widgets" / "sub" / "user.py", "from .helper import VALUE\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    helper = next(m for m in report.modules if m.module == "widgets.sub.helper")
    assert helper.python_refs > 0
    assert not helper.is_dead_path_candidate


def test_init_py_is_never_a_candidate(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py", "VALUE = 1\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    assert all(m.module != "widgets" for m in report.modules)


def test_render_and_render_markdown_and_to_dict_are_consistent(tmp_path):
    _write(tmp_path / "widgets" / "__init__.py")
    _write(tmp_path / "widgets" / "orphan.py", "VALUE = 1\n")

    report = detect_dead_paths(root=tmp_path, candidate_packages=("widgets",))

    assert "widgets.orphan" in report.render()
    assert "widgets.orphan" in report.render_markdown()
    as_dict = report.to_dict()
    assert as_dict["modules_checked"] == 1
    assert as_dict["candidates"] == [{"module": "widgets.orphan", "file": "widgets/orphan.py"}]


def test_real_repo_ignorelist_is_valid_yaml_and_modules_exist():
    """Regression guard: every ignorelisted module must still exist and be a
    real candidate package module — stale ignorelist entries should be
    removed, not left to silently mask nothing.
    """
    project_root = Path(__file__).resolve().parent.parent
    report = detect_dead_paths(root=project_root)
    ignored = {m.module: m for m in report.modules if m.ignored_reason is not None}
    for module, mref in ignored.items():
        assert (project_root / mref.file).exists(), f"{module} ignorelisted but file missing"
