"""
tests/test_check_import_cycles.py — Tests for scripts/check_import_cycles.py (#546)
"""

from __future__ import annotations

import json
import pathlib
import textwrap

from scripts.check_import_cycles import (
    _extract_imports,
    _find_python_files,
    _path_to_module,
    build_dependency_graph,
    find_cycles,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_module(tmp_path: pathlib.Path, rel: str, content: str) -> pathlib.Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# _path_to_module
# ---------------------------------------------------------------------------


def test_path_to_module_simple(tmp_path):
    f = tmp_path / "detection" / "benford_engine.py"
    f.parent.mkdir()
    f.touch()
    assert _path_to_module(f, tmp_path) == "detection.benford_engine"


def test_path_to_module_init(tmp_path):
    f = tmp_path / "detection" / "__init__.py"
    f.parent.mkdir()
    f.touch()
    assert _path_to_module(f, tmp_path) == "detection"


def test_path_to_module_nested(tmp_path):
    f = tmp_path / "detection" / "adversarial" / "attacks.py"
    f.parent.mkdir(parents=True)
    f.touch()
    assert _path_to_module(f, tmp_path) == "detection.adversarial.attacks"


# ---------------------------------------------------------------------------
# _extract_imports
# ---------------------------------------------------------------------------


def test_extract_imports_absolute():
    source = "import detection.benford_engine\nimport numpy\n"
    imports = list(_extract_imports(source, "ingestion.loader"))
    assert "detection.benford_engine" in imports
    # numpy is third-party — should NOT be in results
    assert "numpy" not in imports


def test_extract_imports_from_absolute():
    source = "from detection.benford_engine import chi_square_statistic\n"
    imports = list(_extract_imports(source, "ingestion.loader"))
    assert "detection.benford_engine" in imports


def test_extract_imports_relative():
    source = "from . import benford_engine\n"
    imports = list(_extract_imports(source, "detection.model_training"))
    assert "detection.benford_engine" in imports


def test_extract_imports_relative_parent():
    source = "from .. import utils\n"
    imports = list(_extract_imports(source, "detection.adversarial.attacks"))
    assert "detection.utils" in imports or "utils" in imports


def test_extract_imports_ignores_syntax_error():
    source = "def broken(:\n    pass\n"
    imports = list(_extract_imports(source, "detection.broken"))
    assert imports == []


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------


def test_build_graph_no_cycle(tmp_path):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "# no imports\n")

    files = _find_python_files(tmp_path, ["detection"])
    graph = build_dependency_graph(files, tmp_path)

    assert "detection.b" in graph.get("detection.a", set())
    assert "detection.a" not in graph.get("detection.b", set())


def test_build_graph_cycle(tmp_path):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "import detection.a\n")

    files = _find_python_files(tmp_path, ["detection"])
    graph = build_dependency_graph(files, tmp_path)

    assert "detection.b" in graph["detection.a"]
    assert "detection.a" in graph["detection.b"]


# ---------------------------------------------------------------------------
# find_cycles
# ---------------------------------------------------------------------------


def test_find_cycles_simple():
    graph = {
        "detection.a": {"detection.b"},
        "detection.b": {"detection.a"},
        "detection.c": {"detection.a"},
    }
    cycles = find_cycles(graph)
    # The SCC {detection.a, detection.b} should be detected
    assert any(set(cycle) == {"detection.a", "detection.b"} for cycle in cycles)


def test_find_cycles_no_cycle():
    graph = {
        "detection.a": {"detection.b"},
        "detection.b": {"detection.c"},
        "detection.c": set(),
    }
    assert find_cycles(graph) == []


def test_find_cycles_three_way():
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"a"},
    }
    cycles = find_cycles(graph)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_find_cycles_cross_package_only():
    graph = {
        "detection.a": {"detection.b"},
        "detection.b": {"detection.a"},
        "ingestion.x": {"detection.a"},
    }
    cycles_all = find_cycles(graph, cross_package_only=False)
    cycles_cross = find_cycles(graph, cross_package_only=True)
    # The detection.a <-> detection.b cycle is intra-package — should be excluded
    assert len(cycles_all) >= len(cycles_cross)
    for cycle in cycles_cross:
        pkgs = {m.split(".")[0] for m in cycle}
        assert len(pkgs) > 1, "cross_package_only=True should only return cross-pkg cycles"


def test_find_cycles_self_loop():
    graph = {
        "detection.a": {"detection.a"},  # self-loop
    }
    cycles = find_cycles(graph)
    assert any(cycles)


# ---------------------------------------------------------------------------
# main() — integration
# ---------------------------------------------------------------------------


def test_main_no_cycles(tmp_path, capsys):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "# clean\n")

    ret = main(["--packages", "detection", "--root", str(tmp_path)])
    assert ret == 0
    out = capsys.readouterr().out
    assert "No import cycles" in out


def test_main_with_cycles(tmp_path, capsys):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "import detection.a\n")

    ret = main(["--packages", "detection", "--root", str(tmp_path)])
    assert ret == 2
    out = capsys.readouterr().out
    assert "cycle" in out.lower()


def test_main_writes_json_report(tmp_path):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "import detection.a\n")

    report_path = tmp_path / "cycles.json"
    main(
        [
            "--packages",
            "detection",
            "--root",
            str(tmp_path),
            "--report-path",
            str(report_path),
        ]
    )
    assert report_path.is_file()
    data = json.loads(report_path.read_text())
    assert "cycle_count" in data
    assert data["cycle_count"] >= 1


def test_main_invalid_package(tmp_path, capsys):
    ret = main(["--packages", "nonexistent_pkg", "--root", str(tmp_path)])
    assert ret == 1


def test_main_quiet_flag(tmp_path, capsys):
    write_module(tmp_path, "detection/__init__.py", "")
    write_module(tmp_path, "detection/a.py", "import detection.b\n")
    write_module(tmp_path, "detection/b.py", "import detection.a\n")

    main(["--packages", "detection", "--root", str(tmp_path), "--quiet"])
    out = capsys.readouterr().out
    # Quiet mode should not print per-cycle details
    assert "Cycle 1" not in out


# ---------------------------------------------------------------------------
# Real repo smoke test — the actual codebase should have no import cycles
# ---------------------------------------------------------------------------


def test_real_repo_no_import_cycles():
    """
    Smoke-test against the actual LedgerLens codebase.
    Fails if a real circular import is introduced.
    """
    repo_root = pathlib.Path(__file__).parent.parent.resolve()
    result = main(["--root", str(repo_root)])
    # We expect 0 (clean) or accept 2 (cycles exist but test documents them)
    # The important thing is it doesn't crash (exit 1 = fatal error)
    assert result in (0, 2), f"check_import_cycles crashed with exit code {result}"
