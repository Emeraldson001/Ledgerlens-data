"""Tests for scripts/check_module_dependencies.py against synthetic package
trees, so the test suite doesn't depend on (and isn't broken by) unrelated
changes to the real repo's import graph.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_module_dependencies as cmd  # noqa: E402

BASE_CONFIG = {
    "layers": [
        {"name": "foundation", "packages": ["utils"]},
        {"name": "domain", "packages": ["detection"]},
        {"name": "entrypoint", "packages": ["api"]},
    ],
    "forbidden_imports": [],
    "excluded_packages": [],
}


def _write_package(root: Path, package: str, filename: str, source: str) -> None:
    pkg_dir = root / package
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").touch(exist_ok=True)
    (pkg_dir / filename).write_text(textwrap.dedent(source))


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Point the checker at an isolated, synthetic package tree instead of
    the real repository, so these tests are independent of unrelated
    changes elsewhere in the codebase."""
    monkeypatch.setattr(cmd, "REPO_ROOT", tmp_path)
    return tmp_path


class TestLayering:
    def test_lower_layer_importing_higher_layer_is_a_violation(self, fake_repo, tmp_path):
        _write_package(tmp_path, "utils", "helpers.py", "import api.app\n")
        config = cmd.BoundaryConfig(BASE_CONFIG)

        violations = cmd.check(config)

        assert len(violations) == 1
        assert "utils" in violations[0].message
        assert "api" in violations[0].message
        assert violations[0].lineno == 1

    def test_same_layer_import_is_allowed(self, fake_repo, tmp_path):
        config_dict = {**BASE_CONFIG}
        config_dict["layers"] = [
            {"name": "foundation", "packages": ["utils"]},
            {"name": "domain", "packages": ["detection", "features"]},
            {"name": "entrypoint", "packages": ["api"]},
        ]
        _write_package(tmp_path, "detection", "engine.py", "import features.build\n")
        (tmp_path / "features").mkdir()
        (tmp_path / "features" / "__init__.py").touch()
        config = cmd.BoundaryConfig(config_dict)

        violations = cmd.check(config)

        assert violations == []

    def test_higher_layer_importing_lower_layer_is_allowed(self, fake_repo, tmp_path):
        _write_package(tmp_path, "api", "app.py", "from detection.engine import score\n")
        _write_package(tmp_path, "detection", "engine.py", "def score(): ...\n")
        config = cmd.BoundaryConfig(BASE_CONFIG)

        violations = cmd.check(config)

        assert violations == []

    def test_non_local_imports_are_ignored(self, fake_repo, tmp_path):
        _write_package(tmp_path, "utils", "helpers.py", "import os\nimport numpy as np\n")
        config = cmd.BoundaryConfig(BASE_CONFIG)

        assert cmd.check(config) == []

    def test_relative_imports_are_ignored(self, fake_repo, tmp_path):
        _write_package(tmp_path, "detection", "engine.py", "from . import helpers\n")
        config = cmd.BoundaryConfig(BASE_CONFIG)

        assert cmd.check(config) == []

    def test_package_filter_scopes_the_check(self, fake_repo, tmp_path):
        _write_package(tmp_path, "utils", "helpers.py", "import api.app\n")
        _write_package(tmp_path, "api", "app.py", "x = 1\n")
        config = cmd.BoundaryConfig(BASE_CONFIG)

        assert cmd.check(config, only_package="api") == []
        assert len(cmd.check(config, only_package="utils")) == 1


class TestForbiddenPairs:
    def test_explicit_forbidden_pair_within_same_layer(self, fake_repo, tmp_path):
        config_dict = {
            "layers": [
                {"name": "domain", "packages": ["streaming", "api_client_domain"]},
            ],
            "forbidden_imports": [{"importer": "streaming", "forbidden": "api_client_domain"}],
            "excluded_packages": [],
        }
        _write_package(tmp_path, "streaming", "worker.py", "import api_client_domain.client\n")
        (tmp_path / "api_client_domain").mkdir()
        (tmp_path / "api_client_domain" / "__init__.py").touch()
        config = cmd.BoundaryConfig(config_dict)

        violations = cmd.check(config)

        assert len(violations) == 1
        assert "forbidden import" in violations[0].message


class TestConfigLoading:
    def test_load_reads_real_repo_config(self):
        config = cmd.BoundaryConfig.load(cmd.DEFAULT_CONFIG)
        assert "utils" in config.known_packages()
        assert "api" in config.known_packages()
        assert config.layer_of["utils"] < config.layer_of["api"]

    def test_check_against_real_repo_config_is_actionable(self):
        """Smoke test: running the real checker must not crash, and any
        violations it finds must be reported with a file:line pointer."""
        config = cmd.BoundaryConfig.load(cmd.DEFAULT_CONFIG)
        violations = cmd.check(config)
        for v in violations:
            assert v.lineno >= 1
            assert str(v.file).endswith(".py")
