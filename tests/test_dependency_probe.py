"""
tests/test_dependency_probe.py — Tests for utils/dependency_probe.py (#542)
"""

from __future__ import annotations

import json

import pytest

from utils.dependency_probe import (
    DEPENDENCY_GROUPS,
    MissingDependencyError,
    PackageStatus,
    ProbeReport,
    ProbeResult,
    _cli_main,
    _probe_package,
    probe,
    probe_all,
    require,
)

# ---------------------------------------------------------------------------
# _probe_package
# ---------------------------------------------------------------------------


def test_probe_package_present():
    result = _probe_package("sys", "sys")
    assert result.available is True
    assert result.import_name == "sys"


def test_probe_package_missing():
    result = _probe_package("_ledgerlens_nonexistent_xyz", "nonexistent-pkg")
    assert result.available is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# probe()
# ---------------------------------------------------------------------------


def test_probe_known_group_returns_result():
    result = probe("ml_core")
    assert isinstance(result, ProbeResult)
    assert result.group == "ml_core"


def test_probe_unknown_group_raises():
    with pytest.raises(KeyError):
        probe("not_a_real_group_xyz")


def test_probe_result_missing_property():
    # Fabricate a result with one missing package
    status_ok = PackageStatus("sys", "sys", available=True)
    status_bad = PackageStatus("_fake", "fake-pkg", available=False, error="nope")
    result = ProbeResult(
        group="test_group",
        available=False,
        packages=[status_ok, status_bad],
        install_hint="pip install fake-pkg",
    )
    assert len(result.missing) == 1
    assert result.missing[0].import_name == "_fake"


def test_probe_result_str_available():
    result = ProbeResult(
        group="ml_core",
        available=True,
        packages=[PackageStatus("numpy", "numpy", available=True, version="1.26")],
        install_hint="",
    )
    s = str(result)
    assert "[✓]" in s
    assert "ml_core" in s


def test_probe_result_str_missing():
    result = ProbeResult(
        group="fake_group",
        available=False,
        packages=[PackageStatus("_fake", "fake-pkg", available=False)],
        install_hint="pip install fake-pkg",
    )
    s = str(result)
    assert "[✗]" in s
    assert "pip install" in s


# ---------------------------------------------------------------------------
# probe_all()
# ---------------------------------------------------------------------------


def test_probe_all_returns_all_groups():
    report = probe_all()
    assert isinstance(report, ProbeReport)
    assert len(report.results) == len(DEPENDENCY_GROUPS)


def test_probe_all_subset():
    report = probe_all(["ml_core", "redis"])
    assert len(report.results) == 2
    groups = {r.group for r in report.results}
    assert groups == {"ml_core", "redis"}


def test_probe_all_to_dict():
    report = probe_all(["ml_core"])
    d = report.to_dict()
    assert "all_available" in d
    assert "groups" in d
    assert isinstance(d["groups"], list)
    assert d["groups"][0]["group"] == "ml_core"


def test_probe_all_summary_contains_groups():
    report = probe_all(["ml_core"])
    summary = report.summary()
    assert "ml_core" in summary
    assert "Dependency Probe Report" in summary


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------


def test_require_stdlib_always_present():
    """sys is always present — require('ml_core') should work if sklearn installed."""
    # We use a group that is always safe to check without caring about result
    # Just verify it doesn't raise TypeError or similar
    try:
        require("ml_core")
    except MissingDependencyError:
        pass  # acceptable if sklearn is not installed in test env


def test_require_raises_missing_dependency_error():
    """Inject a fake group to test the error path."""
    import utils.dependency_probe as mod

    # Temporarily register a fake group
    original = dict(mod.DEPENDENCY_GROUPS)
    mod.DEPENDENCY_GROUPS["_test_missing_xyz"] = [("_nope_xyz", "nope-pkg")]
    try:
        with pytest.raises(MissingDependencyError) as exc_info:
            require("_test_missing_xyz")
        err = exc_info.value
        assert "_test_missing_xyz" in err.group
        assert "pip install" in err.install_hint
        assert "nope-pkg" in err.install_hint
    finally:
        mod.DEPENDENCY_GROUPS.clear()
        mod.DEPENDENCY_GROUPS.update(original)


def test_missing_dependency_error_attributes():
    status = PackageStatus("_fake", "fake-pkg", available=False, error="not found")
    result = ProbeResult(
        group="g1",
        available=False,
        packages=[status],
        install_hint="pip install fake-pkg",
    )
    err = MissingDependencyError("g1", result)
    assert err.group == "g1"
    assert err.install_hint == "pip install fake-pkg"
    assert "g1" in str(err)
    assert "fake-pkg" in str(err)


# ---------------------------------------------------------------------------
# CLI (_cli_main)
# ---------------------------------------------------------------------------


def test_cli_all_groups_runs(capsys):
    ret = _cli_main([])
    # 0 = all available, 2 = some missing — both valid; 1 = fatal error (bad)
    assert ret in (0, 2)


def test_cli_specific_group(capsys):
    ret = _cli_main(["--groups", "ml_core"])
    assert ret in (0, 2)
    out = capsys.readouterr().out
    assert "ml_core" in out


def test_cli_json_output(capsys):
    _cli_main(["--groups", "ml_core", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "groups" in data
    assert data["groups"][0]["group"] == "ml_core"


def test_cli_unknown_group_returns_1(capsys):
    ret = _cli_main(["--groups", "definitely_not_a_group_xyz"])
    assert ret == 1


def test_cli_require_missing_returns_2():
    import utils.dependency_probe as mod

    original = dict(mod.DEPENDENCY_GROUPS)
    mod.DEPENDENCY_GROUPS["_ci_test_missing"] = [("_nope_999", "nope-pkg-999")]
    try:
        ret = _cli_main(["--groups", "_ci_test_missing", "--require", "_ci_test_missing"])
        assert ret == 2
    finally:
        mod.DEPENDENCY_GROUPS.clear()
        mod.DEPENDENCY_GROUPS.update(original)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_all_groups_have_at_least_one_package():
    for group, specs in DEPENDENCY_GROUPS.items():
        assert len(specs) >= 1, f"Group '{group}' has no packages"


def test_all_groups_have_string_keys():
    for group in DEPENDENCY_GROUPS:
        assert isinstance(group, str)
        assert group, "Group name must be non-empty"


def test_all_specs_are_tuples_of_strings():
    for group, specs in DEPENDENCY_GROUPS.items():
        for spec in specs:
            assert len(spec) == 2, f"Group '{group}' spec should be (import_name, pip_name)"
            import_name, pip_name = spec
            assert isinstance(import_name, str) and import_name
            assert isinstance(pip_name, str) and pip_name
