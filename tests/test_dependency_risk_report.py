from __future__ import annotations

import json
from pathlib import Path

from scripts.dependency_risk_report import build_report, main


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "numpy>=1.0\nrequests>=2.0\nnot-core==1.0\n", encoding="utf-8"
    )
    lockfile = tmp_path / "requirements.lock"
    lockfile.write_text("numpy==1.2\nrequests==2.1\nnot-core==1.0\n", encoding="utf-8")
    return requirements, lockfile


def test_build_report_selects_core_packages_and_lock_versions(tmp_path):
    requirements, lockfile = _inputs(tmp_path)
    report = build_report(requirements, lockfile)

    assert [package.name for package in report.packages] == ["numpy", "requests"]
    assert report.packages[0].locked_version == "1.2"
    assert report.packages[0].risk == "medium"
    assert report.high_risk_count == 0


def test_missing_core_lock_entry_is_high_risk(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=1.0\n", encoding="utf-8")
    lockfile = tmp_path / "requirements.lock"
    lockfile.write_text("", encoding="utf-8")

    report = build_report(requirements, lockfile)

    assert report.packages[0].locked_version is None
    assert report.packages[0].risk == "high"
    assert report.high_risk_count == 1


def test_osv_advisories_raise_risk_without_network(monkeypatch, tmp_path):
    requirements, lockfile = _inputs(tmp_path)

    def fake_query(name: str, version: str):
        from scripts.dependency_risk_report import Advisory

        return [Advisory("PYSEC-1", "test advisory")]

    monkeypatch.setattr("scripts.dependency_risk_report._query_osv", fake_query)
    report = build_report(requirements, lockfile, osv=True)

    numpy = next(package for package in report.packages if package.name == "numpy")
    assert numpy.risk == "high"
    assert numpy.advisories[0].identifier == "PYSEC-1"


def test_main_writes_json_report(tmp_path):
    requirements, lockfile = _inputs(tmp_path)
    output = tmp_path / "report.json"

    assert main(
        [
            "--requirements",
            str(requirements),
            "--lockfile",
            str(lockfile),
            "--output",
            str(output),
        ]
    ) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["packages"][0]["name"] == "numpy"
    assert data["osv_requested"] is False


def test_all_includes_non_core_packages(tmp_path):
    requirements, lockfile = _inputs(tmp_path)
    report = build_report(requirements, lockfile, core_only=False)

    assert {package.name for package in report.packages} == {"numpy", "requests", "not-core"}
