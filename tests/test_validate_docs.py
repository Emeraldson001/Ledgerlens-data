"""Tests for scripts/validate_docs.py (Issue #512)."""

from pathlib import Path

from scripts.validate_docs import (
    discover_advanced_contributor_docs,
    validate_doc,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_missing_doc_is_a_violation(tmp_path):
    missing = tmp_path / "docs" / "does_not_exist.md"
    violations = validate_doc(missing, tmp_path)
    assert len(violations) == 1
    assert "does not exist" in violations[0].message


def test_empty_doc_is_a_violation(tmp_path):
    doc = _write(tmp_path / "empty.md", "   \n")
    violations = validate_doc(doc, tmp_path)
    assert any("empty" in v.message for v in violations)


def test_doc_without_heading_is_a_violation(tmp_path):
    doc = _write(tmp_path / "no_heading.md", "Just a paragraph with no heading.\n")
    violations = validate_doc(doc, tmp_path)
    assert any("no Markdown heading" in v.message for v in violations)


def test_valid_doc_passes(tmp_path):
    doc = _write(
        tmp_path / "valid.md",
        "# Title\n\nSome content.\n\n```python\nx = 1 + 1\n```\n",
    )
    assert validate_doc(doc, tmp_path) == []


def test_broken_local_link_is_a_violation(tmp_path):
    doc = _write(
        tmp_path / "broken_link.md",
        "# Title\n\nSee [other doc](missing.md) for details.\n",
    )
    violations = validate_doc(doc, tmp_path)
    assert any("broken local link" in v.message for v in violations)


def test_valid_local_link_passes(tmp_path):
    _write(tmp_path / "target.md", "# Target\n")
    doc = _write(
        tmp_path / "linking.md",
        "# Title\n\nSee [target](target.md) for details.\n",
    )
    assert validate_doc(doc, tmp_path) == []


def test_external_link_is_not_checked(tmp_path):
    doc = _write(
        tmp_path / "external.md",
        "# Title\n\nSee [external](https://example.com/does-not-matter) for details.\n",
    )
    assert validate_doc(doc, tmp_path) == []


def test_invalid_python_code_block_is_a_violation(tmp_path):
    doc = _write(
        tmp_path / "bad_code.md",
        "# Title\n\n```python\ndef broken(:\n    pass\n```\n",
    )
    violations = validate_doc(doc, tmp_path)
    assert any("invalid Python" in v.message for v in violations)


def test_template_placeholder_code_block_is_tolerated(tmp_path):
    doc = _write(
        tmp_path / "template.md",
        "# Title\n\n```python\n"
        "def compute_<group>_features(wallet: str, <data>: <Type>):\n    ...\n"
        "```\n",
    )
    assert validate_doc(doc, tmp_path) == []


def test_dict_entry_fragment_code_block_is_tolerated(tmp_path):
    doc = _write(
        tmp_path / "fragment.md",
        '# Title\n\n```python\n"my_key": (\n    "description"\n),\n```\n',
    )
    assert validate_doc(doc, tmp_path) == []


def test_discover_finds_direct_local_md_links(tmp_path):
    _write(tmp_path / "docs" / "guide.md", "# Guide\n")
    _write(
        tmp_path / "CONTRIBUTING.md",
        "# Contributing\n\nSee [guide](docs/guide.md) and "
        "[external](https://example.com) and [anchor-only](#section).\n",
    )
    docs = discover_advanced_contributor_docs(tmp_path)
    relative = {d.relative_to(tmp_path) for d in docs}
    assert Path("CONTRIBUTING.md") in relative
    assert Path("docs/guide.md") in relative
    assert len(docs) == 2


def test_discover_handles_missing_entry_point(tmp_path):
    docs = discover_advanced_contributor_docs(tmp_path, entry_point="NOPE.md")
    assert docs == [tmp_path / "NOPE.md"]


def test_real_repo_advanced_contributor_docs_pass_validation():
    """Regression guard: the real repo's advanced contributor path docs must
    always pass validation (Issue #512 acceptance criterion)."""
    repo_root = Path(__file__).resolve().parent.parent
    docs = discover_advanced_contributor_docs(repo_root)
    assert len(docs) >= 3

    all_violations = []
    for doc in docs:
        all_violations.extend(validate_doc(doc, repo_root))

    assert all_violations == [], [f"{v.file}:{v.line}: {v.message}" for v in all_violations]
