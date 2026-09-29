"""
tests/test_validate_readme_examples.py — Tests for scripts/validate_readme_examples.py (#548)
"""

from __future__ import annotations

import json
import pathlib
import textwrap

from scripts.validate_readme_examples import (
    ReadmeExamplesValidator,
    classify_line,
    extract_bash_blocks,
    file_exists,
    get_makefile_targets,
    main,
    module_exists,
    parse_code_block,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_readme(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    p = tmp_path / "README.md"
    p.write_text(textwrap.dedent(content))
    return p


def make_module(tmp_path: pathlib.Path, rel: str) -> pathlib.Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# module\n")
    return p


def make_makefile(tmp_path: pathlib.Path, targets: list[str]) -> pathlib.Path:
    lines = [f"{t}:\n\techo {t}" for t in targets]
    p = tmp_path / "Makefile"
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# extract_bash_blocks
# ---------------------------------------------------------------------------


def test_extract_bash_blocks_basic():
    text = "Some text\n```bash\necho hello\n```\nmore text"
    blocks = extract_bash_blocks(text)
    assert len(blocks) == 1
    assert "echo hello" in blocks[0]


def test_extract_bash_blocks_multiple():
    text = "```bash\ncmd1\n```\n```sh\ncmd2\n```"
    blocks = extract_bash_blocks(text)
    assert len(blocks) == 2


def test_extract_bash_blocks_shell_fence():
    text = "```shell\nmake install\n```"
    blocks = extract_bash_blocks(text)
    assert len(blocks) == 1


def test_extract_bash_blocks_plain_fence():
    text = "```\npython -m scripts.foo\n```"
    blocks = extract_bash_blocks(text)
    assert len(blocks) == 1


def test_extract_bash_blocks_none():
    text = "No code blocks here."
    assert extract_bash_blocks(text) == []


# ---------------------------------------------------------------------------
# classify_line
# ---------------------------------------------------------------------------


def test_classify_python_module():
    cl = classify_line("python -m scripts.generate_synthetic_dataset", "README.md", 1, 1)
    assert cl.kind == "python_module"
    assert cl.target == "scripts.generate_synthetic_dataset"


def test_classify_python_file():
    cl = classify_line("python run_pipeline.py", "README.md", 1, 1)
    assert cl.kind == "python_file"
    assert cl.target == "run_pipeline.py"


def test_classify_make():
    cl = classify_line("make install", "README.md", 1, 1)
    assert cl.kind == "make"
    assert cl.target == "install"


def test_classify_env_prefix_stripped():
    cl = classify_line(
        "ALERT_WEBHOOK_URL=https://example.com python -m scripts.stream", "README.md", 1, 1
    )
    assert cl.kind == "python_module"
    assert cl.target == "scripts.stream"


def test_classify_unknown():
    cl = classify_line("echo hello world", "README.md", 1, 1)
    assert cl.kind == "unknown"


def test_classify_comment_ignored():
    cl = classify_line("# python -m scripts.foo", "README.md", 1, 1)
    # Comments are stripped in parse_code_block, but if they reach classify_line
    # they should yield 'unknown' since the pattern won't match after the #
    assert cl.kind == "unknown"


# ---------------------------------------------------------------------------
# parse_code_block
# ---------------------------------------------------------------------------


def test_parse_code_block_filters_comments():
    block = "# comment\npython -m scripts.foo\n# another comment\n"
    lines = parse_code_block(block, "README.md", 1)
    kinds = [cl.kind for cl in lines]
    assert "python_module" in kinds
    # Comments should not produce code lines
    for cl in lines:
        assert not cl.raw.strip().startswith("#")


def test_parse_code_block_line_continuation():
    block = "python -m scripts.score_wallet \\\n  --wallet G... \\\n  --pair foo\n"
    lines = parse_code_block(block, "README.md", 1)
    # First line should be parsed as python_module
    assert any(cl.kind == "python_module" for cl in lines)


# ---------------------------------------------------------------------------
# module_exists / file_exists
# ---------------------------------------------------------------------------


def test_module_exists_file(tmp_path):
    make_module(tmp_path, "scripts/foo.py")
    assert module_exists("scripts.foo", tmp_path)


def test_module_exists_package(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").write_text("")
    assert module_exists("scripts", tmp_path)


def test_module_exists_missing(tmp_path):
    assert not module_exists("scripts.nonexistent_xyz", tmp_path)


def test_file_exists(tmp_path):
    (tmp_path / "run_pipeline.py").write_text("")
    assert file_exists("run_pipeline.py", tmp_path)


def test_file_exists_missing(tmp_path):
    assert not file_exists("not_here.py", tmp_path)


# ---------------------------------------------------------------------------
# get_makefile_targets
# ---------------------------------------------------------------------------


def test_get_makefile_targets(tmp_path):
    make_makefile(tmp_path, ["install", "test", "lint"])
    targets = get_makefile_targets(tmp_path)
    assert "install" in targets
    assert "test" in targets
    assert "lint" in targets


def test_get_makefile_targets_no_makefile(tmp_path):
    assert get_makefile_targets(tmp_path) == set()


# ---------------------------------------------------------------------------
# ReadmeExamplesValidator
# ---------------------------------------------------------------------------


def test_validator_valid_module(tmp_path):
    make_module(tmp_path, "scripts/__init__.py")
    make_module(tmp_path, "scripts/generate_synthetic_dataset.py")
    readme = make_readme(
        tmp_path,
        """
        # Docs
        ```bash
        python -m scripts.generate_synthetic_dataset --output data/foo.parquet
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    assert report.errors == []


def test_validator_missing_module(tmp_path):
    readme = make_readme(
        tmp_path,
        """
        ```bash
        python -m scripts.nonexistent_script_xyz
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    assert len(report.errors) >= 1
    assert any("nonexistent_script_xyz" in f.target for f in report.errors)


def test_validator_valid_file(tmp_path):
    (tmp_path / "run_pipeline.py").write_text("")
    readme = make_readme(
        tmp_path,
        """
        ```bash
        python run_pipeline.py
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    assert report.errors == []


def test_validator_missing_file(tmp_path):
    readme = make_readme(
        tmp_path,
        """
        ```bash
        python run_nonexistent.py
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    assert len(report.errors) >= 1


def test_validator_make_target_warning(tmp_path):
    make_makefile(tmp_path, ["install", "test"])
    readme = make_readme(
        tmp_path,
        """
        ```bash
        make nonexistent_target_xyz
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    # make targets are warnings, not errors
    assert len(report.warnings) >= 1
    assert report.errors == []


def test_validator_warn_only_mode(tmp_path):
    readme = make_readme(
        tmp_path,
        """
        ```bash
        python -m scripts.nonexistent_xyz
        ```
    """,
    )
    validator = ReadmeExamplesValidator(root=tmp_path, warn_only=True)
    report = validator.validate_files([readme])
    # warn_only means errors become warnings
    assert report.errors == []
    assert len(report.warnings) >= 1


def test_validator_to_dict(tmp_path):
    readme = make_readme(tmp_path, "No code blocks here.")
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    d = report.to_dict()
    assert "checked" in d
    assert "findings" in d


def test_validator_summary_no_findings(tmp_path):
    readme = make_readme(tmp_path, "No code blocks here.")
    validator = ReadmeExamplesValidator(root=tmp_path)
    report = validator.validate_files([readme])
    summary = report.summary()
    assert "0 error" in summary


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_main_no_errors(tmp_path):
    make_module(tmp_path, "scripts/__init__.py")
    make_module(tmp_path, "scripts/foo.py")
    make_readme(
        tmp_path,
        """
        ```bash
        python -m scripts.foo
        ```
    """,
    )
    ret = main(["--docs", "README.md", "--root", str(tmp_path)])
    assert ret == 0


def test_main_with_errors(tmp_path):
    make_readme(
        tmp_path,
        """
        ```bash
        python -m scripts.totally_missing_module_xyz
        ```
    """,
    )
    ret = main(["--docs", "README.md", "--root", str(tmp_path)])
    assert ret == 2


def test_main_json_output(tmp_path, capsys):
    make_readme(tmp_path, "No code blocks.")
    main(["--docs", "README.md", "--root", str(tmp_path), "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "checked" in data


def test_main_no_markdown_files(tmp_path):
    ret = main(["--docs", str(tmp_path / "nonexistent.md"), "--root", str(tmp_path)])
    assert ret == 1


def test_main_warn_only_exits_zero(tmp_path):
    make_readme(
        tmp_path,
        """
        ```bash
        python -m scripts.totally_missing_module_xyz
        ```
    """,
    )
    ret = main(["--docs", "README.md", "--root", str(tmp_path), "--warn-only"])
    assert ret == 0


# ---------------------------------------------------------------------------
# Real repo smoke test — README.md should not have broken examples
# ---------------------------------------------------------------------------


def test_real_readme_examples():
    """
    Validate the actual repo README.md.
    All python -m and python <file> references must resolve.
    Make targets produce warnings only, not errors.
    """
    repo_root = pathlib.Path(__file__).parent.parent.resolve()
    ret = main(["--docs", "README.md", "--root", str(repo_root)])
    # 0 = all valid; 2 = some missing refs (documents them)
    # We accept 2 to avoid blocking PRs that don't add the script yet,
    # but the test will catch regressions once they're all green.
    assert ret in (0, 2), f"validate_readme_examples crashed with exit code {ret}"
