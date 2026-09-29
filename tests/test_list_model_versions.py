import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def mock_archive_dir(tmp_path):
    """Create a temporary archive directory with fixture model versions."""
    archive = tmp_path / "models" / "archive"
    archive.mkdir(parents=True)

    # Create version 1
    v1_dir = archive / "v1.0.0"
    v1_dir.mkdir()
    v1_dir.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "random_forest": {"auc_roc": 0.8923, "f1": 0.8756},
                "xgboost": {"auc_roc": 0.8956, "f1": 0.8834},
            }
        )
    )
    v1_dir.joinpath("model_metadata.json").write_text(
        json.dumps({"trained_at": "2024-01-15T10:30:00", "n_training_rows": 1000, "n_test_rows": 250})
    )

    # Create version 2
    v2_dir = archive / "v1.0.1"
    v2_dir.mkdir()
    v2_dir.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "random_forest": {"auc_roc": 0.8945, "f1": 0.8812},
                "xgboost": {"auc_roc": 0.8978, "f1": 0.8901},
            }
        )
    )
    v2_dir.joinpath("model_metadata.json").write_text(
        json.dumps({"trained_at": "2024-01-20T14:45:00", "n_training_rows": 1200, "n_test_rows": 300})
    )

    return str(archive)


def test_format_table_renders_table(mock_archive_dir, capsys):
    """Test --format table renders aligned text table."""
    from scripts.list_model_versions import main

    main(["--archive-dir", mock_archive_dir, "--format", "table"])

    captured = capsys.readouterr()
    output = captured.out

    # Should contain header
    assert "Version" in output
    assert "Trained At" in output
    assert "AUC-ROC" in output
    assert "F1" in output

    # Should contain version data
    assert "v1.0.0" in output
    assert "v1.0.1" in output
    assert "2024-01-15" in output or "2024-01-20" in output

    # Should have separator line
    assert "---" in output


def test_format_json_renders_json(mock_archive_dir, capsys):
    """Test --format json renders JSON output."""
    from scripts.list_model_versions import main

    main(["--archive-dir", mock_archive_dir, "--format", "json"])

    captured = capsys.readouterr()
    output = captured.out

    # Parse as JSON
    data = json.loads(output)

    # Should be a list
    assert isinstance(data, list)
    assert len(data) == 2

    # Should have expected structure
    versions = {v["version"] for v in data}
    assert "v1.0.0" in versions
    assert "v1.0.1" in versions

    # Should have metrics
    for v in data:
        assert "metrics" in v
        assert "trained_at" in v


def test_format_table_is_default(mock_archive_dir, capsys):
    """Test that table format is the default when --format is not specified."""
    from scripts.list_model_versions import main

    main(["--archive-dir", mock_archive_dir])

    captured = capsys.readouterr()
    output = captured.out

    # Should contain table elements
    assert "Version" in output
    assert "---" in output


def test_format_with_max_rows(mock_archive_dir, capsys):
    """Test --format table respects --max-rows limit."""
    from scripts.list_model_versions import main

    main(["--archive-dir", mock_archive_dir, "--format", "table", "--max-rows", "1"])

    captured = capsys.readouterr()
    output = captured.out

    # Should contain only one version in the output
    lines = output.strip().split("\n")
    version_lines = [l for l in lines if l and not l.startswith("-") and "Version" not in l and "Trained" not in l]

    # Should have at most 1 version line (plus header and separator)
    assert len(version_lines) == 1


def test_format_json_with_max_rows(mock_archive_dir, capsys):
    """Test --format json respects --max-rows limit."""
    from scripts.list_model_versions import main

    main(["--archive-dir", mock_archive_dir, "--format", "json", "--max-rows", "1"])

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    # Should have exactly 1 version
    assert len(data) == 1


def test_empty_archive_shows_message(tmp_path, capsys):
    """Test that empty archive directory shows appropriate message."""
    from scripts.list_model_versions import main

    empty_archive = tmp_path / "models" / "archive"
    empty_archive.mkdir(parents=True)

    main(["--archive-dir", str(empty_archive), "--format", "table"])

    captured = capsys.readouterr()
    assert "No archived model versions found" in captured.out
