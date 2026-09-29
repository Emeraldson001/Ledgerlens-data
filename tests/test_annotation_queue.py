import json

import pytest

from detection.active_learning.queue_io import load_queue, save_queue

SECRET = "super-secret-key-123"


@pytest.fixture
def base_annotations():
    return [{"wallet": "GABC...", "label": "wash_trade"}, {"wallet": "GXYZ...", "label": "organic"}]


def test_load_queue_success(tmp_path, base_annotations):
    """AC: load_queue succeeds on file written by save_queue with matching secret"""
    file_path = tmp_path / "valid_queue.json"
    save_queue(file_path, base_annotations, SECRET)

    annotations = load_queue(file_path, SECRET)
    assert len(annotations) == 2
    assert annotations[0]["wallet"] == "GABC..."


def test_load_queue_tampered_body(tmp_path, base_annotations):
    """AC: load_queue raises ValueError on tampered file body"""
    file_path = tmp_path / "tampered_body.json"
    save_queue(file_path, base_annotations, SECRET)

    # Simulate an attacker changing a wallet label in transit
    raw_data = json.loads(file_path.read_text())
    raw_data["annotations"][0]["label"] = "organic"  # Poisoned target
    file_path.write_text(json.dumps(raw_data))

    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)


def test_load_queue_tampered_mac(tmp_path, base_annotations):
    """AC: load_queue raises ValueError on an invalid/arbitrary MAC string"""
    file_path = tmp_path / "tampered_mac.json"
    save_queue(file_path, base_annotations, SECRET)

    # Overwrite the signature block with garbage values
    raw_data = json.loads(file_path.read_text())
    raw_data["_hmac"] = "badmac12345"
    file_path.write_text(json.dumps(raw_data))

    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)


def test_load_queue_missing_mac(tmp_path, base_annotations):
    """AC: load_queue raises ValueError when the MAC attribute is missing"""
    file_path = tmp_path / "missing_mac.json"
    # Write a clean file object but completely strip the signature hook
    file_path.write_text(json.dumps({"annotations": base_annotations}))

    with pytest.raises(ValueError, match="Annotation queue HMAC mismatch"):
        load_queue(file_path, SECRET)


def test_load_queue_empty_secret_skips(tmp_path, base_annotations, caplog):
    """AC: Empty secret skips verification with WARNING"""
    file_path = tmp_path / "unverified.json"
    save_queue(file_path, base_annotations, secret=SECRET)

    import logging

    with caplog.at_level(logging.WARNING):
        # Passing an empty string secret overrides verification checks
        annotations = load_queue(file_path, secret="")

    assert len(annotations) == 2
    assert "Skipping signature verification" in caplog.text


def test_annotate_dry_run_preview(tmp_path, monkeypatch, capsys):
    """Test --dry-run flag previews export without writing file."""
    import sys
    from datetime import datetime

    from detection.active_learning.annotation_queue import AnnotationQueue
    from scripts.annotate import main

    # Setup test queue with annotated items
    queue_file = tmp_path / "queue.json"
    queue_data = [
        {
            "wallet": "GABC123",
            "label": 1,
            "annotator_id": "alice",
            "annotated_at": "2024-01-01T12:00:00",
            "status": "annotated",
        },
        {
            "wallet": "GXYZ456",
            "label": 0,
            "annotator_id": "alice",
            "annotated_at": "2024-01-02T12:00:00",
            "status": "annotated",
        },
    ]

    # Compute HMACs for valid annotations
    from detection.active_learning.annotation_queue import _compute_hmac
    from utils.secrets_config import get_secret
    from utils.secrets_manager import SecretType

    for item in queue_data:
        item["annotation_hmac"] = _compute_hmac(
            item["wallet"],
            item["label"],
            item["annotator_id"],
            item["annotated_at"],
        )

    queue_file.write_text(json.dumps(queue_data))

    # Test --dry-run (no file written)
    export_file = tmp_path / "export.parquet"
    monkeypatch.setattr(sys, "argv", ["annotate.py", "--export", str(export_file), "--dry-run", "--queue", str(queue_file)])
    main()

    captured = capsys.readouterr()
    assert "Export Preview" in captured.out
    assert "Total rows: 2" in captured.out
    assert not export_file.exists(), "File should not be created with --dry-run"

    # Test normal export (file written)
    monkeypatch.setattr(sys, "argv", ["annotate.py", "--export", str(export_file), "--queue", str(queue_file)])
    main()

    captured = capsys.readouterr()
    assert "Exported 2 annotated rows" in captured.out
    assert export_file.exists(), "File should be created without --dry-run"
