"""Unit and integration tests for dataset and model input lineage tracking (Issue #463)."""

from pathlib import Path

import pandas as pd
import pytest

from data.lineage import (
    DataArtifactMetadata,
    LineageNodeType,
    LineageTracker,
    compute_file_sha256,
)


@pytest.fixture
def tmp_dataset_files(tmp_path: Path):
    """Fixture creating temporary dataset artifacts for lineage testing."""
    raw_path = tmp_path / "raw_trades.csv"
    raw_df = pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "amount": [100.0, 250.0, 50.0],
            "account": ["G111", "G222", "G333"],
        }
    )
    raw_df.to_csv(raw_path, index=False)

    proc_path = tmp_path / "processed_features.parquet"
    proc_df = pd.DataFrame(
        {
            "account": ["G111", "G222", "G333"],
            "benford_score": [0.02, 0.85, 0.12],
            "is_anomaly": [0, 1, 0],
        }
    )
    proc_df.to_parquet(proc_path, index=False)

    return raw_path, proc_path


def test_artifact_metadata_from_file(tmp_dataset_files):
    raw_path, proc_path = tmp_dataset_files

    meta_csv = DataArtifactMetadata.from_file(raw_path)
    assert meta_csv.format == "csv"
    assert meta_csv.row_count == 3
    assert meta_csv.column_count == 3
    assert meta_csv.sha256 == compute_file_sha256(raw_path)
    assert meta_csv.schema_hash is not None

    meta_pq = DataArtifactMetadata.from_file(proc_path)
    assert meta_pq.format == "parquet"
    assert meta_pq.row_count == 3
    assert meta_pq.columns == ["account", "benford_score", "is_anomaly"]


def test_lineage_tracker_dag_construction(tmp_dataset_files):
    raw_path, proc_path = tmp_dataset_files
    tracker = LineageTracker(graph_name="test_pipeline_graph")

    # 1. Register Source Dataset
    src_node = tracker.register_dataset(
        name="raw_trades_ingest",
        filepath=raw_path,
        node_type=LineageNodeType.SOURCE_DATASET,
        metadata={"source": "stellar_horizon"},
    )
    assert src_node.node_type == LineageNodeType.SOURCE_DATASET
    assert src_node.artifact is not None
    assert src_node.artifact.row_count == 3

    # 2. Add transformation step to source node
    tracker.add_transformation_step(
        node_id=src_node.node_id,
        step_name="clean_nulls",
        transform_type="filtering",
        parameters={"drop_nulls": True},
    )
    assert len(src_node.steps) == 1
    assert src_node.steps[0].step_name == "clean_nulls"

    # 3. Register Derived Feature Dataset
    proc_node = tracker.register_dataset(
        name="processed_features",
        filepath=proc_path,
        node_type=LineageNodeType.DERIVED_DATASET,
        parent_ids=[src_node.node_id],
        metadata={"window_hours": 24},
    )
    assert proc_node.parents == [src_node.node_id]

    # 4. Register Model Input
    model_input_node = tracker.register_model_input(
        model_name="wash_trade_detector_v1",
        dataset_node_id=proc_node.node_id,
        feature_columns=["benford_score"],
        hyperparams={"n_estimators": 100, "lr": 0.01},
    )
    assert model_input_node.node_type == LineageNodeType.MODEL_INPUT
    assert model_input_node.parents == [proc_node.node_id]
    assert model_input_node.metadata["model_name"] == "wash_trade_detector_v1"

    # Ancestor resolution
    ancestors = tracker.get_ancestors(model_input_node.node_id)
    ancestor_ids = [a.node_id for a in ancestors]
    assert src_node.node_id in ancestor_ids
    assert proc_node.node_id in ancestor_ids

    # Descendant resolution
    descendants = tracker.get_descendants(src_node.node_id)
    descendant_ids = [d.node_id for d in descendants]
    assert proc_node.node_id in descendant_ids
    assert model_input_node.node_id in descendant_ids


def test_lineage_integrity_validation(tmp_dataset_files, tmp_path):
    raw_path, proc_path = tmp_dataset_files
    tracker = LineageTracker()

    n1 = tracker.register_dataset("raw_data", filepath=raw_path)
    n2 = tracker.register_dataset("proc_data", filepath=proc_path, parent_ids=[n1.node_id])

    # Healthy state check
    diag = tracker.validate_integrity()
    assert diag["is_healthy"] is True
    assert diag["valid_artifacts_count"] == 2
    assert len(diag["missing_files"]) == 0
    assert len(diag["checksum_mismatches"]) == 0

    # Tamper with file
    raw_path.write_text("corrupted content", encoding="utf-8")
    diag_tampered = tracker.validate_integrity(n1.node_id)
    assert diag_tampered["is_healthy"] is False
    assert len(diag_tampered["checksum_mismatches"]) == 1
    assert diag_tampered["checksum_mismatches"][0]["node_id"] == n1.node_id

    # Delete file
    proc_path.unlink()
    diag_deleted = tracker.validate_integrity(n2.node_id)
    assert diag_deleted["is_healthy"] is False
    assert len(diag_deleted["missing_files"]) == 1
    assert diag_deleted["missing_files"][0]["node_id"] == n2.node_id


def test_lineage_json_export_import_sidecar(tmp_dataset_files, tmp_path):
    raw_path, proc_path = tmp_dataset_files
    tracker = LineageTracker(graph_name="export_test")

    src = tracker.register_dataset("raw", filepath=raw_path)
    tracker.register_dataset("proc", filepath=proc_path, parent_ids=[src.node_id])

    # Export to JSON
    json_path = tmp_path / "lineage.json"
    tracker.export_json(json_path)
    assert json_path.exists()

    # Import back
    reloaded = LineageTracker.from_json(json_path)
    assert reloaded.graph_name == "export_test"
    assert len(reloaded.nodes) == 2
    assert src.node_id in reloaded.nodes
    assert reloaded.nodes[src.node_id].name == "raw"

    # Save sidecar
    sidecar_path = tracker.save_sidecar(proc_path)
    assert sidecar_path.name == "processed_features.parquet.lineage.json"
    assert sidecar_path.exists()
    sidecar_tracker = LineageTracker.from_json(sidecar_path)
    assert len(sidecar_tracker.nodes) == 2


def test_lineage_error_handling(tmp_path):
    tracker = LineageTracker()

    with pytest.raises(ValueError, match="Parent node 'missing_id' not found"):
        tracker.register_dataset("orphan", parent_ids=["missing_id"])

    with pytest.raises(KeyError, match="Node ID 'invalid' not found"):
        tracker.get_node("invalid")

    with pytest.raises(KeyError, match="Node ID 'invalid' does not exist"):
        tracker.add_transformation_step("invalid", "step", "transform")

    non_existent_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        DataArtifactMetadata.from_file(non_existent_file)
