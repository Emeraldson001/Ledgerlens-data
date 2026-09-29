"""Dataset and Model Input Lineage Tracking (Issue #463).

Provides a durable, reusable, and cryptographically verifiable lineage graph
for datasets, feature store extractions, and model inputs/outputs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class LineageNodeType(StrEnum):
    SOURCE_DATASET = "SOURCE_DATASET"
    DERIVED_DATASET = "DERIVED_DATASET"
    FEATURE_STORE = "FEATURE_STORE"
    MODEL_INPUT = "MODEL_INPUT"
    MODEL_OUTPUT = "MODEL_OUTPUT"


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 digest of a file in chunks."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found for SHA256 calculation: {filepath}")
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class DataArtifactMetadata:
    """Metadata and checksum information for a data file artifact."""

    path: str
    sha256: str
    file_size_bytes: int
    row_count: int | None = None
    column_count: int | None = None
    format: str = "unknown"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat()
    )
    schema_hash: str | None = None
    columns: list[str] | None = None

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        row_count: int | None = None,
        column_count: int | None = None,
        columns: list[str] | None = None,
    ) -> DataArtifactMetadata:
        """Construct DataArtifactMetadata by inspecting the file on disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {filepath}")

        file_size = path.stat().st_size
        sha256_val = compute_file_sha256(path)
        ext = path.suffix.lower().lstrip(".") or "binary"

        # Auto-detect dimensions for CSV / Parquet if pandas is available and counts not provided
        if (row_count is None or columns is None) and ext in ("parquet", "csv"):
            try:
                import pandas as pd

                if ext == "parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                if row_count is None:
                    row_count = len(df)
                if columns is None:
                    columns = list(df.columns)
                if column_count is None:
                    column_count = len(columns)
            except Exception:
                pass

        if column_count is None and columns is not None:
            column_count = len(columns)

        schema_hash = None
        if columns:
            col_str = ",".join(sorted(columns))
            schema_hash = hashlib.sha256(col_str.encode("utf-8")).hexdigest()[:16]

        return cls(
            path=str(path.resolve()),
            sha256=sha256_val,
            file_size_bytes=file_size,
            row_count=row_count,
            column_count=column_count,
            format=ext,
            schema_hash=schema_hash,
            columns=columns,
        )


@dataclass
class TransformationStep:
    """A record of a single data transformation or processing operation."""

    step_name: str
    transform_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float | None = None
    commit_hash: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat()
    )


@dataclass
class LineageNode:
    """A single node in the dataset/model input lineage DAG."""

    node_id: str
    name: str
    node_type: LineageNodeType
    artifact: DataArtifactMetadata | None = None
    parents: list[str] = field(default_factory=list)
    steps: list[TransformationStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert LineageNode to a JSON-serializable dictionary."""
        data = asdict(self)
        data["node_type"] = (
            self.node_type.value if isinstance(self.node_type, LineageNodeType) else self.node_type
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageNode:
        """Construct LineageNode from dictionary representation."""
        node_data = dict(data)
        node_data["node_type"] = LineageNodeType(node_data["node_type"])
        if node_data.get("artifact"):
            node_data["artifact"] = DataArtifactMetadata(**node_data["artifact"])
        if node_data.get("steps"):
            node_data["steps"] = [TransformationStep(**s) for s in node_data["steps"]]
        return cls(**node_data)


class LineageTracker:
    """DAG Lineage tracker for datasets, features, and model training inputs."""

    def __init__(self, graph_name: str = "ledgerlens_lineage") -> None:
        self.graph_name = graph_name
        self.nodes: dict[str, LineageNode] = {}
        self.created_at: str = datetime.datetime.now(tz=datetime.UTC).isoformat()

    def register_dataset(
        self,
        name: str,
        filepath: str | Path | None = None,
        node_type: LineageNodeType | str = LineageNodeType.DERIVED_DATASET,
        parent_ids: list[str] | None = None,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        row_count: int | None = None,
        columns: list[str] | None = None,
    ) -> LineageNode:
        """Register a dataset in the lineage DAG.

        If ``filepath`` is provided and exists, SHA256 and file metadata are computed.
        """
        if parent_ids is None:
            parent_ids = []
        for pid in parent_ids:
            if pid not in self.nodes:
                raise ValueError(f"Parent node '{pid}' not found in lineage graph.")

        if isinstance(node_type, str):
            node_type = LineageNodeType(node_type)

        artifact = None
        if filepath is not None:
            path = Path(filepath)
            if path.exists():
                artifact = DataArtifactMetadata.from_file(
                    path, row_count=row_count, columns=columns
                )

        if node_id is None:
            raw_key = f"{name}:{filepath}:{self.created_at}:{len(self.nodes)}"
            node_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]

        node = LineageNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            artifact=artifact,
            parents=parent_ids,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        return node

    def register_model_input(
        self,
        model_name: str,
        dataset_node_id: str,
        feature_columns: list[str] | None = None,
        hyperparams: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Register a model training input tied to a dataset node."""
        if dataset_node_id not in self.nodes:
            raise ValueError(f"Dataset node '{dataset_node_id}' not found in lineage graph.")

        node_id = f"model_input_{model_name}_{hashlib.sha256(model_name.encode()).hexdigest()[:8]}"
        meta = {
            "model_name": model_name,
            "feature_columns": feature_columns or [],
            "hyperparams": hyperparams or {},
        }
        if metadata:
            meta.update(metadata)

        node = LineageNode(
            node_id=node_id,
            name=f"ModelInput:{model_name}",
            node_type=LineageNodeType.MODEL_INPUT,
            parents=[dataset_node_id],
            metadata=meta,
        )
        self.nodes[node_id] = node
        return node

    def add_transformation_step(
        self,
        node_id: str,
        step_name: str,
        transform_type: str,
        parameters: dict[str, Any] | None = None,
        execution_time_seconds: float | None = None,
        commit_hash: str | None = None,
    ) -> TransformationStep:
        """Record a transformation step on an existing node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node ID '{node_id}' does not exist.")
        step = TransformationStep(
            step_name=step_name,
            transform_type=transform_type,
            parameters=parameters or {},
            execution_time_seconds=execution_time_seconds,
            commit_hash=commit_hash,
        )
        self.nodes[node_id].steps.append(step)
        return step

    def get_node(self, node_id: str) -> LineageNode:
        """Retrieve node by ID."""
        if node_id not in self.nodes:
            raise KeyError(f"Node ID '{node_id}' not found.")
        return self.nodes[node_id]

    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Get all ancestor nodes in topological order."""
        if node_id not in self.nodes:
            raise KeyError(f"Node ID '{node_id}' not found.")

        visited: set[str] = set()
        ancestors: list[LineageNode] = []

        def _dfs(nid: str) -> None:
            curr = self.nodes[nid]
            for pid in curr.parents:
                if pid not in visited and pid in self.nodes:
                    visited.add(pid)
                    _dfs(pid)
                    ancestors.append(self.nodes[pid])

        _dfs(node_id)
        return ancestors

    def get_descendants(self, node_id: str) -> list[LineageNode]:
        """Get all descendant nodes."""
        if node_id not in self.nodes:
            raise KeyError(f"Node ID '{node_id}' not found.")

        descendants: list[LineageNode] = []
        for nid, node in self.nodes.items():
            if node_id in node.parents:
                descendants.append(node)
                descendants.extend(self.get_descendants(nid))
        return descendants

    def validate_integrity(self, node_id: str | None = None) -> dict[str, Any]:
        """Validate artifact existence and SHA-256 checksums across graph or for specific node.

        Returns diagnostics detailing valid artifacts, tampered artifacts, and missing files.
        """
        targets = [self.nodes[node_id]] if node_id else list(self.nodes.values())

        valid_nodes: list[str] = []
        missing_files: list[dict[str, str]] = []
        checksum_mismatches: list[dict[str, str]] = []

        for node in targets:
            if node.artifact is None:
                continue
            art_path = Path(node.artifact.path)
            if not art_path.exists():
                missing_files.append(
                    {
                        "node_id": node.node_id,
                        "name": node.name,
                        "expected_path": node.artifact.path,
                    }
                )
            else:
                curr_hash = compute_file_sha256(art_path)
                if curr_hash != node.artifact.sha256:
                    checksum_mismatches.append(
                        {
                            "node_id": node.node_id,
                            "name": node.name,
                            "expected_sha256": node.artifact.sha256,
                            "actual_sha256": curr_hash,
                            "path": node.artifact.path,
                        }
                    )
                else:
                    valid_nodes.append(node.node_id)

        is_healthy = len(missing_files) == 0 and len(checksum_mismatches) == 0
        return {
            "is_healthy": is_healthy,
            "total_nodes_checked": len(targets),
            "valid_artifacts_count": len(valid_nodes),
            "missing_files": missing_files,
            "checksum_mismatches": checksum_mismatches,
            "validated_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete graph to dict."""
        return {
            "graph_name": self.graph_name,
            "created_at": self.created_at,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageTracker:
        """Deserialize LineageTracker from dict."""
        tracker = cls(graph_name=data.get("graph_name", "ledgerlens_lineage"))
        tracker.created_at = data.get("created_at", tracker.created_at)
        nodes_dict = data.get("nodes", {})
        for nid, n_data in nodes_dict.items():
            tracker.nodes[nid] = LineageNode.from_dict(n_data)
        return tracker

    def export_json(self, filepath: str | Path | None = None) -> str:
        """Export lineage graph to JSON string or file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath is not None:
            Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, filepath_or_str: str | Path) -> LineageTracker:
        """Load LineageTracker from JSON string or file."""
        path = Path(str(filepath_or_str))
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(filepath_or_str)
        data = json.loads(content)
        return cls.from_dict(data)

    def save_sidecar(self, artifact_path: str | Path) -> Path:
        """Write `.lineage.json` sidecar alongside specified artifact file."""
        art_path = Path(artifact_path)
        sidecar_path = art_path.with_name(f"{art_path.name}.lineage.json")
        self.export_json(sidecar_path)
        return sidecar_path
