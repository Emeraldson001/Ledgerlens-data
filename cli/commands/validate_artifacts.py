import json
import os
from typing import Any


def validate_artifacts(artifacts_dir: str = "artifacts") -> dict[str, Any]:
    if not os.path.exists(artifacts_dir):
        return {"status": "FAIL", "error": f"Artifacts directory '{artifacts_dir}' does not exist."}

    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    if not os.path.exists(metadata_path):
        return {"status": "FAIL", "error": f"Required artifact file missing: {metadata_path}"}

    try:
        with open(metadata_path) as f:
            data = json.load(f)

        required_keys = ["model_version", "feature_schema_hash"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            return {"status": "FAIL", "error": f"Metadata missing required fields: {missing_keys}"}
    except Exception as e:
        return {"status": "FAIL", "error": f"Failed to parse metadata JSON: {str(e)}"}

    return {
        "status": "PASS",
        "artifacts_dir": artifacts_dir,
        "version": data.get("model_version"),
        "schema_hash": data.get("feature_schema_hash"),
    }
