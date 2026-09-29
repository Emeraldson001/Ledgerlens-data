"""Lightweight contracts shared by model training and inference."""

from __future__ import annotations

import hashlib

FEATURE_COLUMNS_EXCLUDE = {"wallet", "label", "profile"}


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Compute a stable SHA-256 hash of the sorted feature names."""
    schema = "\n".join(sorted(feature_columns))
    return f"sha256:{hashlib.sha256(schema.encode()).hexdigest()}"
