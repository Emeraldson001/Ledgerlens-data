"""Feature-contract validation for model upgrades and rollbacks.

Model artifacts are only interchangeable when their input feature contracts
are compatible.  This module compares the metadata sidecars without loading
the (potentially large) model artifacts, so it can be used by CI, deployment
gates, and local contributor tooling.

Legacy metadata that predates dtype recording is supported.  Such comparisons
are explicitly reported as ``names_only`` rather than silently claiming that
types were checked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FEATURE_CONTRACT_VERSION = 1


class FeatureContractError(ValueError):
    """Raised when model metadata does not contain a valid feature contract."""


def compute_feature_contract_hash(
    feature_columns: list[str],
    feature_dtypes: Mapping[str, str],
    contract_version: int = FEATURE_CONTRACT_VERSION,
) -> str:
    """Return a deterministic hash covering feature order, names, and dtypes."""
    payload = {
        "version": contract_version,
        "features": [
            {"name": column, "dtype": feature_dtypes[column]} for column in feature_columns
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _compute_legacy_schema_hash(feature_columns: list[str]) -> str:
    canonical = "\n".join(sorted(feature_columns))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True)
class FeatureContract:
    """The feature portion of a ``model_metadata.json`` sidecar."""

    columns: tuple[str, ...]
    dtypes: Mapping[str, str]
    contract_version: int
    schema_hash: str | None = None
    contract_hash: str | None = None
    source: str = "model metadata"

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        source: str = "model metadata",
    ) -> FeatureContract:
        columns_raw = metadata.get("feature_columns")
        if not isinstance(columns_raw, list) or not columns_raw:
            raise FeatureContractError(f"{source}: 'feature_columns' must be a non-empty list")
        if any(not isinstance(column, str) or not column for column in columns_raw):
            raise FeatureContractError(f"{source}: every feature column must be a non-empty string")
        if len(columns_raw) != len(set(columns_raw)):
            duplicates = sorted(
                column for column in set(columns_raw) if columns_raw.count(column) > 1
            )
            raise FeatureContractError(f"{source}: duplicate feature columns: {duplicates}")

        version_raw = metadata.get("feature_contract_version", 0)
        if isinstance(version_raw, bool) or not isinstance(version_raw, int) or version_raw < 0:
            raise FeatureContractError(
                f"{source}: 'feature_contract_version' must be a non-negative integer"
            )
        if version_raw > FEATURE_CONTRACT_VERSION:
            raise FeatureContractError(
                f"{source}: feature contract version {version_raw} is newer than "
                f"the supported version {FEATURE_CONTRACT_VERSION}"
            )

        dtypes_raw = metadata.get("feature_dtypes", {})
        if not isinstance(dtypes_raw, dict):
            raise FeatureContractError(f"{source}: 'feature_dtypes' must be an object")
        unknown_dtype_columns = sorted(set(dtypes_raw) - set(columns_raw))
        if unknown_dtype_columns:
            raise FeatureContractError(
                f"{source}: feature_dtypes contains unknown columns: " f"{unknown_dtype_columns}"
            )
        if any(not isinstance(value, str) or not value for value in dtypes_raw.values()):
            raise FeatureContractError(f"{source}: every feature dtype must be a non-empty string")

        schema_hash = metadata.get("feature_schema_hash")
        contract_hash = metadata.get("feature_contract_hash")
        if schema_hash is not None and not isinstance(schema_hash, str):
            raise FeatureContractError(f"{source}: 'feature_schema_hash' must be a string")
        if contract_hash is not None and not isinstance(contract_hash, str):
            raise FeatureContractError(f"{source}: 'feature_contract_hash' must be a string")
        if version_raw >= 1:
            missing_dtypes = sorted(set(columns_raw) - set(dtypes_raw))
            if missing_dtypes:
                raise FeatureContractError(
                    f"{source}: feature contract version {version_raw} requires "
                    f"dtypes for every feature; missing {missing_dtypes}"
                )
            if contract_hash is None:
                raise FeatureContractError(
                    f"{source}: feature contract version {version_raw} requires "
                    "'feature_contract_hash'"
                )

        return cls(
            columns=tuple(columns_raw),
            dtypes=dict(dtypes_raw),
            contract_version=version_raw,
            schema_hash=schema_hash,
            contract_hash=contract_hash,
            source=source,
        )

    def integrity_errors(self) -> list[str]:
        """Return metadata self-consistency errors without raising."""
        errors: list[str] = []
        columns = list(self.columns)
        expected_schema_hash = _compute_legacy_schema_hash(columns)
        if self.schema_hash is not None and self.schema_hash != expected_schema_hash:
            errors.append(f"{self.source}: feature_schema_hash does not match feature_columns")

        if self.contract_hash is not None:
            missing_dtypes = sorted(set(columns) - set(self.dtypes))
            if missing_dtypes:
                errors.append(
                    f"{self.source}: feature_contract_hash cannot be verified because "
                    f"dtypes are missing for {missing_dtypes}"
                )
            else:
                expected_contract_hash = compute_feature_contract_hash(
                    columns,
                    self.dtypes,
                    self.contract_version,
                )
                if self.contract_hash != expected_contract_hash:
                    errors.append(
                        f"{self.source}: feature_contract_hash does not match feature "
                        "order, names, and dtypes"
                    )
        return errors


@dataclass(frozen=True)
class FeatureCompatibilityReport:
    """Actionable result of comparing a candidate contract to a reference."""

    compatible: bool
    status: str
    validation_scope: str
    added_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    reordered_features: tuple[str, ...]
    dtype_changes: Mapping[str, tuple[str, str]]
    unchecked_dtypes: tuple[str, ...]
    errors: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "status": self.status,
            "validation_scope": self.validation_scope,
            "added_features": list(self.added_features),
            "removed_features": list(self.removed_features),
            "reordered_features": list(self.reordered_features),
            "dtype_changes": {
                name: {"reference": values[0], "candidate": values[1]}
                for name, values in self.dtype_changes.items()
            },
            "unchecked_dtypes": list(self.unchecked_dtypes),
            "errors": list(self.errors),
            "diagnostics": list(self.diagnostics),
        }


def compare_feature_contracts(
    reference: FeatureContract,
    candidate: FeatureContract,
    *,
    allow_additive: bool = False,
) -> FeatureCompatibilityReport:
    """Compare a candidate model's features against a reference model.

    By default, compatibility is strict: the candidate must preserve the same
    feature names, relative order, and recorded dtypes.  ``allow_additive``
    permits new candidate-only features, while still rejecting removals,
    shared-feature reordering, and type changes.
    """
    reference_columns = list(reference.columns)
    candidate_columns = list(candidate.columns)
    reference_set = set(reference_columns)
    candidate_set = set(candidate_columns)

    added = tuple(column for column in candidate_columns if column not in reference_set)
    removed = tuple(column for column in reference_columns if column not in candidate_set)

    shared_reference = [column for column in reference_columns if column in candidate_set]
    shared_candidate = [column for column in candidate_columns if column in reference_set]
    reordered = tuple(shared_candidate) if shared_reference != shared_candidate else ()

    dtype_changes: dict[str, tuple[str, str]] = {}
    unchecked_dtypes: list[str] = []
    for column in shared_reference:
        reference_dtype = reference.dtypes.get(column)
        candidate_dtype = candidate.dtypes.get(column)
        if reference_dtype is None or candidate_dtype is None:
            unchecked_dtypes.append(column)
        elif reference_dtype != candidate_dtype:
            dtype_changes[column] = (reference_dtype, candidate_dtype)

    errors = reference.integrity_errors() + candidate.integrity_errors()
    breaking = bool(
        errors or removed or reordered or dtype_changes or (added and not allow_additive)
    )

    if errors:
        status = "invalid_metadata"
    elif breaking:
        status = "incompatible"
    elif added:
        status = "additive_compatible"
    elif unchecked_dtypes:
        status = "compatible_names_only"
    else:
        status = "identical"

    diagnostics: list[str] = []
    if added:
        qualifier = "allowed by policy" if allow_additive else "not allowed by strict policy"
        diagnostics.append(f"Candidate adds features {list(added)} ({qualifier}).")
    if removed:
        diagnostics.append(
            f"Candidate removes reference features {list(removed)}; existing inputs "
            "and rollback models may no longer be interchangeable."
        )
    if reordered:
        diagnostics.append(
            f"Shared feature order changed to {list(reordered)}; positional model "
            "artifacts may score the wrong values."
        )
    if dtype_changes:
        changes = ", ".join(f"{name}: {old} -> {new}" for name, (old, new) in dtype_changes.items())
        diagnostics.append(f"Feature dtype changes detected ({changes}).")
    if unchecked_dtypes:
        diagnostics.append(
            "Dtype compatibility was not checked for "
            f"{unchecked_dtypes}; regenerate legacy metadata to enable full validation."
        )
    diagnostics.extend(errors)

    validation_scope = "names_and_dtypes" if not unchecked_dtypes else "names_only"
    return FeatureCompatibilityReport(
        compatible=not breaking,
        status=status,
        validation_scope=validation_scope,
        added_features=added,
        removed_features=removed,
        reordered_features=reordered,
        dtype_changes=dtype_changes,
        unchecked_dtypes=tuple(unchecked_dtypes),
        errors=tuple(errors),
        diagnostics=tuple(diagnostics),
    )


def validate_feature_compatibility(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    *,
    allow_additive: bool = False,
    reference_source: str = "reference metadata",
    candidate_source: str = "candidate metadata",
) -> FeatureCompatibilityReport:
    """Build contracts from metadata mappings and compare them."""
    reference = FeatureContract.from_metadata(
        reference_metadata,
        source=reference_source,
    )
    candidate = FeatureContract.from_metadata(
        candidate_metadata,
        source=candidate_source,
    )
    return compare_feature_contracts(
        reference,
        candidate,
        allow_additive=allow_additive,
    )
