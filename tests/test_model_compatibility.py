"""Focused tests for model-version feature contract validation."""

import json

import pytest

from detection.model_compatibility import (
    FEATURE_CONTRACT_VERSION,
    FeatureContractError,
    compute_feature_contract_hash,
    validate_feature_compatibility,
)
from scripts.validate_model_compatibility import main


def _metadata(
    columns: list[str],
    dtypes: dict[str, str] | None = None,
) -> dict:
    dtypes = dtypes or {column: "float64" for column in columns}
    return {
        "feature_columns": columns,
        "feature_dtypes": dtypes,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_contract_hash": compute_feature_contract_hash(columns, dtypes),
    }


def test_identical_contracts_are_fully_validated():
    reference = _metadata(["amount", "velocity"])

    report = validate_feature_compatibility(reference, dict(reference))

    assert report.compatible is True
    assert report.status == "identical"
    assert report.validation_scope == "names_and_dtypes"
    assert report.diagnostics == ()


def test_additive_change_requires_explicit_policy():
    reference = _metadata(["amount"])
    candidate = _metadata(["amount", "velocity"])

    strict_report = validate_feature_compatibility(reference, candidate)
    additive_report = validate_feature_compatibility(
        reference,
        candidate,
        allow_additive=True,
    )

    assert strict_report.compatible is False
    assert strict_report.added_features == ("velocity",)
    assert additive_report.compatible is True
    assert additive_report.status == "additive_compatible"


@pytest.mark.parametrize(
    ("candidate", "field"),
    [
        (_metadata(["amount"]), "removed_features"),
        (_metadata(["velocity", "amount"]), "reordered_features"),
        (
            _metadata(
                ["amount", "velocity"],
                {"amount": "float32", "velocity": "float64"},
            ),
            "dtype_changes",
        ),
    ],
)
def test_breaking_changes_are_rejected(candidate, field):
    reference = _metadata(["amount", "velocity"])

    report = validate_feature_compatibility(reference, candidate)

    assert report.compatible is False
    assert report.status == "incompatible"
    assert getattr(report, field)
    assert report.diagnostics


def test_legacy_metadata_is_reported_as_names_only():
    legacy = {"feature_columns": ["amount", "velocity"]}

    report = validate_feature_compatibility(legacy, dict(legacy))

    assert report.compatible is True
    assert report.status == "compatible_names_only"
    assert report.validation_scope == "names_only"
    assert report.unchecked_dtypes == ("amount", "velocity")


def test_tampered_contract_hash_is_rejected():
    reference = _metadata(["amount"])
    candidate = _metadata(["amount"])
    candidate["feature_contract_hash"] = "sha256:not-the-contract"

    report = validate_feature_compatibility(reference, candidate)

    assert report.compatible is False
    assert report.status == "invalid_metadata"
    assert "feature_contract_hash" in report.errors[0]


def test_tampered_legacy_schema_hash_is_rejected():
    reference = {"feature_columns": ["amount"]}
    candidate = {
        "feature_columns": ["amount"],
        "feature_schema_hash": "sha256:not-the-schema",
    }

    report = validate_feature_compatibility(reference, candidate)

    assert report.compatible is False
    assert report.status == "invalid_metadata"
    assert "feature_schema_hash" in report.errors[0]


def test_duplicate_feature_names_are_invalid():
    with pytest.raises(FeatureContractError, match="duplicate"):
        validate_feature_compatibility(
            {"feature_columns": ["amount", "amount"]},
            {"feature_columns": ["amount"]},
        )


def test_versioned_contract_requires_complete_dtypes_and_hash():
    with pytest.raises(FeatureContractError, match="requires dtypes"):
        validate_feature_compatibility(
            {
                "feature_columns": ["amount"],
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
            },
            _metadata(["amount"]),
        )


def test_future_contract_version_is_rejected():
    candidate = _metadata(["amount"])
    candidate["feature_contract_version"] = FEATURE_CONTRACT_VERSION + 1

    with pytest.raises(FeatureContractError, match="newer than the supported version"):
        validate_feature_compatibility(_metadata(["amount"]), candidate)


def test_cli_emits_json_and_nonzero_exit_for_incompatible_contract(
    tmp_path,
    capsys,
):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    (reference_dir / "model_metadata.json").write_text(
        json.dumps(_metadata(["amount", "velocity"]))
    )
    (candidate_dir / "model_metadata.json").write_text(json.dumps(_metadata(["amount"])))

    exit_code = main(
        [
            "--reference",
            str(reference_dir),
            "--candidate",
            str(candidate_dir),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload[0]["status"] == "incompatible"
    assert payload[0]["removed_features"] == ["velocity"]
