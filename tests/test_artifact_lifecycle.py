import json

import pytest

from detection.artifact_lifecycle import (
    ArtifactNotFoundError,
    ArtifactStage,
    IntegrityCheckError,
    InvalidTransitionError,
    ModelArtifactRegistry,
    TrustVerificationRequiredError,
)


@pytest.fixture()
def artifact_file(tmp_path):
    path = tmp_path / "rf.joblib"
    path.write_bytes(b"fake-model-bytes-v1")
    return str(path)


@pytest.fixture()
def registry(tmp_path):
    # Grand 2 (issue #671): promote() requires a trust_verifier. A no-op
    # stub is fine for tests that only exercise the state-machine mechanics
    # (register/validate/promote/rollback transitions) — the *real* verifier
    # wiring (detection.model_governance.make_trust_verifier) is covered by
    # tests/test_model_governance.py.
    return ModelArtifactRegistry(
        manifest_path=str(tmp_path / "artifact_manifest.json"),
        trust_verifier=lambda record: None,
    )


def test_register_creates_staged_version(registry, artifact_file):
    registry.register("rf", artifact_file, metrics={"auc": 0.9})
    versions = registry.list_versions("rf")
    assert len(versions) == 1
    assert versions[0].stage == ArtifactStage.STAGED
    assert versions[0].metrics == {"auc": 0.9}


def test_full_lifecycle_happy_path(registry, artifact_file):
    version = registry.register("rf", artifact_file)
    registry.validate("rf", version)
    registry.promote("rf", version)

    active = registry.get_active("rf")
    assert active.version == version
    assert active.stage == ArtifactStage.PROMOTED


def test_promote_supersedes_previous_active(registry, artifact_file, tmp_path):
    v1 = registry.register("rf", artifact_file)
    registry.validate("rf", v1)
    registry.promote("rf", v1)

    artifact2 = tmp_path / "rf_v2.joblib"
    artifact2.write_bytes(b"fake-model-bytes-v2")
    v2 = registry.register("rf", str(artifact2))
    registry.validate("rf", v2)
    registry.promote("rf", v2)

    assert registry.get_active("rf").version == v2
    v1_record = registry._get("rf", v1)
    assert v1_record.stage == ArtifactStage.DEPRECATED


def test_invalid_transition_raises_with_diagnostics(registry, artifact_file):
    version = registry.register("rf", artifact_file)
    with pytest.raises(InvalidTransitionError) as excinfo:
        registry.promote("rf", version)  # staged -> promoted is illegal
    assert "staged" in str(excinfo.value)
    assert "promoted" in str(excinfo.value)


def test_unknown_artifact_raises_not_found(registry):
    with pytest.raises(ArtifactNotFoundError):
        registry.get_active("does-not-exist")


def test_rollback_reactivates_parent(registry, artifact_file, tmp_path):
    v1 = registry.register("rf", artifact_file)
    registry.validate("rf", v1)
    registry.promote("rf", v1)

    artifact2 = tmp_path / "rf_v2.joblib"
    artifact2.write_bytes(b"fake-model-bytes-v2")
    v2 = registry.register("rf", str(artifact2))
    registry.validate("rf", v2)
    registry.promote("rf", v2)

    rolled_back = registry.rollback("rf", reason="regression on canary")
    assert rolled_back.version == v2
    assert rolled_back.stage == ArtifactStage.ROLLED_BACK
    assert rolled_back.rollback_reason == "regression on canary"

    active = registry.get_active("rf")
    assert active.version == v1


def test_verify_integrity_detects_tampering(registry, artifact_file):
    version = registry.register("rf", artifact_file)
    with open(artifact_file, "wb") as f:
        f.write(b"tampered-bytes")
    with pytest.raises(IntegrityCheckError):
        registry.verify_integrity("rf", version)


def test_manifest_persists_across_instances(tmp_path, artifact_file):
    manifest_path = str(tmp_path / "artifact_manifest.json")
    registry_a = ModelArtifactRegistry(manifest_path=manifest_path)
    version = registry_a.register("rf", artifact_file)

    registry_b = ModelArtifactRegistry(manifest_path=manifest_path)
    versions = registry_b.list_versions("rf")
    assert len(versions) == 1
    assert versions[0].version == version

    with open(manifest_path) as f:
        raw = json.load(f)
    assert "rf" in raw
    assert version in raw["rf"]


def test_register_missing_artifact_raises_file_not_found(registry, tmp_path):
    missing = str(tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError):
        registry.register("rf", missing)


# ---------------------------------------------------------------------------
# Grand 2 (issue #671) — trust-chain-gated promotion
# ---------------------------------------------------------------------------


def test_promote_without_trust_verifier_raises(tmp_path, artifact_file):
    registry = ModelArtifactRegistry(manifest_path=str(tmp_path / "manifest.json"))
    version = registry.register("rf", artifact_file)
    registry.validate("rf", version)

    with pytest.raises(TrustVerificationRequiredError):
        registry.promote("rf", version)

    # The failed promotion must not have mutated the record's stage.
    assert registry._get("rf", version).stage == ArtifactStage.VALIDATED


def test_promote_calls_trust_verifier_with_record(registry, artifact_file):
    seen = []
    registry._trust_verifier = lambda record: seen.append((record.name, record.version))

    version = registry.register("rf", artifact_file)
    registry.validate("rf", version)
    registry.promote("rf", version)

    assert seen == [("rf", version)]
    assert registry.get_active("rf").version == version


def test_promote_leaves_stage_untouched_when_trust_verifier_raises(tmp_path, artifact_file):
    class _FakeTrustError(Exception):
        pass

    def _failing_verifier(record):
        raise _FakeTrustError(f"tampered: {record.name}")

    registry = ModelArtifactRegistry(
        manifest_path=str(tmp_path / "manifest.json"), trust_verifier=_failing_verifier
    )
    version = registry.register("rf", artifact_file)
    registry.validate("rf", version)

    with pytest.raises(_FakeTrustError, match="tampered"):
        registry.promote("rf", version)

    # Stage must remain VALIDATED, not PROMOTED — a failed trust check must
    # never be recorded as a successful promotion.
    assert registry._get("rf", version).stage == ArtifactStage.VALIDATED
    with pytest.raises(ArtifactNotFoundError):
        registry.get_active("rf")
