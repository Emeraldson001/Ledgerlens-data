"""Tests for scripts/manage_artifact_lifecycle.py — authenticated, audited
promote/rollback CLI (Grand 2 / issue #671, Task E).
"""

import os

import pytest

from config import config
from detection import model_governance as mg
from detection.persistence import PromotionAuditLog, get_engine, get_session_factory
from scripts.manage_artifact_lifecycle import main


@pytest.fixture(autouse=True)
def _governance_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PROMOTION_SECRET", "test-secret")
    monkeypatch.setattr(config, "MODEL_PROMOTION_AUTHORIZED_ACTORS", "alice")
    monkeypatch.setattr(config, "RISK_SCORE_DB_URL", f"sqlite:///{tmp_path}/cli_gov.db")


@pytest.fixture()
def artifact_file(tmp_path):
    path = tmp_path / "rf.joblib"
    path.write_bytes(b"fake-model-bytes-v1")
    return str(path)


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["manage_artifact_lifecycle.py", *argv])
    main()


class TestPromoteAuthorization:
    def test_promote_without_credential_is_denied_and_audited(
        self, tmp_path, artifact_file, monkeypatch
    ):
        manifest_path = str(tmp_path / "manifest.json")
        monkeypatch.setattr(
            "sys.argv",
            [
                "manage_artifact_lifecycle.py",
                "--manifest-path",
                manifest_path,
                "register",
                "--name",
                "rf",
                "--artifact-path",
                artifact_file,
            ],
        )
        main()
        monkeypatch.setattr(
            "sys.argv",
            [
                "manage_artifact_lifecycle.py",
                "--manifest-path",
                manifest_path,
                "validate",
                "--name",
                "rf",
                "--version",
                _read_only_version(manifest_path),
            ],
        )
        main()

        with pytest.raises(SystemExit):
            _run(
                monkeypatch,
                [
                    "--manifest-path",
                    manifest_path,
                    "promote",
                    "--name",
                    "rf",
                    "--version",
                    _read_only_version(manifest_path),
                    "--actor",
                    "mallory",
                    "--credential",
                    "wrong",
                ],
            )

        audit_log = PromotionAuditLog(get_session_factory(get_engine()))
        rows = audit_log.recent()
        assert any(r.actor == "mallory" and not r.success for r in rows)

    def test_authorized_promote_succeeds_and_is_audited(self, tmp_path, artifact_file, monkeypatch):
        manifest_path = str(tmp_path / "manifest.json")
        _run(
            monkeypatch,
            [
                "--manifest-path",
                manifest_path,
                "register",
                "--name",
                "rf",
                "--artifact-path",
                artifact_file,
            ],
        )
        version = _read_only_version(manifest_path)
        _run(
            monkeypatch,
            ["--manifest-path", manifest_path, "validate", "--name", "rf", "--version", version],
        )

        # A signed, transparency-logged copy of the artifact must exist at
        # the SAME path recorded in the registry for make_trust_verifier to
        # succeed (it re-verifies via ModelArtifactVerifier against the
        # directory containing record.artifact_path).
        public_key_path = _sign_and_register(artifact_file, "rf")
        monkeypatch.setattr(config, "TRUSTED_SIGNING_PUBLIC_KEY_PATH", public_key_path)

        _run(
            monkeypatch,
            [
                "--manifest-path",
                manifest_path,
                "promote",
                "--name",
                "rf",
                "--version",
                version,
                "--actor",
                "alice",
                "--credential",
                mg.expected_credential("alice"),
            ],
        )

        audit_log = PromotionAuditLog(get_session_factory(get_engine()))
        rows = audit_log.recent()
        assert any(r.actor == "alice" and r.action == "promote" and r.success for r in rows)


def _read_only_version(manifest_path: str) -> str:
    import json

    with open(manifest_path) as f:
        data = json.load(f)
    return next(iter(data["rf"]))


def _sign_and_register(artifact_path: str, model_name: str) -> str:
    """Returns the public key PEM path to configure as TRUSTED_SIGNING_PUBLIC_KEY_PATH."""
    """Sign the metrics.json expected next to *artifact_path* and register
    it in the default transparency log, so make_trust_verifier can succeed."""
    import hashlib
    import json

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from detection.persistence import get_default_transparency_log, sign_metrics

    model_dir = os.path.dirname(artifact_path)
    sha = hashlib.sha256()
    with open(artifact_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({model_name: {"artifact_sha256": sha.hexdigest()}}, f)

    private_key = Ed25519PrivateKey.generate()
    key_path = os.path.join(model_dir, "signing_key.pem")
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    sign_metrics(metrics_path, key_path)

    public_key = private_key.public_key()
    public_path = os.path.join(model_dir, "public_key.pem")
    with open(public_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    log = get_default_transparency_log()
    log.append(model_name, sha.hexdigest())
    return public_path
