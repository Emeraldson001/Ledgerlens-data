"""Tests for detection/model_governance.py — the single gated promotion/
rollback path for production ML models (Grand 2 / issue #671).

Covers: authorization (fail-closed), the offline regression gate, the
production-write guard, end-to-end gated promotion (happy path, tampered
artifact, regression, unauthorized actor), and rollback (happy path,
unauthorized, no target, tampered archive).
"""

import json
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config import config
from detection import model_governance as mg
from detection.persistence import (
    ModelVersionRecord,
    PromotionAuditLog,
    get_engine,
    get_session_factory,
)

MODEL_NAMES = ["random_forest"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_keypair():
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def _write_private_key(tmp_path, private_key, name="signing_key.pem"):
    path = str(tmp_path / name)
    with open(path, "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    return path


def _make_bundle(base_dir, contents: dict[str, bytes] | None = None) -> str:
    """A minimal, unsigned candidate/production directory: {name}.joblib +
    an empty metrics.json (no model_metadata.json, so the compatibility
    gate soft-skips — these tests exercise the trust/regression/auth gates,
    not the compatibility contract, which is covered in
    tests/test_artifact_compatibility.py)."""
    os.makedirs(base_dir, exist_ok=True)
    contents = contents or {n: f"fake-{n}-v1".encode() for n in MODEL_NAMES}
    for name, data in contents.items():
        with open(os.path.join(base_dir, f"{name}.joblib"), "wb") as f:
            f.write(data)
    with open(os.path.join(base_dir, "metrics.json"), "w") as f:
        json.dump({}, f)
    return base_dir


@pytest.fixture()
def gov_session_factory(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/gov.db")
    return get_session_factory(engine)


@pytest.fixture()
def signing(tmp_path):
    private_key, public_key = _gen_keypair()
    key_path = _write_private_key(tmp_path, private_key)
    return {"private_key_path": key_path, "public_key": public_key}


@pytest.fixture(autouse=True)
def _authorized_actor(monkeypatch):
    """Every test in this module gets a working authorization config by
    default; individual tests override to exercise denial paths."""
    monkeypatch.setattr(config, "MODEL_PROMOTION_SECRET", "test-secret")
    monkeypatch.setattr(config, "MODEL_PROMOTION_AUTHORIZED_ACTORS", "alice,retrain-pipeline")
    monkeypatch.setattr(config, "MODEL_PROMOTION_SYSTEM_ACTOR", "retrain-pipeline")


def _old_new_metrics(old_auc=0.90, new_auc=0.91):
    old = {n: {"auc_roc": old_auc, "f1": 0.80} for n in MODEL_NAMES}
    new = {n: {"auc_roc": new_auc, "f1": 0.80} for n in MODEL_NAMES}
    return old, new


# ---------------------------------------------------------------------------
# authorize_actor — fails closed
# ---------------------------------------------------------------------------


class TestAuthorizeActor:
    def test_valid_actor_and_credential_succeeds(self):
        credential = mg.expected_credential("alice")
        mg.authorize_actor("alice", credential)  # must not raise

    def test_unknown_actor_rejected(self):
        credential = mg.expected_credential("mallory")
        with pytest.raises(mg.UnauthorizedPromotionError, match="not in"):
            mg.authorize_actor("mallory", credential)

    def test_wrong_credential_rejected(self):
        with pytest.raises(mg.UnauthorizedPromotionError, match="invalid credential"):
            mg.authorize_actor("alice", "not-the-right-hmac")

    def test_empty_actor_rejected(self):
        with pytest.raises(mg.UnauthorizedPromotionError, match="actor must not be empty"):
            mg.authorize_actor("", "anything")

    def test_fails_closed_when_secret_not_configured(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_PROMOTION_SECRET", "")
        with pytest.raises(mg.UnauthorizedPromotionError, match="MODEL_PROMOTION_SECRET"):
            mg.authorize_actor("alice", "irrelevant")

    def test_system_actor_credential_round_trips(self):
        actor, credential = mg.system_actor_credential()
        assert actor == "retrain-pipeline"
        mg.authorize_actor(actor, credential)  # must not raise


# ---------------------------------------------------------------------------
# evaluate_regression_gate
# ---------------------------------------------------------------------------


class TestEvaluateRegressionGate:
    def test_no_prior_metrics_always_approves(self):
        decision = mg.evaluate_regression_gate(None, {"rf": {"auc_roc": 0.5, "f1": 0.5}})
        assert decision.approved is True

    def test_regression_beyond_tolerance_blocks(self):
        old, new = _old_new_metrics(old_auc=0.90, new_auc=0.80)
        decision = mg.evaluate_regression_gate(old, new, model_names=MODEL_NAMES)
        assert decision.approved is False
        assert "AUC-ROC" in decision.reason

    def test_within_tolerance_approves(self):
        old, new = _old_new_metrics(old_auc=0.90, new_auc=0.895)
        decision = mg.evaluate_regression_gate(old, new, model_names=MODEL_NAMES, tolerance=0.01)
        assert decision.approved is True

    def test_missing_candidate_metrics_blocks(self):
        old, _ = _old_new_metrics()
        decision = mg.evaluate_regression_gate(old, {}, model_names=MODEL_NAMES)
        assert decision.approved is False


# ---------------------------------------------------------------------------
# guard_production_write
# ---------------------------------------------------------------------------


class TestGuardProductionWrite:
    def test_non_production_dir_always_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path / "prod"))
        mg.guard_production_write(str(tmp_path / "staging"))  # must not raise

    def test_bootstrap_allowed_when_production_empty(self, tmp_path, monkeypatch):
        prod = str(tmp_path / "prod")
        os.makedirs(prod)
        monkeypatch.setattr(config, "MODEL_DIR", prod)
        mg.guard_production_write(prod)  # no metrics.json yet — must not raise

    def test_blocks_when_production_already_has_metrics_json(self, tmp_path, monkeypatch):
        prod = str(tmp_path / "prod")
        os.makedirs(prod)
        with open(os.path.join(prod, "metrics.json"), "w") as f:
            f.write("{}")
        monkeypatch.setattr(config, "MODEL_DIR", prod)
        with pytest.raises(mg.UngatedProductionWriteError):
            mg.guard_production_write(prod)


# ---------------------------------------------------------------------------
# promote_candidate
# ---------------------------------------------------------------------------


class TestPromoteCandidate:
    def test_happy_path_publishes_and_records_version(self, tmp_path, signing, gov_session_factory):
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()

        record = mg.promote_candidate(
            candidate_dir=candidate_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            reason="test promotion",
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )

        assert os.path.exists(os.path.join(model_dir, "random_forest.joblib"))
        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == b"fake-random_forest-v1"

        assert record.status == "production"
        assert record.promoted_by == "alice"

        with gov_session_factory() as session:
            row = session.query(ModelVersionRecord).filter_by(version_id=record.version_id).one()
            assert row.status == "production"
            assert row.promoted_by == "alice"

        audit_log = PromotionAuditLog(gov_session_factory)
        rows = audit_log.recent()
        assert any(r.action == "promote" and r.success for r in rows)

    def test_wrong_public_key_blocks_promotion_and_writes_nothing(
        self, tmp_path, signing, gov_session_factory
    ):
        """A candidate signed with one key but verified against a different
        (wrong) trusted public key must fail the trust chain — the same
        failure mode as an attacker substituting the signing key, or an
        operator misconfiguring TRUSTED_SIGNING_PUBLIC_KEY_PATH."""
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()
        _, wrong_public_key = _gen_keypair()

        with pytest.raises(mg.ArtifactTrustError, match="signature verification failed"):
            mg.promote_candidate(
                candidate_dir=candidate_dir,
                model_dir=model_dir,
                actor="alice",
                credential=mg.expected_credential("alice"),
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                model_names=MODEL_NAMES,
                signing_key_path=signing["private_key_path"],
                public_key=wrong_public_key,
                session_factory=gov_session_factory,
            )

        assert not os.path.exists(model_dir) or not os.listdir(model_dir)
        audit_log = PromotionAuditLog(gov_session_factory)
        rows = audit_log.recent()
        assert any(r.action == "promote" and not r.success for r in rows)

    def test_tampered_artifact_at_rest_is_caught_on_reload(
        self, tmp_path, signing, gov_session_factory
    ):
        """promote_candidate signs whatever bytes exist at promotion time (it
        cannot detect tampering that happened *before* it ran — that is a
        training-pipeline supply-chain concern, out of scope per the issue).
        What it does guarantee is that whatever it publishes is verifiable
        through the exact code path RiskScorer uses at load time — so a
        *post-promotion* tamper at rest must be caught by the load-time
        verifier, using the same transparency log this promotion wrote to.
        """
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()

        from detection.persistence import get_default_transparency_log

        transparency_log = get_default_transparency_log(gov_session_factory)

        mg.promote_candidate(
            candidate_dir=candidate_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            transparency_log=transparency_log,
            session_factory=gov_session_factory,
        )

        # Tamper production at rest, post-promotion.
        with open(os.path.join(model_dir, "random_forest.joblib"), "wb") as f:
            f.write(b"tampered-after-promotion")

        from detection.persistence import ModelArtifactVerifier, ModelIntegrityError

        # No expected_sha256 was captured ahead of time (a realistic loader
        # doesn't have one to compare against either), so the tamper is
        # caught by the *other* independent leg of the trust chain: the
        # freshly-computed hash of the tampered bytes was never registered
        # in the transparency log.
        with pytest.raises(ModelIntegrityError, match="not in the transparency log"):
            ModelArtifactVerifier(transparency_log, model_dir).verify(
                "random_forest", public_key=signing["public_key"]
            )

    def test_regression_blocks_promotion_and_writes_nothing(
        self, tmp_path, signing, gov_session_factory
    ):
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics(old_auc=0.95, new_auc=0.70)

        with pytest.raises(mg.RegressionGateError):
            mg.promote_candidate(
                candidate_dir=candidate_dir,
                model_dir=model_dir,
                actor="alice",
                credential=mg.expected_credential("alice"),
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                model_names=MODEL_NAMES,
                signing_key_path=signing["private_key_path"],
                public_key=signing["public_key"],
                session_factory=gov_session_factory,
            )

        assert not os.path.exists(model_dir) or not os.listdir(model_dir)
        # Regression must be caught before any signing/trust-chain work —
        # metrics.json in the candidate must be untouched (still empty).
        with open(os.path.join(candidate_dir, "metrics.json")) as f:
            assert json.load(f) == {}

    def test_unauthorized_actor_blocks_promotion_before_any_crypto_work(
        self, tmp_path, signing, gov_session_factory
    ):
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()

        with pytest.raises(mg.UnauthorizedPromotionError):
            mg.promote_candidate(
                candidate_dir=candidate_dir,
                model_dir=model_dir,
                actor="mallory",
                credential="wrong",
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                model_names=MODEL_NAMES,
                signing_key_path=signing["private_key_path"],
                public_key=signing["public_key"],
                session_factory=gov_session_factory,
            )

        assert not os.path.exists(model_dir) or not os.listdir(model_dir)
        with open(os.path.join(candidate_dir, "metrics.json")) as f:
            assert json.load(f) == {}, "unauthorized attempt must not sign the candidate"

        audit_log = PromotionAuditLog(gov_session_factory)
        rows = audit_log.recent()
        assert any(r.actor == "mallory" and not r.success for r in rows)

    def test_first_ever_promotion_has_no_prior_metrics_to_regress_against(
        self, tmp_path, signing, gov_session_factory
    ):
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")

        record = mg.promote_candidate(
            candidate_dir=candidate_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=None,
            new_metrics={"random_forest": {"auc_roc": 0.5, "f1": 0.5}},
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )
        assert record.status == "production"
        assert record.parent_version_id is None


# ---------------------------------------------------------------------------
# rollback_production
# ---------------------------------------------------------------------------


class TestRollbackProduction:
    def _promote_two_versions(self, tmp_path, signing, gov_session_factory):
        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()

        v1_dir = _make_bundle(str(tmp_path / "v1"), {"random_forest": b"v1-bytes"})
        v1 = mg.promote_candidate(
            candidate_dir=v1_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=None,
            new_metrics=old_metrics,
            reason="v1",
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )

        v2_dir = _make_bundle(str(tmp_path / "v2"), {"random_forest": b"v2-bytes"})
        v2 = mg.promote_candidate(
            candidate_dir=v2_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            reason="v2",
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )
        return model_dir, v1, v2

    def test_rollback_restores_prior_version_bytes(self, tmp_path, signing, gov_session_factory):
        model_dir, v1, v2 = self._promote_two_versions(tmp_path, signing, gov_session_factory)

        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == b"v2-bytes"

        restored = mg.rollback_production(
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            reason="v2 regressed in production",
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )

        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == b"v1-bytes"
        assert restored.status == "production"
        assert restored.parent_version_id == v2.version_id

        with gov_session_factory() as session:
            v2_row = session.query(ModelVersionRecord).filter_by(version_id=v2.version_id).one()
            assert v2_row.status == "rolled_back"
            assert v2_row.rolled_back_by == "alice"

    def test_rollback_requires_authorization(self, tmp_path, signing, gov_session_factory):
        model_dir, _, _ = self._promote_two_versions(tmp_path, signing, gov_session_factory)

        with pytest.raises(mg.UnauthorizedPromotionError):
            mg.rollback_production(
                model_dir=model_dir,
                actor="mallory",
                credential="wrong",
                reason="malicious rollback",
                public_key=signing["public_key"],
                session_factory=gov_session_factory,
            )

        # Production must be untouched by a denied rollback attempt.
        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == b"v2-bytes"

    def test_rollback_with_no_prior_version_raises(self, tmp_path, signing, gov_session_factory):
        candidate_dir = _make_bundle(str(tmp_path / "candidate"))
        model_dir = str(tmp_path / "production")
        mg.promote_candidate(
            candidate_dir=candidate_dir,
            model_dir=model_dir,
            actor="alice",
            credential=mg.expected_credential("alice"),
            old_metrics=None,
            new_metrics={"random_forest": {"auc_roc": 0.5, "f1": 0.5}},
            model_names=MODEL_NAMES,
            signing_key_path=signing["private_key_path"],
            public_key=signing["public_key"],
            session_factory=gov_session_factory,
        )

        with pytest.raises(mg.NoRollbackTargetError):
            mg.rollback_production(
                model_dir=model_dir,
                actor="alice",
                credential=mg.expected_credential("alice"),
                reason="nothing to roll back to",
                public_key=signing["public_key"],
                session_factory=gov_session_factory,
            )

    def test_rollback_rejects_tampered_archive(self, tmp_path, signing, gov_session_factory):
        model_dir, v1, v2 = self._promote_two_versions(tmp_path, signing, gov_session_factory)

        # v2's own training_metadata.archive_path is the pre-promotion
        # snapshot taken right before v2 went live — i.e. v1's content. That
        # is what a rollback-to-v1 actually restores from (see
        # rollback_production's archive_source resolution). Tamper it at
        # rest — rollback must re-verify the trust chain of the archive, not
        # just trust it because it once passed.
        with gov_session_factory() as session:
            v2_row = session.query(ModelVersionRecord).filter_by(version_id=v2.version_id).one()
            archive_path = json.loads(v2_row.training_metadata)["archive_path"]

        with open(os.path.join(archive_path, "random_forest.joblib"), "wb") as f:
            f.write(b"tampered-in-archive")

        with pytest.raises(mg.ArtifactTrustError):
            mg.rollback_production(
                model_dir=model_dir,
                actor="alice",
                credential=mg.expected_credential("alice"),
                reason="attempt rollback to tampered archive",
                public_key=signing["public_key"],
                session_factory=gov_session_factory,
            )

        # Production must remain on v2 — a failed rollback must not partially apply.
        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            assert f.read() == b"v2-bytes"


# ---------------------------------------------------------------------------
# Concurrency: two simultaneous promotions targeting the same model_dir
# ---------------------------------------------------------------------------


class TestConcurrentPromotions:
    def test_two_simultaneous_promotions_do_not_interleave(
        self, tmp_path, signing, gov_session_factory
    ):
        """Two candidates promoted concurrently to the same model_dir must
        not interleave their archive/publish/registry steps: production
        must end up wholly on one candidate's bytes (not a mix), exactly one
        ModelVersionRecord ends up with status="production", and the other
        is correctly marked "deprecated"."""
        import threading

        model_dir = str(tmp_path / "production")
        old_metrics, new_metrics = _old_new_metrics()

        candidate_a = _make_bundle(
            str(tmp_path / "candidate_a"), {"random_forest": b"AAAAAAAAAAAAAAAA"}
        )
        candidate_b = _make_bundle(
            str(tmp_path / "candidate_b"), {"random_forest": b"BBBBBBBBBBBBBBBB"}
        )

        results: list = []
        barrier = threading.Barrier(2)

        def _promote(candidate_dir):
            barrier.wait()
            try:
                record = mg.promote_candidate(
                    candidate_dir=candidate_dir,
                    model_dir=model_dir,
                    actor="alice",
                    credential=mg.expected_credential("alice"),
                    old_metrics=old_metrics,
                    new_metrics=new_metrics,
                    model_names=MODEL_NAMES,
                    signing_key_path=signing["private_key_path"],
                    public_key=signing["public_key"],
                    session_factory=gov_session_factory,
                )
                results.append(("ok", record.version_id))
            except Exception as exc:  # noqa: BLE001
                results.append(("error", str(exc)))

        threads = [
            threading.Thread(target=_promote, args=(candidate_a,)),
            threading.Thread(target=_promote, args=(candidate_b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(results) == 2
        assert all(outcome == "ok" for outcome, _ in results), results

        # Production must be wholly one candidate's bytes, never a mix.
        with open(os.path.join(model_dir, "random_forest.joblib"), "rb") as f:
            final_bytes = f.read()
        assert final_bytes in (b"AAAAAAAAAAAAAAAA", b"BBBBBBBBBBBBBBBB")

        with gov_session_factory() as session:
            rows = session.query(ModelVersionRecord).all()
            assert len(rows) == 2
            statuses = sorted(r.status for r in rows)
            assert statuses == ["deprecated", "production"]
            production_row = next(r for r in rows if r.status == "production")
            deprecated_row = next(r for r in rows if r.status == "deprecated")
            # The production row's parent must be the deprecated one — the
            # lock forces one promotion's "previous production" lookup to
            # see the other's already-committed row, never a stale read.
            assert production_row.parent_version_id == deprecated_row.version_id
