"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for tests
os.environ.setdefault("MODEL_DIR", "./models")
os.environ.setdefault("RISK_SCORE_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("WATCHED_ASSET_PAIRS", "USDC:native,BTC:native,XLM:native")
os.environ.setdefault("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
os.environ.setdefault("MIN_TRADES_FOR_SCORING", "20")

# Hypothesis configuration for property-based tests (issue #205)
from hypothesis import HealthCheck, settings

# Configure Hypothesis for CI environment
settings.register_profile(
    "ci",
    max_examples=500,  # Reasonable number for CI
    deadline=5000,  # 5 seconds per example
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

settings.register_profile(
    "dev",
    max_examples=50,  # Faster for local development
    deadline=2000,
)

# Select profile based on environment
import os

if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
    settings.load_profile("ci")
else:
    settings.load_profile("dev")

# Typed deployment-mode fixtures (Issue #543) — reusable, validated Config
# overlays for local/testnet/production. Each fixture restores prior Config
# state on teardown, so tests using it never leak overrides into other tests.
import pytest

from config.deployment_modes import DeploymentMode, apply_deployment_mode


@pytest.fixture
def local_deployment_config():
    with apply_deployment_mode(DeploymentMode.LOCAL) as fixture:
        yield fixture


@pytest.fixture
def testnet_deployment_config():
    with apply_deployment_mode(DeploymentMode.TESTNET) as fixture:
        yield fixture


@pytest.fixture
def production_deployment_config():
    with apply_deployment_mode(DeploymentMode.PRODUCTION) as fixture:
        yield fixture


# Source package integrity check (Issue #540) — runs once before any test
# collects. A structurally broken tree (missing __init__.py, unresolved
# merge conflict markers, syntax errors) fails the whole session immediately
# with a single readable report instead of surfacing as a scatter of
# unrelated collection/import errors across the suite.
def pytest_sessionstart(session):  # noqa: ARG001
    from utils.package_integrity import check_source_package_integrity

    report = check_source_package_integrity(root=project_root)
    if not report.ok:
        pytest.exit(f"\n{report.render()}", returncode=1)


# ---------------------------------------------------------------------------
# Model-artifact signing helper (Grand 2 / issue #671)
#
# RiskScorer._load_models now hard-blocks on any model that fails the
# Ed25519 signature + transparency-log trust chain (detection.persistence
# .ModelArtifactVerifier) instead of logging a warning and loading it
# unverified. Every test that builds a model directory with save_models()/
# save_training_artifacts() and then constructs a RiskScorer against it must
# sign that directory and pass the resulting (public_key, transparency_log)
# into RiskScorer(..., public_key=..., transparency_log=...) — this helper
# does both steps in one call so fixtures stay short.
# ---------------------------------------------------------------------------


def sign_and_trust_models(model_dir: str, model_names: list[str] | None = None):
    """Sign every model.joblib in *model_dir* with a fresh ephemeral Ed25519
    keypair and register each hash in a new in-memory transparency log.

    Requires ``metrics.json`` (with each model's ``artifact_sha256``) to
    already exist in *model_dir* — i.e. call
    ``detection.model_training.save_training_artifacts`` before this.

    Returns ``(public_key, transparency_log)`` ready to pass to
    ``RiskScorer(model_dir=..., public_key=public_key, transparency_log=transparency_log)``.
    """
    import json
    import os
    import tempfile

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from detection.persistence import TransparencyLog, get_engine, get_session_factory, sign_metrics

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    if model_names is None:
        model_names = [
            name
            for name in metrics
            if isinstance(metrics[name], dict)
            and os.path.exists(os.path.join(model_dir, f"{name}.joblib"))
        ]

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    fd, key_path = tempfile.mkstemp(suffix=".pem", dir=model_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )

    sign_metrics(metrics_path, key_path)

    engine = get_engine("sqlite:///:memory:")
    transparency_log = TransparencyLog(get_session_factory(engine))
    for name in model_names:
        sha = metrics[name]["artifact_sha256"]
        transparency_log.append(name, sha)

    return public_key, transparency_log


def write_minimal_metrics(model_dir: str, model_names: list[str]) -> None:
    """Write a bare-bones ``metrics.json`` (just ``artifact_sha256`` per
    model, no training metrics) for a *model_dir* that only has
    ``{name}.joblib`` files — e.g. built with a bare ``save_models()`` call
    with no ``feature_columns``/``feature_schema_hash`` and no
    ``save_training_artifacts`` call. Lets such fixtures still be signed via
    ``sign_and_trust_models`` (which requires a pre-existing metrics.json)
    while deliberately keeping ``model_metadata.json`` absent, when a test's
    actual point is metadata-absence rather than trust-chain absence.
    """
    import hashlib
    import json
    import os

    metrics: dict = {}
    for name in model_names:
        path = os.path.join(model_dir, f"{name}.joblib")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        metrics[name] = {"artifact_sha256": h.hexdigest()}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def build_signed_model_dir(
    training_output: dict, model_dir: str, data_path: str = "test-data.parquet"
):
    """Write full production-shaped artifacts for *training_output* (the dict
    returned by ``detection.model_training.train_models``) into *model_dir*
    — per-model ``_artifact_manifest.json`` files (so the compatibility gate
    has something real to check, not just a soft "skip" on a bare model
    directory) plus ``metrics.json``/``model_metadata.json`` — then sign and
    register them.

    Returns ``(public_key, transparency_log)``, ready for
    ``RiskScorer(model_dir=..., public_key=..., transparency_log=...)``.
    """
    from detection.model_training import (
        compute_feature_schema_hash,
        save_models,
        save_training_artifacts,
    )

    feature_cols = training_output["feature_columns"]
    feature_hash = compute_feature_schema_hash(feature_cols)
    save_models(
        training_output["results"],
        model_dir,
        feature_columns=feature_cols,
        feature_schema_hash=feature_hash,
    )
    save_training_artifacts(training_output, data_path, model_dir)
    return sign_and_trust_models(model_dir)
