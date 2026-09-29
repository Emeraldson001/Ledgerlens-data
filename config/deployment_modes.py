"""Typed configuration fixtures for named deployment modes.

LedgerLens' ``Config`` (config.py) is a flat, env-var-driven surface with
500+ attributes. Nothing enforces that a given *deployment mode* — local
development, Testnet staging, or Public-network production — actually sets
a coherent, valid combination of those attributes. In practice contributors
hand-roll ``.env`` files per environment and discover missing/incoherent
values only when ``Config.validate()`` (or worse, a runtime call) blows up.

This module gives each supported deployment mode a typed, reusable fixture:

    * :class:`DeploymentMode` — the closed set of supported modes.
    * :class:`DeploymentModeFixture` — a typed contract describing the
      config overrides, and validation requirements for a mode.
    * :func:`get_deployment_mode_fixture` — typed lookup with an actionable
      error listing valid modes.
    * :func:`apply_deployment_mode` — a context manager that overlays a
      fixture's overrides onto a ``Config``-like class, validates the
      result, and restores the previous values on exit. Safe to nest in
      tests or use for local bootstrapping (``scripts/``, notebooks).

Adding a new deployment mode means adding one entry to
``DEPLOYMENT_MODE_FIXTURES`` — every consumer (tests, CLI tooling, docs)
picks it up automatically.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

from importlib import import_module
from typing import Any

#: Environment variable naming the deployment mode a process is running under.
#: Absent means :attr:`DeploymentMode.LOCAL` -- safety checks that harden
#: production must therefore be *opt-in to relax*, never opt-in to enforce.
DEPLOYMENT_MODE_ENV_VAR = "LEDGERLENS_DEPLOYMENT_MODE"


class DeploymentMode(StrEnum):
    """Closed set of deployment modes LedgerLens ships fixtures for."""

    LOCAL = "local"
    TESTNET = "testnet"
    PRODUCTION = "production"


class UnknownDeploymentModeError(ValueError):
    """Raised when a deployment mode name/value has no registered fixture."""

    def __init__(self, requested: str, known: tuple[str, ...]):
        self.requested = requested
        self.known = known
        super().__init__(
            f"Unknown deployment mode {requested!r}. " f"Known modes: {', '.join(sorted(known))}."
        )


class DeploymentModeValidationError(RuntimeError):
    """Raised when a fixture's overrides fail ``Config.validate()``.

    Carries the mode name and the underlying validation error so a failure
    points straight at which fixture is inconsistent and why, rather than
    surfacing as an opaque ``OSError`` deep in application startup.
    """

    def __init__(self, mode: DeploymentMode, cause: Exception):
        self.mode = mode
        self.__cause__ = cause
        super().__init__(
            f"Deployment mode fixture {mode.value!r} produced an invalid " f"configuration: {cause}"
        )


@dataclass(frozen=True)
class DeploymentModeFixture:
    """Typed, reusable contract for a single deployment mode.

    Attributes:
        mode: The mode this fixture describes.
        description: Human-readable summary of when this mode applies.
        overrides: ``Config`` attribute name -> value to set for this mode.
        require_onchain: Passed through to ``Config.validate()`` — whether
            on-chain submission credentials must be present.
    """

    mode: DeploymentMode
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    require_onchain: bool = False


DEPLOYMENT_MODE_FIXTURES: dict[DeploymentMode, DeploymentModeFixture] = {
    DeploymentMode.LOCAL: DeploymentModeFixture(
        mode=DeploymentMode.LOCAL,
        description=(
            "Single-developer local run against Testnet with no on-chain "
            "submission and no external brokers — SSE ingestion, sqlite "
            "storage, stdout alerting."
        ),
        overrides={
            "STELLAR_NETWORK": "TESTNET",
            "HORIZON_URL": "https://horizon-testnet.stellar.org",
            "HORIZON_DEV_MODE": True,
            "STREAMING_BACKEND": "sse",
            "RISK_SCORE_DB_URL": "sqlite:///ledgerlens.local.db",
            "ALERT_CHANNEL": "stdout",
            "WATCHED_ASSET_PAIRS": [("USDC", "native")],
            "MODEL_DIR": "./models",
            "LEDGERLENS_CONTRACT_ID": "",
            "LEDGERLENS_SUBMITTER_SECRET": "",
        },
        require_onchain=False,
    ),
    DeploymentMode.TESTNET: DeploymentModeFixture(
        mode=DeploymentMode.TESTNET,
        description=(
            "Shared Testnet staging deployment — Kafka streaming backend, "
            "on-chain score submission enabled against the Testnet "
            "ledgerlens-score contract, webhook alerting."
        ),
        overrides={
            "STELLAR_NETWORK": "TESTNET",
            "HORIZON_URL": "https://horizon-testnet.stellar.org",
            "HORIZON_DEV_MODE": False,
            "STREAMING_BACKEND": "kafka",
            "SOROBAN_RPC_URL": "https://soroban-testnet.stellar.org",
            "RISK_SCORE_DB_URL": "postgresql://ledgerlens:ledgerlens@localhost:5432/ledgerlens_staging",
            "ALERT_CHANNEL": "webhook",
            "WATCHED_ASSET_PAIRS": [("USDC", "native")],
            "MODEL_DIR": "./models",
            "LEDGERLENS_CONTRACT_ID": "testnet-contract-placeholder",
            "LEDGERLENS_SUBMITTER_SECRET": "testnet-secret-placeholder",
        },
        require_onchain=True,
    ),
    DeploymentMode.PRODUCTION: DeploymentModeFixture(
        mode=DeploymentMode.PRODUCTION,
        description=(
            "Public-network production deployment — Kafka streaming, "
            "on-chain submission required, Postgres storage, webhook "
            "alerting. HORIZON_DEV_MODE is always disabled."
        ),
        overrides={
            "STELLAR_NETWORK": "PUBLIC",
            "HORIZON_URL": "https://horizon.stellar.org",
            "HORIZON_DEV_MODE": False,
            "STREAMING_BACKEND": "kafka",
            "SOROBAN_RPC_URL": "https://soroban-rpc.stellar.org",
            "RISK_SCORE_DB_URL": "postgresql://ledgerlens:ledgerlens@localhost:5432/ledgerlens_production",
            "ALERT_CHANNEL": "webhook",
            "WATCHED_ASSET_PAIRS": [
                ("USDC", "GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")
            ],
            "MODEL_DIR": "./models",
            "LEDGERLENS_CONTRACT_ID": "production-contract-placeholder",
            "LEDGERLENS_SUBMITTER_SECRET": "production-secret-placeholder",
        },
        require_onchain=True,
    ),
}


def current_deployment_mode() -> DeploymentMode:
    """Return the mode this process is running under.

    Reads :data:`DEPLOYMENT_MODE_ENV_VAR`, defaulting to
    :attr:`DeploymentMode.LOCAL`. An unrecognised value raises rather than
    quietly degrading to local, so a typo in a production manifest cannot
    switch production safety checks off.
    """
    raw = os.getenv(DEPLOYMENT_MODE_ENV_VAR, "").strip().lower()
    if not raw:
        return DeploymentMode.LOCAL
    try:
        return DeploymentMode(raw)
    except ValueError as exc:
        known = tuple(m.value for m in DeploymentMode)
        raise UnknownDeploymentModeError(raw, known) from exc


def is_production() -> bool:
    """True when this process is running in production mode."""
    return current_deployment_mode() is DeploymentMode.PRODUCTION


def get_deployment_mode_fixture(mode: DeploymentMode | str) -> DeploymentModeFixture:
    """Look up the typed fixture for ``mode``.

    Accepts either a :class:`DeploymentMode` or its string value (e.g.
    ``"testnet"``) so it's easy to drive from a CLI flag or env var.

    Raises:
        UnknownDeploymentModeError: if ``mode`` has no registered fixture.
    """
    if isinstance(mode, DeploymentMode):
        return DEPLOYMENT_MODE_FIXTURES[mode]
    try:
        resolved = DeploymentMode(mode)
    except ValueError as exc:
        known = tuple(m.value for m in DeploymentMode)
        raise UnknownDeploymentModeError(str(mode), known) from exc
    return DEPLOYMENT_MODE_FIXTURES[resolved]


@contextmanager
def apply_deployment_mode(
    mode: DeploymentMode | str,
    config_cls: Any = None,
    *,
    validate: bool = True,
) -> Iterator[DeploymentModeFixture]:
    """Overlay a deployment mode fixture onto ``config_cls`` for the block.

    Every overridden attribute is restored to its prior value on exit
    (success or exception), so this is safe to use as a pytest fixture or
    a nested context in scripts. When ``validate`` is true (the default),
    the overlaid configuration is checked with ``Config.validate()``
    immediately — a fixture that doesn't produce a valid config fails loudly
    at the call site instead of surfacing later as an unrelated runtime error.

    Args:
        mode: The deployment mode to apply.
        config_cls: The config class to mutate. Defaults to
            ``config.Config`` (imported lazily to avoid a hard import-time
            dependency from this module on the rest of the app).
        validate: Whether to call ``config_cls.validate()`` after applying
            overrides.

    Yields:
        The :class:`DeploymentModeFixture` that was applied.

    Raises:
        UnknownDeploymentModeError: if ``mode`` is not registered.
        DeploymentModeValidationError: if ``validate`` is true and the
            resulting configuration fails ``Config.validate()``.
    """
    if config_cls is None:
        config_cls = import_module("config").Config

    fixture = get_deployment_mode_fixture(mode)

    sentinel = object()
    previous: dict[str, Any] = {
        name: getattr(config_cls, name, sentinel) for name in fixture.overrides
    }

    for name, value in fixture.overrides.items():
        setattr(config_cls, name, value)

    try:
        if validate:
            try:
                config_cls.validate(require_onchain=fixture.require_onchain)
            except Exception as exc:  # noqa: BLE001 - re-raised as typed error below
                raise DeploymentModeValidationError(fixture.mode, exc) from exc
        yield fixture
    finally:
        for name, value in previous.items():
            if value is sentinel:
                delattr(config_cls, name)
            else:
                setattr(config_cls, name, value)
