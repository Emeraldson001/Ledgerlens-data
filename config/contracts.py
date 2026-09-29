"""Environment configuration contracts, one per LedgerLens runtime mode.

`config.py` centralizes ~150 environment variables, but historically only
`run_pipeline.py` called `Config.validate()` — and it called it *before*
parsing `--submit-onchain`, so the on-chain-only vars (`LEDGERLENS_CONTRACT_ID`,
`LEDGERLENS_SUBMITTER_SECRET`) were never actually enforced. Every other entry
point (the REST API, the SSE/Kafka streaming pipelines, the WebSocket server,
training scripts) read `config.*` directly with no startup check at all, so a
missing var surfaced as a `KeyError`/`FileNotFoundError`/auth-that-always-fails
deep inside a request handler or a background thread — often only when a
client happened to exercise that code path.

This module is the fix: one `ModeContract` per runtime mode, each a small list
of checks against the *effective* configuration for that mode (env vars plus
any CLI-provided context, e.g. `--alert-channel`). `validate_mode()` collects
every violation and raises a single `OSError` describing all of them, mirroring
the "collect all errors, then raise once" shape of `Config.validate()` — so
one run tells a maintainer everything that's wrong, not just the first thing.

Usage:
    from config.contracts import validate_mode

    validate_mode("api")
    validate_mode("streaming_kafka", alert_channel=args.alert_channel,
                   role=args.role, backend=args.backend)

Developer entry point: `python -m scripts.check_env --mode api` (or `--all`)
runs these same contracts against the current environment without starting
the service — see scripts/check_env.py.  Use `--list-modes` to see every
registered mode and its required variables at a glance.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable
from typing import Any

from config import Config

Check = Callable[[type[Config], dict[str, Any]], "str | list[str] | None"]


class ModeContract:
    """A named runtime mode plus the checks its effective config must pass."""

    def __init__(self, mode: str, description: str, checks: list[Check]):
        self.mode = mode
        self.description = description
        self.checks = checks


# ---------------------------------------------------------------------------
# Reusable check builders
# ---------------------------------------------------------------------------


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def require_nonempty(attr: str, reason: str) -> Check:
    """`config.<attr>` must be set to a non-empty value."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        if _is_empty(getattr(cls, attr)):
            return f"{attr} is not set — required {reason}."
        return None

    return check


def require_positive(attr: str, reason: str) -> Check:
    """`config.<attr>` must be a number > 0."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        value = getattr(cls, attr)
        if value is None or value <= 0:
            return f"{attr}={value!r} must be > 0 — required {reason}."
        return None

    return check


def require_range(
    attr: str, low: float, high: float, reason: str, inclusive: bool = False
) -> Check:
    """`config.<attr>` must fall within (low, high), or [low, high] if inclusive."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        value = getattr(cls, attr)
        ok = (low <= value <= high) if inclusive else (low < value < high)
        if not ok:
            bounds = f"[{low}, {high}]" if inclusive else f"({low}, {high})"
            return f"{attr}={value!r} must be in {bounds} — required {reason}."
        return None

    return check


def require_choice(attr: str, choices: tuple[str, ...], reason: str) -> Check:
    """`config.<attr>` must be one of `choices`."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        value = getattr(cls, attr)
        if value not in choices:
            return f"{attr}={value!r} must be one of {choices} — required {reason}."
        return None

    return check


def require_file_exists(attr: str, reason: str) -> Check:
    """`config.<attr>` must be set and point at an existing file."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        value = getattr(cls, attr)
        if _is_empty(value):
            return f"{attr} is not set — required {reason}."
        if not os.path.isfile(value):
            return f"{attr}={value!r} does not point to an existing file — required {reason}."
        return None

    return check


def require_paired(attr_a: str, attr_b: str, reason: str) -> Check:
    """If either `config.<attr_a>` or `config.<attr_b>` is set, both must be."""

    def check(cls: type[Config], ctx: dict[str, Any]) -> str | None:
        a, b = getattr(cls, attr_a), getattr(cls, attr_b)
        if _is_empty(a) != _is_empty(b):
            set_one, unset_one = (attr_a, attr_b) if not _is_empty(a) else (attr_b, attr_a)
            return f"{set_one} is set but {unset_one} is not — {reason} require both or neither."
        return None

    return check


def core_pipeline_errors(
    require_onchain: bool,
) -> Callable[[type[Config], dict[str, Any]], list[str]]:
    """Wrap `Config._core_errors`, the baseline checks shared by both pipeline modes.

    Returns a list (possibly empty) rather than a single `str | None` like the
    other check builders, so each underlying error keeps its own bullet point
    in `validate_mode()`'s output instead of being squashed onto one line.
    """

    def check(cls: type[Config], ctx: dict[str, Any]) -> list[str]:
        return cls._core_errors(require_onchain=require_onchain)

    return check


# ---------------------------------------------------------------------------
# Mode-specific context-aware checks
# ---------------------------------------------------------------------------
# Some requirements depend on a CLI flag rather than an env var directly
# (e.g. scripts/stream.py's --alert-channel/--role/--backend can override
# config.ALERT_CHANNEL/STREAMING_BACKEND for that process only). Callers pass
# these as keyword context to validate_mode(); each check below falls back to
# the corresponding Config attribute when the context key isn't supplied, so
# `check_env.py` can validate the configured defaults with no context at all.


def _alert_channel_needs_webhook_url(cls: type[Config], ctx: dict[str, Any]) -> str | None:
    channel = ctx.get("alert_channel", cls.ALERT_CHANNEL)
    if channel == "webhook" and _is_empty(cls.ALERT_WEBHOOK_URL):
        return "ALERT_WEBHOOK_URL is not set — required when the alert channel is 'webhook'."
    return None


def _watched_pairs_needed_unless_kafka_worker(cls: type[Config], ctx: dict[str, Any]) -> str | None:
    # scripts/stream.py: a Kafka *worker* discovers topics dynamically and
    # never reads WATCHED_ASSET_PAIRS; every other role/backend combination
    # (producer, SSE, "all") needs it to know what to stream.
    role = ctx.get("role", "all")
    if role == "worker":
        return None
    if _is_empty(cls.WATCHED_ASSET_PAIRS):
        return "WATCHED_ASSET_PAIRS is not set — required to know which pairs to stream."
    return None


def _effective_model_dir_nonempty(cls: type[Config], ctx: dict[str, Any]) -> str | None:
    # detection/model_training.py: `--model-dir` overrides config.MODEL_DIR for
    # that run, so a bare MODEL_DIR env check would false-positive when a
    # caller only ever passes --model-dir on the CLI.
    model_dir = ctx.get("model_dir") or cls.MODEL_DIR
    if _is_empty(model_dir):
        return "MODEL_DIR is not set (and no --model-dir override was given) — trained artifacts need somewhere to write."
    return None


def _model_dir_needed_unless_kafka_producer(cls: type[Config], ctx: dict[str, Any]) -> str | None:
    # scripts/stream.py: a pure Kafka producer (backend=kafka, role=producer)
    # only relays trades onto a topic and never loads a scorer/model.
    backend = ctx.get("backend", cls.STREAMING_BACKEND)
    role = ctx.get("role", "all")
    if backend == "kafka" and role == "producer":
        return None
    if _is_empty(cls.MODEL_DIR):
        return "MODEL_DIR is not set — the streaming scorer needs trained models to load."
    return None


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

_CONTRACTS: dict[str, ModeContract] = {
    "pipeline": ModeContract(
        mode="pipeline",
        description="python run_pipeline.py (no --submit-onchain)",
        checks=[core_pipeline_errors(require_onchain=False)],
    ),
    "pipeline_onchain": ModeContract(
        mode="pipeline_onchain",
        description="python run_pipeline.py --submit-onchain",
        checks=[
            core_pipeline_errors(require_onchain=True),
            require_nonempty("SOROBAN_RPC_URL", "to reach the Soroban RPC endpoint for submission"),
            require_choice(
                "STELLAR_NETWORK", ("PUBLIC", "TESTNET"), "to select the correct network passphrase"
            ),
        ],
    ),
    "api": ModeContract(
        mode="api",
        description="uvicorn api.app:app",
        checks=[
            require_nonempty("RISK_SCORE_DB_URL", "to read persisted risk scores"),
            require_nonempty("MODEL_DIR", "to load SHAP explainers for /latest"),
            require_nonempty(
                "API_KEYS",
                "otherwise every request gets 401 Unauthorized — set at least one "
                "bcrypt-hashed key (comma-separated)",
            ),
            require_positive("API_RATE_LIMIT_RPM", "to configure the per-key rate limiter"),
        ],
    ),
    "streaming_sse": ModeContract(
        mode="streaming_sse",
        description="python -m scripts.stream --backend sse",
        checks=[
            _watched_pairs_needed_unless_kafka_worker,
            _alert_channel_needs_webhook_url,
            require_nonempty("MODEL_DIR", "the streaming scorer needs trained models to load"),
        ],
    ),
    "streaming_kafka": ModeContract(
        mode="streaming_kafka",
        description="python -m scripts.stream --backend kafka (or scripts.kafka_workers)",
        checks=[
            _watched_pairs_needed_unless_kafka_worker,
            _alert_channel_needs_webhook_url,
            _model_dir_needed_unless_kafka_producer,
            require_nonempty("KAFKA_BOOTSTRAP_SERVERS", "to connect to the Kafka cluster"),
            require_file_exists(
                "TRADE_AVRO_SCHEMA_PATH", "to encode/decode trade events on the wire"
            ),
            require_paired(
                "KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD", "Kafka SASL authentication"
            ),
        ],
    ),
    "ws_server": ModeContract(
        mode="ws_server",
        description="streaming.ws_server.start_ws_server_thread(...)",
        checks=[
            require_file_exists(
                "JWT_PUBLIC_KEY_PATH",
                "to verify client JWTs — without it every connection fails auth at "
                "first use instead of at startup",
            ),
            require_positive("WS_MAX_CLIENTS", "to bound concurrent connections"),
        ],
    ),
    "training": ModeContract(
        mode="training",
        description="detection/model_training.py, training/train.py",
        checks=[
            _effective_model_dir_nonempty,
            require_range("CALIBRATION_SPLIT", 0.0, 1.0, "to hold out a calibration set"),
            require_positive("DP_TARGET_EPSILON", "DP-SGD training budget"),
            require_range("DP_TARGET_DELTA", 0.0, 1.0, "DP-SGD training budget"),
        ],
    ),
}

RUNTIME_MODES: tuple[str, ...] = tuple(_CONTRACTS)

_CORE_PIPELINE_ATTRS: list[str] = [
    "WATCHED_ASSET_PAIRS",
    "RISK_SCORE_DB_URL",
    "MODEL_DIR",
    "DP_AGGREGATOR_EPSILON",
    "DP_AGGREGATOR_DELTA",
]

_CORE_PIPELINE_ATTRS_ONCHAIN: list[str] = [
    "LEDGERLENS_CONTRACT_ID",
    "LEDGERLENS_SUBMITTER_SECRET",
]


def _check_attrs(check: Check) -> list[str]:
    """Return the config attribute names a single check validates."""
    attrs: list[str] = []
    # Builder-generated closures capture the attr name(s) in closure cells.
    if hasattr(check, "__closure__") and check.__closure__:
        for cell in check.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, str) and val.isupper() and "_" in val:
                attrs.append(val)
    if attrs:
        return sorted(set(attrs))
    # Named functions reference cls.ATTR_NAME in their body.
    try:
        src = inspect.getsource(check)
        attrs = re.findall(r"cls\.([A-Z_][A-Z_0-9]*)", src)
    except (OSError, TypeError):
        pass
    return sorted(set(attrs))


def list_modes(
    config_cls: type[Config] = Config,
) -> list[dict[str, Any]]:
    """Return a structured description of every registered runtime mode.

    Each dict contains:

    * ``mode`` — the mode name (e.g. ``"api"``).
    * ``description`` — human-readable entry-point hint.
    * ``required_attrs`` — sorted list of config attributes the mode requires.
    * ``conditional_attrs`` — attrs required only in certain contexts.
    """
    result = []
    for name, contract in _CONTRACTS.items():
        required: list[str] = []
        conditional: list[str] = []
        for chk in contract.checks:
            src = ""
            try:
                src = inspect.getsource(chk)
            except (OSError, TypeError):
                pass
            if "_core_errors" in src:
                required.extend(_CORE_PIPELINE_ATTRS)
                require_onchain = False
                if hasattr(chk, "__closure__") and chk.__closure__:
                    for cell in chk.__closure__:
                        try:
                            val = cell.cell_contents
                        except ValueError:
                            continue
                        if isinstance(val, bool):
                            require_onchain = val
                            break
                if require_onchain:
                    conditional.extend(_CORE_PIPELINE_ATTRS_ONCHAIN)
            else:
                required.extend(_check_attrs(chk))
        required = sorted(set(required))
        conditional = sorted(set(conditional))
        # Remove conditionals from required if present
        required = [a for a in required if a not in conditional]
        result.append(
            {
                "mode": name,
                "description": contract.description,
                "required_attrs": required,
                "conditional_attrs": conditional,
            }
        )
    return result


def validate_mode(mode: str, *, config_cls: type[Config] = Config, **context: Any) -> None:
    """Raise `OSError` listing every violation of `mode`'s config contract.

    `context` supplies values a caller only knows at runtime (CLI flags that
    override env-derived config for that process), e.g.:
        validate_mode("streaming_kafka", alert_channel=args.alert_channel,
                       role=args.role, backend=args.backend)
    Any key a check needs but that isn't supplied falls back to the matching
    `Config` attribute, so `validate_mode(mode)` alone validates the
    configured defaults.
    """
    contract = _CONTRACTS.get(mode)
    if contract is None:
        raise ValueError(f"Unknown runtime mode {mode!r}. Known modes: {sorted(RUNTIME_MODES)}")

    errors: list[str] = []
    for check in contract.checks:
        result = check(config_cls, context)
        if isinstance(result, list):
            errors.extend(result)
        elif result:
            errors.append(result)

    if errors:
        raise OSError(
            f"LedgerLens configuration errors for mode={mode!r} ({contract.description}):\n- "
            + "\n- ".join(errors)
        )
