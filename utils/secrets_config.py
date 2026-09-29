"""Secrets-safe configuration wrapper for config.py.

This module provides a backward-compatible wrapper that uses the secrets
manager for sensitive configuration values while maintaining the existing
Config interface.

Usage
-----
In application code, import config as usual::

    from config import config

The Config class automatically uses SecretsManager for sensitive values.

For explicit secrets management::

    from utils.secrets_config import get_secret, rotate_secret

    submitter_secret = get_secret("LEDGERLENS_SUBMITTER_SECRET", required=True)
    rotate_secret("LEDGERLENS_SUBMITTER_SECRET", new_value)

Migration
---------
This wrapper provides a migration path from direct os.getenv() calls to
the secrets manager without breaking existing code. The Config class
delegates to SecretsManager for sensitive values automatically.
"""

from __future__ import annotations

from utils.secrets_manager import (
    SecretNotFoundError,
    SecretType,
    SecretValidationError,
    get_secrets_manager,
    register_ledgerlens_secrets,
)

# Initialize and configure the global secrets manager
_secrets_manager = get_secrets_manager()
register_ledgerlens_secrets(_secrets_manager)


# Map of config attribute names to their secret types
_SECRET_ATTRIBUTES = {
    "LEDGERLENS_SUBMITTER_SECRET": SecretType.STELLAR_SECRET,
    "KAFKA_SASL_PASSWORD": SecretType.PASSWORD,
    "KAFKA_SASL_USERNAME": SecretType.RAW,  # Username is not sensitive
    "MODEL_SIGNING_PRIVATE_KEY_PATH": SecretType.FILEPATH,
    "ANNOTATION_HMAC_SECRET": SecretType.HMAC_SECRET,
    "AUDIT_VERIFY_PUBLIC_KEY_PATH": SecretType.FILEPATH,
    "JWT_PUBLIC_KEY_PATH": SecretType.FILEPATH,
    "FORENSIC_REPORT_ENCRYPTION_KEY": SecretType.HMAC_SECRET,
    "MODEL_WATERMARK_KEY": SecretType.HMAC_SECRET,
}


def get_secret(
    name: str,
    secret_type: SecretType | None = None,
    required: bool = False,
    default: str | None = None,
) -> str | None:
    """Get a secret value using the secrets manager.

    Args:
        name: Secret name (environment variable name)
        secret_type: Type of secret (auto-detected from _SECRET_ATTRIBUTES if None)
        required: Whether the secret is required
        default: Default value if not found and not required

    Returns:
        Secret value or None/default if not found

    Raises:
        SecretNotFoundError: If required secret is missing
        SecretValidationError: If secret validation fails
    """
    # Auto-detect secret type if not provided
    if secret_type is None:
        secret_type = _SECRET_ATTRIBUTES.get(name, SecretType.RAW)

    try:
        return _secrets_manager.get_secret(
            name, secret_type=secret_type, required=required, default=default
        )
    except SecretNotFoundError:
        if required:
            raise
        return default


def rotate_secret(name: str, new_value: str, new_version: int | None = None) -> None:
    """Rotate a secret to a new value.

    Args:
        name: Secret name
        new_value: New secret value
        new_version: Optional version number (auto-increments if None)

    Raises:
        SecretRotationError: If rotation fails
        SecretValidationError: If new value is invalid
    """
    _secrets_manager.rotate_secret(name, new_value, new_version)


def verify_secrets() -> dict[str, str | None]:
    """Verify all registered secrets are present and valid.

    Returns:
        Dictionary mapping secret names to error messages (None if valid)
    """
    return _secrets_manager.verify_all_secrets()


def is_secret_configured(name: str) -> bool:
    """Check if a secret is configured (without accessing it).

    Args:
        name: Secret name

    Returns:
        True if secret exists and is non-empty
    """
    try:
        value = get_secret(name, required=False)
        return value is not None and value != ""
    except (SecretNotFoundError, SecretValidationError):
        return False
