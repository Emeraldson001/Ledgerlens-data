"""Secrets-safe configuration handling for integrations.

This module provides a secure, auditable, and rotation-ready secrets management
system for the LedgerLens data pipeline. It replaces direct `os.getenv()` calls
for sensitive credentials with a typed, validated provider layer.

Architecture
------------
- **SecretProvider protocol**: Abstract interface for secret sources (env vars,
  files, external vaults)
- **Validation layer**: Pattern-based validators for each secret type (Stellar
  keys, HMAC secrets, API keys) with strength checking
- **Audit trail**: Tamper-evident HMAC-signed log of all secret access events
- **Rotation support**: Version tracking, graceful fallback during rotation,
  rollback capability

Usage
-----
Replace direct environment variable access::

    # OLD (insecure)
    submitter_secret = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

    # NEW (secure)
    from utils.secrets_manager import get_secrets_manager
    mgr = get_secrets_manager()
    submitter_secret = mgr.get_secret("LEDGERLENS_SUBMITTER_SECRET")

The manager automatically validates the secret format, logs the access event,
and raises `SecretValidationError` if the secret is malformed or missing when
required.

Secret Types
------------
- `stellar_secret`: Stellar secret key (S... format, base32, 56 chars)
- `hmac_secret`: HMAC key (minimum 32 hex characters for 128-bit security)
- `api_key`: Generic API key (minimum 32 characters)
- `password`: Generic password (minimum 16 characters, complexity checks)
- `filepath`: Path to a secret file (existence validation)
- `raw`: No validation (use sparingly)

Migration
---------
See docs/secrets_management.md for step-by-step migration guide and examples.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecretError(Exception):
    """Base exception for secrets management errors."""


class SecretNotFoundError(SecretError):
    """Raised when a required secret is not found."""


class SecretValidationError(SecretError):
    """Raised when a secret fails validation (malformed, too weak, etc.)."""


class SecretRotationError(SecretError):
    """Raised when secret rotation fails."""


# ---------------------------------------------------------------------------
# Secret Types and Validation
# ---------------------------------------------------------------------------


class SecretType(StrEnum):
    """Enum of supported secret types with their validation rules."""

    STELLAR_SECRET = "stellar_secret"
    HMAC_SECRET = "hmac_secret"
    API_KEY = "api_key"
    PASSWORD = "password"
    FILEPATH = "filepath"
    RAW = "raw"


@dataclass
class SecretMetadata:
    """Metadata about a secret access event for audit logging."""

    secret_name: str
    secret_type: SecretType
    accessed_at: datetime
    caller_module: str
    caller_function: str
    version: int = 1
    redacted_value_hash: str | None = None  # SHA-256 of the secret value


@dataclass
class SecretDefinition:
    """Configuration for a managed secret."""

    name: str
    secret_type: SecretType
    required: bool = True
    default: str | None = None
    env_var: str | None = None  # Override environment variable name
    description: str = ""
    min_length: int | None = None
    allow_rotation: bool = True
    current_version: int = 1


class SecretValidator:
    """Validates secrets according to their type and security requirements."""

    # Stellar secret keys are base32, always start with 'S', exactly 56 chars
    STELLAR_SECRET_PATTERN = re.compile(r"^S[A-Z2-7]{55}$")

    # HMAC secrets should be hex-encoded, minimum 32 chars (128-bit)
    HMAC_SECRET_PATTERN = re.compile(r"^[0-9a-fA-F]{32,}$")

    # API keys should be alphanumeric + common symbols, minimum 32 chars
    API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{32,}$")

    @classmethod
    def validate(cls, value: str, secret_type: SecretType, min_length: int | None = None) -> None:
        """Validate a secret value according to its type.

        Args:
            value: The secret value to validate
            secret_type: Type of secret (determines validation rules)
            min_length: Optional minimum length override

        Raises:
            SecretValidationError: If validation fails
        """
        if not value:
            raise SecretValidationError("Secret value is empty")

        if secret_type == SecretType.RAW:
            return  # No validation for raw secrets

        # Check minimum length
        effective_min_length = min_length or cls._get_default_min_length(secret_type)
        if len(value) < effective_min_length:
            raise SecretValidationError(
                f"Secret is too short: {len(value)} chars, "
                f"minimum {effective_min_length} required for {secret_type.value}"
            )

        # Type-specific validation
        if secret_type == SecretType.STELLAR_SECRET:
            cls._validate_stellar_secret(value)
        elif secret_type == SecretType.HMAC_SECRET:
            cls._validate_hmac_secret(value)
        elif secret_type == SecretType.API_KEY:
            cls._validate_api_key(value)
        elif secret_type == SecretType.PASSWORD:
            cls._validate_password(value)
        elif secret_type == SecretType.FILEPATH:
            cls._validate_filepath(value)

    @staticmethod
    def _get_default_min_length(secret_type: SecretType) -> int:
        """Get default minimum length for a secret type."""
        return {
            SecretType.STELLAR_SECRET: 56,
            SecretType.HMAC_SECRET: 32,
            SecretType.API_KEY: 32,
            SecretType.PASSWORD: 16,
            SecretType.FILEPATH: 1,
            SecretType.RAW: 0,
        }[secret_type]

    @classmethod
    def _validate_stellar_secret(cls, value: str) -> None:
        """Validate Stellar secret key format."""
        if not cls.STELLAR_SECRET_PATTERN.match(value):
            raise SecretValidationError(
                "Invalid Stellar secret key format. Must be base32, start with 'S', "
                "exactly 56 characters. Never commit Stellar secrets to version control."
            )

    @classmethod
    def _validate_hmac_secret(cls, value: str) -> None:
        """Validate HMAC secret format and strength."""
        if not cls.HMAC_SECRET_PATTERN.match(value):
            raise SecretValidationError(
                "Invalid HMAC secret format. Must be hex-encoded (0-9, a-f), "
                "minimum 32 characters (128-bit). Generate with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )

    @classmethod
    def _validate_api_key(cls, value: str) -> None:
        """Validate API key format."""
        if not cls.API_KEY_PATTERN.match(value):
            raise SecretValidationError(
                "Invalid API key format. Must be alphanumeric with _-. characters, "
                "minimum 32 characters."
            )

    @classmethod
    def _validate_password(cls, value: str) -> None:
        """Validate password strength."""
        if len(value) < 16:
            raise SecretValidationError("Password must be at least 16 characters")

        # Check for basic complexity
        has_upper = any(c.isupper() for c in value)
        has_lower = any(c.islower() for c in value)
        has_digit = any(c.isdigit() for c in value)
        has_special = any(not c.isalnum() for c in value)

        if sum([has_upper, has_lower, has_digit, has_special]) < 3:
            raise SecretValidationError(
                "Password must contain at least 3 of: uppercase, lowercase, "
                "digits, special characters"
            )

    @classmethod
    def _validate_filepath(cls, value: str) -> None:
        """Validate that a filepath exists."""
        path = Path(value)
        if not path.exists():
            raise SecretValidationError("Secret file does not exist")
        if not path.is_file():
            raise SecretValidationError("Secret path is not a file")


# ---------------------------------------------------------------------------
# Secret Providers
# ---------------------------------------------------------------------------


class SecretProvider(ABC):
    """Abstract interface for secret sources."""

    @abstractmethod
    def get(self, name: str) -> str | None:
        """Retrieve a secret by name. Returns None if not found."""
        pass

    @abstractmethod
    def set(self, name: str, value: str, version: int = 1) -> None:
        """Store a secret. Used for rotation."""
        pass

    @abstractmethod
    def list_versions(self, name: str) -> list[int]:
        """List available versions of a secret."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Reads secrets from environment variables (default provider)."""

    def get(self, name: str) -> str | None:
        return os.getenv(name)

    def set(self, name: str, value: str, version: int = 1) -> None:
        """Environment provider doesn't support setting (read-only)."""
        raise NotImplementedError(
            "EnvironmentSecretProvider is read-only. Use FileSecretProvider "
            "or an external secrets manager for rotation support."
        )

    def list_versions(self, name: str) -> list[int]:
        """Environment provider only has current version."""
        return [1] if self.get(name) is not None else []


class FileSecretProvider(SecretProvider):
    """Reads secrets from individual files in a directory.

    Each secret is stored in a file named after the secret key.
    Supports versioning through filename suffixes (e.g., SECRET.v2).
    """

    def __init__(self, secrets_dir: Path):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

    def get(self, name: str) -> str | None:
        """Get the latest version of a secret."""
        versions = self.list_versions(name)
        if not versions:
            return None

        latest_version = max(versions)
        return self._read_version(name, latest_version)

    def set(self, name: str, value: str, version: int = 1) -> None:
        """Store a secret at a specific version."""
        if version == 1:
            filepath = self.secrets_dir / name
        else:
            filepath = self.secrets_dir / f"{name}.v{version}"

        # Write with restrictive permissions (owner read/write only)
        filepath.write_text(value)
        filepath.chmod(0o600)

    def list_versions(self, name: str) -> list[int]:
        """List all available versions of a secret."""
        versions = []

        # Check for base file (version 1)
        base_path = self.secrets_dir / name
        if base_path.exists():
            versions.append(1)

        # Check for versioned files
        for path in self.secrets_dir.glob(f"{name}.v*"):
            version_str = path.name.split(".v")[-1]
            try:
                versions.append(int(version_str))
            except ValueError:
                continue

        return sorted(versions)

    def _read_version(self, name: str, version: int) -> str:
        """Read a specific version of a secret."""
        if version == 1:
            filepath = self.secrets_dir / name
        else:
            filepath = self.secrets_dir / f"{name}.v{version}"

        return filepath.read_text().strip()


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------


class SecretAuditLogger:
    """Logs secret access events with HMAC-signed tamper-evident trail."""

    def __init__(self, log_path: Path, hmac_key: str | None = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.hmac_key = (hmac_key or os.getenv("SECRETS_AUDIT_HMAC_KEY", "")).encode()

        if not self.hmac_key:
            warnings.warn(
                "SECRETS_AUDIT_HMAC_KEY not set - audit log will not be tamper-evident",
                stacklevel=2,
            )

    def log_access(self, metadata: SecretMetadata) -> None:
        """Log a secret access event with HMAC signature."""
        # Create log entry
        entry = {
            "timestamp": metadata.accessed_at.isoformat(),
            "secret_name": metadata.secret_name,
            "secret_type": metadata.secret_type.value,
            "caller_module": metadata.caller_module,
            "caller_function": metadata.caller_function,
            "version": metadata.version,
            "redacted_value_hash": metadata.redacted_value_hash,
        }

        # Add HMAC if key is available
        if self.hmac_key:
            entry_str = "|".join(str(v) for v in entry.values())
            signature = hmac_lib.new(self.hmac_key, entry_str.encode(), hashlib.sha256).hexdigest()
            entry["hmac_sha256"] = signature

        # Append to log file (NDJSON format)
        import json

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def verify_log_integrity(self) -> tuple[int, int]:
        """Verify HMAC signatures in the audit log.

        Returns:
            Tuple of (valid_entries, invalid_entries) counts
        """
        if not self.hmac_key:
            raise SecretError("Cannot verify log integrity without HMAC key")

        import json

        valid_count = 0
        invalid_count = 0

        with open(self.log_path) as f:
            for line in f:
                entry = json.loads(line)
                if "hmac_sha256" not in entry:
                    invalid_count += 1
                    continue

                stored_hmac = entry.pop("hmac_sha256")
                entry_str = "|".join(str(v) for v in entry.values())
                computed_hmac = hmac_lib.new(
                    self.hmac_key, entry_str.encode(), hashlib.sha256
                ).hexdigest()

                if hmac_lib.compare_digest(stored_hmac, computed_hmac):
                    valid_count += 1
                else:
                    invalid_count += 1

        return valid_count, invalid_count


# ---------------------------------------------------------------------------
# Secrets Manager (Main Interface)
# ---------------------------------------------------------------------------


class SecretsManager:
    """Central secrets management with validation, audit, and rotation support.

    This is the primary interface that application code should use to access
    secrets. It coordinates the provider, validator, and audit logger.
    """

    def __init__(
        self,
        provider: SecretProvider | None = None,
        audit_logger: SecretAuditLogger | None = None,
        enable_validation: bool = True,
    ):
        self.provider = provider or EnvironmentSecretProvider()
        self.audit_logger = audit_logger
        self.enable_validation = enable_validation
        self._definitions: dict[str, SecretDefinition] = {}

    def register_secret(self, definition: SecretDefinition) -> None:
        """Register a secret definition with the manager."""
        self._definitions[definition.name] = definition

    def get_secret(
        self,
        name: str,
        secret_type: SecretType = SecretType.RAW,
        required: bool = True,
        default: str | None = None,
    ) -> str | None:
        """Get a secret value with validation and audit logging.

        Args:
            name: Secret name (usually environment variable name)
            secret_type: Type of secret for validation
            required: Whether the secret is required (raises if missing)
            default: Default value if secret not found (only used if not required)

        Returns:
            Secret value, or None if not required and not found

        Raises:
            SecretNotFoundError: If required secret is not found
            SecretValidationError: If secret validation fails
        """
        # Get from registered definition if available
        definition = self._definitions.get(name)
        if definition:
            secret_type = definition.secret_type
            required = definition.required
            default = definition.default
            name = definition.env_var or name

        # Retrieve from provider
        value = self.provider.get(name)

        # Handle missing secrets
        if value is None or value == "":
            if required:
                raise SecretNotFoundError(
                    f"Required secret '{name}' not found. "
                    f"Set the {name} environment variable or configure a file-based provider."
                )
            return default

        # Validate if enabled
        if self.enable_validation:
            min_length = definition.min_length if definition else None
            SecretValidator.validate(value, secret_type, min_length)

        # Audit log the access
        if self.audit_logger:
            import inspect

            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None
            metadata = SecretMetadata(
                secret_name=name,
                secret_type=secret_type,
                accessed_at=datetime.utcnow(),
                caller_module=caller_frame.f_globals["__name__"] if caller_frame else "unknown",
                caller_function=caller_frame.f_code.co_name if caller_frame else "unknown",
                version=definition.current_version if definition else 1,
                redacted_value_hash=hashlib.sha256(value.encode()).hexdigest()[:16],
            )
            self.audit_logger.log_access(metadata)

        return value

    def rotate_secret(self, name: str, new_value: str, new_version: int | None = None) -> None:
        """Rotate a secret to a new value.

        Args:
            name: Secret name
            new_value: New secret value
            new_version: Version number (auto-increments if not provided)

        Raises:
            SecretRotationError: If rotation is not allowed or fails
        """
        definition = self._definitions.get(name)
        if definition and not definition.allow_rotation:
            raise SecretRotationError(f"Secret '{name}' is not configured to allow rotation")

        # Determine new version
        if new_version is None:
            existing_versions = self.provider.list_versions(name)
            new_version = max(existing_versions, default=0) + 1

        # Validate new value
        secret_type = definition.secret_type if definition else SecretType.RAW
        if self.enable_validation:
            min_length = definition.min_length if definition else None
            SecretValidator.validate(new_value, secret_type, min_length)

        # Store new version
        try:
            self.provider.set(name, new_value, new_version)
        except NotImplementedError as exc:
            raise SecretRotationError(
                f"Provider {type(self.provider).__name__} does not support rotation"
            ) from exc

        # Update definition version
        if definition:
            definition.current_version = new_version

        logger.info(
            f"Successfully rotated secret '{name}' to version {new_version}",
            extra={"secret_name": name, "new_version": new_version},
        )

    def verify_all_secrets(self) -> dict[str, str | None]:
        """Verify all registered secrets are present and valid.

        Returns:
            Dictionary mapping secret names to error messages (None if valid)
        """
        results = {}
        for name, _definition in self._definitions.items():
            try:
                self.get_secret(name)
                results[name] = None
            except (SecretNotFoundError, SecretValidationError) as e:
                results[name] = str(e)
        return results


# ---------------------------------------------------------------------------
# Global Instance and Helper
# ---------------------------------------------------------------------------


_global_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = _create_default_manager()
    return _global_manager


def _create_default_manager() -> SecretsManager:
    """Create default secrets manager with configuration from environment."""
    # Determine provider
    secrets_dir = os.getenv("SECRETS_DIR")
    if secrets_dir:
        provider = FileSecretProvider(Path(secrets_dir))
        logger.info(f"Using FileSecretProvider with directory: {secrets_dir}")
    else:
        provider = EnvironmentSecretProvider()
        logger.info("Using EnvironmentSecretProvider (default)")

    # Setup audit logging
    audit_log_path = os.getenv("SECRETS_AUDIT_LOG", "data/secrets_audit.ndjson")
    audit_hmac_key = os.getenv("SECRETS_AUDIT_HMAC_KEY")
    audit_logger = SecretAuditLogger(Path(audit_log_path), audit_hmac_key)

    # Create manager
    enable_validation = os.getenv("SECRETS_VALIDATION_ENABLED", "true").lower() == "true"
    return SecretsManager(provider, audit_logger, enable_validation)


def configure_secrets_manager(manager: SecretsManager) -> None:
    """Set the global secrets manager instance (for testing/custom configs)."""
    global _global_manager
    _global_manager = manager


# ---------------------------------------------------------------------------
# Common Secret Definitions (for registration)
# ---------------------------------------------------------------------------


def register_ledgerlens_secrets(manager: SecretsManager) -> None:
    """Register all LedgerLens secrets with the manager."""

    secrets = [
        SecretDefinition(
            name="LEDGERLENS_SUBMITTER_SECRET",
            secret_type=SecretType.STELLAR_SECRET,
            required=False,  # Only required for on-chain submission
            description="Stellar secret key for submitting scores to Soroban contract",
        ),
        SecretDefinition(
            name="KAFKA_SASL_PASSWORD",
            secret_type=SecretType.PASSWORD,
            required=False,  # Only required when Kafka SASL is enabled
            description="Kafka SASL authentication password",
        ),
        SecretDefinition(
            name="MODEL_SIGNING_PRIVATE_KEY_PATH",
            secret_type=SecretType.FILEPATH,
            required=False,  # Only required for model signing
            description="Path to Ed25519 private key for model artifact signing",
        ),
        SecretDefinition(
            name="ANNOTATION_HMAC_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            required=False,  # Only required for annotation queue
            description="HMAC key for annotation queue integrity verification",
        ),
        SecretDefinition(
            name="FORENSIC_REPORT_ENCRYPTION_KEY",
            secret_type=SecretType.HMAC_SECRET,
            required=False,  # Only required for encrypted reports
            description="AES-256 key for forensic report field encryption",
            min_length=64,  # 256 bits = 64 hex chars
        ),
        SecretDefinition(
            name="OPENAI_API_KEY",
            secret_type=SecretType.API_KEY,
            required=False,  # Only required for narrative generation
            description="OpenAI API key for narrative generation",
        ),
        SecretDefinition(
            name="ANTHROPIC_API_KEY",
            secret_type=SecretType.API_KEY,
            required=False,  # Only required for narrative generation
            description="Anthropic API key for narrative generation",
        ),
        SecretDefinition(
            name="FEDERATED_CA_KEY_PEM",
            secret_type=SecretType.RAW,  # PEM format, validated separately
            required=False,  # Only required for federated cert management
            description="CA private key in PEM format for federated learning",
        ),
        SecretDefinition(
            name="EVENT_HMAC_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            required=False,  # Only required for event signing
            description="HMAC key for Soroban event signature verification",
        ),
    ]

    for secret_def in secrets:
        manager.register_secret(secret_def)
