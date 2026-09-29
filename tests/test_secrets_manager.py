"""Comprehensive test suite for secrets management system.

Tests cover:
- Secret provider implementations (environment, file-based)
- Validation for each secret type
- Audit trail with HMAC integrity
- Secret rotation and versioning
- Error handling and edge cases
- Integration with SecretsManager
"""

import json
from datetime import datetime

import pytest

from utils.secrets_manager import (
    EnvironmentSecretProvider,
    FileSecretProvider,
    SecretAuditLogger,
    SecretDefinition,
    SecretError,
    SecretNotFoundError,
    SecretRotationError,
    SecretsManager,
    SecretType,
    SecretValidationError,
    SecretValidator,
    configure_secrets_manager,
    get_secrets_manager,
    register_ledgerlens_secrets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_secrets_dir(tmp_path):
    """Provide a temporary directory for file-based secrets."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    return secrets_dir


@pytest.fixture
def temp_audit_log(tmp_path):
    """Provide a temporary audit log file."""
    return tmp_path / "secrets_audit.ndjson"


@pytest.fixture
def hmac_key():
    """Provide a test HMAC key."""
    return "test_hmac_key_for_audit_trail_integrity_checking"


@pytest.fixture
def file_provider(temp_secrets_dir):
    """Provide a FileSecretProvider instance."""
    return FileSecretProvider(temp_secrets_dir)


@pytest.fixture
def audit_logger(temp_audit_log, hmac_key):
    """Provide a SecretAuditLogger instance."""
    return SecretAuditLogger(temp_audit_log, hmac_key)


@pytest.fixture
def secrets_manager(file_provider, audit_logger):
    """Provide a SecretsManager instance."""
    return SecretsManager(file_provider, audit_logger, enable_validation=True)


# ---------------------------------------------------------------------------
# SecretValidator Tests
# ---------------------------------------------------------------------------


class TestSecretValidator:
    """Test secret validation for different secret types."""

    def test_validate_stellar_secret_valid(self):
        """Valid Stellar secret key passes validation."""
        valid_secret = "SBZVF2CTUDTHHDKJP3UEKQRC2XLUJMCG3DL5HGJ2YPTPZXC7QCMQW2W3"
        SecretValidator.validate(valid_secret, SecretType.STELLAR_SECRET)

    def test_validate_stellar_secret_invalid_format(self):
        """Invalid Stellar secret format raises ValidationError."""
        with pytest.raises(SecretValidationError, match="Invalid Stellar secret"):
            SecretValidator.validate("not_a_stellar_secret", SecretType.STELLAR_SECRET)

    def test_validate_stellar_secret_wrong_length(self):
        """Stellar secret with wrong length raises ValidationError."""
        with pytest.raises(SecretValidationError, match="too short"):
            SecretValidator.validate("S" + "A" * 50, SecretType.STELLAR_SECRET)

    def test_validate_stellar_secret_wrong_prefix(self):
        """Stellar secret not starting with S raises ValidationError."""
        with pytest.raises(SecretValidationError, match="Invalid Stellar secret"):
            SecretValidator.validate("G" + "A" * 55, SecretType.STELLAR_SECRET)

    def test_validate_hmac_secret_valid(self):
        """Valid HMAC secret passes validation."""
        valid_hmac = "a" * 64  # 256-bit hex
        SecretValidator.validate(valid_hmac, SecretType.HMAC_SECRET)

    def test_validate_hmac_secret_minimum_length(self):
        """HMAC secret with minimum length (32 chars) passes."""
        valid_hmac = "a" * 32  # 128-bit hex
        SecretValidator.validate(valid_hmac, SecretType.HMAC_SECRET)

    def test_validate_hmac_secret_too_short(self):
        """HMAC secret shorter than 32 chars raises ValidationError."""
        with pytest.raises(SecretValidationError, match="too short"):
            SecretValidator.validate("a" * 31, SecretType.HMAC_SECRET)

    def test_validate_hmac_secret_invalid_chars(self):
        """HMAC secret with non-hex chars raises ValidationError."""
        with pytest.raises(SecretValidationError, match="Invalid HMAC secret"):
            SecretValidator.validate("x" * 32, SecretType.HMAC_SECRET)

    def test_validate_api_key_valid(self):
        """Valid API key passes validation."""
        valid_key = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        SecretValidator.validate(valid_key, SecretType.API_KEY)

    def test_validate_api_key_too_short(self):
        """API key shorter than 32 chars raises ValidationError."""
        with pytest.raises(SecretValidationError, match="too short"):
            SecretValidator.validate("short_key", SecretType.API_KEY)

    def test_validate_api_key_invalid_chars(self):
        """API key with invalid chars raises ValidationError."""
        with pytest.raises(SecretValidationError, match="Invalid API key"):
            SecretValidator.validate("a" * 32 + "!@#$", SecretType.API_KEY)

    def test_validate_password_valid(self):
        """Valid password passes validation."""
        valid_password = "MySecure!Password123"
        SecretValidator.validate(valid_password, SecretType.PASSWORD)

    def test_validate_password_too_short(self):
        """Password shorter than 16 chars raises ValidationError."""
        with pytest.raises(SecretValidationError, match="at least 16 characters"):
            SecretValidator.validate("Short1!", SecretType.PASSWORD)

    def test_validate_password_weak_complexity(self):
        """Password without sufficient complexity raises ValidationError."""
        with pytest.raises(SecretValidationError, match="at least 3 of"):
            SecretValidator.validate("a" * 16, SecretType.PASSWORD)

    def test_validate_filepath_exists(self, tmp_path):
        """Valid filepath passes validation."""
        test_file = tmp_path / "test_secret.txt"
        test_file.write_text("secret_content")
        SecretValidator.validate(str(test_file), SecretType.FILEPATH)

    def test_validate_filepath_not_exists(self):
        """Non-existent filepath raises ValidationError."""
        with pytest.raises(SecretValidationError, match="does not exist"):
            SecretValidator.validate("/nonexistent/path", SecretType.FILEPATH)

    def test_validate_filepath_is_directory(self, tmp_path):
        """Directory path raises ValidationError."""
        with pytest.raises(SecretValidationError, match="not a file"):
            SecretValidator.validate(str(tmp_path), SecretType.FILEPATH)

    def test_validate_raw_no_validation(self):
        """Raw secrets pass validation regardless of content."""
        SecretValidator.validate("anything goes!", SecretType.RAW)
        SecretValidator.validate("", SecretType.RAW)  # Even empty

    def test_validate_empty_value_raises(self):
        """Empty value raises ValidationError for non-RAW types."""
        with pytest.raises(SecretValidationError, match="empty"):
            SecretValidator.validate("", SecretType.HMAC_SECRET)


# ---------------------------------------------------------------------------
# EnvironmentSecretProvider Tests
# ---------------------------------------------------------------------------


class TestEnvironmentSecretProvider:
    """Test environment variable secret provider."""

    def test_get_existing_env_var(self, monkeypatch):
        """Get returns value of existing environment variable."""
        monkeypatch.setenv("TEST_SECRET", "test_value")
        provider = EnvironmentSecretProvider()
        assert provider.get("TEST_SECRET") == "test_value"

    def test_get_missing_env_var(self):
        """Get returns None for missing environment variable."""
        provider = EnvironmentSecretProvider()
        assert provider.get("NONEXISTENT_SECRET") is None

    def test_set_raises_not_implemented(self):
        """Set raises NotImplementedError (read-only provider)."""
        provider = EnvironmentSecretProvider()
        with pytest.raises(NotImplementedError, match="read-only"):
            provider.set("TEST_SECRET", "value")

    def test_list_versions_existing(self, monkeypatch):
        """List versions returns [1] for existing secret."""
        monkeypatch.setenv("TEST_SECRET", "test_value")
        provider = EnvironmentSecretProvider()
        assert provider.list_versions("TEST_SECRET") == [1]

    def test_list_versions_missing(self):
        """List versions returns [] for missing secret."""
        provider = EnvironmentSecretProvider()
        assert provider.list_versions("NONEXISTENT_SECRET") == []


# ---------------------------------------------------------------------------
# FileSecretProvider Tests
# ---------------------------------------------------------------------------


class TestFileSecretProvider:
    """Test file-based secret provider."""

    def test_set_and_get_secret(self, file_provider, temp_secrets_dir):
        """Set and get a secret successfully."""
        file_provider.set("TEST_SECRET", "test_value")
        assert file_provider.get("TEST_SECRET") == "test_value"

        # Verify file was created
        secret_file = temp_secrets_dir / "TEST_SECRET"
        assert secret_file.exists()
        assert secret_file.read_text() == "test_value"

    def test_file_permissions(self, file_provider, temp_secrets_dir):
        """Secret files have restrictive permissions (0o600)."""
        file_provider.set("TEST_SECRET", "test_value")
        secret_file = temp_secrets_dir / "TEST_SECRET"

        # Check permissions (owner read/write only)
        stat_info = secret_file.stat()
        assert stat_info.st_mode & 0o777 == 0o600

    def test_get_missing_secret(self, file_provider):
        """Get returns None for missing secret."""
        assert file_provider.get("NONEXISTENT") is None

    def test_versioned_secrets(self, file_provider):
        """Set and get versioned secrets."""
        file_provider.set("TEST_SECRET", "v1_value", version=1)
        file_provider.set("TEST_SECRET", "v2_value", version=2)
        file_provider.set("TEST_SECRET", "v3_value", version=3)

        # Get returns latest version
        assert file_provider.get("TEST_SECRET") == "v3_value"

    def test_list_versions(self, file_provider):
        """List all versions of a secret."""
        file_provider.set("TEST_SECRET", "v1", version=1)
        file_provider.set("TEST_SECRET", "v2", version=2)
        file_provider.set("TEST_SECRET", "v3", version=3)

        versions = file_provider.list_versions("TEST_SECRET")
        assert versions == [1, 2, 3]

    def test_list_versions_empty(self, file_provider):
        """List versions returns [] for non-existent secret."""
        assert file_provider.list_versions("NONEXISTENT") == []

    def test_read_specific_version(self, file_provider):
        """Read a specific version of a secret."""
        file_provider.set("TEST_SECRET", "v1_value", version=1)
        file_provider.set("TEST_SECRET", "v2_value", version=2)

        assert file_provider._read_version("TEST_SECRET", 1) == "v1_value"
        assert file_provider._read_version("TEST_SECRET", 2) == "v2_value"

    def test_whitespace_trimming(self, file_provider):
        """Get strips whitespace from secret values."""
        file_provider.set("TEST_SECRET", "  test_value  \n")
        assert file_provider.get("TEST_SECRET") == "test_value"


# ---------------------------------------------------------------------------
# SecretAuditLogger Tests
# ---------------------------------------------------------------------------


class TestSecretAuditLogger:
    """Test audit logging with HMAC integrity."""

    def test_log_access(self, audit_logger, temp_audit_log):
        """Log a secret access event."""
        from utils.secrets_manager import SecretMetadata

        metadata = SecretMetadata(
            secret_name="TEST_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            accessed_at=datetime(2024, 1, 1, 12, 0, 0),
            caller_module="test_module",
            caller_function="test_function",
            version=1,
            redacted_value_hash="abc123",
        )

        audit_logger.log_access(metadata)

        # Verify log entry was written
        assert temp_audit_log.exists()
        log_content = temp_audit_log.read_text()
        assert "TEST_SECRET" in log_content
        assert "test_module" in log_content

    def test_log_contains_hmac(self, audit_logger, temp_audit_log):
        """Log entry contains HMAC signature."""
        from utils.secrets_manager import SecretMetadata

        metadata = SecretMetadata(
            secret_name="TEST_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            accessed_at=datetime(2024, 1, 1, 12, 0, 0),
            caller_module="test_module",
            caller_function="test_function",
        )

        audit_logger.log_access(metadata)

        with open(temp_audit_log) as f:
            entry = json.loads(f.read())
            assert "hmac_sha256" in entry
            assert len(entry["hmac_sha256"]) == 64  # SHA-256 hex

    def test_verify_log_integrity_valid(self, audit_logger, temp_audit_log):
        """Verify integrity of valid log entries."""
        from utils.secrets_manager import SecretMetadata

        # Log multiple entries
        for i in range(3):
            metadata = SecretMetadata(
                secret_name=f"SECRET_{i}",
                secret_type=SecretType.API_KEY,
                accessed_at=datetime.utcnow(),
                caller_module="test",
                caller_function="test",
            )
            audit_logger.log_access(metadata)

        # Verify all entries are valid
        valid, invalid = audit_logger.verify_log_integrity()
        assert valid == 3
        assert invalid == 0

    def test_verify_log_integrity_tampered(self, audit_logger, temp_audit_log, hmac_key):
        """Detect tampered log entries."""
        from utils.secrets_manager import SecretMetadata

        # Log an entry
        metadata = SecretMetadata(
            secret_name="TEST_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            accessed_at=datetime.utcnow(),
            caller_module="test",
            caller_function="test",
        )
        audit_logger.log_access(metadata)

        # Tamper with the log
        with open(temp_audit_log) as f:
            entry = json.loads(f.read())

        entry["secret_name"] = "TAMPERED_SECRET"

        with open(temp_audit_log, "w") as f:
            f.write(json.dumps(entry) + "\n")

        # Verify should detect tampering
        valid, invalid = audit_logger.verify_log_integrity()
        assert valid == 0
        assert invalid == 1

    def test_log_without_hmac_key(self, temp_audit_log):
        """Logger without HMAC key logs without signature."""
        logger = SecretAuditLogger(temp_audit_log, hmac_key=None)

        from utils.secrets_manager import SecretMetadata

        metadata = SecretMetadata(
            secret_name="TEST_SECRET",
            secret_type=SecretType.RAW,
            accessed_at=datetime.utcnow(),
            caller_module="test",
            caller_function="test",
        )

        with pytest.warns(UserWarning, match="HMAC key"):
            logger.log_access(metadata)

        with open(temp_audit_log) as f:
            entry = json.loads(f.read())
            assert "hmac_sha256" not in entry

    def test_verify_without_hmac_key_raises(self, temp_audit_log):
        """Verify raises without HMAC key."""
        logger = SecretAuditLogger(temp_audit_log, hmac_key=None)

        with pytest.raises(SecretError, match="verify log integrity without HMAC"):
            logger.verify_log_integrity()


# ---------------------------------------------------------------------------
# SecretsManager Tests
# ---------------------------------------------------------------------------


class TestSecretsManager:
    """Test the main SecretsManager interface."""

    def test_get_secret_from_provider(self, file_provider):
        """Get secret retrieves from provider."""
        file_provider.set("TEST_SECRET", "test_value")
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        value = manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW, required=False)
        assert value == "test_value"

    def test_get_secret_required_missing_raises(self, file_provider):
        """Get required secret that's missing raises SecretNotFoundError."""
        manager = SecretsManager(file_provider, audit_logger=None)

        with pytest.raises(SecretNotFoundError, match="Required secret"):
            manager.get_secret("NONEXISTENT", secret_type=SecretType.RAW, required=True)

    def test_get_secret_not_required_returns_default(self, file_provider):
        """Get non-required missing secret returns default."""
        manager = SecretsManager(file_provider, audit_logger=None)

        value = manager.get_secret(
            "NONEXISTENT", secret_type=SecretType.RAW, required=False, default="default_value"
        )
        assert value == "default_value"

    def test_get_secret_validates_when_enabled(self, file_provider):
        """Get secret validates format when validation enabled."""
        file_provider.set("TEST_SECRET", "invalid_stellar_secret")
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=True)

        with pytest.raises(SecretValidationError, match="Invalid Stellar"):
            manager.get_secret("TEST_SECRET", secret_type=SecretType.STELLAR_SECRET)

    def test_get_secret_skips_validation_when_disabled(self, file_provider):
        """Get secret skips validation when disabled."""
        file_provider.set("TEST_SECRET", "invalid_stellar_secret")
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        # Should not raise
        value = manager.get_secret("TEST_SECRET", secret_type=SecretType.STELLAR_SECRET)
        assert value == "invalid_stellar_secret"

    def test_get_secret_logs_access(self, file_provider, audit_logger, temp_audit_log):
        """Get secret logs access event."""
        file_provider.set("TEST_SECRET", "test_value")
        manager = SecretsManager(file_provider, audit_logger, enable_validation=False)

        manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)

        # Verify audit log was written
        assert temp_audit_log.exists()
        with open(temp_audit_log) as f:
            entry = json.loads(f.read())
            assert entry["secret_name"] == "TEST_SECRET"
            assert "redacted_value_hash" in entry

    def test_register_secret(self, secrets_manager):
        """Register a secret definition."""
        definition = SecretDefinition(
            name="MY_SECRET",
            secret_type=SecretType.API_KEY,
            required=True,
            description="Test secret",
        )

        secrets_manager.register_secret(definition)
        assert "MY_SECRET" in secrets_manager._definitions

    def test_get_secret_uses_registered_definition(self, file_provider):
        """Get secret uses registered definition for type and requirements."""
        file_provider.set("MY_SECRET", "a" * 32)
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=True)

        definition = SecretDefinition(
            name="MY_SECRET",
            secret_type=SecretType.API_KEY,
            required=True,
        )
        manager.register_secret(definition)

        # Should use definition's secret_type
        value = manager.get_secret("MY_SECRET")
        assert value == "a" * 32

    def test_rotate_secret(self, file_provider):
        """Rotate a secret to new value and version."""
        file_provider.set("TEST_SECRET", "v1_value", version=1)
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        definition = SecretDefinition(
            name="TEST_SECRET",
            secret_type=SecretType.RAW,
            allow_rotation=True,
        )
        manager.register_secret(definition)

        manager.rotate_secret("TEST_SECRET", "v2_value")

        # Should have new version
        assert file_provider.get("TEST_SECRET") == "v2_value"
        assert 2 in file_provider.list_versions("TEST_SECRET")

    def test_rotate_secret_validates_new_value(self, file_provider):
        """Rotate validates new secret value."""
        file_provider.set("TEST_SECRET", "a" * 32, version=1)
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=True)

        definition = SecretDefinition(
            name="TEST_SECRET",
            secret_type=SecretType.API_KEY,
            allow_rotation=True,
        )
        manager.register_secret(definition)

        # Try to rotate to invalid value
        with pytest.raises(SecretValidationError):
            manager.rotate_secret("TEST_SECRET", "invalid!")

    def test_rotate_secret_not_allowed_raises(self, file_provider):
        """Rotate raises when rotation not allowed."""
        file_provider.set("TEST_SECRET", "value", version=1)
        manager = SecretsManager(file_provider, audit_logger=None)

        definition = SecretDefinition(
            name="TEST_SECRET",
            secret_type=SecretType.RAW,
            allow_rotation=False,
        )
        manager.register_secret(definition)

        with pytest.raises(SecretRotationError, match="not configured to allow rotation"):
            manager.rotate_secret("TEST_SECRET", "new_value")

    def test_rotate_secret_with_readonly_provider_raises(self, monkeypatch):
        """Rotate with read-only provider raises."""
        monkeypatch.setenv("TEST_SECRET", "value")
        provider = EnvironmentSecretProvider()
        manager = SecretsManager(provider, audit_logger=None)

        with pytest.raises(SecretRotationError, match="does not support rotation"):
            manager.rotate_secret("TEST_SECRET", "new_value")

    def test_verify_all_secrets(self, file_provider):
        """Verify all registered secrets."""
        file_provider.set("VALID_SECRET", "a" * 32)
        file_provider.set("INVALID_SECRET", "short")

        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=True)

        # Register secrets
        manager.register_secret(
            SecretDefinition(
                name="VALID_SECRET",
                secret_type=SecretType.API_KEY,
                required=True,
            )
        )
        manager.register_secret(
            SecretDefinition(
                name="INVALID_SECRET",
                secret_type=SecretType.API_KEY,
                required=True,
            )
        )
        manager.register_secret(
            SecretDefinition(
                name="MISSING_SECRET",
                secret_type=SecretType.API_KEY,
                required=True,
            )
        )

        results = manager.verify_all_secrets()

        # VALID_SECRET should pass
        assert results["VALID_SECRET"] is None

        # INVALID_SECRET should fail validation
        assert "too short" in results["INVALID_SECRET"]

        # MISSING_SECRET should not be found
        assert "not found" in results["MISSING_SECRET"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestSecretsManagerIntegration:
    """Test integration scenarios."""

    def test_full_lifecycle_with_audit(self, temp_secrets_dir, temp_audit_log, hmac_key):
        """Test full secret lifecycle with audit trail."""
        # Setup
        provider = FileSecretProvider(temp_secrets_dir)
        audit_logger = SecretAuditLogger(temp_audit_log, hmac_key)
        manager = SecretsManager(provider, audit_logger, enable_validation=True)

        # Register secret
        definition = SecretDefinition(
            name="TEST_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            required=True,
            allow_rotation=True,
        )
        manager.register_secret(definition)

        # Set initial value
        provider.set("TEST_SECRET", "a" * 64, version=1)

        # Access secret (should log)
        value = manager.get_secret("TEST_SECRET")
        assert value == "a" * 64

        # Rotate secret
        manager.rotate_secret("TEST_SECRET", "b" * 64, new_version=2)

        # Access again
        new_value = manager.get_secret("TEST_SECRET")
        assert new_value == "b" * 64

        # Verify audit trail
        valid, invalid = audit_logger.verify_log_integrity()
        assert valid == 2  # Two access events
        assert invalid == 0

    def test_register_ledgerlens_secrets(self, secrets_manager):
        """Register all LedgerLens secrets."""
        register_ledgerlens_secrets(secrets_manager)

        # Verify all expected secrets are registered
        expected_secrets = [
            "LEDGERLENS_SUBMITTER_SECRET",
            "KAFKA_SASL_PASSWORD",
            "MODEL_SIGNING_PRIVATE_KEY_PATH",
            "ANNOTATION_HMAC_SECRET",
            "FORENSIC_REPORT_ENCRYPTION_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "FEDERATED_CA_KEY_PEM",
            "EVENT_HMAC_SECRET",
        ]

        for secret_name in expected_secrets:
            assert secret_name in secrets_manager._definitions

    def test_global_manager_singleton(self):
        """Global manager is a singleton."""
        manager1 = get_secrets_manager()
        manager2 = get_secrets_manager()
        assert manager1 is manager2

    def test_configure_custom_manager(self, secrets_manager):
        """Configure custom manager as global instance."""
        configure_secrets_manager(secrets_manager)
        assert get_secrets_manager() is secrets_manager


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_secret_value(self, file_provider):
        """Empty secret value is handled correctly."""
        file_provider.set("TEST_SECRET", "")
        manager = SecretsManager(file_provider, audit_logger=None)

        # Should raise when required
        with pytest.raises(SecretNotFoundError):
            manager.get_secret("TEST_SECRET", required=True)

        # Should return default when not required
        value = manager.get_secret("TEST_SECRET", required=False, default="default")
        assert value == "default"

    def test_secret_with_newlines(self, file_provider):
        """Secret with embedded newlines is preserved."""
        secret_with_newlines = "line1\nline2\nline3"
        file_provider.set("TEST_SECRET", secret_with_newlines)
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        value = manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)
        assert value == secret_with_newlines

    def test_secret_with_unicode(self, file_provider):
        """Secret with unicode characters is handled correctly."""
        unicode_secret = "secret_with_émojis_🔐_and_special_chars_漢字"
        file_provider.set("TEST_SECRET", unicode_secret)
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        value = manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)
        assert value == unicode_secret

    def test_concurrent_access(self, file_provider):
        """Multiple threads can access secrets safely."""
        import threading

        file_provider.set("TEST_SECRET", "test_value")
        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        results = []
        errors = []

        def access_secret():
            try:
                value = manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)
                results.append(value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_secret) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(v == "test_value" for v in results)
        assert len(errors) == 0

    def test_version_rollback_scenario(self, file_provider):
        """Simulate rolling back to a previous secret version."""
        # Create multiple versions
        file_provider.set("TEST_SECRET", "v1", version=1)
        file_provider.set("TEST_SECRET", "v2_bad", version=2)
        file_provider.set("TEST_SECRET", "v1", version=3)  # Rollback to v1 value

        manager = SecretsManager(file_provider, audit_logger=None, enable_validation=False)

        # Latest version should be v3 with v1's value
        value = manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)
        assert value == "v1"

    def test_audit_log_directory_creation(self, tmp_path):
        """Audit logger creates parent directories if needed."""
        nested_log = tmp_path / "nested" / "dir" / "audit.ndjson"
        audit_logger = SecretAuditLogger(nested_log, "test_key")

        from utils.secrets_manager import SecretMetadata

        metadata = SecretMetadata(
            secret_name="TEST",
            secret_type=SecretType.RAW,
            accessed_at=datetime.utcnow(),
            caller_module="test",
            caller_function="test",
        )

        audit_logger.log_access(metadata)
        assert nested_log.exists()


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------


class TestSecurity:
    """Test security-related behavior."""

    def test_redacted_value_hash_in_audit(self, file_provider, audit_logger, temp_audit_log):
        """Audit log contains hash of secret value, not plaintext."""
        file_provider.set("TEST_SECRET", "sensitive_value")
        manager = SecretsManager(file_provider, audit_logger, enable_validation=False)

        manager.get_secret("TEST_SECRET", secret_type=SecretType.RAW)

        # Check audit log
        with open(temp_audit_log) as f:
            entry = json.loads(f.read())

            # Should have redacted hash
            assert "redacted_value_hash" in entry

            # Should NOT contain actual secret value
            log_content = temp_audit_log.read_text()
            assert "sensitive_value" not in log_content

    def test_hmac_prevents_tampering(self, audit_logger, temp_audit_log):
        """HMAC signature prevents undetected log tampering."""
        from utils.secrets_manager import SecretMetadata

        # Log original entry
        metadata = SecretMetadata(
            secret_name="ORIGINAL_NAME",
            secret_type=SecretType.API_KEY,
            accessed_at=datetime.utcnow(),
            caller_module="test",
            caller_function="test",
        )
        audit_logger.log_access(metadata)

        # Tamper with secret name
        with open(temp_audit_log) as f:
            entry = json.loads(f.read())

        entry["hmac_sha256"]
        entry["secret_name"] = "TAMPERED_NAME"

        with open(temp_audit_log, "w") as f:
            f.write(json.dumps(entry) + "\n")

        # Verify should detect tampering
        valid, invalid = audit_logger.verify_log_integrity()
        assert valid == 0
        assert invalid == 1

    def test_file_permissions_secure(self, file_provider, temp_secrets_dir):
        """Secret files are created with secure permissions."""
        file_provider.set("SECURE_SECRET", "sensitive_data")

        secret_file = temp_secrets_dir / "SECURE_SECRET"
        stat_info = secret_file.stat()

        # Should be 0o600 (owner read/write only)
        perms = stat_info.st_mode & 0o777
        assert perms == 0o600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
