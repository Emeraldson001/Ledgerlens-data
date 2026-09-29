import logging
import os
import unittest
from io import StringIO
from pathlib import Path

from cli.diagnostics import run_diagnostics
from utils.logging import SecretsRedactingFormatter
from utils.secrets import SecretString, mask_secret, sanitize_config, sanitize_text, sanitize_url
from utils.secrets_manager import SecretType, SecretValidator, SecretValidationError


class TestSecretsSafety(unittest.TestCase):

    def test_secret_string_masking(self):
        s = SecretString("super_secret_password_123")
        self.assertEqual(s.expose(), "super_secret_password_123")
        self.assertEqual(repr(s), "<SecretString [REDACTED]>")
        self.assertNotIn("super_secret_password_123", str(s))
        self.assertTrue(s == "super_secret_password_123")

    def test_mask_secret(self):
        self.assertEqual(mask_secret("123456789"), "*****6789")
        self.assertEqual(mask_secret("123"), "***")
        self.assertEqual(mask_secret(""), "")

    def test_sanitize_url(self):
        db_url = "postgresql://db_user:super_secret_pass@localhost:5432/ledgerlens"
        sanitized = sanitize_url(db_url)
        self.assertNotIn("super_secret_pass", sanitized)
        self.assertIn("db_user:****@localhost:5432/ledgerlens", sanitized)

    def test_sanitize_text_patterns(self):
        text = "Failed connect with api_key=secret_key_abc123 and Bearer eyJhbGciOiJIUzI1NiIn1"
        sanitized = sanitize_text(text)
        self.assertNotIn("secret_key_abc123", sanitized)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIn1", sanitized)

    def test_sanitize_config_dict(self):
        cfg = {
            "HORIZON_URL": "https://horizon.stellar.org",
            "KAFKA_SASL_PASSWORD": "my_kafka_password",
            "RISK_SCORE_DB_URL": "postgres://admin:secret123@localhost/db",
        }
        clean = sanitize_config(cfg)
        self.assertNotIn("my_kafka_password", clean["KAFKA_SASL_PASSWORD"])
        self.assertNotIn("secret123", clean["RISK_SCORE_DB_URL"])
        self.assertEqual(clean["HORIZON_URL"], "https://horizon.stellar.org")

    def test_log_redaction(self):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SecretsRedactingFormatter("%(message)s"))
        logger = logging.getLogger("test_redact")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info("Connecting to postgresql://user:my_secret_pass@localhost/db")
        log_output = stream.getvalue()
        self.assertNotIn("my_secret_pass", log_output)
        self.assertIn("****@", log_output)

    def test_diagnostics_sanitizes_urls(self):
        os.environ["RISK_SCORE_DB_URL"] = "postgresql://user:dbpass123@localhost:5432/db"
        os.environ["HORIZON_URL"] = "https://horizon.stellar.org"
        os.environ["KAFKA_SASL_PASSWORD"] = "secret_sasl_pass"

        diag = run_diagnostics()
        details = diag["checks"]["environment"]["details"]
        self.assertNotIn("dbpass123", details["RISK_SCORE_DB_URL"])
        self.assertNotIn("secret_sasl_pass", details["KAFKA_SASL_PASSWORD"])


class TestSecretsValidationErrorRedaction(unittest.TestCase):
    """Verify that secret values never appear in validation error messages."""

    def test_stellar_secret_too_short_redacts_value(self):
        """Stellar secret too short error does not include the secret value."""
        short_secret = "S" + "A" * 50  # Too short
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(short_secret, SecretType.STELLAR_SECRET)
        error_msg = str(ctx.exception)
        self.assertNotIn(short_secret, error_msg)
        self.assertIn("too short", error_msg)

    def test_stellar_secret_invalid_format_redacts_value(self):
        """Stellar secret invalid format error does not include the secret value."""
        invalid_secret = "not_a_stellar_secret_value_12345"
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(invalid_secret, SecretType.STELLAR_SECRET)
        error_msg = str(ctx.exception)
        self.assertNotIn(invalid_secret, error_msg)
        self.assertIn("Invalid Stellar secret", error_msg)

    def test_hmac_secret_too_short_redacts_value(self):
        """HMAC secret too short error does not include the secret value."""
        short_hmac = "a" * 31  # Too short
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(short_hmac, SecretType.HMAC_SECRET)
        error_msg = str(ctx.exception)
        self.assertNotIn(short_hmac, error_msg)
        self.assertIn("too short", error_msg)

    def test_hmac_secret_invalid_chars_redacts_value(self):
        """HMAC secret invalid chars error does not include the secret value."""
        invalid_hmac = "x" * 32  # Non-hex characters
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(invalid_hmac, SecretType.HMAC_SECRET)
        error_msg = str(ctx.exception)
        self.assertNotIn(invalid_hmac, error_msg)
        self.assertIn("Invalid HMAC secret", error_msg)

    def test_api_key_too_short_redacts_value(self):
        """API key too short error does not include the secret value."""
        short_key = "short_key"
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(short_key, SecretType.API_KEY)
        error_msg = str(ctx.exception)
        self.assertNotIn(short_key, error_msg)
        self.assertIn("too short", error_msg)

    def test_api_key_invalid_chars_redacts_value(self):
        """API key invalid chars error does not include the secret value."""
        invalid_key = "a" * 32 + "!@#$"
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(invalid_key, SecretType.API_KEY)
        error_msg = str(ctx.exception)
        self.assertNotIn(invalid_key, error_msg)
        self.assertIn("Invalid API key", error_msg)

    def test_password_too_short_redacts_value(self):
        """Password too short error does not include the secret value."""
        short_password = "Short1!"
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(short_password, SecretType.PASSWORD)
        error_msg = str(ctx.exception)
        self.assertNotIn(short_password, error_msg)
        self.assertIn("at least 16 characters", error_msg)

    def test_password_weak_complexity_redacts_value(self):
        """Password weak complexity error does not include the secret value."""
        weak_password = "a" * 16  # Only lowercase, no complexity
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(weak_password, SecretType.PASSWORD)
        error_msg = str(ctx.exception)
        self.assertNotIn(weak_password, error_msg)
        self.assertIn("at least 3 of", error_msg)

    def test_filepath_not_exists_redacts_value(self, tmp_path):
        """Filepath not exists error does not include the filepath value."""
        nonexistent_path = "/nonexistent/path/to/secret/file_12345"
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(nonexistent_path, SecretType.FILEPATH)
        error_msg = str(ctx.exception)
        self.assertNotIn(nonexistent_path, error_msg)
        self.assertIn("does not exist", error_msg)

    def test_filepath_is_directory_redacts_value(self, tmp_path):
        """Filepath is directory error does not include the directory path value."""
        dir_path = str(tmp_path / "test_dir")
        os.makedirs(dir_path, exist_ok=True)
        with self.assertRaises(SecretValidationError) as ctx:
            SecretValidator.validate(dir_path, SecretType.FILEPATH)
        error_msg = str(ctx.exception)
        self.assertNotIn(dir_path, error_msg)
        self.assertIn("not a file", error_msg)


if __name__ == "__main__":
    unittest.main()
