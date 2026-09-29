# Secrets Management Integration Example

This document demonstrates how to integrate the secrets management system into LedgerLens integrations.

## Example: Contract Client Integration

### Before (Direct os.getenv)

```python
# integrations/contract_client.py (OLD)
import os
from config import config

class LedgerLensContractClient:
    def __init__(self, submitter_secret: str | None = None):
        # Direct environment variable access - no validation
        self.submitter_secret = submitter_secret or os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")
        
    def submit_score(self, wallet: str, asset_pair: str, risk_score: dict) -> object:
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")
        
        # Use secret...
```

###Problems with this approach:
- No validation of Stellar secret format
- Secrets logged in error messages
- No audit trail of access
- No rotation support

### After (With Secrets Manager)

```python
# integrations/contract_client.py (NEW)
from utils.secrets_config import get_secret
from utils.secrets_manager import SecretType, SecretValidationError
from config import config

class LedgerLensContractClient:
    def __init__(self, submitter_secret: str | None = None):
        if submitter_secret is None:
            # Use secrets manager with validation
            try:
                self.submitter_secret = get_secret(
                    "LEDGERLENS_SUBMITTER_SECRET",
                    secret_type=SecretType.STELLAR_SECRET,
                    required=False  # Only required when submitting
                )
            except SecretValidationError as e:
                raise ValueError(
                    f"Invalid LEDGERLENS_SUBMITTER_SECRET: {e}. "
                    "Must be a valid Stellar secret key (S... format, 56 chars)"
                ) from e
        else:
            self.submitter_secret = submitter_secret
        
    def submit_score(self, wallet: str, asset_pair: str, risk_score: dict) -> object:
        if not self.submitter_secret:
            raise ValueError(
                "LEDGERLENS_SUBMITTER_SECRET is not configured. "
                "Set the environment variable or use file-based secrets."
            )
        
        # Access is automatically logged to audit trail
        # Use secret...
```

### Benefits:
- Automatic Stellar key format validation
- Access logged to tamper-evident audit trail
- Better error messages with actionable guidance
- Rotation support without code changes

## Example: Kafka Integration

### Before

```python
# streaming/kafka_worker.py (OLD)
import os

def create_kafka_consumer():
    config = {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
        'sasl_mechanism': 'PLAIN',
        'security_protocol': 'SASL_SSL',
        'sasl_plain_username': os.getenv('KAFKA_SASL_USERNAME'),
        'sasl_plain_password': os.getenv('KAFKA_SASL_PASSWORD', ''),  # Unsafe default
    }
    return KafkaConsumer(**config)
```

### After

```python
# streaming/kafka_worker.py (NEW)
from utils.secrets_config import get_secret, is_secret_configured
from utils.secrets_manager import SecretType

def create_kafka_consumer():
    # Check if SASL auth is configured
    if is_secret_configured('KAFKA_SASL_USERNAME'):
        config = {
            'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'sasl_mechanism': 'PLAIN',
            'security_protocol': 'SASL_SSL',
            'sasl_plain_username': get_secret('KAFKA_SASL_USERNAME', required=True),
            'sasl_plain_password': get_secret(
                'KAFKA_SASL_PASSWORD',
                secret_type=SecretType.PASSWORD,
                required=True
            ),
        }
    else:
        # No SASL auth configured
        config = {
            'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
        }
    
    return KafkaConsumer(**config)
```

## Example: Model Signing

### Before

```python
# detection/audit_trail.py (OLD)
import os
from pathlib import Path
from cryptography.hazmat.primitives import serialization

def load_signing_key():
    key_path = os.getenv("MODEL_SIGNING_PRIVATE_KEY_PATH", "")
    if not key_path:
        raise ValueError("MODEL_SIGNING_PRIVATE_KEY_PATH not set")
    
    # No existence check
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)
```

### After

```python
# detection/audit_trail.py (NEW)
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from utils.secrets_config import get_secret
from utils.secrets_manager import SecretType, SecretValidationError

def load_signing_key():
    try:
        key_path = get_secret(
            "MODEL_SIGNING_PRIVATE_KEY_PATH",
            secret_type=SecretType.FILEPATH,
            required=True
        )
    except SecretValidationError as e:
        raise ValueError(
            f"MODEL_SIGNING_PRIVATE_KEY_PATH is invalid: {e}. "
            "Ensure the file exists and is accessible."
        ) from e
    
    # File existence already validated by secrets manager
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)
```

## Example: Config.py Integration

### Update config.py to use secrets manager for sensitive values:

```python
# config.py
import os
from dotenv import load_dotenv
from utils.secrets_config import get_secret, is_secret_configured
from utils.secrets_manager import SecretType

load_dotenv()

class Config:
    # ... other config ...
    
    # Use secrets manager for sensitive values
    @property
    def LEDGERLENS_SUBMITTER_SECRET(self) -> str:
        return get_secret(
            "LEDGERLENS_SUBMITTER_SECRET",
            secret_type=SecretType.STELLAR_SECRET,
            required=False,
            default=""
        ) or ""
    
    @property
    def KAFKA_SASL_PASSWORD(self) -> str | None:
        return get_secret(
            "KAFKA_SASL_PASSWORD",
            secret_type=SecretType.PASSWORD,
            required=False
        )
    
    @property
    def ANNOTATION_HMAC_SECRET(self) -> str:
        return get_secret(
            "ANNOTATION_HMAC_SECRET",
            secret_type=SecretType.HMAC_SECRET,
            required=False,
            default=""
        ) or ""

config = Config()
```

## Testing Integration

### Unit Test Example

```python
# tests/test_my_integration.py
import pytest
from utils.secrets_manager import FileSecretProvider, SecretsManager

@pytest.fixture
def secrets_manager(tmp_path):
    """Provide a test secrets manager with file provider."""
    provider = FileSecretProvider(tmp_path / "secrets")
    manager = SecretsManager(provider, audit_logger=None, enable_validation=True)
    return manager, provider

def test_contract_client_with_secrets(secrets_manager):
    """Test contract client uses secrets manager correctly."""
    manager, provider = secrets_manager
    
    # Set a valid test secret
    test_secret = "SBZVF2CTUDTHHDKJP3UEKQRC2XLUJMCG3DL5HGJ2YPTPZXC7QCMQW2W3"
    provider.set("LEDGERLENS_SUBMITTER_SECRET", test_secret)
    
    # Test code should retrieve and use the secret
    from integrations.contract_client import LedgerLensContractClient
    client = LedgerLensContractClient()
    
    assert client.submitter_secret == test_secret
```

## Deployment Checklist

### Pre-Deployment

- [ ] Run `python -m scripts.validate_secrets` to validate all secrets
- [ ] Verify audit log configuration: `SECRETS_AUDIT_LOG` and `SECRETS_AUDIT_HMAC_KEY`
- [ ] Enable validation in production: `SECRETS_VALIDATION_ENABLED=true`
- [ ] Use file-based secrets: `SECRETS_DIR=/run/secrets`
- [ ] Set restrictive file permissions: `chmod 600 /run/secrets/*`

### Production Configuration

```bash
# /etc/ledgerlens/secrets.env
export SECRETS_DIR=/run/secrets
export SECRETS_AUDIT_LOG=/var/log/ledgerlens/secrets_audit.ndjson
export SECRETS_AUDIT_HMAC_KEY=<generate-with-secrets.token_hex(32)>
export SECRETS_VALIDATION_ENABLED=true
```

### CI/CD Integration

```yaml
# .github/workflows/deploy.yml
- name: Validate secrets configuration
  run: |
    python -m scripts.validate_secrets
    if [ $? -ne 0 ]; then
      echo "Secrets validation failed!"
      exit 1
    fi

- name: Verify audit log integrity
  run: python -m scripts.validate_secrets --verify-audit-log
```

## Common Patterns

### Optional vs Required Secrets

```python
# Required - raises if missing
stellar_secret = get_secret("LEDGERLENS_SUBMITTER_SECRET", required=True)

# Optional - returns None if missing
api_key = get_secret("OPENAI_API_KEY", required=False)

# Optional with default
fallback_key = get_secret("OPTIONAL_KEY", required=False, default="default_value")
```

### Conditional Secret Requirements

```python
from utils.secrets_config import is_secret_configured, get_secret

if is_secret_configured("KAFKA_SASL_USERNAME"):
    # SASL is enabled, password is now required
    kafka_password = get_secret("KAFKA_SASL_PASSWORD", required=True)
else:
    # SASL not configured, skip password
    kafka_password = None
```

### Error Handling

```python
from utils.secrets_manager import SecretNotFoundError, SecretValidationError
from utils.secrets_config import get_secret

try:
    secret = get_secret("MY_SECRET", required=True)
except SecretNotFoundError:
    logger.error("MY_SECRET not configured - check deployment documentation")
    sys.exit(1)
except SecretValidationError as e:
    logger.error(f"MY_SECRET is invalid: {e}")
    sys.exit(1)
```

## Migration Checklist

- [ ] Identify all secrets in codebase: `grep -r "os.getenv.*SECRET"`
- [ ] Update imports to use `utils.secrets_config`
- [ ] Replace `os.getenv()` calls with `get_secret()`
- [ ] Add type hints for secret types (Stellar, HMAC, API key, etc.)
- [ ] Update tests to use `FileSecretProvider` fixtures
- [ ] Run test suite: `pytest tests/ -v`
- [ ] Validate configuration: `python -m scripts.validate_secrets`
- [ ] Update documentation with new secret requirements
- [ ] Train team on secrets management system
- [ ] Update deployment procedures for file-based secrets

## Support

For questions or issues:
- Review [secrets_management.md](secrets_management.md)
- Run `python -m scripts.validate_secrets --help`
- Check audit log: `tail -f data/secrets_audit.ndjson`
- Open an issue with the `security` label
