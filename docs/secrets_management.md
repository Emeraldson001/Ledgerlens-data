# Secrets Management Guide

This document describes the secrets-safe configuration handling system for LedgerLens integrations.

## Overview

The secrets management system provides:

- **Type-safe validation**: Each secret type (Stellar keys, HMAC secrets, API keys, passwords) has format validation
- **Audit trail**: All secret access events are logged with HMAC-signed tamper-evident records
- **Rotation support**: Secrets can be rotated with versioning and graceful fallback
- **Provider abstraction**: Supports environment variables, file-based secrets, and external vaults
- **Security by default**: Secrets are never logged in plaintext, files have restrictive permissions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Code                         │
│                  (config.py, integrations/)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    SecretsManager                            │
│  - Registration & validation                                 │
│  - Access control                                            │
│  - Rotation coordination                                     │
└──┬────────────────┬────────────────────────────────────┬────┘
   │                │                                     │
   ▼                ▼                                     ▼
┌─────────┐  ┌──────────────┐                   ┌──────────────┐
│Provider │  │  Validator   │                   │ AuditLogger  │
│         │  │              │                   │              │
│ - Env   │  │ - Patterns   │                   │ - HMAC       │
│ - File  │  │ - Strength   │                   │ - NDJSON     │
│ - Vault │  │ - Existence  │                   │ - Integrity  │
└─────────┘  └──────────────┘                   └──────────────┘
```

## Quick Start

### Basic Usage

```python
from utils.secrets_config import get_secret

# Get a secret with automatic validation
submitter_secret = get_secret(
    "LEDGERLENS_SUBMITTER_SECRET",
    required=True
)

# Optional secret with default
api_key = get_secret(
    "OPENAI_API_KEY",
    required=False,
    default=None
)
```

### Validation

Secrets are automatically validated based on their type:

```python
from utils.secrets_manager import SecretType, SecretValidator

# This will raise SecretValidationError if invalid
SecretValidator.validate(
    "SBZVF2CTUDTHHDKJP3UEKQRC2XLUJMCG3DL5HGJ2YPTPZXC7QCMQW2W3",
    SecretType.STELLAR_SECRET
)
```

### Rotation

```python
from utils.secrets_config import rotate_secret

# Rotate to a new value (auto-increments version)
rotate_secret("ANNOTATION_HMAC_SECRET", new_value="..." )
```

## Secret Types

### Stellar Secret Keys

**Format**: Base32, starts with 'S', exactly 56 characters

**Example**: `SBZVF2CTUDTHHDKJP3UEKQRC2XLUJMCG3DL5HGJ2YPTPZXC7QCMQW2W3`

**Generation**:
```python
from stellar_sdk import Keypair
keypair = Keypair.random()
print(keypair.secret)
```

### HMAC Secrets

**Format**: Hex-encoded, minimum 32 characters (128-bit)

**Example**: `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2`

**Generation**:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### API Keys

**Format**: Alphanumeric with `_-.` characters, minimum 32 characters

**Example**: `api_key_1234567890abcdef1234567890abcdef` (example only, not real)

### Passwords

**Format**: Minimum 16 characters with at least 3 of: uppercase, lowercase, digits, special characters

**Example**: `MySecure!Password123`

### File Paths

**Format**: Path to an existing file

**Example**: `/etc/ledgerlens/signing_key.pem`

## Configuration

### Environment Variables

Configure the secrets management system:

```bash
# Provider selection
SECRETS_DIR=/path/to/secrets  # Use file-based provider (optional)

# Audit logging
SECRETS_AUDIT_LOG=data/secrets_audit.ndjson
SECRETS_AUDIT_HMAC_KEY=<64-char-hex>  # For tamper-evident audit trail

# Validation
SECRETS_VALIDATION_ENABLED=true  # Enable/disable validation
```

### File-Based Secrets

Store secrets in individual files:

```bash
# Create secrets directory
mkdir -p /etc/ledgerlens/secrets
chmod 700 /etc/ledgerlens/secrets

# Store secrets (one per file)
echo "SBZVF2..." > /etc/ledgerlens/secrets/LEDGERLENS_SUBMITTER_SECRET
chmod 600 /etc/ledgerlens/secrets/LEDGERLENS_SUBMITTER_SECRET

# Configure application
export SECRETS_DIR=/etc/ledgerlens/secrets
```

## Migration Guide

### Step 1: Audit Current Usage

Find all secrets in the codebase:

```bash
grep -r "os.getenv.*SECRET\|os.getenv.*PASSWORD\|os.getenv.*KEY" .
```

### Step 2: Update Code

Replace direct `os.getenv()` calls:

```python
# BEFORE
import os
submitter_secret = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

# AFTER
from utils.secrets_config import get_secret
submitter_secret = get_secret("LEDGERLENS_SUBMITTER_SECRET", required=True)
```

### Step 3: Validate Configuration

Run the validation tool:

```bash
python -m scripts.validate_secrets
```

### Step 4: Test

Ensure all integration tests pass:

```bash
pytest tests/test_secrets_manager.py -v
pytest tests/test_contract_client.py -v
```

## CLI Tools

### Validate Secrets

Check all secrets are properly configured:

```bash
# Validate all secrets
python -m scripts.validate_secrets

# Validate specific secrets
python -m scripts.validate_secrets --secrets LEDGERLENS_SUBMITTER_SECRET

# Verify audit log integrity
python -m scripts.validate_secrets --verify-audit-log

# Generate JSON report
python -m scripts.validate_secrets --report --report-output config_report.json
```

### Exit Codes

- `0`: All secrets valid
- `1`: Warnings found (optional secrets missing)
- `2`: Errors found (required secrets missing or invalid)

## Security Best Practices

### 1. Never Commit Secrets

Add to `.gitignore`:

```
# Secrets
*.pem
*.key
*_secret
*_password
secrets/
.env
```

### 2. Use File-Based Secrets in Production

Environment variables can be exposed in process listings. Use file-based secrets:

```bash
export SECRETS_DIR=/run/secrets  # Docker secrets, Kubernetes secrets
```

### 3. Enable Audit Logging

Always configure HMAC-signed audit trails in production:

```bash
export SECRETS_AUDIT_LOG=/var/log/ledgerlens/secrets_audit.ndjson
export SECRETS_AUDIT_HMAC_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

### 4. Rotate Regularly

Rotate sensitive secrets on a schedule:

```python
from utils.secrets_config import rotate_secret

# Rotate annotation HMAC secret
new_hmac = secrets.token_hex(32)
rotate_secret("ANNOTATION_HMAC_SECRET", new_hmac)
```

### 5. Validate Before Deployment

Add to CI/CD pipeline:

```yaml
- name: Validate secrets configuration
  run: python -m scripts.validate_secrets
```

## Audit Trail

All secret access events are logged with HMAC signatures for tamper detection.

### Log Format

NDJSON (newline-delimited JSON):

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "secret_name": "LEDGERLENS_SUBMITTER_SECRET",
  "secret_type": "stellar_secret",
  "caller_module": "integrations.contract_client",
  "caller_function": "submit_score",
  "version": 1,
  "redacted_value_hash": "a1b2c3d4e5f6a7b8",
  "hmac_sha256": "..."
}
```

### Verifying Integrity

```python
from utils.secrets_manager import SecretAuditLogger

audit_logger = SecretAuditLogger("data/secrets_audit.ndjson", hmac_key)
valid, invalid = audit_logger.verify_log_integrity()

if invalid > 0:
    print(f"WARNING: {invalid} tampered entries detected!")
```

## Troubleshooting

### "Required secret not found"

**Cause**: Environment variable or secret file is not set

**Solution**:
```bash
# Check if secret is configured
python -m scripts.validate_secrets --secrets LEDGERLENS_SUBMITTER_SECRET

# Set the secret
export LEDGERLENS_SUBMITTER_SECRET="S..."
```

### "Invalid Stellar secret key format"

**Cause**: Secret doesn't match Stellar format

**Solution**: Verify the secret starts with 'S' and is exactly 56 base32 characters

### "Audit log integrity verification failed"

**Cause**: Log file has been tampered with or HMAC key changed

**Solution**: Investigate potential security incident, review access logs

### "Provider does not support rotation"

**Cause**: Using EnvironmentSecretProvider (read-only)

**Solution**: Switch to FileSecretProvider:
```bash
export SECRETS_DIR=/path/to/secrets
```

## API Reference

See module docstrings for complete API documentation:

- `utils/secrets_manager.py` - Core secrets management
- `utils/secrets_config.py` - Configuration wrapper
- `scripts/validate_secrets.py` - CLI validation tool

## Related Documentation

- [Security Threat Model](security_threat_model.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Configuration Reference](../README.md#configuration)
