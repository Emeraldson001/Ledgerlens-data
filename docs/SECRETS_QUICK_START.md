# Secrets Management Quick Start

5-minute guide to using the LedgerLens secrets management system.

## Installation

No installation needed - the system is included in the repository.

## Basic Usage

### Get a Secret

```python
from utils.secrets_config import get_secret

# Required secret (raises if missing)
secret = get_secret("LEDGERLENS_SUBMITTER_SECRET", required=True)

# Optional secret with default
api_key = get_secret("OPENAI_API_KEY", required=False, default=None)
```

### Validate Configuration

```bash
# Check all secrets are properly configured
python -m scripts.validate_secrets

# Verify audit log hasn't been tampered with
python -m scripts.validate_secrets --verify-audit-log
```

### Generate a Secret

```bash
# Stellar secret key
python -c "from stellar_sdk import Keypair; print(Keypair.random().secret)"

# HMAC secret (64 hex chars for 256-bit)
python -c "import secrets; print(secrets.token_hex(32))"

# API key (32+ chars)
python -c "import secrets; print('sk_' + secrets.token_urlsafe(32))"
```

## Secret Types

| Type | Example | Generation |
|------|---------|------------|
| Stellar Secret | `SBZVF2CT...` (56 chars) | `Keypair.random().secret` |
| HMAC Secret | `a1b2c3d4...` (64 hex) | `secrets.token_hex(32)` |
| API Key | `sk_live_...` (32+ chars) | Manual from provider |
| Password | `MySecure!Pass123` (16+ chars) | `secrets.token_urlsafe(16)` |
| File Path | `/path/to/key.pem` | N/A |

## Configuration

### Environment Variables (Default)

```bash
# Set secrets as environment variables
export LEDGERLENS_SUBMITTER_SECRET="SBZVF2CT..."
export ANNOTATION_HMAC_SECRET="a1b2c3d4..."
```

### File-Based Secrets (Production)

```bash
# 1. Create secrets directory
mkdir -p /run/secrets
chmod 700 /run/secrets

# 2. Store secrets (one per file)
echo "SBZVF2..." > /run/secrets/LEDGERLENS_SUBMITTER_SECRET
chmod 600 /run/secrets/LEDGERLENS_SUBMITTER_SECRET

# 3. Configure application
export SECRETS_DIR=/run/secrets
```

### Enable Audit Trail

```bash
# Generate HMAC key for audit log integrity
export SECRETS_AUDIT_HMAC_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Set audit log path (default: data/secrets_audit.ndjson)
export SECRETS_AUDIT_LOG=/var/log/ledgerlens/secrets_audit.ndjson
```

## Common Tasks

### Check if Secret is Configured

```python
from utils.secrets_config import is_secret_configured

if is_secret_configured("OPENAI_API_KEY"):
    # API key is available, use it
    pass
else:
    # Fall back to alternative
    pass
```

### Rotate a Secret

```python
from utils.secrets_config import rotate_secret
import secrets

# Generate new value
new_hmac = secrets.token_hex(32)

# Rotate (auto-increments version)
rotate_secret("ANNOTATION_HMAC_SECRET", new_hmac)
```

### Validate All Secrets

```python
from utils.secrets_config import verify_secrets

results = verify_secrets()

for secret_name, error in results.items():
    if error:
        print(f"❌ {secret_name}: {error}")
    else:
        print(f"✅ {secret_name}: valid")
```

## Troubleshooting

### "Required secret not found"

**Fix**: Set the environment variable or create the secret file

```bash
export LEDGERLENS_SUBMITTER_SECRET="your-secret-here"
# or
echo "your-secret" > /run/secrets/LEDGERLENS_SUBMITTER_SECRET
```

### "Invalid Stellar secret key format"

**Fix**: Verify format is correct (starts with S, 56 base32 chars)

```bash
# Generate new valid secret
python -c "from stellar_sdk import Keypair; print(Keypair.random().secret)"
```

### "Invalid HMAC secret format"

**Fix**: Must be hex-encoded, minimum 32 chars

```bash
# Generate 256-bit HMAC secret
python -c "import secrets; print(secrets.token_hex(32))"
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Validate secrets
  run: python -m scripts.validate_secrets
  
- name: Verify audit log
  run: python -m scripts.validate_secrets --verify-audit-log
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: validate-secrets
      name: Validate secrets
      entry: python -m scripts.validate_secrets
      language: system
```

## Production Checklist

- [ ] Use file-based secrets: `export SECRETS_DIR=/run/secrets`
- [ ] Enable audit logging: `export SECRETS_AUDIT_HMAC_KEY=...`
- [ ] Set restrictive permissions: `chmod 600 /run/secrets/*`
- [ ] Enable validation: `export SECRETS_VALIDATION_ENABLED=true`
- [ ] Run validation: `python -m scripts.validate_secrets`
- [ ] Test rotation: Rotate a non-critical secret first
- [ ] Monitor audit log: `tail -f $SECRETS_AUDIT_LOG`

## Security Best Practices

1. **Never commit secrets to version control**
   ```bash
   # Add to .gitignore
   echo "secrets/" >> .gitignore
   echo "*.pem" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use file-based secrets in production** (not environment variables)

3. **Rotate secrets regularly** (every 90 days minimum)

4. **Enable audit logging** with HMAC for tamper detection

5. **Validate before deployment**
   ```bash
   python -m scripts.validate_secrets || exit 1
   ```

## Getting Help

- **Full documentation**: See [docs/secrets_management.md](secrets_management.md)
- **Integration examples**: See [docs/secrets_integration_example.md](secrets_integration_example.md)
- **CLI help**: `python -m scripts.validate_secrets --help`
- **Issues**: Open GitHub issue with `security` label

## Next Steps

1. Read [docs/secrets_management.md](secrets_management.md) for complete guide
2. Review [docs/secrets_integration_example.md](secrets_integration_example.md) for integration patterns
3. Run `python -m scripts.validate_secrets` to check current configuration
4. Update code to use `get_secret()` instead of `os.getenv()` for sensitive values
5. Configure audit logging for production deployments

---

**For full details, see [docs/secrets_management.md](secrets_management.md)**
