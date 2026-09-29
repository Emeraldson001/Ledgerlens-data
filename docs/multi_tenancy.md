# Multi-Tenant Namespace Isolation

LedgerLens supports multi-tenant deployments where each exchange client operates with isolated risk score configurations.

## Architecture

Tenant isolation is enforced at multiple layers:

- **Redis keys**: All Redis keys are prefixed with `tenant_id:` to prevent cross-tenant data leakage
- **Prometheus metrics**: Every metric includes a `tenant` label for per-tenant observability
- **Database records**: Risk scores are scoped by tenant_id
- **Configuration**: Each tenant has custom risk thresholds, Benford parameters, and asset pair whitelists

## YAML Schema

Tenant configuration is loaded from `config/tenants.yaml`. Every tenant entry must include the following keys:

| Key | Type | Required | Default | Description | Example |
|---|---|---|---|---|---|
| `risk_threshold` | integer | Yes | — | Risk score (0–100) at or above which a wallet is flagged as suspicious. Used by the static threshold strategy and as a fallback. | `70` |
| `benford_min_sample` | integer | Yes | — | Minimum number of trades required before Benford analysis is enabled for this tenant. | `100` |
| `alert_channels` | list of strings | Yes | — | List of alert destinations (`stdout`, `webhook`, `email`, etc.). Used by notification handlers. | `["stdout", "webhook"]` |
| `asset_pair_whitelist` | list of strings | Yes | — | Permitted asset pairs in `CODE:ISSUER/CODE:ISSUER` format. Only whitelisted pairs are scored for this tenant. | `["USDC:GA5Z…/XLM:native"]` |
| `threshold_strategy` | string | No | `"static"` | Strategy for computing the risk threshold. Options: `"static"` (fixed threshold), `"statistical"` (data-driven). | `"statistical"` |
| `threshold_config` | object | No | `{}` | Configuration object passed to the threshold strategy builder. Contents depend on `threshold_strategy`. | `{"recall_floor": 0.85, "target_metric": "f1"}` |

### Minimal Two-Tenant Configuration

```yaml
tenants:
  exchange_a:
    risk_threshold: 70
    threshold_strategy: static
    threshold_config: {}
    benford_min_sample: 100
    alert_channels:
      - stdout
    asset_pair_whitelist:
      - USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native
  
  exchange_b:
    risk_threshold: 75
    threshold_strategy: statistical
    threshold_config:
      recall_floor: 0.85
      target_metric: f1
    benford_min_sample: 50
    alert_channels:
      - webhook
    asset_pair_whitelist:
      - USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native
      - BTC:GBTCHP4GPA3WJGVEUQWEPY7NVZLKJ2QNCU3C3ST6MRTIYBFPP5A5K47B/XLM:native
```

## Onboarding a New Tenant

1. Add an entry to `config/tenants.yaml` with the tenant's configuration, including all required keys (see YAML Schema table above)
2. Restart the pipeline to load the new tenant configuration
3. Use the `TenantContext` class in request handlers to inject tenant-specific behavior

## Security

Tenant IDs are validated against the allowlist in `tenants.yaml`. Arbitrary strings are not accepted as tenant IDs to prevent injection attacks.
