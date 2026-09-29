# Typed Deployment Mode Fixtures

`config/deployment_modes.py` (Issue #543) gives each supported deployment
mode — `local`, `testnet`, `production` — a typed, reusable, validated
fixture instead of hand-rolled `.env` files that drift out of sync with
what `Config.validate()` actually requires.

## Why

`Config` (config.py) is a flat, env-var-driven surface with 500+
attributes. Nothing previously enforced that a given deployment mode set a
*coherent* combination of them — e.g. that `production` never accidentally
ships with `HORIZON_DEV_MODE=True`, or that `testnet` sets an on-chain
contract ID when on-chain submission is required. Contributors discovered
missing/incoherent values only at runtime.

## Usage

```python
from config.deployment_modes import DeploymentMode, apply_deployment_mode

with apply_deployment_mode(DeploymentMode.TESTNET) as fixture:
    # Config now reflects the testnet fixture's overrides, and has already
    # been validated with Config.validate(require_onchain=fixture.require_onchain).
    run_pipeline()
# Every overridden attribute is restored to its prior value on exit.
```

In tests, use the equivalent pytest fixtures registered in
`tests/conftest.py`:

```python
def test_something(testnet_deployment_config):
    assert Config.STELLAR_NETWORK == "TESTNET"
```

## Adding a new mode

Add one `DeploymentModeFixture` entry to `DEPLOYMENT_MODE_FIXTURES` in
`config/deployment_modes.py`. Every consumer — tests, scripts,
`apply_deployment_mode` — picks it up automatically; there is nothing else
to wire up.

## Validation

`apply_deployment_mode(..., validate=True)` (the default) calls
`Config.validate()` with the fixture's `require_onchain` flag immediately
after applying overrides. A fixture that doesn't produce a valid
configuration raises `DeploymentModeValidationError` naming the mode and
the underlying `Config.validate()` failure — the diagnostic points
directly at which fixture is inconsistent and why, instead of surfacing as
an unrelated runtime error downstream.

## Local validation commands

```bash
pytest tests/test_deployment_modes.py -v
```
