# Contributing to ledgerlens-data

Thanks for your interest in contributing to LedgerLens! This repo holds the
data ingestion and fraud-detection layer — see the README's
[Organization Map](README.md#organization-map) for how it fits with the
other LedgerLens repos.

## Security

Before implementing changes that touch API endpoints, model loading, training data, or database persistence, review the [Security Threat Model](docs/security_threat_model.md) for STRIDE analysis and attack surface identification. High-risk components may require security architect review.

## Development setup

```bash
git clone https://github.com/<org>/ledgerlens-data.git
cd ledgerlens-data
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env  # then edit as needed
```

## Running checks locally

```bash
make lint     # ruff + black --check
make format   # ruff --fix + black
make test     # pytest (unit tests only — no network)
make check-env-example  # verify .env.example covers every config.py variable
```

Optionally install the pre-commit hooks so checks run automatically:

```bash
pip install pre-commit
pre-commit install
```

### Unit tests vs integration tests

`make test` runs `pytest tests/` and **never** hits the Testnet. All tests
under `tests/integration/` are automatically skipped unless
`LEDGERLENS_INTEGRATION_TESTS=1` is set.

To run the live Testnet integration tests locally:

```bash
# 1. Deploy the contract (once per testnet reset / keypair rotation)
python -m scripts.testnet_setup \
    --wasm-path ledgerlens_score.wasm \
    --wasm-sha256 <sha256-from-release> \
    --salt ci-testnet

# 2. Run integration tests
export LEDGERLENS_INTEGRATION_TESTS=1
export $(grep -v '^#' .env.testnet | xargs)
pytest tests/integration/ -v --timeout=120
```

See [`tests/integration/README.md`](tests/integration/README.md) for full
setup instructions, required environment variables, WASM version details,
and Testnet fee estimates.

The `testnet-integration.yml` CI workflow runs these tests on a weekly
schedule (Sundays 03:00 UTC) and on manual `workflow_dispatch` — it does
**not** run on pull requests so it never blocks a PR merge.

### Running a subset of tests

With 300+ files under `tests/`, running the full suite on every iteration is
slow. Use these `pytest` invocations to scope a run down while you iterate:

```bash
# A single test file
pytest tests/test_benford.py

# A single test function (-k matches by substring)
pytest tests/test_benford.py -k test_chi_square_statistic

# Everything except the slower integration and fuzz suites
pytest tests/ --ignore=tests/integration --ignore=tests/fuzz
```

`pyproject.toml` defines these pytest markers (`-m 'not integration and not
slow'` to exclude both):

| Marker | Meaning |
|---|---|
| `integration` | Live Testnet integration tests — deselect with `-m "not integration"` (also skipped automatically unless `LEDGERLENS_INTEGRATION_TESTS=1` is set, see above) |
| `slow` | Tests that run PPO training — deselect with `-m "not slow"` |
| `concurrency` | Concurrency validation tests for streaming workers — included by default |

`tests/fuzz/` is a separate atheris-based fuzzing suite, not run by plain
`pytest`; see [`tests/fuzz/README.md`](tests/fuzz/README.md) and `make fuzz`.

## Pull requests

- Keep PRs focused on a single logical change.
- Add or update tests for any behavior change.
- Run `make lint` and `make test` before opening a PR — CI runs the same
  checks on Python 3.11 and 3.12.
- If you change a shared contract (`RiskScore` shape, asset pair ID format,
  feature schema — see the README's "Shared Contracts" section), call that
  out in the PR description so consuming repos (`ledgerlens-core`,
  `ledgerlens-api`, `ledgerlens-contract`, `ledgerlens-dashboard`) can be
  updated.

### API contract compatibility

Any PR that touches `api/app.py` or anything under `contracts/` is gated by
[`scripts/check_api_compatibility.py`](scripts/check_api_compatibility.py),
which diffs the public API surface against the committed baseline and fails
the build on a breaking change. The check runs in CI via
[`.github/workflows/api-compatibility.yml`](.github/workflows/api-compatibility.yml)
on every pull request that modifies those paths.

Run it locally before opening such a PR:

```bash
python scripts/check_api_compatibility.py
```

#### Intentionally shipping a breaking change

A breaking change is only allowed when it is deliberate and acknowledged:

1. **Bump the API version.** Update the version constant in `api/app.py`
   (and the matching entry in `contracts/`) so the new surface is published
   under a new major/minor version.
2. **Regenerate the baseline.** Re-run the checker with the update flag to
   record the new contract as the accepted baseline:

   ```bash
   python scripts/check_api_compatibility.py --update-baseline
   ```

   Commit the regenerated baseline alongside the version bump so the CI gate
   sees the change as acknowledged rather than accidental.
3. **Notify downstream consumers.** Call out the break in the PR description
   and in the `CHANGELOG.md` entry, and notify the consuming repos
   (`ledgerlens-core`, `ledgerlens-api`, `ledgerlens-contract`,
   `ledgerlens-dashboard`) so they can pin or migrate before the release.

Without the version bump and regenerated baseline, the CI job fails and the
PR cannot merge.

### Changelog entries

Every PR that touches high-impact paths must include a `CHANGELOG.md` entry
under `## [Unreleased]`. The entry is enforced by
[`scripts/validate_changelog.py`](scripts/validate_changelog.py) in CI.

**Required format (Keep a Changelog):**

```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Behavior change description

### Fixed
- Bug fix description
```

- Entries must start with `- ` and be grouped under one of the standard
  subsections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
  `Security`.
- If your PR only touches documentation, CI configuration, or tests, a
  changelog entry is not required — but you still need to check the
  "Added a CHANGELOG.md entry, or this PR is exempt" box in the PR template.
- The CI job
  [`.github/workflows/changelog-validation.yml`](.github/workflows/changelog-validation.yml)
  runs `python scripts/validate_changelog.py --check-pr` on every pull
  request and fails the build if a high-impact path changed without a
  matching entry.

## Security

See [`docs/security_threat_model.md`](docs/security_threat_model.md) for the comprehensive STRIDE-based threat model. Key mitigations:

- **Model integrity:** Ed25519 signatures on `metrics.json`; SHA-256 verification of `.joblib` files
- **Label poisoning:** HMAC-SHA256 on annotations; baseline distribution tracking
- **Model inversion:** Gaussian-mechanism DP on SHAP explanations; per-wallet query budgeting
- **Byzantine robustness:** Trimmed-mean ensemble voting
- **Credential security:** Never commit signing keys or API credentials to version control

All security-relevant PRs must reference the threat model and document which mitigations are affected.

## Code style

- Formatting/linting is enforced by `ruff` and `black` (see
  `pyproject.toml`). Line length is 100.
- Favor small, composable functions following the existing module layout:
  `ingestion/` for data acquisition, `detection/` for scoring logic,
  `tests/` mirrors both.
- **Adding a new top-level module?** Also add a corresponding pattern to
  [`.github/CODEOWNERS`](.github/CODEOWNERS) — see
  [`.github/review-checklists.md`](.github/review-checklists.md) for the
  expected review-gate entry and an example.
- New feature columns added to `detection/feature_engineering.py` must be
  documented in the README's feature tables and accounted for in
  `detection/model_training.py::FEATURE_COLUMNS_EXCLUDE` handling.
- **Adding a new ML feature?** Follow the end-to-end guide in
  [`docs/contributor_feature_guide.md`](docs/contributor_feature_guide.md).
  It covers naming conventions, function signatures, range validation,
  dataset card updates, SHAP integration, and required test patterns —
  with a complete worked example using `counterparty_variance`.
- **Replacing a trained model artifact?** Run `make validate-artifacts`
  before committing — see
  [`docs/artifact_backward_compatibility.md`](docs/artifact_backward_compatibility.md)
  for the backward compatibility rules enforced against archived versions.
- **Deprecating a public function or class?** Use the `@deprecated` decorator
  documented in
  [`docs/deprecation_policy.md`](docs/deprecation_policy.md) and run
  `make check-deprecations` so removal versions are tracked and enforced.
- Run `make validate-docs` after editing any doc linked from this file — it
  checks for dead links, missing headings, and broken code examples.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include the asset
pair, wallet, and time window if reporting a detection accuracy issue —
that's usually enough to reproduce a Benford/feature calculation locally.

## Mutation testing

LedgerLens uses [mutmut](https://github.com/boxed/mutmut) to measure test
*effectiveness*, not just coverage. A mutation score of **≥ 80%** is
enforced in CI on the core scoring path:

- `detection/benford_engine.py`
- `detection/feature_engineering.py`
- `detection/model_inference.py`

### Running mutation tests locally

```bash
# Full run — same as CI (may take 10–15 minutes)
make mutation-test

# Run only and inspect results
mutmut run \
  --paths-to-mutate "detection/benford_en

/* … truncated 2671 chars — edit only what you need near the top … */
