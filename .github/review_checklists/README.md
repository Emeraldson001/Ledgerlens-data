# Code Review Checklists

This directory contains reviewer checklists for high-risk categories of changes
in the LedgerLens-data repository. These checklists are triggered automatically
by the `.github/workflows/code-review-checklist.yml` workflow, which classifies
each PR's changed files and activates the relevant checklist jobs.

## Checklists

| File | When to use |
|------|-------------|
| [`feature_schema_changes.md`](feature_schema_changes.md) | `detection/feature_engineering.py`, `data/feature_dictionary.md`, `data/feature_ranges.json` |
| [`model_changes.md`](model_changes.md) | `detection/model_training.py`, `detection/model_inference.py`, `detection/ensemble_calibrator.py`, `models/**` |
| [`privacy_security_changes.md`](privacy_security_changes.md) | `detection/differential_privacy.py`, `detection/shap_explainer.py`, `detection/privacy/**`, `detection/adversarial/**`, `detection/audit_trail.py`, `utils/field_encryption.py` |
| [`data_persistence_changes.md`](data_persistence_changes.md) | `ingestion/data_models.py`, `data/trade_avro_schema.json`, `detection/persistence.py`, `detection/risk_score_store.py`, `data/**`, `models/**` |
| [`contract_integration_changes.md`](contract_integration_changes.md) | `integrations/**` |

## Risk levels

The workflow classifies each PR into a risk level:

| Level | Triggered by | Required reviewers |
|-------|--------------|--------------------|
| 🔴 **Critical** | privacy, data schema, or Soroban contract changes | 2 reviewers, including `@Ledger-Lenz/security` |
| 🟠 **High** | model training/inference or persistence changes | 2 reviewers from the owning team |
| 🟡 **Medium** | feature engineering changes | 1 reviewer from `@Ledger-Lenz/ml-core` |
| 🟢 **Low** | documentation, tests, monitoring | Standard 1 reviewer |

## How the automation works

1. A PR is opened touching any path in the `paths:` list of `code-review-checklist.yml`
2. The `classify-changes` job diffs the PR against `origin/main` and sets output flags
3. Conditional jobs run only for the triggered categories:
   - `feature-schema-checklist` — validates feature dict, ranges, backward compat
   - `model-change-checklist` — checks artifact integrity, BFT voting, metrics
   - `privacy-security-checklist` — validates DP params, audit trail, MI defence
   - `data-asset-checklist` — checks Avro compat, migration scripts, ground truth
   - `contract-integration-checklist` — verifies RiskScore alignment, testnet
4. `generate-review-summary` prints a consolidated report at the end

## Using checklists manually

Copy the relevant checklist into your PR description or a review comment and
tick items off as you verify them. For critical-risk PRs, both required reviewers
should independently complete the checklist before approving.
