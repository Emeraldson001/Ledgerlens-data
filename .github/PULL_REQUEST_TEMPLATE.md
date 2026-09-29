## Summary

What does this PR change and why?

## Type of change

<!-- Check all that apply -->
- [ ] Bug fix
- [ ] New feature
- [ ] Data/schema change (affects feature columns, data models, Avro schema)
- [ ] Model change (affects training, inference, or evaluation metrics)
- [ ] Infrastructure / CI change
- [ ] Documentation only
- [ ] Security fix

## Changelog

<!-- REQUIRED: paste the entry you added to CHANGELOG.md under [Unreleased] -->
<!-- If this PR touches detection/feature_engineering.py, detection/model_*.py, -->
<!-- ingestion/data_models.py, data/*, or models/* an entry is mandatory.       -->

```
### Added / Changed / Fixed / ...
- <your entry here>
```

## Impact assessment

### Data / feature schema changes
<!-- If feature_engineering.py, data_models.py, or data/trade_avro_schema.json changed: -->
- [ ] No schema changes
- [ ] New feature columns added (update data/feature_dictionary.md)
- [ ] Feature columns removed or renamed (update feature_schema_hash; note downstream impact)
- [ ] Avro schema evolved (backward-compatible / breaking)

### Model changes
<!-- If model_training.py, model_inference.py, or models/* changed: -->
- [ ] No model changes
- [ ] Model retrained — include before/after AUC-ROC / F1 in the summary above
- [ ] Inference API changed — update ledgerlens-core if RiskScore shape changed
- [ ] Feature schema hash will change — all consumers need to reload models

### Downstream impact
<!-- Check any systems affected by this change -->
- [ ] `ledgerlens-api` — REST API response shape may change
- [ ] `ledgerlens-dashboard` — visualization or SHAP field names may change
- [ ] `ledgerlens-contract` — on-chain RiskScore struct may change
- [ ] `ledgerlens-core` — shared types or thresholds may change
- [ ] None

## Checklist

- [ ] `make test` passes locally
- [ ] `make lint` and `make format` are clean
- [ ] New/changed behavior has test coverage
- [ ] Added a `CHANGELOG.md` entry under `[Unreleased]`, or this PR is exempt because it only touches docs/CI/tests (see [Changelog entries](CONTRIBUTING.md#changelog-entries))
- [ ] If a shared contract changed (`RiskScore`, asset pair format, feature
      schema), linked issues/PRs in `ledgerlens-core` and downstream repos

<!--
If this PR touches a high-risk path (wire schema, shared contracts, feature
or training code, DB schema, tenant config, model metadata), CI will ask you
to add a short note under a specific heading. See
.github/review-checklists.md — or run `make check-review-gates` to see which
apply before you push.
-->

