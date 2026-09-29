# Data Quality Validation Framework

## Why

Validation logic in this repo is currently scattered and ad hoc: range
checks live next to `data/feature_ranges.json`, config presence checks are
bespoke in `tests/test_config_validation.py`, and trade-record shape checks
are implicit in `ingestion/data_models.py`. Each new importer or feature
pipeline that needs to validate incoming data re-derives its own checks,
and failures are usually reported as an opaque exception rather than "row
17, field `score`, value -5 is below minimum 0."

## What this adds

`utils/data_quality.py` provides a small rule-composition contract:

- `ValidationRule` — the `Protocol` every rule satisfies: a `name` and a
  `check(record) -> str | None` method.
- Built-in rules: `RequiredFieldRule`, `TypeRule`, `RangeRule`, `RegexRule`.
- `RangeRule.from_feature_ranges(path)` — optionally builds `RangeRule`
  instances directly from the existing `data/feature_ranges.json`, so
  bounds don't have to be duplicated by hand; returns `[]` (not an
  exception) if the file is absent, so it's safe to use in environments
  without that fixture.
- `DataQualityValidator` — composes rules, runs them against a single
  record (`validate`) or a batch (`validate_batch`), and returns a
  `ValidationReport` with **every** issue found, tagged with the failing
  rule name, field, message, and (for batches) record index.

```python
from utils.data_quality import DataQualityValidator, RequiredFieldRule, RangeRule

validator = DataQualityValidator([
    RequiredFieldRule("wallet"),
    RequiredFieldRule("score"),
    RangeRule("score", minimum=0, maximum=100),
])
report = validator.validate_batch(incoming_records)
if not report.passed:
    for issue in report.issues:
        log.warning("record %s failed %s: %s", issue.record_index, issue.rule_name, issue.message)
```

## Developer commands

```
pytest tests/test_data_quality.py -v

python -m scripts.validate_dataset --input data/some_export.jsonl \
    --required wallet --required score --range score:0:100
```

`scripts/validate_dataset.py` runs the framework over a JSON-lines file and
exits non-zero on any validation failure, so it can be dropped into a CI
step or a pre-ingest gate ahead of a bulk import (pairs naturally with
`ingestion/batch_processor.py` from this same change set — validate a
sample before running `BatchProcessor.run` over the full dataset).

Tests cover: passing/failing on required/type/range/regex rules, bool
values correctly rejected by numeric type/range checks, absent fields
being silent for type/range rules (so `RequiredFieldRule` is the single
source of truth for "missing" diagnostics), batch aggregation with
per-record indices, `fail_fast` early exit, loading rules from
`feature_ranges.json`, and the missing-file fallback.

## Design tradeoffs

- **Composable rule objects, not a schema DSL.** Matches the repo's
  existing preference for small explicit Python objects (e.g. the
  dataclass-based models in `ingestion/data_models.py`) over introducing a
  schema language or new dependency (e.g. `jsonschema`, `pydantic`
  validators) purely for this.
- **Non-raising `check()` contract.** Rules return a message string instead
  of raising, so a `DataQualityValidator` can always finish a full batch
  and report every issue at once, rather than stopping at the first
  exception — important for triaging a bad ingest run in one pass instead
  of a fix-one-fail-again loop.
- **Follow-up work:** wiring `RangeRule.from_feature_ranges()` into the
  feature pipeline (`features/feature_pipeline.py`) as an optional runtime
  guard, and adding a `SchemaRule` that validates against
  `data/trade_avro_schema.json` directly for parity with the Avro codec
  path (`ingestion/avro_codec.py`).
