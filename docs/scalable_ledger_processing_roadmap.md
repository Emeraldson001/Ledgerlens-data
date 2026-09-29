# Scalable ledger data processing roadmap

This roadmap defines the foundation for adding ledger providers, reusable
features, and reproducible model workflows without changing existing pipeline
behavior.

## Contracts

- Source connectors implement `LedgerSourceConnector` and return validated
  `SourceBatch` pages with provider-owned cursors.
- Detection pipelines consume feature data through `FeatureStoreBackend` using
  `FeatureKey` and `FeatureRecord` instead of depending on a specific online
  cache implementation.
- Model experiments record dataset hash, feature schema hash, parameters,
  metrics, artifacts, and optional Git SHA with `JsonlExperimentTracker`.

## Processing path

1. A provider connector fetches typed ledger records and emits a resumable
   cursor.
2. Feature builders produce wallet/pair/window feature vectors.
3. A feature store backend writes and serves schema-versioned `FeatureRecord`
   objects to batch and streaming detectors.
4. Training jobs log deterministic experiment metadata before model promotion.

## Near-term scale checkpoints

- Add provider adapters behind the connector contract rather than new one-off
  ingestion entry points.
- Promote feature schema version changes with compatibility checks before
  detectors consume cached data.
- Keep experiment logs append-only so CI and reviewers can audit training
  inputs without a hosted tracking dependency.
