"""End-to-end detection workflow examples for LedgerLens.

This package contains runnable examples that exercise the full detection
pipeline from raw trade data through to a LedgerLens Risk Score, without
requiring a live Stellar Horizon connection.  Each example:

- Generates synthetic trade data that mimics a specific on-chain pattern
- Runs the full detection stack (Benford engine, feature engineering, model
  inference, SHAP explainer)
- Prints the resulting risk score and top SHAP attributions

These examples double as integration smoke-tests: ``pytest examples/`` runs
all of them in CI to ensure the pipeline does not regress.

Run any example directly::

    python -m examples.e2e_clean_trading
    python -m examples.e2e_wash_trading_ring
    python -m examples.e2e_benford_anomaly
    python -m examples.e2e_cross_venue_coordination
    python -m examples.e2e_full_pipeline
"""
