"""The single write gate for ``config.MODEL_DIR`` (Grand 2 / issue #671).

Split out from ``detection.model_governance`` into its own leaf module (no
dependency on ``detection.model_training`` or anything else in the model
lifecycle) so that ``detection.model_training`` can call it without creating
an import cycle: ``model_training`` needs this guard, and
``detection.model_governance`` needs ``model_training.MODEL_REGISTRY`` — both
importing from each other directly would be circular.
``detection.model_governance`` re-exports both names below, so
``detection.model_governance.guard_production_write`` remains the documented
public entry point; this module is an implementation detail.
"""

from __future__ import annotations

import os

from config import config


class UngatedProductionWriteError(Exception):
    """Raised when code tries to write model artifacts to production MODEL_DIR
    outside the gated promotion path (``detection.model_governance.promote_candidate``)."""


def guard_production_write(target_dir: str) -> None:
    """Raise :class:`UngatedProductionWriteError` if *target_dir* is the live
    production ``MODEL_DIR`` and it already holds a promoted artifact.

    Called from ``detection.model_training.save_models`` and
    ``save_training_artifacts`` — the only two functions elsewhere in the
    codebase that write trained model files to disk. Training into any other
    directory (the normal case: a staging/candidate directory, then
    ``promote_candidate`` to go live) is always allowed.
    """
    prod_dir = os.path.abspath(config.MODEL_DIR)
    if os.path.abspath(target_dir) != prod_dir:
        return
    if not os.path.exists(os.path.join(prod_dir, "metrics.json")):
        # Nothing promoted yet (e.g. first-ever training run, or a fresh
        # environment) — allow bootstrapping production directly.
        return
    raise UngatedProductionWriteError(
        f"Refusing to write model artifacts directly to the production MODEL_DIR "
        f"({prod_dir}) — it already holds a promoted artifact. Train into a "
        "separate staging directory and call "
        "detection.model_governance.promote_candidate(...) to publish it, so the "
        "trust-chain and regression gates run before production is overwritten."
    )
