"""Scenario Store — Issue #536.

Persists and retrieves named anomaly scenarios for the scenario replay system.
Each scenario captures a frozen snapshot of:
  - The trade records that triggered the detection
  - The feature row computed from those trades
  - The risk score that was produced
  - Ground-truth metadata (label, campaign id, notes)

Scenarios are stored as newline-delimited JSON in
``data/scenarios/<scenario_id>.json`` so they are human-readable, diff-friendly,
and require no extra database dependency.

This module is imported by ``scripts/replay_scenario.py``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

SCENARIOS_DIR = Path("data/scenarios")


def _scenario_path(scenario_id: str) -> Path:
    return SCENARIOS_DIR / f"{scenario_id}.json"


def _derive_id(trades: list[dict[str, Any]], wallet: str, pair_id: str) -> str:
    """Stable scenario ID derived from wallet + pair + sorted trade IDs."""
    trade_ids = sorted(str(t.get("id", t.get("trade_id", ""))) for t in trades)
    raw = f"{wallet}|{pair_id}|{'|'.join(trade_ids)}"
    return "sc_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


class ScenarioStore:
    """CRUD interface for anomaly scenarios."""

    def __init__(self, scenarios_dir: Path | str = SCENARIOS_DIR) -> None:
        self._dir = Path(scenarios_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        wallet: str,
        pair_id: str,
        trades: list[dict[str, Any]],
        features: dict[str, Any],
        risk_score: dict[str, Any],
        label: int | None = None,
        campaign_id: str | None = None,
        notes: str = "",
        scenario_id: str | None = None,
    ) -> str:
        """Persist a scenario and return its ``scenario_id``."""
        sid = scenario_id or _derive_id(trades, wallet, pair_id)
        record = {
            "scenario_id": sid,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "wallet": wallet,
            "pair_id": pair_id,
            "label": label,
            "campaign_id": campaign_id,
            "notes": notes,
            "risk_score": risk_score,
            "features": features,
            "trades": trades,
        }
        path = self._dir / f"{sid}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        logger.info("Scenario saved: %s → %s", sid, path)
        return sid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, scenario_id: str) -> dict[str, Any]:
        """Load a scenario by ID.  Raises FileNotFoundError if not found."""
        path = self._dir / f"{scenario_id}.json"
        return json.loads(path.read_text())

    def list_scenarios(
        self,
        campaign_id: str | None = None,
        label: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return metadata (no trades/features) for all stored scenarios."""
        results = []
        for p in sorted(self._dir.glob("sc_*.json")):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if campaign_id is not None and data.get("campaign_id") != campaign_id:
                continue
            if label is not None and data.get("label") != label:
                continue
            results.append(
                {
                    "scenario_id": data["scenario_id"],
                    "created_at": data.get("created_at"),
                    "wallet": data.get("wallet"),
                    "pair_id": data.get("pair_id"),
                    "label": data.get("label"),
                    "campaign_id": data.get("campaign_id"),
                    "score": data.get("risk_score", {}).get("score"),
                    "notes": data.get("notes", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, scenario_id: str) -> bool:
        path = self._dir / f"{scenario_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Scenario deleted: %s", scenario_id)
            return True
        return False
