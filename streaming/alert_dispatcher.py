"""Alert dispatcher: threshold check, deduplication, and outbound delivery.

Supports three delivery channels:
  - stdout  — structured single-line log (local dev / CI)
  - webhook — HTTP POST to ALERT_WEBHOOK_URL (must be https://)
  - websocket — push to an injected ws_client handle
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any

import requests

from config import config
from streaming.alert_ledger import AlertDeliveryLedger
from utils.logging import get_logger

if TYPE_CHECKING:
    from streaming.rl_threshold_controller import ThresholdController

logger = get_logger(__name__)


class AlertDispatcher:
    """Filter, deduplicate, and deliver risk-score alerts."""

    def __init__(
        self,
        channel: str = "stdout",
        webhook_url: str | None = None,
        ws_client: Any = None,
        alert_cooldown_seconds: int = 3600,
        threshold: int | None = None,
        threshold_controller: ThresholdController | None = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        delivery_ledger: AlertDeliveryLedger | None = None,
    ):
        if channel not in ("stdout", "webhook", "websocket"):
            raise ValueError(f"Unknown alert channel: {channel!r}")

        self._channel = channel
        self._webhook_url = (
            webhook_url if webhook_url is not None else os.getenv("ALERT_WEBHOOK_URL")
        )
        self._ws_client = ws_client
        self._alert_cooldown_seconds = alert_cooldown_seconds
        self._threshold = threshold if threshold is not None else config.RISK_SCORE_FLAG_THRESHOLD
        self._threshold_controller = threshold_controller
        self._max_retries = max_retries
        self._base_delay = base_delay
        # Optional — no ledger means no reconciliation coverage but zero
        # behavior/side-effect change from before this dispatcher gained a
        # ledger (Issue #670, required scope E). The live entry points
        # (scripts/stream.py, scripts/kafka_workers.py) pass a real one.
        self._delivery_ledger = delivery_ledger

        if channel == "webhook":
            if not self._webhook_url:
                raise ValueError("ALERT_WEBHOOK_URL is required when alert channel is 'webhook'")
            if not self._webhook_url.startswith("https://"):
                raise ValueError(
                    "ALERT_WEBHOOK_URL must start with https:// — got " f"{self._webhook_url!r}"
                )

        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        """Deliver an alert if *risk_score* exceeds threshold and wallet is not cooling down."""
        if risk_score["score"] < self._get_threshold(pair_id):
            return

        with self._lock:
            now = time.time()
            if wallet in self._cooldowns and now < self._cooldowns[wallet]:
                if self._delivery_ledger is not None:
                    self._delivery_ledger.record(
                        wallet,
                        pair_id,
                        risk_score,
                        "suppressed_cooldown",
                        channel=self._channel,
                        reason=f"cooldown active until {self._cooldowns[wallet]:.0f}",
                    )
                return
            self._cooldowns[wallet] = now + self._alert_cooldown_seconds

        self._deliver(wallet, risk_score, pair_id)

    # ------------------------------------------------------------------
    # Internal delivery
    # ------------------------------------------------------------------

    def _get_threshold(self, asset: str) -> float:
        if self._threshold_controller is not None:
            return self._threshold_controller.get_threshold(asset)
        return float(self._threshold)

    def _deliver(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        if self._channel == "stdout":
            self._deliver_stdout(wallet, risk_score, pair_id)
        elif self._channel == "webhook":
            self._deliver_webhook(wallet, risk_score, pair_id)
        elif self._channel == "websocket":
            self._deliver_websocket(wallet, risk_score, pair_id)

    def _deliver_stdout(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        # Human-readable line on real stdout — this is what the "stdout" channel
        # name promises, and what operators tailing the process expect to see.
        print(
            f"[ALERT] wallet={wallet} pair={pair_id}"
            f" score={risk_score['score']}"
            f" benford={risk_score.get('benford_flag')}"
            f" ml={risk_score.get('ml_flag')}"
            f" confidence={risk_score.get('confidence')}"
        )
        # Structured JSON log for aggregation/observability tooling.
        logger.info(
            "Alert dispatched",
            extra={
                "wallet": wallet,
                "pair_id": pair_id,
                "score": risk_score["score"],
                "benford_flag": risk_score.get("benford_flag"),
                "ml_flag": risk_score.get("ml_flag"),
                "confidence": risk_score.get("confidence"),
                "score_lower": risk_score.get("score_lower"),
                "score_upper": risk_score.get("score_upper"),
                "coverage_guarantee": risk_score.get("coverage_guarantee"),
            },
        )
        if self._delivery_ledger is not None:
            self._delivery_ledger.record(wallet, pair_id, risk_score, "delivered", channel="stdout")

    def _write_to_dead_letter(
        self, payload: dict, *, wallet: str, risk_score: dict, pair_id: str, reason: str
    ) -> None:
        try:
            path = config.ALERT_DEAD_LETTER_PATH
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logger.error("Failed to write alert to dead-letter file: %s", exc)
        if self._delivery_ledger is not None:
            self._delivery_ledger.record(
                wallet, pair_id, risk_score, "dead_lettered", channel="webhook", reason=reason
            )

    def _deliver_webhook(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        payload = {**risk_score, "wallet": wallet, "pair_id": pair_id}
        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.post(self._webhook_url or "", json=payload, timeout=5)
                resp.raise_for_status()
                if self._delivery_ledger is not None:
                    self._delivery_ledger.record(
                        wallet, pair_id, risk_score, "delivered", channel="webhook"
                    )
                return
            except requests.HTTPError as exc:
                status_code = exc.response.status_code
                if 400 <= status_code < 500:
                    logger.warning(
                        "Webhook delivery failed (HTTP %s) — client error, will not retry",
                        status_code,
                    )
                    self._write_to_dead_letter(
                        payload,
                        wallet=wallet,
                        risk_score=risk_score,
                        pair_id=pair_id,
                        reason=f"HTTP {status_code} client error",
                    )
                    return
                else:
                    logger.warning(
                        "Webhook delivery failed (HTTP %s) on attempt %d",
                        status_code,
                        attempt + 1,
                    )
            except requests.RequestException as exc:
                logger.warning(
                    "Webhook delivery failed on attempt %d: %s",
                    attempt + 1,
                    type(exc).__name__,
                )

            if attempt < self._max_retries:
                delay = self._base_delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                logger.error("Webhook delivery failed after %d retries", self._max_retries)
                self._write_to_dead_letter(
                    payload,
                    wallet=wallet,
                    risk_score=risk_score,
                    pair_id=pair_id,
                    reason=f"exhausted {self._max_retries} retries",
                )

    def _deliver_websocket(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        payload = {**risk_score, "wallet": wallet, "pair_id": pair_id}
        self._ws_client.send(json.dumps(payload))
        if self._delivery_ledger is not None:
            self._delivery_ledger.record(
                wallet, pair_id, risk_score, "delivered", channel="websocket"
            )
