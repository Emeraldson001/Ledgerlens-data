"""Soroban event listener for governance and emergency-pause contract events.

Polls the Soroban RPC ``getEvents`` endpoint for two event types:

- ``threshold_changed`` (topic symbol ``t_changed``) — updates the local
  ``config.RISK_SCORE_FLAG_THRESHOLD`` at runtime when M-of-N keyholders
  approve a new value via the governance contract.
- ``contract_paused`` (topic symbol ``c_paused``) — signals the local scoring
  pipeline to halt until an ``contract_unpaused`` event is received.

Usage::

    listener = SorobanEventListener(
        governance_contract_id="C...",
        pause_contract_id="C...",
    )
    listener.run_forever()          # blocking loop
    # Or use asyncio:
    await listener.poll_once()
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests
from sqlalchemy import Integer, String, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Score-oracle event parsing, persistence, and stale-score alerting
# ---------------------------------------------------------------------------
#
# Separate from the governance/pause polling above: this section listens for
# score-oracle contract events (``score_read``, ``score_updated``,
# ``threshold_updated``) emitted by the on-chain LedgerLens score consumer
# contract, persists them (with wallet/consumer addresses HMAC-hashed for
# privacy) and raises a "stale score consumption" alert when a caller reads a
# score that has since drifted materially from the current score.

EVENT_HMAC_SECRET = os.getenv("EVENT_HMAC_SECRET", "ledgerlens-soroban-event-hmac-default")

STALE_SCORE_ALERT_THRESHOLD = 20

# Ledgers a Soroban event must be buried under before we treat it as final.
# Stellar's SCP gives fast probabilistic finality, but an event read straight
# off the tip can still be superseded; acting on one that later disappears
# would leave local state describing chain history that never happened. Events
# newer than this are held back entirely: not dispatched, not persisted, and
# the watermark is not advanced past them, so the next poll re-reads them.
DEFAULT_CONFIRMATION_DEPTH = 10

_KNOWN_EVENT_TYPES = {"score_read", "score_updated", "threshold_updated"}


def _hash_address(address: str) -> str:
    """HMAC-SHA256 hex digest of a Stellar address, keyed by EVENT_HMAC_SECRET."""
    return _hmac.new(EVENT_HMAC_SECRET.encode(), address.encode(), hashlib.sha256).hexdigest()


def _parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _decode_value_map(value: dict | None) -> dict[str, Any]:
    """Flatten a Soroban ScVal map (``{"type": "map", "value": [...]}``) into
    a plain ``{key: value}`` dict, keyed by each entry's symbol name."""
    if not value or value.get("type") != "map":
        return {}
    result: dict[str, Any] = {}
    for entry in value.get("value", []):
        key = entry.get("key", {}).get("value")
        if key is None:
            continue
        result[key] = entry.get("val", {}).get("value")
    return result


@dataclass
class ContractEvent:
    """A parsed, privacy-scrubbed score-oracle contract event."""

    event_type: str
    ledger_sequence: int
    timestamp: datetime
    # Soroban's own paging token for this event: globally unique and stable
    # across re-reads, which is what makes ingestion idempotent. Synthesised
    # deterministically when the source shape does not carry one.
    event_id: str | None = None
    contract_id: str | None = None
    score: int | None = None
    asset_pair: str | None = None
    wallet_id_hash: str | None = None
    consumer_address_hash: str | None = None
    old_threshold: int | None = None
    new_threshold: int | None = None


class _EventBase(DeclarativeBase):
    pass


class ContractEventRecord(_EventBase):
    """Append-only persisted record of one score-oracle contract event."""

    __tablename__ = "soroban_contract_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # UNIQUE is the actual idempotency guarantee: a replayed batch after a
    # partial-batch crash re-inserts the same event_id and is rejected by the
    # database rather than relying on the caller to have checked first.
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, unique=True, index=True)
    contract_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ledger_sequence: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    asset_pair: Mapped[str | None] = mapped_column(String, nullable=True)
    wallet_id_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    consumer_address_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    old_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    new_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)


class _WatermarkRecord(_EventBase):
    """Per-contract last-processed ledger sequence, for resumable polling."""

    __tablename__ = "soroban_event_watermarks"

    contract_id: Mapped[str] = mapped_column(String, primary_key=True)
    ledger_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


def _get_session_factory(db_url: str) -> sessionmaker:
    engine = create_engine(db_url)
    _EventBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _event_identity(
    raw: dict,
    payload: dict,
    event_type: str,
    ledger_sequence: int,
    timestamp_str: str | None,
) -> str:
    """Return a stable unique identity for one event.

    Soroban RPC supplies ``id`` (its paging token), which is already unique and
    stable across re-reads -- exactly what deduplication needs. The Horizon
    effects fallback shape carries no such field, so derive one by hashing the
    event's own content. Both are stable under replay, which is the property
    that matters; neither collapses two distinct events that happen to share a
    ledger sequence, which a ``(contract_id, ledger_sequence)`` key would.
    """
    native_id = raw.get("id") or payload.get("id")
    if native_id:
        return str(native_id)

    digest = hashlib.sha256()
    for part in (
        payload.get("contractId") or raw.get("contractId") or "",
        event_type,
        str(ledger_sequence),
        timestamp_str or "",
        _canonical_repr(payload.get("topic")),
        _canonical_repr(payload.get("value")),
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"")
    return f"synthetic:{digest.hexdigest()}"


def _canonical_repr(value: Any) -> str:
    """Deterministic string form of a nested topic/value structure."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def parse_contract_event(raw: dict) -> ContractEvent | None:
    """Parse a raw Soroban RPC (or Horizon effects fallback) event dict.

    Returns ``None`` for events with no topics or an unrecognised event type
    (topic[0]) — these are ignored rather than raising, since new on-chain
    event types may be added without breaking the listener.
    """
    if not raw:
        return None

    # Horizon effects shape nests the Soroban payload under "data" and puts
    # the close timestamp at the top level under "created_at".
    if "data" in raw and "topic" in raw.get("data", {}):
        payload = raw["data"]
        timestamp_str = raw.get("created_at")
    else:
        payload = raw
        timestamp_str = raw.get("ledgerClosedAt")

    topics = payload.get("topic") or []
    if not topics:
        return None

    event_type = topics[0].get("value")
    if event_type not in _KNOWN_EVENT_TYPES:
        return None

    ledger_sequence = int(payload.get("ledger", 0))
    timestamp = _parse_timestamp(timestamp_str) if timestamp_str else datetime.now(UTC)
    contract_id = payload.get("contractId") or raw.get("contractId")
    event_id = _event_identity(raw, payload, event_type, ledger_sequence, timestamp_str)

    wallet_id_hash = None
    if len(topics) > 1 and topics[1].get("type") == "address":
        wallet_id_hash = _hash_address(topics[1]["value"])

    fields = _decode_value_map(payload.get("value"))
    consumer_raw = fields.get("consumer")

    return ContractEvent(
        event_type=event_type,
        ledger_sequence=ledger_sequence,
        timestamp=timestamp,
        event_id=event_id,
        contract_id=contract_id,
        score=fields.get("score"),
        asset_pair=fields.get("asset_pair"),
        wallet_id_hash=wallet_id_hash,
        consumer_address_hash=_hash_address(consumer_raw) if consumer_raw else None,
        old_threshold=fields.get("old_threshold"),
        new_threshold=fields.get("new_threshold"),
    )


def persist_event(event: ContractEvent, session_factory: sessionmaker) -> bool:
    """Append *event* to ``soroban_contract_events``, ignoring a replay.

    Returns ``True`` when a new row was written, ``False`` when this event was
    already stored. Idempotency matters because the watermark only advances at
    the end of a batch: a crash partway through means the surviving events are
    re-read on restart, and without this every restart duplicated them.

    The pre-check is an optimisation, not the guarantee -- two workers racing
    can both pass it. The UNIQUE constraint on ``event_id`` is what actually
    prevents the duplicate, so the ``IntegrityError`` path is the real one.
    """
    with session_factory() as session:
        if event.event_id is not None:
            existing = session.scalar(
                select(ContractEventRecord.id).where(ContractEventRecord.event_id == event.event_id)
            )
            if existing is not None:
                return False

        session.add(
            ContractEventRecord(
                event_id=event.event_id,
                contract_id=event.contract_id,
                event_type=event.event_type,
                ledger_sequence=event.ledger_sequence,
                timestamp=event.timestamp.isoformat(),
                score=event.score,
                asset_pair=event.asset_pair,
                wallet_id_hash=event.wallet_id_hash,
                consumer_address_hash=event.consumer_address_hash,
                old_threshold=event.old_threshold,
                new_threshold=event.new_threshold,
            )
        )
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return False
        return True


def get_watermark(contract_id: str, session_factory: sessionmaker) -> int:
    """Return the last-processed ledger sequence for *contract_id* (0 if none)."""
    with session_factory() as session:
        row = session.get(_WatermarkRecord, contract_id)
        return row.ledger_sequence if row is not None else 0


def set_watermark(contract_id: str, ledger_sequence: int, session_factory: sessionmaker) -> None:
    """Persist the last-processed ledger sequence for *contract_id*."""
    with session_factory() as session:
        row = session.get(_WatermarkRecord, contract_id)
        if row is None:
            session.add(_WatermarkRecord(contract_id=contract_id, ledger_sequence=ledger_sequence))
        else:
            row.ledger_sequence = ledger_sequence
        session.commit()


def fetch_latest_ledger(rpc_url: str, timeout: int = 10) -> int | None:
    """Return the current ledger sequence, or ``None`` if it cannot be read.

    ``None`` is deliberately distinct from ``0``: callers treat an unknown tip
    as "cannot establish finality" and hold events back rather than assuming
    everything is confirmed.
    """
    try:
        resp = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getLatestLedger"},
            timeout=timeout,
        )
        resp.raise_for_status()
        sequence = resp.json().get("result", {}).get("sequence")
        return int(sequence) if sequence is not None else None
    except Exception:
        logger.warning("Could not read latest ledger from %s", rpc_url, exc_info=True)
        return None


def split_confirmed(
    raw_events: list[dict],
    latest_ledger: int | None,
    confirmation_depth: int = DEFAULT_CONFIRMATION_DEPTH,
) -> tuple[list[dict], list[dict]]:
    """Partition *raw_events* into (confirmed, pending) by confirmation depth.

    An event is confirmed once ``latest_ledger - event_ledger >= depth``.
    When the tip is unknown, or the depth is zero, behaviour degrades to the
    caller's explicit choice: an unknown tip holds everything back (safe), a
    zero depth confirms everything (opt-out, preserving the old behaviour).
    """
    if confirmation_depth <= 0:
        return list(raw_events), []
    if latest_ledger is None:
        return [], list(raw_events)

    cutoff = latest_ledger - confirmation_depth
    confirmed: list[dict] = []
    pending: list[dict] = []
    for raw in raw_events:
        payload = raw.get("data") if "data" in raw and "topic" in raw.get("data", {}) else raw
        ledger = int(payload.get("ledger", raw.get("ledger", 0)) or 0)
        (confirmed if ledger <= cutoff else pending).append(raw)
    return confirmed, pending


def check_stale_score_alert(
    event: ContractEvent,
    current_score: int | None,
    dispatcher: Any,
    threshold: int = STALE_SCORE_ALERT_THRESHOLD,
) -> bool:
    """Fire a stale-score-consumption alert if *event* is a ``score_read``
    whose consumed score has since drifted from *current_score* by more than
    *threshold* points. Returns whether the alert fired.
    """
    if event is None or event.event_type != "score_read":
        return False
    if current_score is None or event.score is None:
        return False

    delta = abs(current_score - event.score)
    if delta <= threshold:
        return False

    dispatcher.dispatch(
        event,
        {
            "delta": delta,
            "stale_consumption": True,
            "consumed_score": event.score,
            "current_score": current_score,
            "wallet_id_hash": event.wallet_id_hash,
        },
    )
    return True


class ScoreOracleEventListener:
    """Polls a Soroban score-oracle contract, persisting events and raising
    stale-score-consumption alerts.

    Distinct from :class:`SorobanEventListener` (governance/pause polling
    above): this listener tracks a per-contract watermark for resumable
    polling and can run its poll loop on a background thread via
    :meth:`start_background` / :meth:`stop`.
    """

    def __init__(
        self,
        contract_id: str,
        db_url: str,
        rpc_url: str | None = None,
        dispatcher: Any | None = None,
        current_score_fn: Callable[[str], int | None] | None = None,
        stale_threshold: int = STALE_SCORE_ALERT_THRESHOLD,
        poll_interval: float = 5.0,
        confirmation_depth: int = DEFAULT_CONFIRMATION_DEPTH,
    ) -> None:
        self.contract_id = contract_id
        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.poll_interval = poll_interval
        self.confirmation_depth = confirmation_depth
        self._session_factory = _get_session_factory(db_url)
        self._dispatcher = dispatcher
        self._current_score_fn = current_score_fn
        self._stale_threshold = stale_threshold
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _fetch_raw_events(self) -> list[dict]:
        """Fetch raw events since the last watermark via Soroban RPC getEvents."""
        watermark = get_watermark(self.contract_id, self._session_factory)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getEvents",
            "params": {
                "startLedger": watermark,
                "filters": [{"type": "contract", "contractIds": [self.contract_id]}],
                "pagination": {"limit": 200},
            },
        }
        resp = requests.post(self.rpc_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get("events", [])

    def process_batch(
        self,
        raw_events: list[dict],
        latest_ledger: int | None = None,
    ) -> list[ContractEvent]:
        """Parse, persist, and alert on a batch of raw events; advance the
        watermark to the highest ledger sequence seen. Returns the parsed
        events (unrecognised raw events are silently skipped).

        When *latest_ledger* is supplied, events not yet buried under
        ``confirmation_depth`` ledgers are held back entirely -- not persisted,
        not alerted on, and the watermark is not advanced past them -- so a
        later poll re-reads them once final. :meth:`poll_once` supplies it.

        This method performs no network I/O: the chain tip is an argument, not
        something fetched here, so callers that already hold a batch of raw
        events can process it offline and tests need no RPC endpoint.
        """
        if self.confirmation_depth > 0 and latest_ledger is not None:
            raw_events, pending = split_confirmed(
                raw_events, latest_ledger, self.confirmation_depth
            )
            if pending:
                logger.debug(
                    "Holding %d unconfirmed event(s) below %d-ledger confirmation depth",
                    len(pending),
                    self.confirmation_depth,
                )

        parsed_events: list[ContractEvent] = []
        max_ledger: int | None = None

        for raw in raw_events:
            event = parse_contract_event(raw)
            if event is None:
                continue

            persist_event(event, self._session_factory)
            parsed_events.append(event)
            max_ledger = (
                event.ledger_sequence
                if max_ledger is None
                else max(max_ledger, event.ledger_sequence)
            )

            if (
                event.event_type == "score_read"
                and self._dispatcher is not None
                and self._current_score_fn is not None
            ):
                current_score = self._current_score_fn(event.wallet_id_hash)
                check_stale_score_alert(
                    event, current_score, self._dispatcher, threshold=self._stale_threshold
                )

        if max_ledger is not None:
            set_watermark(self.contract_id, max_ledger, self._session_factory)
        return parsed_events

    def start_background(self) -> threading.Thread:
        """Start the poll loop on a daemon background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the background poll loop to stop after its current iteration."""
        self._stop_event.set()

    def poll_once(self) -> list[ContractEvent]:
        """Fetch one batch from the chain and process the confirmed part.

        This is the only path that reads from the chain, so it is where
        finality is enforced. If the tip cannot be read the batch is held back
        wholesale rather than assumed final -- the watermark does not advance,
        so nothing is lost, it is merely deferred to the next poll.
        """
        raw_events = self._fetch_raw_events()
        if not raw_events:
            # Nothing to place relative to the tip, so do not pay for it.
            return []

        latest_ledger = fetch_latest_ledger(self.rpc_url) if self.confirmation_depth > 0 else None
        if self.confirmation_depth > 0 and latest_ledger is None:
            logger.warning(
                "Chain tip unavailable; deferring %d event(s) rather than "
                "treating them as final",
                len(raw_events),
            )
            return []
        return self.process_batch(raw_events, latest_ledger=latest_ledger)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("ScoreOracleEventListener: poll error")
            self._stop_event.wait(self.poll_interval)


_DEFAULT_POLL_INTERVAL = 5  # seconds (≈1 Stellar ledger close)


class SorobanEventListener:
    """Long-polls Soroban RPC for governance and pause events.

    Parameters
    ----------
    governance_contract_id:
        Contract ID of the ThresholdGovernanceContract (issue #238).
    pause_contract_id:
        Contract ID of the EmergencyPauseContract (issue #241).
    rpc_url:
        Soroban RPC endpoint; defaults to ``config.SOROBAN_RPC_URL``.
    poll_interval:
        Seconds between ``getEvents`` calls.
    on_threshold_changed:
        Optional callback invoked with the new threshold value (int)
        whenever a ``threshold_changed`` event is received.
    on_contract_paused:
        Optional callback invoked with the pause reason (str) whenever
        a ``contract_paused`` event is received.
    on_contract_unpaused:
        Optional callback invoked (no args) when an ``contract_unpaused``
        event is received.
    """

    def __init__(
        self,
        governance_contract_id: str,
        pause_contract_id: str,
        rpc_url: str | None = None,
        poll_interval: int = _DEFAULT_POLL_INTERVAL,
        on_threshold_changed: Callable[[int], None] | None = None,
        on_contract_paused: Callable[[str], None] | None = None,
        on_contract_unpaused: Callable[[], None] | None = None,
        db_url: str | None = None,
        confirmation_depth: int = DEFAULT_CONFIRMATION_DEPTH,
    ) -> None:
        self.governance_contract_id = governance_contract_id
        self.pause_contract_id = pause_contract_id
        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.poll_interval = poll_interval
        self.confirmation_depth = confirmation_depth
        self._on_threshold_changed = on_threshold_changed or self._default_threshold_handler
        self._on_contract_paused = on_contract_paused or self._default_pause_handler
        self._on_contract_unpaused = on_contract_unpaused or self._default_unpause_handler
        self._paused: bool = False

        # Durable resume position. Without a db_url the cursor stays in memory
        # and a restart rewinds to ledger 0 -- which, because getEvents is
        # bounded to 200 results, silently skips pause/unpause/threshold events
        # rather than replaying them. That is the failure mode this listener
        # exists to prevent, so supply db_url in any real deployment.
        self._session_factory = _get_session_factory(db_url) if db_url else None
        self._watermark_key = f"gov:{governance_contract_id}|pause:{pause_contract_id}"
        self._start_ledger: int = self._load_start_ledger()

    def _load_start_ledger(self) -> int:
        if self._session_factory is None:
            logger.warning(
                "SorobanEventListener has no db_url: resume position is "
                "in-memory only and will reset to ledger 0 on restart, which "
                "can silently skip safety-critical pause events."
            )
            return 0
        return get_watermark(self._watermark_key, self._session_factory)

    def _save_start_ledger(self, ledger: int) -> None:
        self._start_ledger = ledger
        if self._session_factory is not None:
            set_watermark(self._watermark_key, ledger, self._session_factory)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def run_forever(self) -> None:
        """Block and poll indefinitely."""
        logger.info(
            "SorobanEventListener starting (governance=%s pause=%s)",
            self.governance_contract_id,
            self.pause_contract_id,
        )
        while True:
            try:
                self.poll_once()
            except Exception:
                logger.exception("SorobanEventListener: poll error")
            time.sleep(self.poll_interval)

    def poll_once(self) -> list[dict]:
        """Fetch and dispatch new events since the last seen ledger."""
        events = self._fetch_events()
        for event in events:
            self._dispatch(event)
        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_events(self) -> list[dict]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getEvents",
            "params": {
                "startLedger": self._start_ledger,
                "filters": [
                    {
                        "type": "contract",
                        "contractIds": [
                            self.governance_contract_id,
                            self.pause_contract_id,
                        ],
                    }
                ],
                "pagination": {"limit": 200},
            },
        }
        resp = requests.post(self.rpc_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events: list[dict] = data.get("result", {}).get("events", [])
        if not events:
            return events

        if self.confirmation_depth > 0:
            latest_ledger = fetch_latest_ledger(self.rpc_url)
            if latest_ledger is None:
                logger.warning(
                    "Chain tip unavailable; deferring %d governance/pause "
                    "event(s) rather than treating them as final",
                    len(events),
                )
                return []
            events, pending = split_confirmed(events, latest_ledger, self.confirmation_depth)
            if pending:
                logger.debug(
                    "Holding %d unconfirmed governance/pause event(s) below "
                    "%d-ledger confirmation depth",
                    len(pending),
                    self.confirmation_depth,
                )
            if not events:
                return events

        # Advance the cursor past the confirmed events only, and persist it, so
        # a restart resumes here instead of rewinding to ledger 0.
        self._save_start_ledger(events[-1].get("ledger", self._start_ledger) + 1)
        return events

    def _dispatch(self, event: dict) -> None:
        topics: list = event.get("topic", [])
        if not topics:
            return

        # Soroban topic[0] is the event name symbol
        event_name = self._sym_to_str(topics[0]) if topics else ""
        contract_id = event.get("contractId", "")
        value = event.get("value")

        if event_name == "t_changed" and contract_id == self.governance_contract_id:
            new_threshold = self._extract_u32(value)
            if new_threshold is not None:
                logger.info("Governance: threshold_changed → %d", new_threshold)
                self._on_threshold_changed(new_threshold)

        elif event_name == "c_paused" and contract_id == self.pause_contract_id:
            reason = self._extract_string(value) or "unknown"
            logger.warning("Emergency pause received: %s", reason)
            self._paused = True
            self._on_contract_paused(reason)

        elif event_name == "c_unpaused" and contract_id == self.pause_contract_id:
            logger.info("Contract unpaused")
            self._paused = False
            self._on_contract_unpaused()

    @staticmethod
    def _sym_to_str(scval: dict) -> str:
        return scval.get("sym", scval.get("str", ""))

    @staticmethod
    def _extract_u32(scval: dict | None) -> int | None:
        if scval is None:
            return None
        return scval.get("u32")

    @staticmethod
    def _extract_string(scval: dict | None) -> str | None:
        if scval is None:
            return None
        return scval.get("str") or scval.get("sym")

    # ------------------------------------------------------------------
    # Default handlers (update config in-process)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_threshold_handler(new_threshold: int) -> None:
        config.RISK_SCORE_FLAG_THRESHOLD = new_threshold
        logger.info(
            "RISK_SCORE_FLAG_THRESHOLD updated to %d via on-chain governance",
            new_threshold,
        )

    @staticmethod
    def _default_pause_handler(reason: str) -> None:
        logger.critical("Scoring pipeline HALTED by emergency pause: %s", reason)

    @staticmethod
    def _default_unpause_handler() -> None:
        logger.info("Scoring pipeline resumed after emergency unpause")
