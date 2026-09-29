"""Normalized event contract for blockchain ingestion adapters.

LedgerLens ingests raw data from Stellar Horizon today (``ingestion/
horizon_fetcher.py``, ``ingestion/historical_loader.py``) and produces
chain-specific Pydantic models (``ingestion/data_models.py``: ``Trade``,
``OrderBookEvent``, ``AccountActivity``). Downstream feature engineering
and detection code (``features/``, ``detection/``) consume those models
directly.

That works while there is exactly one source chain, but it means every new
ingestion source (an EVM bridge counterpart, a second DEX, a different
Stellar-compatible network) would need to either shoehorn its data into the
existing Stellar-specific models or fork the feature engineering layer per
chain. Neither scales.

This module defines a chain-agnostic **normalized event contract**
(``NormalizedEvent``) and an **adapter contract** (``ChainAdapter``) that
any ingestion source implements to produce it. Feature engineering and
detection code that only needs the normalized shape (timestamp, accounts,
asset, amount, event type) can be written once against
``NormalizedEvent`` regardless of which chain produced the underlying raw
event; chain-specific detail is preserved losslessly in ``raw``.

This does not replace ``ingestion/data_models.py`` -- existing pipelines
that depend on ``Trade``/``OrderBookEvent``/``AccountActivity`` keep
working unchanged. ``StellarAdapter`` (see ``stellar_adapter.py``) adapts
those existing models into the normalized contract for consumers that want
the chain-agnostic shape.
"""

from __future__ import annotations

import abc
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventType(StrEnum):
    """Chain-agnostic classification of a normalized ingestion event."""

    TRADE = "trade"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    TRANSFER = "transfer"
    ACCOUNT_CREATED = "account_created"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"


class NormalizedAsset(BaseModel):
    """Chain-agnostic asset identifier.

    ``symbol`` is the human-readable ticker (e.g. ``"USDC"``); ``issuer``
    disambiguates assets that share a symbol across issuers/contracts (a
    Stellar issuer account, an ERC-20 contract address, ...). ``native`` is
    ``True`` for a chain's base currency (XLM, ETH), which has no issuer.
    """

    symbol: str
    issuer: str | None = None
    native: bool = False

    def asset_id(self) -> str:
        if self.native:
            return f"native:{self.symbol}"
        return f"{self.symbol}:{self.issuer or 'unknown'}"


class NormalizedEvent(BaseModel):
    """Chain-agnostic representation of a single on-chain event.

    This is the shared contract every :class:`ChainAdapter` implementation
    must produce. Fields are intentionally a strict subset of what any
    supported chain can provide -- chain-specific detail that doesn't fit
    (e.g. a Soroban contract invocation's function selector) belongs in
    ``raw``, never as an ad hoc top-level field, so the contract stays
    stable as new chains are added.

    Attributes:
        event_id: globally unique within ``chain`` (chain-native tx/op id).
        chain: adapter-assigned chain identifier, e.g. ``"stellar"``.
        event_type: one of :class:`EventType`.
        occurred_at: event timestamp (ledger/block close time), UTC.
        account: primary account/address associated with the event
            (the initiator for a transfer, the trader for a trade).
        counterparty: secondary account/address, if any (the trade's
            other side, a transfer's recipient).
        asset: primary asset involved.
        counter_asset: second asset for two-sided events (trades, swaps);
            ``None`` for single-asset events (transfers, account creation).
        amount: primary amount, in the asset's natural (non-atomic) units.
        counter_amount: amount of ``counter_asset``, if applicable.
        raw: the untouched source payload, for adapters/consumers that
            need chain-specific detail the normalized shape can't express.
    """

    event_id: str
    chain: str
    event_type: EventType
    occurred_at: datetime
    account: str
    counterparty: str | None = None
    asset: NormalizedAsset
    counter_asset: NormalizedAsset | None = None
    amount: float = Field(ge=0)
    counter_amount: float | None = Field(default=None, ge=0)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chain")
    @classmethod
    def _chain_lowercase(cls, v: str) -> str:
        if not v:
            raise ValueError("chain must be a non-empty string")
        return v.lower()

    def dedup_key(self) -> str:
        """Stable key for cross-adapter deduplication of the same event."""
        return f"{self.chain}:{self.event_id}"


class AdapterValidationError(ValueError):
    """Raised when a raw event cannot be normalized.

    Carries the offending chain name and raw payload (truncated) so a
    failure in a batch ingestion run is attributable to a specific record
    instead of aborting with a bare ``KeyError``/``TypeError`` deep inside
    adapter logic.
    """

    def __init__(self, chain: str, reason: str, raw_event: Any = None):
        self.chain = chain
        self.reason = reason
        self.raw_event = raw_event
        preview = repr(raw_event)
        if len(preview) > 200:
            preview = preview[:200] + "...(truncated)"
        super().__init__(f"[{chain}] failed to normalize event: {reason} (raw={preview})")


class ChainAdapter(abc.ABC):
    """Contract every blockchain ingestion adapter must implement.

    An adapter is responsible for one thing: turning a raw, chain-specific
    event payload into a :class:`NormalizedEvent`. It must not perform
    feature engineering, scoring, or persistence -- those stay in
    ``features/`` and ``detection/`` and operate on the normalized
    contract, independent of which adapter produced it.
    """

    #: Adapter-assigned chain identifier, e.g. "stellar", "ethereum".
    #: Must be a stable, lowercase, non-empty string across releases --
    #: it is embedded in NormalizedEvent.chain and used as the adapter
    #: registry key (see registry.py).
    chain: str

    @abc.abstractmethod
    def normalize(self, raw_event: Any) -> NormalizedEvent:
        """Convert a single raw event into a :class:`NormalizedEvent`.

        Implementations should raise :class:`AdapterValidationError`
        (rather than a bare exception) when *raw_event* cannot be
        normalized, so callers can distinguish a malformed/unsupported
        record from a programming error.
        """
        raise NotImplementedError

    def normalize_batch(
        self, raw_events: list[Any]
    ) -> tuple[list[NormalizedEvent], list[AdapterValidationError]]:
        """Normalize a batch, collecting per-record failures instead of
        aborting the whole batch on the first bad record.

        Returns:
            ``(normalized_events, errors)`` -- every raw event that failed
            normalization is represented in ``errors`` rather than raising,
            so a single malformed record in a large historical backfill
            doesn't discard the rest of the batch.
        """
        normalized: list[NormalizedEvent] = []
        errors: list[AdapterValidationError] = []
        for raw_event in raw_events:
            try:
                normalized.append(self.normalize(raw_event))
            except AdapterValidationError as exc:
                errors.append(exc)
            except Exception as exc:  # noqa: BLE001 - wrap unexpected errors too
                errors.append(AdapterValidationError(self.chain, str(exc), raw_event))
        return normalized, errors
