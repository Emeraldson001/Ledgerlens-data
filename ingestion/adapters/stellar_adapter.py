"""Adapter that normalizes existing Stellar ingestion models.

Wraps the models already produced by the Horizon ingestion pipeline
(``ingestion/data_models.py``: ``Trade``, ``OrderBookEvent``,
``AccountActivity``) into :class:`~ingestion.adapters.base.NormalizedEvent`,
so consumers written against the normalized contract work with the
existing Stellar pipeline without requiring any change to
``horizon_fetcher.py`` / ``historical_loader.py`` / the raw Horizon
response shape.
"""

from __future__ import annotations

from typing import Any

from ingestion.adapters.base import (
    AdapterValidationError,
    ChainAdapter,
    EventType,
    NormalizedAsset,
    NormalizedEvent,
)
from ingestion.data_models import AccountActivity, OrderBookEvent, Trade

_ORDER_ACTION_TO_EVENT_TYPE = {
    "created": EventType.ORDER_PLACED,
    "cancelled": EventType.ORDER_CANCELLED,
    "updated": EventType.ORDER_PLACED,
}


def _asset_from_stellar(asset: Any) -> NormalizedAsset:
    native = asset.issuer is None
    return NormalizedAsset(symbol=asset.code, issuer=asset.issuer, native=native)


class StellarAdapter(ChainAdapter):
    """Normalizes Stellar ``Trade`` / ``OrderBookEvent`` / ``AccountActivity``.

    Accepts any of the three raw model types; dispatch is by ``isinstance``
    rather than a discriminator field because the existing Horizon
    ingestion loaders already hand back typed objects, not raw dicts.
    """

    chain = "stellar"

    def normalize(self, raw_event: Any) -> NormalizedEvent:
        if isinstance(raw_event, Trade):
            return self._normalize_trade(raw_event)
        if isinstance(raw_event, OrderBookEvent):
            return self._normalize_order_book_event(raw_event)
        if isinstance(raw_event, AccountActivity):
            return self._normalize_account_activity(raw_event)
        raise AdapterValidationError(
            self.chain,
            f"unsupported raw event type {type(raw_event).__name__}; expected "
            "Trade, OrderBookEvent, or AccountActivity",
            raw_event,
        )

    def _normalize_trade(self, trade: Trade) -> NormalizedEvent:
        try:
            return NormalizedEvent(
                event_id=trade.trade_id,
                chain=self.chain,
                event_type=EventType.TRADE,
                occurred_at=trade.ledger_close_time,
                account=trade.base_account,
                counterparty=trade.counter_account,
                asset=_asset_from_stellar(trade.base_asset),
                counter_asset=_asset_from_stellar(trade.counter_asset),
                amount=trade.base_amount,
                counter_amount=trade.counter_amount,
                raw=trade.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterValidationError(self.chain, str(exc), trade) from exc

    def _normalize_order_book_event(self, event: OrderBookEvent) -> NormalizedEvent:
        event_type = _ORDER_ACTION_TO_EVENT_TYPE.get(event.action)
        if event_type is None:
            raise AdapterValidationError(
                self.chain, f"unknown order book action {event.action!r}", event
            )
        try:
            return NormalizedEvent(
                event_id=event.event_id,
                chain=self.chain,
                event_type=event_type,
                occurred_at=event.ledger_close_time,
                account=event.account,
                asset=_asset_from_stellar(event.selling),
                counter_asset=_asset_from_stellar(event.buying),
                amount=event.amount,
                raw=event.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterValidationError(self.chain, str(exc), event) from exc

    def _normalize_account_activity(self, activity: AccountActivity) -> NormalizedEvent:
        try:
            return NormalizedEvent(
                event_id=f"account-created:{activity.account_id}",
                chain=self.chain,
                event_type=EventType.ACCOUNT_CREATED,
                occurred_at=activity.account_created_at,
                account=activity.account_id,
                counterparty=activity.funding_account,
                asset=NormalizedAsset(symbol="XLM", native=True),
                amount=0.0,
                raw=activity.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterValidationError(self.chain, str(exc), activity) from exc
