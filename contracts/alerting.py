"""Alert-channel contracts for the streaming / alerts boundary.

Defines the ``AlertChannel`` protocol so that alternative alert
backends (webhook, websocket, Slack, PagerDuty, …) can be
plugged in without changing ``AlertDispatcher``.
"""

from __future__ import annotations

import typing
from typing import Protocol, runtime_checkable


class AlertEvent(typing.TypedDict, total=False):
    """Shape of an alert event flowing through the alerting subsystem.

    Required
    --------
    - ``wallet`` — the flagged wallet
    - ``asset_pair`` — traded pair
    - ``score`` — risk score 0–100
    - ``detectors`` — list of detector names that fired
    - ``timestamp`` — unix seconds
    """

    wallet: str
    asset_pair: str
    score: int
    detectors: list[str]
    timestamp: int
    severity: str
    message: str


@runtime_checkable
class AlertChannel(Protocol):
    """Interface for an alert delivery channel.

    Usage::

        channel: AlertChannel = WebhookChannel(...)
        channel.dispatch(event)
    """

    def dispatch(self, event: AlertEvent) -> None:
        """Deliver *event* through this channel."""
