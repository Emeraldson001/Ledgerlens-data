"""Streaming pipeline package — real-time detection components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streaming.feature_buffer import FeatureBuffer
    from streaming.streaming_scorer import StreamingScorer

__all__ = ["FeatureBuffer", "StreamingScorer"]


def __getattr__(name: str):
    if name == "FeatureBuffer":
        from streaming.feature_buffer import FeatureBuffer as _fb

        return _fb
    if name == "StreamingScorer":
        from streaming.streaming_scorer import StreamingScorer as _ss

        return _ss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
