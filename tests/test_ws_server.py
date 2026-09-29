"""Tests for streaming/ws_server.py bind-host security guard."""

import asyncio

import pytest

from streaming.ws_server import _is_loopback_host, run_ws_server


def test_is_loopback_host():
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("127.0.0.5")
    assert _is_loopback_host("localhost")
    assert _is_loopback_host("::1")

    assert not _is_loopback_host("0.0.0.0")
    assert not _is_loopback_host("192.168.1.10")
    assert not _is_loopback_host("not-an-address")


def test_non_loopback_bind_without_optin_raises(monkeypatch):
    """WS_BIND_HOST=0.0.0.0 with WS_ALLOW_EXTERNAL unset must fail closed."""
    monkeypatch.setenv("WS_BIND_HOST", "0.0.0.0")
    monkeypatch.delenv("WS_ALLOW_EXTERNAL", raising=False)

    with pytest.raises(ValueError, match="WS_ALLOW_EXTERNAL"):
        asyncio.run(run_ws_server())


def test_non_loopback_bind_with_optin_allowed(monkeypatch):
    """Setting WS_ALLOW_EXTERNAL=1 allows the non-loopback bind as documented."""
    monkeypatch.setenv("WS_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("WS_ALLOW_EXTERNAL", "1")

    class _FakeServe:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        "streaming.ws_server.websockets.serve", lambda *a, **k: _FakeServe()
    )

    async def _run() -> None:
        # The guard must pass; run_ws_server then awaits forever, so a timeout
        # (not a ValueError) proves the non-loopback bind was accepted.
        await asyncio.wait_for(run_ws_server(), timeout=0.2)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())
