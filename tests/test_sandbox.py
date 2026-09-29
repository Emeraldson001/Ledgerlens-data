"""Tests for scripts.sandbox — sandboxed execution checks for maintenance scripts."""

import socket

import pytest

from scripts.sandbox import SandboxViolation, dry_run_guard, sandboxed_execution


def test_sandboxed_execution_runs_block_normally():
    calls = []
    with sandboxed_execution():
        calls.append(1)
    assert calls == [1]


def test_sandboxed_execution_blocks_network_when_disallowed():
    with sandboxed_execution(allow_network=False):
        with pytest.raises(SandboxViolation):
            socket.socket()


def test_sandboxed_execution_restores_socket_after_block():
    original = socket.socket
    with sandboxed_execution(allow_network=False):
        pass
    assert socket.socket is original


def test_sandboxed_execution_allows_network_by_default():
    with sandboxed_execution():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()


def test_sandboxed_execution_propagates_underlying_exceptions():
    with pytest.raises(ValueError):
        with sandboxed_execution():
            raise ValueError("boom")


def test_dry_run_guard_skips_action_when_dry_run():
    called = []
    result = dry_run_guard(True, "delete everything", lambda: called.append(1))
    assert called == []
    assert result is None


def test_dry_run_guard_runs_action_when_not_dry_run():
    called = []

    def action():
        called.append(1)
        return "done"

    result = dry_run_guard(False, "delete everything", action)
    assert called == [1]
    assert result == "done"
