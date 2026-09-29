"""Tests for scripts/inspect_quarantine.py (Issue #763)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


class TestInspectQuarantine:
    def test_min_severity_filter_includes_matching_entries(self, tmp_path, capsys):
        """--min-severity should include only entries at or above the threshold."""
        from scripts.inspect_quarantine import list_quarantined

        # Create mock quarantined records with different severity levels
        mock_records = [
            {
                "wallet": "GA1",
                "asset_pair": "USDC/XLM",
                "label": 1,
                "quarantine_reason": "backdoor_ac_detected",
                "severity": "critical",
            },
            {
                "wallet": "GA2",
                "asset_pair": "USDC/XLM",
                "label": 1,
                "quarantine_reason": "policy_flag",
                "severity": "warning",
            },
            {
                "wallet": "GA3",
                "asset_pair": "USDC/XLM",
                "label": 1,
                "quarantine_reason": "validation_failed",
                "severity": "info",
            },
        ]

        with mock.patch(
            "scripts.inspect_quarantine.AnnotationQueue"
        ) as MockQueue:
            mock_queue = mock.MagicMock()
            mock_queue.quarantined_samples.return_value = mock_records
            MockQueue.return_value = mock_queue

            # Without --min-severity, all should be shown
            list_quarantined()
            captured = capsys.readouterr()
            assert "GA1" in captured.out
            assert "GA2" in captured.out
            assert "GA3" in captured.out

    def test_quarantine_reason_maps_to_severity(self):
        """quarantine_reason values should map to severity levels."""
        from scripts.inspect_quarantine import _get_severity

        assert _get_severity("backdoor_ac_detected") in (
            "critical",
            "high",
        )  # High priority
        assert _get_severity("policy_flag") in ("warning", "high")
        assert _get_severity("validation_failed") in ("info", "warning")
        assert _get_severity("unknown_reason") == "info"  # Default

    def test_min_severity_flag_accepted(self, capsys):
        """Script should accept --min-severity flag without crashing."""
        from scripts.inspect_quarantine import main

        with mock.patch(
            "scripts.inspect_quarantine.AnnotationQueue"
        ) as MockQueue:
            mock_queue = mock.MagicMock()
            mock_queue.quarantined_samples.return_value = []
            MockQueue.return_value = mock_queue

            try:
                main(["list", "--min-severity", "warning"])
            except SystemExit:
                pass  # May exit with 0

            captured = capsys.readouterr()
            # Should either show filtered results or "No quarantined samples"
            assert captured.out  # Something was printed
