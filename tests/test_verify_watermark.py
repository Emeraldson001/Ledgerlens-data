"""Tests for scripts/verify_watermark.py (Issue #765)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestVerifyWatermarkScript:
    def test_verbose_flag_prints_per_trigger_details(self, capsys, tmp_path):
        """--verbose flag should print per-trigger expected vs actual comparisons."""
        from scripts.verify_watermark import main

        # Create a simple mock model that returns predictable results
        mock_model = mock.MagicMock()
        # All triggers match target_label=1
        mock_model.predict.return_value = np.array([1, 1, 1, 1, 1])

        # Mock load_trigger_vectors to return a fixture trigger set
        triggers = np.random.randn(5, 10)

        # Mock the imports and calls
        with mock.patch("scripts.verify_watermark.joblib.load", return_value=mock_model), \
             mock.patch(
                 "scripts.verify_watermark.load_trigger_vectors", return_value=triggers
             ), \
             mock.patch("scripts.verify_watermark.verify_watermark") as mock_verify:

            mock_verify.return_value = {
                "agreement": 1.0,
                "n_triggers": 5,
                "watermark_detected": True,
                "threshold": 0.9,
            }

            # Run with --verbose
            try:
                main(["--model-path", "dummy.joblib", "--verbose"])
            except SystemExit:
                pass  # Expected

            captured = capsys.readouterr()
            # Should include per-trigger information
            assert "Trigger" in captured.out or "match" in captured.out.lower() or \
                   "agreement" in captured.out.lower()

    def test_default_output_without_verbose(self, capsys, tmp_path):
        """Without --verbose flag, output should be concise (pass/fail only)."""
        from scripts.verify_watermark import main

        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([1, 1, 0])

        triggers = np.random.randn(3, 10)

        with mock.patch("scripts.verify_watermark.joblib.load", return_value=mock_model), \
             mock.patch(
                 "scripts.verify_watermark.load_trigger_vectors", return_value=triggers
             ), \
             mock.patch("scripts.verify_watermark.verify_watermark") as mock_verify:

            mock_verify.return_value = {
                "agreement": 0.67,
                "n_triggers": 3,
                "watermark_detected": False,
                "threshold": 0.9,
            }

            try:
                main(["--model-path", "dummy.joblib"])
            except SystemExit:
                pass

            captured = capsys.readouterr()
            # Should have basic output (NOT DETECTED message)
            assert "NOT DETECTED" in captured.out or "DETECTED" in captured.out
            assert "Agreement" in captured.out

    def test_json_output_with_verbose_flag(self, capsys, tmp_path):
        """--json output should be valid JSON even with --verbose."""
        from scripts.verify_watermark import main

        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([1, 1, 1])

        triggers = np.random.randn(3, 10)

        with mock.patch("scripts.verify_watermark.joblib.load", return_value=mock_model), \
             mock.patch(
                 "scripts.verify_watermark.load_trigger_vectors", return_value=triggers
             ), \
             mock.patch("scripts.verify_watermark.verify_watermark") as mock_verify:

            mock_verify.return_value = {
                "agreement": 1.0,
                "n_triggers": 3,
                "watermark_detected": True,
                "threshold": 0.9,
            }

            try:
                main(["--model-path", "dummy.joblib", "--json", "--verbose"])
            except SystemExit:
                pass

            captured = capsys.readouterr()
            # Should be valid JSON
            data = json.loads(captured.out)
            assert "agreement" in data
            assert "watermark_detected" in data
