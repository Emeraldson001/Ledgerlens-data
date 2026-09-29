"""Tests for scripts/inspect_importers.py CLI tool."""

import subprocess
import sys
from io import StringIO

import pytest

from ingestion.importer_registry import (
    DataSource,
    DataType,
    ImporterCapability,
    ImporterMetadata,
    get_registry,
    reset_registry,
)
from ingestion.registered_importers import verify_registration


class TestInspectImportersCLI:
    """Test the inspect_importers CLI script."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_list_flag_prints_names_one_per_line(self):
        """Test that --list flag prints importer names, one per line."""
        # Register test importers
        registry = get_registry()

        test_metadata = ImporterMetadata(
            name="test_importer_1",
            description="Test importer 1",
            capabilities=ImporterCapability.STREAMING,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_SSE}),
        )
        registry.register(object, test_metadata)

        test_metadata2 = ImporterMetadata(
            name="test_importer_2",
            description="Test importer 2",
            capabilities=ImporterCapability.BULK,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_REST}),
        )
        registry.register(object, test_metadata2)

        # Run the --list command
        result = subprocess.run(
            [sys.executable, "-m", "scripts.inspect_importers", "--list"],
            capture_output=True,
            text=True,
            cwd="/home/ajidokwu/Desktop/Drips/Jambox/Ledgerlens-data",
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        lines = result.stdout.strip().split("\n")

        # Should contain both importer names
        assert "test_importer_1" in lines
        assert "test_importer_2" in lines

    def test_list_flag_with_real_registry(self):
        """Test --list flag with the actual registered importers."""
        # Verify real importers are registered
        status = verify_registration()
        assert all(status.values()), "Some importers failed to register"

        # Run the --list command
        result = subprocess.run(
            [sys.executable, "-m", "scripts.inspect_importers", "--list"],
            capture_output=True,
            text=True,
            cwd="/home/ajidokwu/Desktop/Drips/Jambox/Ledgerlens-data",
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        lines = result.stdout.strip().split("\n")

        # Should have multiple importers
        assert len(lines) > 0
        # Should contain known importers
        assert "horizon_streamer" in lines
        assert "historical_loader" in lines

    def test_list_flag_no_extra_output(self):
        """Test that --list produces no decoration, just names."""
        registry = get_registry()

        test_metadata = ImporterMetadata(
            name="simple_importer",
            description="Simple test importer",
            capabilities=ImporterCapability.BULK,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_REST}),
        )
        registry.register(object, test_metadata)

        result = subprocess.run(
            [sys.executable, "-m", "scripts.inspect_importers", "--list"],
            capture_output=True,
            text=True,
            cwd="/home/ajidokwu/Desktop/Drips/Jambox/Ledgerlens-data",
        )

        assert result.returncode == 0
        output = result.stdout

        # Should contain the name
        assert "simple_importer" in output

        # Should NOT contain headers or decorations
        assert "Found" not in output
        assert "Description" not in output
        assert "Capabilities" not in output
        assert "registered importers" not in output

    def test_default_list_command_unchanged(self):
        """Test that default 'list' subcommand still produces detailed output."""
        registry = get_registry()

        test_metadata = ImporterMetadata(
            name="test_detailed",
            description="This is a detailed test importer",
            capabilities=ImporterCapability.STREAMING,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_SSE}),
        )
        registry.register(object, test_metadata)

        result = subprocess.run(
            [sys.executable, "-m", "scripts.inspect_importers", "list"],
            capture_output=True,
            text=True,
            cwd="/home/ajidokwu/Desktop/Drips/Jambox/Ledgerlens-data",
        )

        assert result.returncode == 0
        output = result.stdout

        # Should have detailed information
        assert "Found" in output
        assert "test_detailed" in output
        assert "Description" in output
        assert "Capabilities" in output
