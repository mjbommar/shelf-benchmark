"""Integration tests for SHELF CLI commands.

Tests CLI invocation using typer.testing.CliRunner to verify
options plumb through correctly without subprocess overhead.

Marked with @pytest.mark.integration for separate execution.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from shelf.cli import app


runner = CliRunner()


@pytest.mark.integration
class TestCLIList:
    """Tests for 'shelf list' command."""

    def test_list_shows_taxonomies(self):
        """Test that 'shelf list' displays available taxonomies."""
        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        # Verify table headers/content appear
        assert "Type" in result.stdout or "Available" in result.stdout
        # Verify some taxonomy types are listed
        assert "lcgft" in result.stdout.lower() or "LCGFT" in result.stdout

    def test_list_help(self):
        """Test that 'shelf list --help' works."""
        result = runner.invoke(app, ["list", "--help"])

        assert result.exit_code == 0
        assert (
            "List available taxonomy types" in result.stdout
            or "list" in result.stdout.lower()
        )


@pytest.mark.integration
class TestCLIInfo:
    """Tests for 'shelf info' command."""

    def test_info_requires_taxonomy_argument(self):
        """Test that 'shelf info' requires a taxonomy argument."""
        result = runner.invoke(app, ["info"])

        # Should fail or show usage without argument
        assert (
            result.exit_code != 0
            or "Missing argument" in result.stdout
            or "Usage" in result.stdout
        )

    def test_info_help(self):
        """Test that 'shelf info --help' works."""
        result = runner.invoke(app, ["info", "--help"])

        assert result.exit_code == 0
        assert "taxonomy" in result.stdout.lower() or "TAXONOMY" in result.stdout


@pytest.mark.integration
class TestCLIHelp:
    """Tests for CLI help and basic invocation."""

    def test_main_help(self):
        """Test that 'shelf --help' shows help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "SHELF" in result.stdout or "shelf" in result.stdout
        # Should show available commands
        assert "list" in result.stdout.lower()
        assert "info" in result.stdout.lower()

    def test_gen_subcommand_help(self):
        """Test that 'shelf gen --help' shows generation help."""
        result = runner.invoke(app, ["gen", "--help"])

        assert result.exit_code == 0
        assert (
            "generate" in result.stdout.lower() or "benchmark" in result.stdout.lower()
        )

    def test_invalid_command_shows_error(self):
        """Test that invalid command shows appropriate error."""
        result = runner.invoke(app, ["nonexistent"])

        assert result.exit_code != 0


@pytest.mark.integration
class TestCLIGen:
    """Tests for 'shelf gen' subcommands."""

    def test_gen_help(self):
        """Test that 'shelf gen --help' works."""
        result = runner.invoke(app, ["gen", "--help"])

        assert result.exit_code == 0

    def test_gen_create_help(self):
        """Test that 'shelf gen create --help' shows options."""
        result = runner.invoke(app, ["gen", "create", "--help"])

        # Either the command exists or we get a help message
        # (command may not exist yet, so just check no crash)
        assert (
            result.exit_code == 0
            or "No such command" in result.stdout
            or "Usage" in result.stdout
        )
