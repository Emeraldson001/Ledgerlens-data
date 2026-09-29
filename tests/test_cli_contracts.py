"""Tests for scripts/check_cli_contracts.py and scripts/cli_contracts.py."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_cli_contracts as cli_check  # noqa: E402
from cli_contracts import CONTRACTS, CliArgument, CliContract  # noqa: E402


def _write_script(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


class TestExtraction:
    def test_extracts_single_alias_argument(self, tmp_path):
        path = _write_script(
            tmp_path,
            "s.py",
            """
            parser.add_argument("--wallet", help="x")
            """,
        )
        actual = cli_check._extract_actual_arguments(path)
        assert len(actual) == 1
        assert actual[0].aliases == ("--wallet",)
        assert actual[0].required is False

    def test_extracts_multiple_aliases_and_required_flag(self, tmp_path):
        path = _write_script(
            tmp_path,
            "s.py",
            """
            parser.add_argument("-q", "--quiet", action="store_true", required=True)
            """,
        )
        actual = cli_check._extract_actual_arguments(path)
        assert actual[0].aliases == ("-q", "--quiet")
        assert actual[0].required is True
        assert actual[0].matches("--quiet")
        assert actual[0].matches("-q")

    def test_extracts_positional_argument(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'parser.add_argument("wallet")\n')
        actual = cli_check._extract_actual_arguments(path)
        assert actual[0].aliases == ("wallet",)

    def test_ignores_non_add_argument_calls(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'print("add_argument")\nother.method("x")\n')
        assert cli_check._extract_actual_arguments(path) == []


class TestCheckContract:
    def _contract(self, *arguments: CliArgument) -> CliContract:
        return CliContract(
            script="s.py", command="python s.py", description="test", arguments=arguments
        )

    def test_matching_contract_produces_no_diagnostics(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'parser.add_argument("--pair", required=True)\n')
        contract = self._contract(CliArgument("--pair", required=True))
        actual = cli_check._extract_actual_arguments(path)
        assert cli_check.check_contract(contract, actual) == []

    def test_missing_declared_argument_is_flagged(self):
        contract = self._contract(CliArgument("--pair", required=True))
        diagnostics = cli_check.check_contract(contract, actual=[])
        assert len(diagnostics) == 1
        assert "--pair" in diagnostics[0]
        assert "no matching add_argument" in diagnostics[0]

    def test_undeclared_actual_argument_is_flagged(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'parser.add_argument("--secret-new-flag")\n')
        contract = self._contract()
        actual = cli_check._extract_actual_arguments(path)
        diagnostics = cli_check.check_contract(contract, actual)
        assert len(diagnostics) == 1
        assert "--secret-new-flag" in diagnostics[0]
        assert "undeclared" in diagnostics[0]

    def test_required_mismatch_is_flagged(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'parser.add_argument("--pair")\n')
        contract = self._contract(CliArgument("--pair", required=True))
        actual = cli_check._extract_actual_arguments(path)
        diagnostics = cli_check.check_contract(contract, actual)
        assert len(diagnostics) == 1
        assert "required mismatch" in diagnostics[0]

    def test_alias_on_either_side_satisfies_declared_flag(self, tmp_path):
        path = _write_script(tmp_path, "s.py", 'parser.add_argument("-q", "--quiet")\n')
        contract = self._contract(CliArgument("--quiet"))
        actual = cli_check._extract_actual_arguments(path)
        assert cli_check.check_contract(contract, actual) == []


class TestRealContracts:
    def test_every_contracted_script_file_exists(self):
        for name, contract in CONTRACTS.items():
            assert (cli_check.SCRIPTS_DIR / contract.script).is_file(), name

    def test_real_scripts_match_their_declared_contracts(self):
        """Guards against operational CLI drift: fails with an actionable
        diff if a contracted script's flags no longer match
        scripts/cli_contracts.py."""
        all_diagnostics = []
        for _name, contract in CONTRACTS.items():
            actual = cli_check._extract_actual_arguments(cli_check.SCRIPTS_DIR / contract.script)
            all_diagnostics.extend(cli_check.check_contract(contract, actual))
        assert all_diagnostics == [], "\n".join(all_diagnostics)

    def test_every_contract_declares_at_least_one_argument(self):
        for name, contract in CONTRACTS.items():
            assert len(contract.arguments) > 0, name
