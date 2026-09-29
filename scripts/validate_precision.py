#!/usr/bin/env python3
"""CLI tool to validate numeric precision in the codebase.

This tool scans the codebase for potential precision issues:
- Float arithmetic on financial values
- Missing Decimal conversions
- Incorrect stroops handling
- Precision loss in calculations
- Dataset validation

Usage
-----
Scan all Python files::

    python -m scripts.validate_precision

Scan specific modules::

    python -m scripts.validate_precision --modules detection ingestion

Validate a dataset::

    python -m scripts.validate_precision --validate-dataset data/trades.parquet

Fix simple issues automatically::

    python -m scripts.validate_precision --fix

Exit Codes
----------
0: No issues found
1: Warnings found (non-critical)
2: Errors found (critical precision issues)
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Colorize text for terminal output."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


# ---------------------------------------------------------------------------
# AST Analysis
# ---------------------------------------------------------------------------


class PrecisionIssue:
    """Represents a precision issue found in code."""

    def __init__(
        self,
        filepath: Path,
        line_number: int,
        severity: str,
        issue_type: str,
        description: str,
        suggestion: str | None = None,
    ):
        self.filepath = filepath
        self.line_number = line_number
        self.severity = severity  # "error", "warning", "info"
        self.issue_type = issue_type
        self.description = description
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Format issue for display."""
        severity_colors = {
            "error": Colors.RED,
            "warning": Colors.YELLOW,
            "info": Colors.BLUE,
        }
        severity_symbols = {
            "error": "✗",
            "warning": "⚠",
            "info": "ℹ",
        }

        color = severity_colors.get(self.severity, "")
        symbol = severity_symbols.get(self.severity, "?")

        lines = [
            f"{colorize(symbol, color)} {colorize(self.severity.upper(), color)} "
            f"[{self.issue_type}]",
            f"  File: {self.filepath}:{self.line_number}",
            f"  {self.description}",
        ]

        if self.suggestion:
            lines.append(f"  💡 Suggestion: {self.suggestion}")

        return "\n".join(lines)


class PrecisionAnalyzer(ast.NodeVisitor):
    """AST visitor to detect precision issues."""

    FINANCIAL_KEYWORDS = {
        "amount",
        "volume",
        "price",
        "balance",
        "stroops",
        "value",
        "total",
        "fee",
    }

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.issues: list[PrecisionIssue] = []
        self.imports: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Track imports."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            self.imports.add(node.module)
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Check binary operations."""
        # Check for float operations on financial values
        if self._is_arithmetic_op(node.op):
            if self._involves_financial_variable(node):
                # Check if Decimal is being used
                if "decimal" not in self.imports and "DecimalAmount" not in self.imports:
                    self.issues.append(
                        PrecisionIssue(
                            filepath=self.filepath,
                            line_number=node.lineno,
                            severity="warning",
                            issue_type="float_arithmetic",
                            description=(
                                "Arithmetic operation on potential financial value "
                                "without Decimal import"
                            ),
                            suggestion="Import and use DecimalAmount from utils.decimal_guards",
                        )
                    )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for float() on financial values
        if isinstance(node.func, ast.Name) and node.func.id == "float":
            if node.args and self._involves_financial_variable(node.args[0]):
                self.issues.append(
                    PrecisionIssue(
                        filepath=self.filepath,
                        line_number=node.lineno,
                        severity="error",
                        issue_type="float_conversion",
                        description="Converting financial value to float (precision loss)",
                        suggestion="Use DecimalAmount or Decimal instead",
                    )
                )

        # Check for round() with financial values
        if isinstance(node.func, ast.Name) and node.func.id == "round":
            if node.args and self._involves_financial_variable(node.args[0]):
                if "decimal" not in self.imports:
                    self.issues.append(
                        PrecisionIssue(
                            filepath=self.filepath,
                            line_number=node.lineno,
                            severity="warning",
                            issue_type="float_rounding",
                            description="Using round() on financial value (may use float)",
                            suggestion="Use DecimalAmount.round() for precision-safe rounding",
                        )
                    )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check type annotations."""
        # Check for float annotations on financial variables
        if isinstance(node.target, ast.Name):
            var_name = node.target.id.lower()
            if any(keyword in var_name for keyword in self.FINANCIAL_KEYWORDS):
                if isinstance(node.annotation, ast.Name) and node.annotation.id == "float":
                    self.issues.append(
                        PrecisionIssue(
                            filepath=self.filepath,
                            line_number=node.lineno,
                            severity="warning",
                            issue_type="float_annotation",
                            description=f"Financial variable '{node.target.id}' annotated as float",
                            suggestion="Use Decimal or DecimalAmount type annotation",
                        )
                    )

        self.generic_visit(node)

    def _is_arithmetic_op(self, op: ast.operator) -> bool:
        """Check if operator is arithmetic."""
        return isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod))

    def _involves_financial_variable(self, node: ast.AST) -> bool:
        """Check if expression involves financial variable."""
        if isinstance(node, ast.Name):
            var_name = node.id.lower()
            return any(keyword in var_name for keyword in self.FINANCIAL_KEYWORDS)

        if isinstance(node, ast.Attribute):
            attr_name = node.attr.lower()
            return any(keyword in attr_name for keyword in self.FINANCIAL_KEYWORDS)

        if isinstance(node, ast.BinOp):
            return self._involves_financial_variable(
                node.left
            ) or self._involves_financial_variable(node.right)

        return False


# ---------------------------------------------------------------------------
# Codebase Scanning
# ---------------------------------------------------------------------------


def scan_file(filepath: Path) -> list[PrecisionIssue]:
    """Scan a single Python file for precision issues."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        analyzer = PrecisionAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer.issues

    except SyntaxError as e:
        logger.warning(f"Syntax error in {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error scanning {filepath}: {e}")
        return []


def scan_codebase(
    root_dir: Path,
    modules: list[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> list[PrecisionIssue]:
    """Scan the entire codebase for precision issues.

    Parameters
    ----------
    root_dir : Path
        Root directory to scan
    modules : list[str], optional
        Specific modules to scan (e.g., ['detection', 'ingestion'])
    exclude_dirs : set[str], optional
        Directories to exclude from scanning

    Returns
    -------
    list[PrecisionIssue]
        All issues found
    """
    if exclude_dirs is None:
        exclude_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "build",
            "dist",
            "*.egg-info",
        }

    all_issues = []

    # Determine which directories to scan
    if modules:
        scan_dirs = [root_dir / module for module in modules]
    else:
        scan_dirs = [root_dir]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            logger.warning(f"Directory not found: {scan_dir}")
            continue

        # Find all Python files
        python_files = []
        for path in scan_dir.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in path.parts for excluded in exclude_dirs):
                continue
            python_files.append(path)

        logger.info(f"Scanning {len(python_files)} files in {scan_dir}")

        for filepath in python_files:
            issues = scan_file(filepath)
            all_issues.extend(issues)

    return all_issues


# ---------------------------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------------------------


def validate_dataset(filepath: Path) -> list[PrecisionIssue]:
    """Validate numeric precision in a dataset (Parquet, CSV, etc.).

    Parameters
    ----------
    filepath : Path
        Path to dataset file

    Returns
    -------
    list[PrecisionIssue]
        Issues found in the dataset
    """
    issues = []

    try:
        # Load dataset
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        else:
            logger.error(f"Unsupported file format: {filepath.suffix}")
            return issues

        logger.info(f"Loaded dataset with {len(df)} rows, {len(df.columns)} columns")

        # Check for float columns that might be financial
        financial_keywords = [
            "amount",
            "volume",
            "price",
            "balance",
            "value",
            "total",
            "fee",
        ]

        for col in df.columns:
            col_lower = col.lower()

            # Check if column name suggests financial data
            is_financial = any(keyword in col_lower for keyword in financial_keywords)

            if is_financial and df[col].dtype == "float64":
                issues.append(
                    PrecisionIssue(
                        filepath=filepath,
                        line_number=0,
                        severity="warning",
                        issue_type="float_column",
                        description=(
                            f"Column '{col}' appears to be financial data "
                            f"but is stored as float64"
                        ),
                        suggestion=(
                            "Consider using Decimal or storing as int64 " "(stroops for Stellar)"
                        ),
                    )
                )

                # Check for precision issues
                sample_values = df[col].dropna().head(100)
                for idx, value in sample_values.items():
                    # Check if value has more than 7 decimal places
                    str_value = f"{value:.15f}".rstrip("0")
                    if "." in str_value:
                        decimal_places = len(str_value.split(".")[1])
                        if decimal_places > 7:
                            issues.append(
                                PrecisionIssue(
                                    filepath=filepath,
                                    line_number=int(idx) + 2,  # +2 for header and 1-indexing
                                    severity="info",
                                    issue_type="excess_precision",
                                    description=(
                                        f"Column '{col}' row {idx}: value has "
                                        f"{decimal_places} decimal places (Stellar uses 7)"
                                    ),
                                    suggestion="Round to 7 decimal places or store as stroops",
                                )
                            )
                            break  # Only report once per column

    except Exception as e:
        logger.error(f"Error validating dataset {filepath}: {e}")

    return issues


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_report(issues: list[PrecisionIssue]) -> dict[str, Any]:
    """Generate summary report of issues.

    Parameters
    ----------
    issues : list[PrecisionIssue]
        All issues found

    Returns
    -------
    dict
        Report summary with counts and groupings
    """
    report = {
        "total": len(issues),
        "by_severity": {"error": 0, "warning": 0, "info": 0},
        "by_type": {},
        "by_file": {},
    }

    for issue in issues:
        # Count by severity
        report["by_severity"][issue.severity] += 1

        # Count by type
        if issue.issue_type not in report["by_type"]:
            report["by_type"][issue.issue_type] = 0
        report["by_type"][issue.issue_type] += 1

        # Count by file
        filepath_str = str(issue.filepath)
        if filepath_str not in report["by_file"]:
            report["by_file"][filepath_str] = 0
        report["by_file"][filepath_str] += 1

    return report


def print_report(issues: list[PrecisionIssue], report: dict[str, Any]) -> None:
    """Print formatted report to console.

    Parameters
    ----------
    issues : list[PrecisionIssue]
        All issues found
    report : dict
        Summary report
    """
    print("\n" + "=" * 80)
    print(colorize("Numeric Precision Validation Report", Colors.BOLD))
    print("=" * 80 + "\n")

    # Summary
    print(colorize("Summary", Colors.BOLD))
    print(f"  Total issues: {report['total']}")
    print(f"  Errors:   {colorize(str(report['by_severity']['error']), Colors.RED)}")
    print(f"  Warnings: {colorize(str(report['by_severity']['warning']), Colors.YELLOW)}")
    print(f"  Info:     {colorize(str(report['by_severity']['info']), Colors.BLUE)}")
    print()

    # Issues by type
    if report["by_type"]:
        print(colorize("Issues by Type", Colors.BOLD))
        for issue_type, count in sorted(
            report["by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {issue_type}: {count}")
        print()

    # Files with most issues
    if report["by_file"]:
        print(colorize("Files with Most Issues", Colors.BOLD))
        sorted_files = sorted(report["by_file"].items(), key=lambda x: x[1], reverse=True)[:10]
        for filepath, count in sorted_files:
            print(f"  {count:3d} issues: {filepath}")
        print()

    # Detailed issues
    if issues:
        print(colorize("Detailed Issues", Colors.BOLD))
        print()

        # Sort by severity (errors first)
        severity_order = {"error": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(issues, key=lambda x: severity_order[x.severity])

        for issue in sorted_issues:
            print(issue)
            print()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate numeric precision in LedgerLens codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific modules to scan (e.g., detection ingestion)",
    )

    parser.add_argument(
        "--validate-dataset",
        type=Path,
        help="Validate a specific dataset file (Parquet or CSV)",
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the codebase (default: current directory)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix simple issues automatically (not implemented yet)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not detailed issues",
    )

    args = parser.parse_args()

    # Validate dataset if requested
    if args.validate_dataset:
        logger.info(f"Validating dataset: {args.validate_dataset}")
        issues = validate_dataset(args.validate_dataset)
    else:
        # Scan codebase
        logger.info(f"Scanning codebase in {args.root_dir}")
        issues = scan_codebase(args.root_dir, modules=args.modules)

    # Generate report
    report = generate_report(issues)

    # Output results
    if args.json:
        import json

        output = {
            "summary": report,
            "issues": [
                {
                    "file": str(issue.filepath),
                    "line": issue.line_number,
                    "severity": issue.severity,
                    "type": issue.issue_type,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                }
                for issue in issues
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if not args.quiet:
            print_report(issues, report)
        else:
            print(f"Found {report['total']} issues:")
            print(f"  Errors: {report['by_severity']['error']}")
            print(f"  Warnings: {report['by_severity']['warning']}")
            print(f"  Info: {report['by_severity']['info']}")

    # Fix issues if requested
    if args.fix:
        logger.warning("Auto-fix not implemented yet")

    # Determine exit code
    if report["by_severity"]["error"] > 0:
        return 2  # Critical errors
    elif report["by_severity"]["warning"] > 0:
        return 1  # Warnings
    else:
        return 0  # All clear


if __name__ == "__main__":
    sys.exit(main())
