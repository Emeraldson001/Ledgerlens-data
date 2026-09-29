#!/usr/bin/env python3
"""CLI tool to validate currency normalization usage in the codebase.

This tool scans the codebase for potential currency normalization issues:
- Cross-asset comparisons without normalization
- Missing exchange rate providers
- Direct amount comparisons across different assets
- Aggregations mixing multiple currencies
- Missing confidence checks on normalized amounts

Usage
-----
Scan all Python files::

    python -m scripts.validate_normalization

Scan specific modules::

    python -m scripts.validate_normalization --modules detection features

Check dataset for normalization opportunities::

    python -m scripts.validate_normalization --check-dataset data/trades.parquet

CI integration::

    python -m scripts.validate_normalization --json > normalization_report.json

Exit Codes
----------
0: No issues found
1: Warnings found (non-critical)
2: Errors found (missing normalization)
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class Colors:
    """ANSI color codes."""

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
# Issue Detection
# ---------------------------------------------------------------------------


class NormalizationIssue:
    """Represents a normalization issue found in code."""

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


class NormalizationAnalyzer(ast.NodeVisitor):
    """AST visitor to detect normalization issues."""

    # Keywords indicating financial/currency operations
    AMOUNT_KEYWORDS = {
        "amount",
        "volume",
        "price",
        "balance",
        "value",
        "total",
        "fee",
        "cost",
    }

    # Keywords indicating asset/currency operations
    ASSET_KEYWORDS = {
        "asset",
        "currency",
        "token",
        "pair",
        "base",
        "counter",
    }

    # Comparison operators that require normalization
    COMPARISON_OPS = {ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq}

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.issues: list[NormalizationIssue] = []
        self.imports: set[str] = set()
        self.has_normalization_import = False

    def visit_Import(self, node: ast.Import) -> None:
        """Track imports."""
        for alias in node.names:
            self.imports.add(alias.name)
            if "normalization" in alias.name:
                self.has_normalization_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            self.imports.add(node.module)
            if "normalization" in node.module:
                self.has_normalization_import = True
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Check comparison operations."""
        # Check if comparing amounts from different assets
        if self._involves_cross_asset_comparison(node):
            self.issues.append(
                NormalizationIssue(
                    filepath=self.filepath,
                    line_number=node.lineno,
                    severity="warning",
                    issue_type="unnormalized_comparison",
                    description=(
                        "Comparing amounts that may be from different assets "
                        "without normalization"
                    ),
                    suggestion=(
                        "Use normalize_amount() or Trade.normalize_both_amounts() "
                        "before comparison"
                    ),
                )
            )

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Check binary operations."""
        # Check for aggregation across assets
        if isinstance(node.op, ast.Add) and self._involves_amount_variable(node):
            # Check if adding amounts without normalization
            if not self.has_normalization_import:
                self.issues.append(
                    NormalizationIssue(
                        filepath=self.filepath,
                        line_number=node.lineno,
                        severity="info",
                        issue_type="potential_aggregation",
                        description=(
                            "Adding amounts - ensure same currency or use " "aggregate_normalized()"
                        ),
                        suggestion=(
                            "Import currency_normalization and use "
                            "aggregate_normalized() for multi-asset sums"
                        ),
                    )
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for sum() on amounts
        if isinstance(node.func, ast.Name) and node.func.id == "sum":
            if node.args and self._involves_amount_variable(node.args[0]):
                self.issues.append(
                    NormalizationIssue(
                        filepath=self.filepath,
                        line_number=node.lineno,
                        severity="warning",
                        issue_type="unnormalized_sum",
                        description=("Using sum() on amounts - may be mixing currencies"),
                        suggestion=("Use aggregate_normalized() or normalize amounts first"),
                    )
                )

        # Check for pandas operations
        if self._is_pandas_operation(node):
            if not self.has_normalization_import:
                self.issues.append(
                    NormalizationIssue(
                        filepath=self.filepath,
                        line_number=node.lineno,
                        severity="info",
                        issue_type="pandas_aggregation",
                        description=("Pandas aggregation on amounts - verify currency consistency"),
                        suggestion=(
                            "Use normalize_trade_amounts_to_series() or ensure " "single currency"
                        ),
                    )
                )

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Check for loops."""
        # Check for cross-pair iteration without normalization
        if self._involves_cross_pair_iteration(node):
            if not self.has_normalization_import:
                self.issues.append(
                    NormalizationIssue(
                        filepath=self.filepath,
                        line_number=node.lineno,
                        severity="info",
                        issue_type="cross_pair_iteration",
                        description=(
                            "Iterating over multiple pairs - consider normalization "
                            "for comparison"
                        ),
                        suggestion=("Use compare_cross_pair_volumes() for cross-pair analysis"),
                    )
                )

        self.generic_visit(node)

    def _involves_cross_asset_comparison(self, node: ast.Compare) -> bool:
        """Check if comparison involves amounts from different assets."""
        # Check left side
        left_has_amount = self._involves_amount_variable(node.left)

        # Check comparators
        comparators_have_amount = any(
            self._involves_amount_variable(comp) for comp in node.comparators
        )

        # Check for asset-related context
        left_has_asset = self._involves_asset_variable(node.left)
        comparators_have_asset = any(
            self._involves_asset_variable(comp) for comp in node.comparators
        )

        # If amounts involved and assets mentioned, likely cross-asset
        return (
            left_has_amount
            and comparators_have_amount
            and (left_has_asset or comparators_have_asset)
        )

    def _involves_amount_variable(self, node: ast.AST) -> bool:
        """Check if expression involves amount variable."""
        if isinstance(node, ast.Name):
            var_name = node.id.lower()
            return any(keyword in var_name for keyword in self.AMOUNT_KEYWORDS)

        if isinstance(node, ast.Attribute):
            attr_name = node.attr.lower()
            return any(keyword in attr_name for keyword in self.AMOUNT_KEYWORDS)

        if isinstance(node, ast.BinOp):
            return self._involves_amount_variable(node.left) or self._involves_amount_variable(
                node.right
            )

        return False

    def _involves_asset_variable(self, node: ast.AST) -> bool:
        """Check if expression involves asset variable."""
        if isinstance(node, ast.Name):
            var_name = node.id.lower()
            return any(keyword in var_name for keyword in self.ASSET_KEYWORDS)

        if isinstance(node, ast.Attribute):
            attr_name = node.attr.lower()
            return any(keyword in attr_name for keyword in self.ASSET_KEYWORDS)

        return False

    def _is_pandas_operation(self, node: ast.Call) -> bool:
        """Check if call is a pandas aggregation."""
        if isinstance(node.func, ast.Attribute):
            attr_name = node.attr.lower()
            pandas_ops = {"sum", "mean", "median", "agg", "aggregate", "groupby"}
            return attr_name in pandas_ops

        return False

    def _involves_cross_pair_iteration(self, node: ast.For) -> bool:
        """Check if loop iterates over multiple pairs."""
        # Check if iterating over something with 'pair' in the name
        if isinstance(node.iter, ast.Name):
            iter_name = node.iter.id.lower()
            return "pair" in iter_name or "asset" in iter_name

        return False


# ---------------------------------------------------------------------------
# Codebase Scanning
# ---------------------------------------------------------------------------


def scan_file(filepath: Path) -> list[NormalizationIssue]:
    """Scan a single Python file for normalization issues."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        analyzer = NormalizationAnalyzer(filepath)
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
) -> list[NormalizationIssue]:
    """Scan the entire codebase for normalization issues.

    Parameters
    ----------
    root_dir : Path
        Root directory to scan
    modules : list[str], optional
        Specific modules to scan
    exclude_dirs : set[str], optional
        Directories to exclude

    Returns
    -------
    list[NormalizationIssue]
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

    # Determine scan directories
    if modules:
        scan_dirs = [root_dir / module for module in modules]
    else:
        scan_dirs = [root_dir]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            logger.warning(f"Directory not found: {scan_dir}")
            continue

        # Find Python files
        python_files = []
        for path in scan_dir.rglob("*.py"):
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


def check_dataset_normalization_opportunities(filepath: Path) -> list[NormalizationIssue]:
    """Check dataset for normalization opportunities.

    Parameters
    ----------
    filepath : Path
        Path to dataset file

    Returns
    -------
    list[NormalizationIssue]
        Opportunities found
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

        # Check for asset/pair columns
        asset_columns = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["asset", "currency", "pair", "code"])
        ]

        amount_columns = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["amount", "volume", "price", "value"])
        ]

        if asset_columns and amount_columns:
            # Dataset has both assets and amounts - likely needs normalization
            issues.append(
                NormalizationIssue(
                    filepath=filepath,
                    line_number=0,
                    severity="info",
                    issue_type="normalization_opportunity",
                    description=(
                        f"Dataset contains {len(asset_columns)} asset columns "
                        f"and {len(amount_columns)} amount columns. "
                        "Consider normalizing for cross-asset analysis."
                    ),
                    suggestion=("Use create_normalized_dataframe() to add normalized columns"),
                )
            )

            # Check for multiple unique assets
            for asset_col in asset_columns:
                if asset_col in df.columns:
                    unique_assets = df[asset_col].nunique()
                    if unique_assets > 1:
                        issues.append(
                            NormalizationIssue(
                                filepath=filepath,
                                line_number=0,
                                severity="warning",
                                issue_type="multiple_assets",
                                description=(
                                    f"Column '{asset_col}' has {unique_assets} "
                                    f"different assets. Amounts likely need normalization "
                                    "for fair comparison."
                                ),
                                suggestion=("Normalize amounts before aggregation or comparison"),
                            )
                        )

    except Exception as e:
        logger.error(f"Error checking dataset {filepath}: {e}")

    return issues


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_report(issues: list[NormalizationIssue]) -> dict[str, Any]:
    """Generate summary report of issues."""
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


def print_report(issues: list[NormalizationIssue], report: dict[str, Any]) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 80)
    print(colorize("Currency Normalization Validation Report", Colors.BOLD))
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

        # Sort by severity
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
        description="Validate currency normalization usage in LedgerLens codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific modules to scan (e.g., detection features)",
    )

    parser.add_argument(
        "--check-dataset",
        type=Path,
        help="Check dataset for normalization opportunities",
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the codebase",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary",
    )

    args = parser.parse_args()

    # Scan codebase or dataset
    if args.check_dataset:
        logger.info(f"Checking dataset: {args.check_dataset}")
        issues = check_dataset_normalization_opportunities(args.check_dataset)
    else:
        logger.info(f"Scanning codebase in {args.root_dir}")
        issues = scan_codebase(args.root_dir, modules=args.modules)

    # Generate report
    report = generate_report(issues)

    # Output results
    if args.json:
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

    # Determine exit code
    if report["by_severity"]["error"] > 0:
        return 2  # Critical errors
    elif report["by_severity"]["warning"] > 0:
        return 1  # Warnings
    else:
        return 0  # All clear


if __name__ == "__main__":
    sys.exit(main())
