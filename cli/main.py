import argparse
import json
import logging
import sys

from cli.commands.validate_artifacts import validate_artifacts
from cli.diagnostics import run_diagnostics


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens-ops", description="Operational Harness for LedgerLens Data Pipelines"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser(
        "healthcheck", help="Run diagnostic health checks on setup and variables"
    )
    health_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    val_parser = subparsers.add_parser(
        "validate-artifacts", help="Validate local model and schema artifacts"
    )
    val_parser.add_argument(
        "--dir", default="artifacts", help="Path to artifacts folder (default: artifacts)"
    )

    return parser


def _format_health_summary(report: dict) -> str:
    lines = [
        f"Overall status: {report.get('overall_status', 'UNKNOWN')}",
        f"Checks run: {report.get('checks', {}).get('environment', {}).get('status', 'unknown')}",
    ]
    env = report.get("checks", {}).get("environment", {})
    if env:
        lines.append(f"Environment: {env.get('status', 'unknown')}")
    streaming = report.get("checks", {}).get("streaming", {})
    if streaming:
        lines.append(f"Streaming: {streaming.get('status', 'unknown')}")
    return "\n".join(lines)


def main(args=None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)
    setup_logging(opts.verbose)

    if opts.command == "healthcheck":
        report = run_diagnostics()
        if getattr(opts, "json", False):
            print(json.dumps(report, indent=2))
        else:
            print(_format_health_summary(report))
        return 0 if report["overall_status"] == "PASS" else 2

    elif opts.command == "validate-artifacts":
        res = validate_artifacts(opts.dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "PASS" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
