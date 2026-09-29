#!/usr/bin/env python3
"""CLI tool to validate secrets configuration.

This tool checks that all required secrets are properly configured, validates
their formats, and reports any issues. It's useful for:
- Pre-deployment configuration validation
- CI/CD pipeline checks
- Troubleshooting configuration issues
- Auditing secrets setup

Usage
-----
Validate all registered secrets::

    python -m scripts.validate_secrets

Validate specific secrets::

    python -m scripts.validate_secrets --secrets LEDGERLENS_SUBMITTER_SECRET KAFKA_SASL_PASSWORD

Check audit log integrity::

    python -m scripts.validate_secrets --verify-audit-log

Generate configuration report::

    python -m scripts.validate_secrets --report

Exit Codes
----------
0: All secrets valid
1: Validation errors found
2: Critical errors (missing required secrets)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from utils.logging import get_logger
from utils.secrets_config import is_secret_configured
from utils.secrets_manager import (
    get_secrets_manager,
    register_ledgerlens_secrets,
)

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


def validate_all_secrets(secrets_filter: list[str] | None = None) -> tuple[int, int, int]:
    """Validate all registered secrets.

    Args:
        secrets_filter: Optional list of specific secret names to validate

    Returns:
        Tuple of (valid_count, warning_count, error_count)
    """
    manager = get_secrets_manager()
    register_ledgerlens_secrets(manager)

    results = manager.verify_all_secrets()

    valid_count = 0
    warning_count = 0
    error_count = 0

    print(colorize("\n=== Secrets Validation Report ===\n", Colors.BOLD))

    for secret_name, error in results.items():
        # Filter if requested
        if secrets_filter and secret_name not in secrets_filter:
            continue

        definition = manager._definitions.get(secret_name)
        required = definition.required if definition else False

        if error is None:
            # Secret is valid
            print(f"{colorize('✓', Colors.GREEN)} {secret_name}: {colorize('VALID', Colors.GREEN)}")
            valid_count += 1
        elif "not found" in error and not required:
            # Optional secret not configured (warning, not error)
            print(
                f"{colorize('⚠', Colors.YELLOW)} {secret_name}: "
                f"{colorize('NOT CONFIGURED', Colors.YELLOW)} (optional)"
            )
            warning_count += 1
        else:
            # Validation error or missing required secret
            severity = "ERROR" if required else "WARNING"
            color = Colors.RED if required else Colors.YELLOW
            symbol = "✗" if required else "⚠"

            print(f"{colorize(symbol, color)} {secret_name}: {colorize(severity, color)}")
            print(f"  {error}")

            if required:
                error_count += 1
            else:
                warning_count += 1

    return valid_count, warning_count, error_count


def verify_audit_log() -> bool:
    """Verify audit log integrity.

    Returns:
        True if audit log is valid, False otherwise
    """
    import os

    from utils.secrets_manager import SecretAuditLogger

    audit_log_path = Path(os.getenv("SECRETS_AUDIT_LOG", "data/secrets_audit.ndjson"))
    audit_hmac_key = os.getenv("SECRETS_AUDIT_HMAC_KEY")

    if not audit_log_path.exists():
        print(colorize("\n⚠ Audit log not found", Colors.YELLOW))
        print(f"  Path: {audit_log_path}")
        return True  # Not an error if log doesn't exist yet

    if not audit_hmac_key:
        print(colorize("\n⚠ SECRETS_AUDIT_HMAC_KEY not set", Colors.YELLOW))
        print("  Audit log integrity cannot be verified without HMAC key")
        return True  # Not an error, just a warning

    print(colorize("\n=== Audit Log Verification ===\n", Colors.BOLD))

    audit_logger = SecretAuditLogger(audit_log_path, audit_hmac_key)

    try:
        valid, invalid = audit_logger.verify_log_integrity()

        if invalid == 0:
            print(f"{colorize('✓', Colors.GREEN)} All {valid} entries verified")
            return True
        else:
            print(
                f"{colorize('✗', Colors.RED)} {invalid} entries failed verification "
                f"({valid} valid)"
            )
            print("  TAMPERED LOG DETECTED - audit trail may be compromised")
            return False

    except Exception as e:
        print(f"{colorize('✗', Colors.RED)} Verification failed: {e}")
        return False


def generate_report(output_path: Path | None = None) -> dict[str, Any]:
    """Generate a comprehensive configuration report.

    Args:
        output_path: Optional path to write JSON report

    Returns:
        Report dictionary
    """
    manager = get_secrets_manager()
    register_ledgerlens_secrets(manager)

    report = {
        "timestamp": str(Path.ctime(Path(__file__))),
        "provider": type(manager.provider).__name__,
        "validation_enabled": manager.enable_validation,
        "secrets": {},
    }

    for name, definition in manager._definitions.items():
        configured = is_secret_configured(name)

        report["secrets"][name] = {
            "type": definition.secret_type.value,
            "required": definition.required,
            "configured": configured,
            "allow_rotation": definition.allow_rotation,
            "description": definition.description,
        }

    if output_path:
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n{colorize('✓', Colors.GREEN)} Report written to {output_path}")

    return report


def print_configuration_summary() -> None:
    """Print a summary of the current configuration."""
    manager = get_secrets_manager()

    print(colorize("\n=== Configuration Summary ===\n", Colors.BOLD))
    print(f"Provider: {colorize(type(manager.provider).__name__, Colors.BLUE)}")
    print(f"Validation: {colorize(str(manager.enable_validation), Colors.BLUE)}")

    if manager.audit_logger:
        print(f"Audit log: {colorize(str(manager.audit_logger.log_path), Colors.BLUE)}")
    else:
        print(f"Audit log: {colorize('DISABLED', Colors.YELLOW)}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate secrets configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--secrets",
        nargs="+",
        help="Specific secrets to validate (validates all if not specified)",
    )

    parser.add_argument(
        "--verify-audit-log",
        action="store_true",
        help="Verify audit log integrity with HMAC signatures",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate configuration report",
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        help="Path to write JSON report (default: stdout)",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    exit_code = 0

    try:
        # Print configuration summary
        print_configuration_summary()

        # Validate secrets
        valid, warnings, errors = validate_all_secrets(args.secrets)

        print(colorize("\n=== Summary ===\n", Colors.BOLD))
        print(f"{colorize('✓', Colors.GREEN)} Valid: {valid}")
        print(f"{colorize('⚠', Colors.YELLOW)} Warnings: {warnings}")
        print(f"{colorize('✗', Colors.RED)} Errors: {errors}")

        # Verify audit log if requested
        if args.verify_audit_log:
            audit_ok = verify_audit_log()
            if not audit_ok:
                exit_code = 1

        # Generate report if requested
        if args.report:
            generate_report(args.report_output)

        # Determine exit code
        if errors > 0:
            exit_code = 2
        elif warnings > 0 and exit_code == 0:
            exit_code = 1

        # Final status
        if exit_code == 0:
            print(colorize("\n✓ All checks passed\n", Colors.GREEN))
        elif exit_code == 1:
            print(colorize("\n⚠ Validation completed with warnings\n", Colors.YELLOW))
        else:
            print(colorize("\n✗ Validation failed with errors\n", Colors.RED))

    except Exception as e:
        logger.exception("Validation failed with exception")
        print(colorize(f"\n✗ Fatal error: {e}\n", Colors.RED))
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
