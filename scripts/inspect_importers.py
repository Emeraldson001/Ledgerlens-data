"""CLI tool for inspecting importer capabilities.

This script provides a command-line interface for querying and validating
the importer capability discovery system. It's useful for:

- Discovering what importers are available
- Finding importers with specific capabilities
- Validating that required capabilities exist
- Debugging importer registration issues
- Generating reports for documentation

Usage Examples
--------------
List all importers:
    $ python -m scripts.inspect_importers list

Show details for a specific importer:
    $ python -m scripts.inspect_importers info horizon_streamer

Find importers by capability:
    $ python -m scripts.inspect_importers find --capability STREAMING
    $ python -m scripts.inspect_importers find --capability "STREAMING|REAL_TIME"

Find importers by data type:
    $ python -m scripts.inspect_importers find --data-type TRADE

Validate requirements:
    $ python -m scripts.inspect_importers validate --capability BULK --data-type TRADE

Show registry statistics:
    $ python -m scripts.inspect_importers stats

Generate markdown report:
    $ python -m scripts.inspect_importers report --output importers.md
"""

import argparse
import json
import sys
from typing import Any

# Ensure registry is populated
import ingestion.registered_importers  # noqa: F401
from ingestion.importer_registry import (
    DataSource,
    DataType,
    ImporterCapability,
    get_registry,
    validate_importer_requirements,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# CLI Commands
# ============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered importers."""
    registry = get_registry()
    importers = registry.list_all()

    if not importers:
        print("No importers registered.")
        return 1

    print(f"Found {len(importers)} registered importers:\n")

    for name in importers:
        info = registry.get_importer_info(name)

        # Format capabilities
        caps_list = []
        for cap in ImporterCapability:
            if cap != ImporterCapability.NONE and info.has_capability(cap):
                caps_list.append(cap.name)
        caps_str = ", ".join(caps_list[:3])
        if len(caps_list) > 3:
            caps_str += f" (+{len(caps_list) - 3} more)"

        # Format data types
        types_str = ", ".join(str(dt) for dt in sorted(info.data_types, key=str))

        print(f"  • {name}")
        print(f"    Description: {info.description.split('.')[0]}...")
        print(f"    Capabilities: {caps_str}")
        print(f"    Data types: {types_str}")
        print()

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed information for a specific importer."""
    registry = get_registry()

    try:
        info = registry.get_importer_info(args.name)
    except KeyError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Importer: {info.name}")
    print(f"Version: {info.version}")
    print("\nDescription:")
    print(f"  {info.description}")
    print()

    # Capabilities
    print("Capabilities:")
    for cap in ImporterCapability:
        if cap != ImporterCapability.NONE and info.has_capability(cap):
            print(f"  ✓ {cap.name}")
    print()

    # Data types
    print("Data Types:")
    for dt in sorted(info.data_types, key=str):
        print(f"  • {dt}")
    print()

    # Data sources
    print("Data Sources:")
    for src in sorted(info.sources, key=str):
        print(f"  • {src}")
    print()

    # Performance
    if info.performance.typical_latency_ms or info.performance.throughput_records_per_sec:
        print("Performance:")
        if info.performance.typical_latency_ms:
            print(f"  Latency: ~{info.performance.typical_latency_ms}ms")
        if info.performance.throughput_records_per_sec:
            print(f"  Throughput: ~{info.performance.throughput_records_per_sec} records/sec")
        if info.performance.memory_overhead_mb:
            print(f"  Memory: ~{info.performance.memory_overhead_mb}MB")
        print()

    # Feature flags
    print("Features:")
    if info.supports_failover:
        print("  ✓ Multi-region failover")
    if info.requires_authentication:
        print("  ⚠ Requires authentication")
    if info.supports_rate_limiting:
        print("  ✓ Rate limiting support")
    print()

    # Dependencies
    if info.dependencies:
        print("Dependencies:")
        for dep in sorted(info.dependencies):
            print(f"  • {dep}")
        print()

    # Module path
    if info.module_path:
        print(f"Module: {info.module_path}")

    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """Find importers matching criteria."""
    registry = get_registry()
    results = []

    # Find by capability
    if args.capability:
        # Parse capability flags
        caps = ImporterCapability.NONE
        for cap_name in args.capability.split("|"):
            cap_name = cap_name.strip()
            try:
                caps |= ImporterCapability[cap_name]
            except KeyError:
                print(f"Error: Unknown capability '{cap_name}'")
                print(
                    f"Available: {', '.join(c.name for c in ImporterCapability if c != ImporterCapability.NONE)}"
                )
                return 1

        results = registry.find_by_capability(caps, require_all=args.require_all)

    # Find by data type
    elif args.data_type:
        try:
            data_type = DataType(args.data_type.lower())
        except ValueError:
            print(f"Error: Unknown data type '{args.data_type}'")
            print(f"Available: {', '.join(dt.value for dt in DataType)}")
            return 1

        results = registry.find_by_data_type(data_type)

    # Find by source
    elif args.source:
        try:
            source = DataSource(args.source.lower())
        except ValueError:
            print(f"Error: Unknown data source '{args.source}'")
            print(f"Available: {', '.join(src.value for src in DataSource)}")
            return 1

        results = registry.find_by_source(source)

    else:
        print("Error: Must specify --capability, --data-type, or --source")
        return 1

    # Display results
    if not results:
        print("No importers found matching criteria.")
        return 1

    print(f"Found {len(results)} matching importer(s):\n")

    for result in results:
        name = result["importer_name"]
        info = result["metadata"]
        score = result["match_score"]

        print(f"  • {name} (match: {score:.0%})")
        print(f"    {info.description.split('.')[0]}...")
        print()

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate that required capabilities are available."""
    required_caps = None
    required_types = None

    # Parse capabilities
    if args.capability:
        required_caps = ImporterCapability.NONE
        for cap_name in args.capability.split("|"):
            cap_name = cap_name.strip()
            try:
                required_caps |= ImporterCapability[cap_name]
            except KeyError:
                print(f"Error: Unknown capability '{cap_name}'")
                return 1

    # Parse data types
    if args.data_type:
        try:
            required_types = [DataType(dt.strip().lower()) for dt in args.data_type.split(",")]
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

    # Validate
    result = validate_importer_requirements(
        required_capabilities=required_caps,
        required_data_types=required_types,
    )

    # Display result
    print(str(result))
    print()

    if result.available_importers:
        print(f"Available importers: {', '.join(sorted(result.available_importers))}")

    return 0 if result.is_valid else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show registry statistics."""
    registry = get_registry()
    stats = registry.get_statistics()

    print("Registry Statistics:")
    print(f"  Total importers: {stats['total_importers']}")
    print(f"  Streaming importers: {stats['streaming_importers']}")
    print(f"  Bulk importers: {stats['bulk_importers']}")
    print(f"  Failover-capable: {stats['failover_capable']}")
    print()

    print("Importers by Data Type:")
    for data_type, count in sorted(stats["importers_by_data_type"].items()):
        print(f"  {data_type}: {count}")
    print()

    print("Importers by Source:")
    for source, count in sorted(stats["importers_by_source"].items()):
        print(f"  {source}: {count}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a markdown report of all importers."""
    registry = get_registry()
    importers = registry.list_all()

    lines = []
    lines.append("# LedgerLens Data Importers")
    lines.append("")
    lines.append("This document provides a comprehensive overview of all data source")
    lines.append("importers available in LedgerLens-data.")
    lines.append("")
    lines.append(f"**Total Importers:** {len(importers)}")
    lines.append("")

    # Statistics
    stats = registry.get_statistics()
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Streaming importers:** {stats['streaming_importers']}")
    lines.append(f"- **Bulk importers:** {stats['bulk_importers']}")
    lines.append(f"- **Failover-capable:** {stats['failover_capable']}")
    lines.append("")

    # Importer details
    lines.append("## Importer Details")
    lines.append("")

    for name in importers:
        info = registry.get_importer_info(name)

        lines.append(f"### {info.name}")
        lines.append("")
        lines.append(f"**Version:** {info.version}")
        lines.append("")
        lines.append(info.description)
        lines.append("")

        # Capabilities
        lines.append("**Capabilities:**")
        for cap in ImporterCapability:
            if cap != ImporterCapability.NONE and info.has_capability(cap):
                lines.append(f"- {cap.name}")
        lines.append("")

        # Data types
        lines.append("**Data Types:**")
        for dt in sorted(info.data_types, key=str):
            lines.append(f"- {dt}")
        lines.append("")

        # Data sources
        lines.append("**Sources:**")
        for src in sorted(info.sources, key=str):
            lines.append(f"- {src}")
        lines.append("")

        # Performance
        if info.performance.typical_latency_ms or info.performance.throughput_records_per_sec:
            lines.append("**Performance:**")
            if info.performance.typical_latency_ms:
                lines.append(f"- Latency: ~{info.performance.typical_latency_ms}ms")
            if info.performance.throughput_records_per_sec:
                lines.append(
                    f"- Throughput: ~{info.performance.throughput_records_per_sec} records/sec"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file or stdout
    content = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Report written to {args.output}")
    else:
        print(content)

    return 0


def cmd_json(args: argparse.Namespace) -> int:
    """Export registry as JSON."""
    registry = get_registry()
    importers = registry.list_all()

    data: dict[str, Any] = {
        "version": "1.0",
        "total_importers": len(importers),
        "importers": {},
    }

    for name in importers:
        info = registry.get_importer_info(name)

        data["importers"][name] = {
            "name": info.name,
            "version": info.version,
            "description": info.description,
            "capabilities": [
                cap.name
                for cap in ImporterCapability
                if cap != ImporterCapability.NONE and info.has_capability(cap)
            ],
            "data_types": [str(dt) for dt in info.data_types],
            "sources": [str(src) for src in info.sources],
            "supports_failover": info.supports_failover,
            "requires_authentication": info.requires_authentication,
            "supports_rate_limiting": info.supports_rate_limiting,
            "performance": (
                {
                    "typical_latency_ms": info.performance.typical_latency_ms,
                    "throughput_records_per_sec": info.performance.throughput_records_per_sec,
                    "memory_overhead_mb": info.performance.memory_overhead_mb,
                    "supports_batching": info.performance.supports_batching,
                }
                if info.performance
                else None
            ),
        }

    # Write to file or stdout
    output = json.dumps(data, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"JSON exported to {args.output}")
    else:
        print(output)

    return 0


def cmd_list_names(args: argparse.Namespace) -> int:
    """Print just the importer names, one per line."""
    registry = get_registry()
    importers = registry.list_all()

    for name in importers:
        print(name)

    return 0


# ============================================================================
# Main entry point
# ============================================================================


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect and query the LedgerLens importer registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--list", action="store_true", help="List all importer names (one per line)")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # list command
    subparsers.add_parser("list", help="List all registered importers")

    # info command
    info_parser = subparsers.add_parser("info", help="Show detailed importer information")
    info_parser.add_argument("name", help="Importer name")

    # find command
    find_parser = subparsers.add_parser("find", help="Find importers by criteria")
    find_group = find_parser.add_mutually_exclusive_group(required=True)
    find_group.add_argument("--capability", help="Capability flags (use | to combine)")
    find_group.add_argument("--data-type", help="Data type")
    find_group.add_argument("--source", help="Data source")
    find_parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all capabilities (AND logic instead of OR)",
    )

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate requirements")
    validate_parser.add_argument("--capability", help="Required capabilities (use | to combine)")
    validate_parser.add_argument("--data-type", help="Required data types (comma-separated)")

    # stats command
    subparsers.add_parser("stats", help="Show registry statistics")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate markdown report")
    report_parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    # json command
    json_parser = subparsers.add_parser("json", help="Export registry as JSON")
    json_parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    if args.list:
        return cmd_list_names(args)

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "find": cmd_find,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "report": cmd_report,
        "json": cmd_json,
    }

    try:
        return commands[args.command](args)
    except Exception as exc:
        logger.exception("Command failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
