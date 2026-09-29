"""
utils/dependency_probe.py — Optional Dependency Probes (Issue #542)
====================================================================
Provides a reusable capability for probing whether optional or
feature-specific Python dependencies are available at runtime, and for
emitting clear, actionable diagnostics when they are not.

Design
------
* Each *feature group* maps to a list of required packages.
* ``probe(group)`` performs a live ``importlib.util.find_spec`` check (no
  side-effect imports) and returns a :class:`ProbeResult`.
* ``require(group)`` raises :class:`MissingDependencyError` with an
  install hint if any package in the group is absent — use this at the top
  of a module that cannot function without the dependency.
* ``probe_all()`` checks every registered group and returns a
  :class:`ProbeReport` suitable for CI gating or developer diagnostics.
* The whole module is importable with zero third-party dependencies.

Feature groups
--------------
The table below lists every group registered in DEPENDENCY_GROUPS.

Group                  Packages                           Install extra
-----                  --------                           -------------
ml_core                scikit-learn, numpy, pandas        pip install -r requirements.txt
xgboost                xgboost                            pip install xgboost
lightgbm               lightgbm                           pip install lightgbm
shap                   shap                               pip install shap
torch                  torch                              pip install torch
torch_geometric        torch_geometric                    pip install torch-geometric
gnn                    torch, torch_geometric             pip install torch torch-geometric
opacus                 opacus                             pip install opacus
kafka                  confluent_kafka, fastavro          pip install confluent-kafka fastavro
redis                  redis                              pip install redis
websockets             websockets                         pip install websockets
stellar_sdk            stellar_sdk                        pip install stellar-sdk
louvain                community                          pip install python-louvain
pymoo                  pymoo                              pip install pymoo
optuna                 optuna                             pip install optuna
causal_learn           causallearn                        pip install causal-learn
dice_ml                dice                               pip install dice-ml
cleanlab               cleanlab                           pip install cleanlab
stable_baselines3      stable_baselines3, gymnasium       pip install stable-baselines3 gymnasium
hnswlib                hnswlib                            pip install hnswlib
weasyprint             weasyprint                         pip install weasyprint
cryptography           cryptography                       pip install cryptography
prometheus             prometheus_client                  pip install prometheus-client

Usage (library)
---------------
    from utils.dependency_probe import probe, require, probe_all, ProbeResult

    # Soft check — log and degrade gracefully
    result = probe("gnn")
    if not result.available:
        logger.warning("GNN unavailable: %s", result.install_hint)

    # Hard check — raise if missing
    require("kafka")  # raises MissingDependencyError with install hint

    # Full diagnostic report
    report = probe_all()
    print(report.summary())

Usage (CLI)
-----------
    # Check all groups
    python -m utils.dependency_probe

    # Check specific groups
    python -m utils.dependency_probe --groups gnn kafka redis

    # Emit JSON report
    python -m utils.dependency_probe --json

    # Fail (exit 2) if any required group is missing
    python -m utils.dependency_probe --require ml_core shap

Exit codes
----------
0  All probed groups are available.
1  Fatal error (bad argument, etc.).
2  One or more probed groups have missing packages.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------

# Each entry: (import_name, pip_package_name)
# import_name  — the name passed to importlib.util.find_spec
# pip_name     — the name used in pip install (may differ from import_name)

DependencySpec = list[tuple[str, str]]

DEPENDENCY_GROUPS: dict[str, DependencySpec] = {
    "ml_core": [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ],
    "xgboost": [
        ("xgboost", "xgboost"),
    ],
    "lightgbm": [
        ("lightgbm", "lightgbm"),
    ],
    "shap": [
        ("shap", "shap"),
    ],
    "torch": [
        ("torch", "torch"),
    ],
    "torch_geometric": [
        ("torch_geometric", "torch-geometric"),
    ],
    "gnn": [
        ("torch", "torch"),
        ("torch_geometric", "torch-geometric"),
    ],
    "opacus": [
        ("opacus", "opacus"),
    ],
    "kafka": [
        ("confluent_kafka", "confluent-kafka"),
        ("fastavro", "fastavro"),
    ],
    "redis": [
        ("redis", "redis"),
    ],
    "websockets": [
        ("websockets", "websockets"),
    ],
    "stellar_sdk": [
        ("stellar_sdk", "stellar-sdk"),
    ],
    "louvain": [
        ("community", "python-louvain"),
    ],
    "pymoo": [
        ("pymoo", "pymoo"),
    ],
    "optuna": [
        ("optuna", "optuna"),
    ],
    "causal_learn": [
        ("causallearn", "causal-learn"),
    ],
    "dice_ml": [
        ("dice", "dice-ml"),
    ],
    "cleanlab": [
        ("cleanlab", "cleanlab"),
    ],
    "stable_baselines3": [
        ("stable_baselines3", "stable-baselines3"),
        ("gymnasium", "gymnasium"),
    ],
    "hnswlib": [
        ("hnswlib", "hnswlib"),
    ],
    "weasyprint": [
        ("weasyprint", "weasyprint"),
    ],
    "cryptography": [
        ("cryptography", "cryptography"),
    ],
    "prometheus": [
        ("prometheus_client", "prometheus-client"),
    ],
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class PackageStatus:
    """Result for a single package within a group."""

    import_name: str
    pip_name: str
    available: bool
    version: str | None = None
    error: str | None = None


@dataclass
class ProbeResult:
    """
    Result for a single feature group probe.

    Attributes
    ----------
    group:          The group name (e.g. ``"gnn"``).
    available:      True when *all* packages in the group are present.
    packages:       Per-package status.
    install_hint:   A ready-to-paste ``pip install`` command for missing packages.
    """

    group: str
    available: bool
    packages: list[PackageStatus] = field(default_factory=list)
    install_hint: str = ""

    @property
    def missing(self) -> list[PackageStatus]:
        return [p for p in self.packages if not p.available]

    def __str__(self) -> str:
        status = "✓" if self.available else "✗"
        lines = [f"  [{status}] {self.group}"]
        for pkg in self.packages:
            mark = "✓" if pkg.available else "✗"
            ver = f" ({pkg.version})" if pkg.version else ""
            lines.append(f"        [{mark}] {pkg.import_name}{ver}")
        if not self.available:
            lines.append(f"        Install: {self.install_hint}")
        return "\n".join(lines)


@dataclass
class ProbeReport:
    """Aggregated results across all probed groups."""

    results: list[ProbeResult] = field(default_factory=list)

    @property
    def all_available(self) -> bool:
        return all(r.available for r in self.results)

    @property
    def missing_groups(self) -> list[ProbeResult]:
        return [r for r in self.results if not r.available]

    def summary(self) -> str:
        lines = ["Dependency Probe Report", "=" * 40]
        for result in self.results:
            lines.append(str(result))
        lines.append("=" * 40)
        n_ok = sum(1 for r in self.results if r.available)
        n_missing = len(self.results) - n_ok
        lines.append(f"  {n_ok} group(s) available, {n_missing} missing.")
        if self.missing_groups:
            lines.append("\nMissing install commands:")
            for r in self.missing_groups:
                lines.append(f"  {r.install_hint}")
        return "\n".join(lines)

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "all_available": self.all_available,
            "groups": [
                {
                    "group": r.group,
                    "available": r.available,
                    "install_hint": r.install_hint,
                    "packages": [
                        {
                            "import_name": p.import_name,
                            "pip_name": p.pip_name,
                            "available": p.available,
                            "version": p.version,
                            "error": p.error,
                        }
                        for p in r.packages
                    ],
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# MissingDependencyError
# ---------------------------------------------------------------------------


class MissingDependencyError(ImportError):
    """
    Raised by :func:`require` when a dependency group is not fully installed.

    Carries both a human-readable message and a machine-readable
    ``install_hint`` attribute.
    """

    def __init__(self, group: str, result: ProbeResult) -> None:
        missing_names = [p.pip_name for p in result.missing]
        msg = (
            f"Feature group '{group}' requires package(s) that are not installed: "
            f"{', '.join(missing_names)}. "
            f"To install: {result.install_hint}"
        )
        super().__init__(msg)
        self.group = group
        self.result = result
        self.install_hint = result.install_hint


# ---------------------------------------------------------------------------
# Probe logic
# ---------------------------------------------------------------------------


def _try_get_version(import_name: str) -> str | None:
    """Return the package version string if available, else None."""
    try:
        import importlib.metadata

        # importlib.metadata uses the *distribution* name which often differs
        # from the import name.  We attempt a few common transforms.
        candidates = [
            import_name,
            import_name.replace("_", "-"),
        ]
        for name in candidates:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def _probe_package(import_name: str, pip_name: str) -> PackageStatus:
    """Check a single package via importlib (no side-effect imports)."""
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return PackageStatus(
                import_name=import_name,
                pip_name=pip_name,
                available=False,
                error="Module spec not found",
            )
        version = _try_get_version(import_name)
        return PackageStatus(
            import_name=import_name,
            pip_name=pip_name,
            available=True,
            version=version,
        )
    except (ModuleNotFoundError, ValueError) as exc:
        return PackageStatus(
            import_name=import_name,
            pip_name=pip_name,
            available=False,
            error=str(exc),
        )


def probe(group: str) -> ProbeResult:
    """
    Probe a single feature group.

    Parameters
    ----------
    group:
        One of the keys in :data:`DEPENDENCY_GROUPS`.

    Returns
    -------
    :class:`ProbeResult`

    Raises
    ------
    KeyError
        If *group* is not registered.
    """
    specs = DEPENDENCY_GROUPS[group]
    statuses = [_probe_package(imp, pip) for imp, pip in specs]
    available = all(s.available for s in statuses)
    missing_pip = [s.pip_name for s in statuses if not s.available]
    install_hint = f"pip install {' '.join(missing_pip)}" if missing_pip else ""
    return ProbeResult(
        group=group,
        available=available,
        packages=statuses,
        install_hint=install_hint,
    )


def probe_all(groups: list[str] | None = None) -> ProbeReport:
    """
    Probe all registered groups (or a subset).

    Parameters
    ----------
    groups:
        List of group names to probe.  Defaults to all keys in
        :data:`DEPENDENCY_GROUPS`.

    Returns
    -------
    :class:`ProbeReport`
    """
    target = groups if groups is not None else list(DEPENDENCY_GROUPS.keys())
    return ProbeReport(results=[probe(g) for g in target])


def require(group: str) -> ProbeResult:
    """
    Ensure a feature group is fully available.

    Parameters
    ----------
    group:
        One of the keys in :data:`DEPENDENCY_GROUPS`.

    Returns
    -------
    :class:`ProbeResult` (all packages available)

    Raises
    ------
    :class:`MissingDependencyError`
        If one or more packages in the group are absent.
    KeyError
        If *group* is not registered.
    """
    result = probe(group)
    if not result.available:
        raise MissingDependencyError(group, result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe optional LedgerLens dependency groups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Dependency groups to probe "
            f"(default: all {len(DEPENDENCY_GROUPS)} groups).  "
            f"Available: {', '.join(sorted(DEPENDENCY_GROUPS))}."
        ),
    )
    parser.add_argument(
        "--require",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Exit with code 2 if any of these groups are not fully installed.  "
            "Useful for CI gate jobs."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the report as JSON instead of human-readable text.",
    )
    return parser.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate group names
    all_unknown: list[str] = []
    for name_list in [args.groups, args.require]:
        if name_list:
            all_unknown.extend(g for g in name_list if g not in DEPENDENCY_GROUPS)
    if all_unknown:
        print(
            f"[dependency_probe] Unknown group(s): {', '.join(set(all_unknown))}.\n"
            f"Available: {', '.join(sorted(DEPENDENCY_GROUPS))}",
            file=sys.stderr,
        )
        return 1

    report = probe_all(args.groups)

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    # Hard-require check
    if args.require:
        require_report = probe_all(args.require)
        if not require_report.all_available:
            if not args.as_json:
                print(
                    "\n[dependency_probe] ✗ Required groups are missing.  " "Cannot continue.",
                    file=sys.stderr,
                )
            return 2

    return 0 if report.all_available else 2


if __name__ == "__main__":
    sys.exit(_cli_main())
