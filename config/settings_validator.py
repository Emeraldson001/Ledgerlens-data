"""Runtime settings validation service.

A small, typed contract for describing the *shape* of configuration a
component expects (required keys, types, ranges, allowed values) and
validating a live settings object or mapping against it at process start —
before a misconfigured `BENFORD_MIN_SAMPLE_SIZE` or an out-of-range
`CONFORMAL_COVERAGE_LEVEL` causes a confusing failure deep inside a pipeline
run instead of a clear, actionable startup error.

This module does not replace `config.Config` (the existing env-driven
settings object) — it validates *any* settings source (a `Config` instance,
a plain dict, argparse `Namespace`, etc.) against declared `SettingSpec`s and
produces a `ValidationReport` naming exactly which setting is wrong and why.

API::

    validator = SettingsValidator([
        SettingSpec("HORIZON_URL", str, required=True, validator=is_url),
        SettingSpec("BENFORD_MIN_SAMPLE_SIZE", int, required=True, min_value=10),
        SettingSpec("CONFORMAL_COVERAGE_LEVEL", float, min_value=0.0, max_value=1.0),
    ])
    report = validator.validate(config.Config)
    if not report.ok:
        raise SettingsValidationError(report)

Run `python -m config.settings_validator` to validate the live `Config`
object against the built-in `DEFAULT_LEDGERLENS_SPECS` registry and print a
human-readable report — useful as a pre-flight CI or deploy-time check.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def is_url(value: Any) -> bool:
    """Loose URL shape check: scheme + netloc present."""
    if not isinstance(value, str) or not _URL_RE.match(value):
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) > 0
    return True


@dataclass(frozen=True)
class SettingsIssue:
    """One concrete problem found in a settings source."""

    setting_name: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.setting_name}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """Typed result of running a `SettingsValidator` over a settings source."""

    issues: list[SettingsIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[SettingsIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[SettingsIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def render(self) -> str:
        if not self.issues:
            return "settings validation: OK (no issues found)"
        lines = [
            f"settings validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


class SettingsValidationError(Exception):
    """Raised by `SettingsValidator.validate_or_raise` when errors are present.

    Carries the full `ValidationReport` so callers (or CI) can print every
    failing setting at once instead of failing on the first one.
    """

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.render())


@dataclass(frozen=True)
class SettingSpec:
    """Declares the expected shape of a single setting.

    Args:
        name: Attribute/key name to look up on the settings source.
        expected_type: Type (or tuple of types) the value must be an
            instance of. `bool` is checked strictly (a `bool` is never
            silently accepted where `int` was intended, and vice versa).
        required: If True, a missing/None value is an error.
        min_value / max_value: Inclusive numeric bounds, checked when set.
        allowed_values: If set, value must be one of these.
        validator: Optional extra predicate; return False to fail validation.
        description: Human-readable hint included in failure messages.
    """

    name: str
    expected_type: type | tuple[type, ...] | None = None
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    validator: Callable[[Any], bool] | None = None
    description: str = ""

    def check(self, value: Any, present: bool) -> list[SettingsIssue]:
        issues: list[SettingsIssue] = []
        hint = f" ({self.description})" if self.description else ""

        if not present or value is None:
            if self.required:
                issues.append(
                    SettingsIssue(self.name, f"required setting is missing{hint}", "error")
                )
            return issues

        if self.expected_type is not None:
            type_ok = isinstance(value, self.expected_type) and not (
                self.expected_type in (int, (int,)) and isinstance(value, bool)
            )
            if not type_ok:
                issues.append(
                    SettingsIssue(
                        self.name,
                        f"expected type {self.expected_type}, got {type(value).__name__}{hint}",
                        "error",
                    )
                )
                return issues  # further checks assume the right type

        if (
            self.min_value is not None
            and isinstance(value, (int, float))
            and value < self.min_value
        ):
            issues.append(
                SettingsIssue(
                    self.name, f"value {value!r} is below minimum {self.min_value!r}{hint}", "error"
                )
            )
        if (
            self.max_value is not None
            and isinstance(value, (int, float))
            and value > self.max_value
        ):
            issues.append(
                SettingsIssue(
                    self.name, f"value {value!r} is above maximum {self.max_value!r}{hint}", "error"
                )
            )
        if self.allowed_values is not None and value not in self.allowed_values:
            issues.append(
                SettingsIssue(
                    self.name,
                    f"value {value!r} not in allowed values {self.allowed_values!r}{hint}",
                    "error",
                )
            )
        if self.validator is not None:
            try:
                passed = self.validator(value)
            except Exception as exc:  # validator itself misbehaved
                issues.append(
                    SettingsIssue(
                        self.name, f"validator raised {exc!r} for value {value!r}{hint}", "error"
                    )
                )
            else:
                if not passed:
                    issues.append(
                        SettingsIssue(
                            self.name, f"value {value!r} failed custom validation{hint}", "error"
                        )
                    )
        return issues


def _lookup(source: Any, name: str) -> tuple[bool, Any]:
    """Reads `name` off `source`, supporting both attribute- and dict-style access."""
    if isinstance(source, dict):
        if name in source:
            return True, source[name]
        return False, None
    if hasattr(source, name):
        return True, getattr(source, name)
    return False, None


class SettingsValidator:
    """Validates a settings source against a list of `SettingSpec`s."""

    def __init__(self, specs: list[SettingSpec]):
        by_name = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate SettingSpec for {spec.name!r}")
            by_name[spec.name] = spec
        self.specs = specs

    def validate(self, source: Any) -> ValidationReport:
        issues: list[SettingsIssue] = []
        for spec in self.specs:
            present, value = _lookup(source, spec.name)
            issues.extend(spec.check(value, present))
        return ValidationReport(issues=issues)

    def validate_or_raise(self, source: Any) -> ValidationReport:
        report = self.validate(source)
        if not report.ok:
            raise SettingsValidationError(report)
        return report


# ---------------------------------------------------------------------------
# Built-in registry covering a subset of `config.Config` settings that are
# known to cause silent bad behavior when misconfigured, rather than a
# loud startup failure.
# ---------------------------------------------------------------------------
DEFAULT_LEDGERLENS_SPECS: list[SettingSpec] = [
    SettingSpec(
        "HORIZON_URL",
        str,
        required=True,
        validator=is_url,
        description="must be a reachable Horizon base URL",
    ),
    SettingSpec(
        "STELLAR_NETWORK",
        str,
        required=True,
        allowed_values=("PUBLIC", "TESTNET", "FUTURENET"),
        description="must match a known Stellar network passphrase alias",
    ),
    SettingSpec(
        "BENFORD_MIN_SAMPLE_SIZE",
        int,
        required=True,
        min_value=10,
        description="samples below 10 produce statistically meaningless Benford metrics",
    ),
    SettingSpec(
        "BENFORD_DRIFT_Z_THRESHOLD",
        float,
        required=False,
        min_value=0.0,
        description="z-score threshold for drift alerts; must be non-negative",
    ),
    SettingSpec(
        "CONFORMAL_COVERAGE_LEVEL",
        float,
        required=False,
        min_value=0.0,
        max_value=1.0,
        description="coverage level is a probability and must be in [0, 1]",
    ),
    SettingSpec(
        "RISK_SCORE_FLAG_THRESHOLD",
        int,
        required=False,
        min_value=0,
        max_value=100,
        description="risk score is 0-100; a value outside that range would "
        "flag every wallet or none silently",
    ),
    SettingSpec(
        "MAD_NONCONFORMITY_THRESHOLD",
        float,
        required=False,
        min_value=0.0,
        description="Benford MAD is non-negative; a negative threshold would "
        "make benford_flag always true",
    ),
]


def validate_default_config() -> ValidationReport:
    """Validates the live `config.Config` object against the built-in registry."""
    from config import Config

    return SettingsValidator(DEFAULT_LEDGERLENS_SPECS).validate(Config)


def _main() -> int:
    report = validate_default_config()
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main())
