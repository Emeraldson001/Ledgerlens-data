# Repository Health Diagnostics

Comprehensive developer diagnostics command for assessing the health and readiness of the LedgerLens-data repository.

## Quick Start

```bash
# Run all diagnostic checks
make diagnose

# Run specific categories
make diagnose CATEGORIES="environment dependencies"

# JSON output for CI/automation
make diagnose-json

# List available checks
python -m scripts.diagnose --list
```

## Overview

The diagnostics system provides actionable insights across five key areas:

1. **Environment** — Configuration contracts, environment variables, required files
2. **Dependencies** — Required packages, optional features, version compatibility
3. **Code Health** — Package integrity, import cycles, git status
4. **Data Artifacts** — Model files, data directories, dataset availability
5. **Runtime** — Database connectivity, external API reachability

## Categories

### Environment (`environment`)

| Check | What It Does |
|-------|--------------|
| `config_contracts` | Validates all runtime mode configuration contracts |
| `critical_env_vars` | Ensures critical environment variables are set |
| `required_files` | Checks that required config files exist |

### Dependencies (`dependencies`)

| Check | What It Does |
|-------|--------------|
| `required_packages` | Verifies core Python packages are installed |
| `optional_dependencies` | Checks availability of optional feature groups (kafka, gnn, rl) |

### Code Health (`code_health`)

| Check | What It Does |
|-------|--------------|
| `package_integrity` | Scans for missing `__init__.py`, syntax errors, merge conflicts |
| `import_cycles` | Detects circular import dependencies |
| `git_status` | Checks for uncommitted changes |

### Data Artifacts (`data_artifacts`)

| Check | What It Does |
|-------|--------------|
| `model_artifacts` | Ensures trained model files and metadata exist |
| `data_directories` | Verifies expected data/models/reports directories are present |

### Runtime (`runtime`)

| Check | What It Does |
|-------|--------------|
| `database_connectivity` | Tests database connection (if `RISK_SCORE_DB_URL` is set) |
| `horizon_api` | Checks Horizon API reachability |

## Usage Examples

### Pre-Deployment Check

```bash
# Check environment and runtime readiness before deploying
python -m scripts.diagnose --categories environment runtime
```

### CI Validation

```bash
# Fail fast on first error
python -m scripts.diagnose --json --fail-fast
```

### Development Workflow

```bash
# Check code health before committing
python -m scripts.diagnose --categories code_health

# Verify dependencies after branch switch
python -m scripts.diagnose --categories dependencies
```

## Output Formats

### Text (Default)

Human-readable summary with failures, warnings, and remediation suggestions:

```
Overall Status: FAIL
Total Checks: 12
  FAIL: 2
    - critical_env_vars: 2 critical environment variable(s) missing
      Fix: Set: HORIZON_URL, MODEL_DIR
    - model_artifacts: 3 model artifact(s) missing
      Fix: Run: python -m detection.model_training
  PASS: 10

Execution time: 850ms
```

### JSON

Machine-readable format for CI/automation:

```json
{
  "overall_status": "fail",
  "total_checks": 12,
  "pass_count": 10,
  "warn_count": 0,
  "fail_count": 2,
  "categories_checked": ["environment", "dependencies"],
  "total_duration_ms": 850.5,
  "checks": [
    {
      "check_name": "critical_env_vars",
      "category": "environment",
      "status": "fail",
      "message": "2 critical environment variable(s) missing",
      "details": {
        "missing_critical": ["HORIZON_URL", "MODEL_DIR"]
      },
      "remediation": "Set: HORIZON_URL, MODEL_DIR",
      "duration_ms": 0.5
    }
  ]
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed or produced warnings only |
| `1` | One or more checks failed or errored |
| `2` | Execution error (invalid arguments, interrupted) |

## Check Statuses

| Status | Meaning | Exit Code Impact |
|--------|---------|------------------|
| `PASS` | Check passed | None — healthy |
| `WARN` | Minor issue detected | None — still healthy |
| `FAIL` | Check failed | Causes exit code 1 |
| `ERROR` | Check raised exception | Causes exit code 1 |
| `SKIP` | Check not applicable | None — healthy |

## CI Integration

The diagnostics system runs automatically in CI (`.github/workflows/ci.yml`) before the test suite. The JSON report is uploaded as a build artifact for historical tracking.

```yaml
- name: Repository health diagnostics
  run: |
    python -m scripts.diagnose \
      --categories environment dependencies code_health data_artifacts \
      --json > reports/ci/diagnostics.json
```

## Adding New Checks

1. Create a new check class in `utils/diagnostics_checks.py`:

```python
class MyNewCheck:
    """Check description."""
    
    name = "my_new_check"
    category = CheckCategory.ENVIRONMENT
    
    def run(self) -> DiagnosticResult:
        # Your check logic here
        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message="Check passed",
        )
```

2. Register it in `_register_all_checks()` at the bottom of the file.

3. Add tests in `tests/test_diagnostics_checks.py`.

## Architecture

The diagnostics system follows a modular, protocol-based architecture:

- **`DiagnosticCheck`** — Protocol defining the check interface
- **`DiagnosticResult`** — Structured result from a single check
- **`DiagnosticReport`** — Aggregated results from all checks
- **`DiagnosticRegistry`** — Central registry of available checks

Each check is self-contained and produces a structured result with:
- Status (pass/warn/fail/error/skip)
- Human-readable message
- Structured details (dict)
- Remediation suggestion
- Execution duration

## Related Commands

| Command | Purpose |
|---------|---------|
| `make check-env` | Validate environment contracts for specific runtime modes |
| `make check-integrity` | Run package integrity check standalone |
| `make check-cycles` | Run import cycle detection standalone |
| `make validate-all` | Run all validation tools (cycles, deps, README, notebooks) |
| `make dependency-risk-report` | Generate the deterministic offline core dependency risk report |
| `make dependency-risk-report-osv` | Enrich the dependency risk report with OSV.dev advisories |

## Dependency risk reporting (Issue #478)

The dependency risk reporter evaluates the direct packages used by the core
scoring, ingestion, API, and security paths. It compares `requirements.txt`
with `requirements.lock` and assigns low/medium/high/critical risk based on
lockfile coverage and optional vulnerability advisories.

```bash
make dependency-risk-report
python scripts/dependency_risk_report.py --format markdown --output reports/dependency_risk.md
python scripts/dependency_risk_report.py --osv --output reports/dependency_risk.json
```

The default command is offline and deterministic, which makes it suitable for
CI. `--osv` performs best-effort OSV.dev lookups; network failures are included
as warnings in the report rather than being hidden.

## Troubleshooting

### "Cannot import diagnostic checks"

The `utils.diagnostics_checks` module failed to import. Check for syntax errors:

```bash
python -c "import utils.diagnostics_checks"
```

### Checks Always Failing in CI

Runtime checks (database, Horizon) skip automatically when environment variables aren't set. If they're failing, ensure the CI workflow isn't setting those variables unexpectedly.

### Slow Execution

Use category filtering to run only the checks you need:

```bash
# Fast pre-commit check (~1s)
python -m scripts.diagnose --categories code_health

# Full diagnostic (~3-5s)
python -m scripts.diagnose
```

## See Also

- `scripts/check_env.py` — Runtime mode configuration validation
- `scripts/check_package_integrity.py` — Standalone package integrity check
- `scripts/check_import_cycles.py` — Standalone import cycle detection
- `utils/dependency_probe.py` — Optional dependency availability probe
- `scripts/dependency_risk_report.py` — Core package lockfile and advisory risk report (Issue #478)
