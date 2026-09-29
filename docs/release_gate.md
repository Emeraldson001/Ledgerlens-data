# Release gate exit codes

`scripts/release_gate.py` evaluates a JUnit XML report against the release
criticality taxonomy in `data/test_criticality.json`.

## Exit codes

| Exit code | Meaning | Operational impact |
|---|---|---|
| `0` | The gate passed. | Release may proceed; no CRITICAL or unapproved HIGH failures were found. |
| `1` | The gate failed. | A CRITICAL test failure was detected, or a HIGH-tier failure was detected while `--allow-high-failures` was not used. |
| `2` | Invalid CLI input or unreadable JUnit/taxonomy input. | The script could not evaluate the report because the file was missing, invalid, or unparsable. |

## Typical usage

```bash
pytest --junitxml=reports/junit.xml
python -m scripts.release_gate --junit reports/junit.xml
```

To allow HIGH-tier failures on a feature branch:

```bash
python -m scripts.release_gate --junit reports/junit.xml --allow-high-failures
```

This keeps the release gate conservative: CRITICAL failures always block, while
HIGH failures are only allowed when the caller explicitly opts in.

## How this matches the workflow

`.github/workflows/release-readiness.yml` does not inspect the script's exit
code directly; it runs a separate set of repository checks and fails the job on
those checks. The release gate therefore provides a clear, consistent contract
for CI jobs that run pytest output through `scripts/release_gate.py` without
changing the release-readiness workflow's semantics.
