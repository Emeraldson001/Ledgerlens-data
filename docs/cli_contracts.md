# CLI Command Contracts for Operational Workflows

## Problem

`scripts/` holds ~49 standalone entry points used for real operational
workflows: scoring a wallet on demand (`score_wallet.py`), replaying a
Kafka topic during an incident (`replay_stream.py`), scaling consumer
workers (`kafka_workers.py`), managing the human annotation queue
(`manage_queue.py`), running a backtest (`backtest.py`), kicking off an
active-learning round (`run_active_learning.py`). Each script owns its own
`argparse` parser with no shared, reviewable definition of what an on-call
operator can rely on -- a required flag silently becoming optional (or vice
versa), or a flag being renamed, was previously only discoverable by
reading the script's source or running `--help`.

## Design

- **`scripts/cli_contracts.py`** declares one `CliContract` per
  operationally-important script: its flags/positionals, which are
  required, and a short description of each -- effectively a machine-checked
  version of the `--help` text, doubling as operator-facing documentation.
- **`scripts/check_cli_contracts.py`** parses the real script with `ast`
  (no execution -- several of these scripts import Kafka clients or ML
  frameworks that shouldn't be required just to lint the CLI surface) and
  extracts every `add_argument(...)` call's alias(es) and `required` flag,
  then diffs it against the declared contract:
  - a contract entry with no matching `add_argument` call -> **missing**
    (renamed/removed without updating the contract)
  - an `add_argument` call not covered by any contract entry -> **undeclared**
    (a new flag shipped without documenting it as part of the operational
    surface)
  - a `required=` mismatch between contract and source -> **required mismatch**

## Validation

```
python scripts/check_cli_contracts.py                     # all contracted scripts
python scripts/check_cli_contracts.py --script backtest.py
make check-cli-contracts                                   # same, via Makefile
pytest tests/test_cli_contracts.py -q                       # unit tests
```

`tests/test_cli_contracts.py` exercises extraction and diffing logic
against synthetic scripts (single/multi-alias arguments, positionals,
missing/undeclared/required-mismatch diagnostics), plus a real-contract
smoke test that all six contracted scripts (`score_wallet.py`,
`backtest.py`, `manage_queue.py`, `kafka_workers.py`, `replay_stream.py`,
`run_active_learning.py`) currently match `scripts/cli_contracts.py`
exactly. Both the standalone script and the pytest run are wired into CI.

## Tradeoffs / follow-up

- Contracts are declared for the 6 scripts with the widest operational
  blast radius (incident response, scoring, backtesting, queue
  management), not all 49 files in `scripts/`. Extending coverage is
  additive: add a `CliContract` entry and rerun the checker.
- Extraction is script-wide rather than per-subcommand -- `manage_queue.py`
  uses subparsers (`list` / `annotate` / `skip` / `export`), and the
  contract treats their flags as one flat set. This was a deliberate
  scope tradeoff to keep the contract model simple; a future iteration
  could add a `subcommand` field to `CliArgument` if per-subcommand
  precision becomes necessary.
- Like the API-compatibility check (`docs/api_compatibility.md`), this is
  static analysis only: it validates the *shape* of the CLI contract, not
  runtime behavior (e.g. it won't catch a flag whose value is silently
  ignored). Runtime coverage for individual scripts already exists in
  `tests/test_score_wallet.py`, `tests/test_backtest.py`, etc.
