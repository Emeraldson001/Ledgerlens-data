# Issue #733 — Validate `annotator_id` is Non-Empty Before Writing to the Annotation Queue

## Summary

`scripts/annotate.py` is the interactive CLI used by human reviewers to label
wallets as wash-trade or clean. Every annotation is written to the queue with
an HMAC that binds together:

```
wallet | label | annotator_id | annotated_at
```

(described in `README.md` §Security). If `--annotator-id` is empty or
whitespace-only, the HMAC input silently degrades to
`wallet|label||annotated_at`, producing:

1. **Corrupt attribution** — all annotations appear to come from an anonymous
   actor; downstream inter-annotator agreement metrics (`IAA`) become
   meaningless.
2. **Ambiguous audit trail** — the security model guarantees that each label
   can be traced back to a named reviewer; an empty ID breaks that guarantee
   without raising any error.
3. **Silent data quality problem** — the annotation is accepted, written to
   disk, and potentially used for retraining before the issue is noticed.

---

## Current Behaviour

### `scripts/annotate.py` — `parse_args` (lines 101–115)

```python
parser.add_argument(
    "--annotator-id",
    default="",
    help="Non-empty annotator identifier (required unless --export)",
)
```

The argument defaults to `""`. The only guard is in `main()`:

```python
if not args.annotator_id:
    print("Error: --annotator-id is required for annotation sessions.", file=sys.stderr)
    sys.exit(1)
```

This check catches a completely empty string (`""`), but it does **not** catch
whitespace-only values such as `--annotator-id "   "` or `--annotator-id $'\t'`,
because `bool("   ")` is `True` in Python.

### `detection/active_learning/annotation_queue.py` — `annotate`

The `AnnotationQueue.annotate` method accepts `annotator_id` as a plain string
and passes it directly into the HMAC computation. There is no secondary
validation at the queue level; it trusts the caller.

---

## Root Cause

The guard in `main()` uses a bare truthiness check (`not args.annotator_id`)
instead of a strip-and-check (`not args.annotator_id.strip()`). The argparse
layer has no custom type validator, so whitespace-only strings pass through
undetected.

---

## Risk

| Input | `not args.annotator_id` | `not args.annotator_id.strip()` | Outcome |
|---|---|---|---|
| `""` (empty) | `True` — caught | `True` — caught | Rejected ✓ |
| `"   "` (spaces) | `False` — **missed** | `True` — caught | Accepted incorrectly ✗ |
| `"\t\n"` (tabs/newlines) | `False` — **missed** | `True` — caught | Accepted incorrectly ✗ |
| `"alice"` | `False` | `False` | Accepted ✓ |

A reviewer running the CLI in a shell script that accidentally passes a
space-padded variable (`--annotator-id " "`) produces annotations with a
whitespace ID that cannot be attributed, and the HMAC over
`wallet|label| |annotated_at` is technically valid but semantically wrong.

---

## Acceptance Criteria

1. `scripts/annotate.py` rejects any `--annotator-id` value that is empty
   **or** whitespace-only at **startup**, before any annotation is recorded,
   with a clear message such as:

   ```
   Error: --annotator-id must be a non-empty, non-whitespace string.
   Usage: python -m scripts.annotate --annotator-id yourname
   ```

2. The rejection happens **before** `AnnotationQueue` is constructed and
   **before** any queue file is opened or written.

3. A test in `tests/test_annotation_queue.py` or a dedicated CLI test covers:
   - Completely empty string (`""`)
   - Whitespace-only string (`"   "`)
   - Tab-only string (`"\t"`)

4. A valid non-whitespace `annotator_id` still passes through without
   modification (no silent strip of leading/trailing whitespace that could
   silently change the HMAC input).

---

## Proposed Implementation

### Option A — Fix the guard in `main()` (minimal, no argparse change)

```python
# scripts/annotate.py  main()

if not args.annotator_id or not args.annotator_id.strip():
    print(
        "Error: --annotator-id must be a non-empty, non-whitespace string.\n"
        "Usage: python -m scripts.annotate --annotator-id yourname",
        file=sys.stderr,
    )
    sys.exit(1)
```

### Option B — Custom argparse `type` validator (preferred — fails at parse time)

Define a validator function and pass it as `type=` in `add_argument`:

```python
def _non_empty_str(value: str) -> str:
    """Argparse type validator: reject empty or whitespace-only strings."""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError(
            "annotator-id must be a non-empty, non-whitespace string"
        )
    return value


parser.add_argument(
    "--annotator-id",
    type=_non_empty_str,
    default="",
    help="Non-empty annotator identifier (required unless --export)",
)
```

Option B is preferred because argparse will print the standard usage block and
a clear error message automatically, and it fires before `main()` body
executes — consistent with how `--lookback-days` validation should work in
issue #732.

### Option C — Add a secondary guard in `AnnotationQueue.annotate` (defence-in-depth)

```python
# detection/active_learning/annotation_queue.py

def annotate(self, wallet: str, *, label: int, annotator_id: str, notes: str = "") -> None:
    if not annotator_id or not annotator_id.strip():
        raise ValueError(
            "annotator_id must be a non-empty, non-whitespace string. "
            "Accepting an empty ID would corrupt the HMAC attribution chain."
        )
    ...
```

Option C alone is insufficient (it fires too late), but combining B + C
provides defence-in-depth: the CLI rejects at parse time, and the queue
rejects at write time as a safety net for programmatic callers.

---

## Suggested Tests

```python
# tests/test_annotation_queue.py  (or a new tests/test_annotate_cli.py)

import subprocess
import sys

def test_annotate_cli_rejects_empty_annotator_id():
    """Issue #733 — empty --annotator-id must exit non-zero before any write."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.annotate", "--annotator-id", ""],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "annotator" in result.stderr.lower() or "annotator" in result.stdout.lower()


def test_annotate_cli_rejects_whitespace_annotator_id():
    """Issue #733 — whitespace-only --annotator-id must exit non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.annotate", "--annotator-id", "   "],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_annotate_cli_rejects_tab_annotator_id():
    """Issue #733 — tab-only --annotator-id must exit non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.annotate", "--annotator-id", "\t"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
```

For unit-level coverage without subprocess overhead:

```python
# tests/test_annotation_queue.py

def test_annotation_queue_rejects_empty_annotator_id(tmp_path):
    """Issue #733 — AnnotationQueue.annotate rejects empty annotator_id."""
    from detection.active_learning.annotation_queue import AnnotationQueue
    queue = AnnotationQueue(queue_path=str(tmp_path / "queue.json"))
    with pytest.raises(ValueError, match="annotator_id"):
        queue.annotate("GABC123", label=1, annotator_id="")


def test_annotation_queue_rejects_whitespace_annotator_id(tmp_path):
    """Issue #733 — AnnotationQueue.annotate rejects whitespace annotator_id."""
    from detection.active_learning.annotation_queue import AnnotationQueue
    queue = AnnotationQueue(queue_path=str(tmp_path / "queue.json"))
    with pytest.raises(ValueError, match="annotator_id"):
        queue.annotate("GABC123", label=1, annotator_id="   ")
```

---

## HMAC Security Context

The annotation queue HMAC is constructed over:

```
{wallet}|{label}|{annotator_id}|{annotated_at}
```

An empty or whitespace `annotator_id` does not break the HMAC integrity check
(`load_queue` will still verify the MAC correctly), but it breaks the
**semantic** guarantee: the MAC now authenticates an attribution-free record.
Future features that rely on per-annotator agreement scores, bias audits, or
reviewer accountability reports will silently produce wrong results on any
record written with a blank ID — without any integrity failure to signal the
problem.

---

## Affected Files

| File | Change type |
|---|---|
| `scripts/annotate.py` | Strengthen `--annotator-id` guard: strip-and-check, or argparse `type=` validator |
| `detection/active_learning/annotation_queue.py` | Add defence-in-depth `annotator_id` check in `annotate()` |
| `tests/test_annotation_queue.py` | Add rejection tests for empty, whitespace, and tab-only annotator IDs |

---

## Related

- Issue #59 — original annotation queue + HMAC design
- `README.md` §Security — HMAC input format documentation
- `detection/active_learning/queue_io.py` — `save_queue` / `load_queue` HMAC implementation
- `tests/test_annotation_queue.py` — existing HMAC integrity tests
- `docs/active_learning.md` — active learning pipeline overview

