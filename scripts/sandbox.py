"""Sandboxed execution checks for maintenance scripts.

Maintenance scripts under `scripts/` (DB migrations, backfills, retraining
jobs, ...) run with full process privileges and no guardrails: a runaway
loop can exhaust memory, a missing `--dry-run` flag can mutate production
data, and network access is unrestricted by default. `sandboxed_execution`
wraps a maintenance script's entrypoint with:

- CPU-time and address-space (memory) resource limits, so a runaway script
  is killed instead of degrading the host.
- Optional network blocking, for scripts that should only touch the DB or
  local filesystem.
- A dry-run guard (`dry_run_guard`) that logs the action a destructive call
  would take instead of executing it, and refuses to run destructive code
  unless the caller explicitly opts out of dry-run.

Usage:
    from scripts.sandbox import sandboxed_execution, dry_run_guard

    def main() -> None:
        args = parse_args()
        with sandboxed_execution(allow_network=False):
            dry_run_guard(
                args.dry_run,
                "ALTER TABLE risk_scores ADD COLUMN ring_id",
                lambda: migrate(get_engine()),
            )
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None  # type: ignore[assignment]

T = TypeVar("T")

DEFAULT_CPU_SECONDS = 300
DEFAULT_MEMORY_MB = 1024


class SandboxViolation(RuntimeError):
    """Raised when a maintenance script exceeds its sandbox constraints."""


def _blocked_socket(*_args: object, **_kwargs: object) -> None:
    raise SandboxViolation(
        "network access is disabled inside this sandboxed_execution() block "
        "(pass allow_network=True if this script legitimately needs it)"
    )


@contextmanager
def sandboxed_execution(
    *,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    allow_network: bool = True,
) -> Iterator[None]:
    """Run a block of a maintenance script under resource + network constraints.

    On POSIX systems, sets soft `RLIMIT_CPU` / `RLIMIT_AS` limits for the
    current process (best-effort — limits already tighter than requested are
    left untouched, matching `resource.setrlimit` semantics). Limits are not
    restored on exit since they only ever tighten the process ceiling, which
    is safe for the remainder of a short-lived script run.

    Raises `SandboxViolation` with an actionable message (which limit was
    hit or which disallowed action was attempted) instead of letting the
    process die with an opaque `MemoryError` or `OSError`.
    """
    start = time.monotonic()

    if resource is not None:
        try:
            soft_cpu, hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)
            new_cpu = (
                cpu_seconds if soft_cpu == resource.RLIM_INFINITY else min(soft_cpu, cpu_seconds)
            )
            resource.setrlimit(resource.RLIMIT_CPU, (new_cpu, hard_cpu))

            memory_bytes = memory_mb * 1024 * 1024
            soft_mem, hard_mem = resource.getrlimit(resource.RLIMIT_AS)
            new_mem = (
                memory_bytes if soft_mem == resource.RLIM_INFINITY else min(soft_mem, memory_bytes)
            )
            resource.setrlimit(resource.RLIMIT_AS, (new_mem, hard_mem))
        except (ValueError, OSError) as exc:
            logger.warning("Could not apply sandbox resource limits: %s", exc)
    else:
        logger.warning("resource module unavailable — CPU/memory limits not enforced")

    original_socket = socket.socket
    if not allow_network:
        socket.socket = _blocked_socket  # type: ignore[assignment]

    try:
        yield
    except MemoryError as exc:
        raise SandboxViolation(
            f"maintenance script exceeded the {memory_mb} MB sandbox memory limit"
        ) from exc
    except SandboxViolation:
        raise
    except Exception:
        elapsed = time.monotonic() - start
        logger.error(
            "Maintenance script failed inside sandboxed_execution after %.2fs "
            "(cpu_limit=%ss, memory_limit=%sMB, network=%s)",
            elapsed,
            cpu_seconds,
            memory_mb,
            allow_network,
        )
        raise
    finally:
        socket.socket = original_socket  # type: ignore[assignment]


def dry_run_guard(dry_run: bool, description: str, action: Callable[[], T]) -> T | None:
    """Log + skip a destructive `action` when `dry_run` is True, else run it.

    Centralizes the "what would this maintenance script do" logging so every
    script reports dry-run actions in the same format instead of each
    inventing its own.
    """
    if dry_run:
        logger.info("[DRY RUN] would execute: %s", description)
        return None
    logger.info("Executing: %s", description)
    return action()
