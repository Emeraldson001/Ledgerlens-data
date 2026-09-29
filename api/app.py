"""FastAPI REST API exposing LedgerLens wallet risk scores.

Endpoints:
    GET /v1/wallets/{address}/scores   — paginated risk score history
    GET /v1/wallets/{address}/latest   — latest score + top-3 features
    GET /v1/health                     — liveness / readiness check

Idempotency contract
--------------------
Write endpoints (e.g. manual review submissions, threshold overrides) accept
an idempotency key via the ``Idempotency-Key`` request header or an
``idempotency_key`` body field. When a key is supplied, the first request is
processed and its response is stored for a configurable TTL window
(``config.API_IDEMPOTENCY_TTL_SECONDS``). A retry that reuses the same key
within that window returns the original stored response verbatim and does
*not* reprocess the underlying write, so a client retry after a network
timeout cannot produce a duplicate write. Keys are scoped per authenticated
tenant. After the TTL expires the key is forgotten and a subsequent request
with the same key is treated as a new write.
"""

import re
import threading
import time
from contextlib import asynccontextmanager

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select

from config import config
from config.contracts import validate_mode
from detection.persistence import RiskScoreRecord, get_session_factory
from detection.risk_score_store import RiskScoreStore
from detection.shap_explainer import ShapExplainer
from streaming.health import HealthStatus, get_health_registry
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stellar address validation
# ---------------------------------------------------------------------------
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")


def _validate_stellar_address(address: str) -> str:
    if not _STELLAR_ACCOUNT_RE.match(address):
        raise HTTPException(status_code=400, detail="Invalid Stellar account address")
    return address


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    for hashed in config.API_KEYS:
        if bcrypt.checkpw(api_key.encode(), hashed.encode()):
            return api_key
    raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Tenant-scoped rate limiting
# ---------------------------------------------------------------------------
# Global limits (above) protect the API as a whole. Per-tenant limits below
# ensure a single high-volume tenant cannot degrade availability for others.
# Each tenant gets its own token bucket, so limiter state is fully isolated
# and no counters are shared across tenants.


class _TokenBucket:
    """Thread-safe token bucket for a single tenant."""

    __slots__ = ("capacity", "refill_per_sec", "tokens", "updated_at", "_lock")

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        """Try to consume ``amount`` tokens; return True if allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            if elapsed > 0:
                self.tokens = min(
                    self.capacity, self.tokens + elapsed * self.refill_per_sec
                )
                self.updated_at = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def utilization(self) -> float:
        """Fraction of the bucket currently consumed (0.0–1.0)."""
        with self._lock:
            if self.capacity <= 0:
                return 0.0
            return max(0.0, min(1.0, 1.0 - (self.tokens / self.capacity)))


class TenantRateLimiter:
    """Per-tenant token-bucket limiter with isolated state per tenant.

    Limits are resolved from ``config/tenant_config.py`` (backed by
    ``config/tenants.yaml``) so operators can set per-tenant overrides.
    """

    def __init__(self, default_rpm: int) -> None:
        self._default_rpm = int(default_rpm)
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()
        # Metrics: per-tenant utilization + throttling counters.
        self._throttled: dict[str, int] = {}
        self._allowed: dict[str, int] = {}

    def _limit_rpm(self, tenant_id: str) -> int:
        """Resolve the effective RPM for a tenant (override or default)."""
        try:
            from config.tenant_config import get_tenant_config

            tenant = get_tenant_config(tenant_id)
            override = getattr(tenant, "rate_limit_rpm", None)
            if override:
                return int(override)
        except Exception:
            # Unknown tenant or config unavailable: fall back to default.
            pass
        return self._default_rpm

    def _bucket_for(self, tenant_id: str) -> _TokenBucket:
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                rpm = self._limit_rpm(tenant_id)
                # Capacity = one minute of burst; refill at the configured RPM.
                bucket = _TokenBucket(capacity=rpm, refill_per_sec=rpm / 60.0)
                self._buckets[tenant_id] = bucket
            return bucket

    def check(self, tenant_id: str) -> bool:
        """Consume one token for ``tenant_id``; return True if allowed."""
        bucket = self._bucket_for(tenant_id)
        allowed = bucket.consume(1.0)
        with self._lock:
            if allowed:
                self._allowed[tenant_id] = self._allowed.get(tenant_id, 0) + 1
            else:
                self._throttled[tenant_id] = self._throttled.get(tenant_id, 0) + 1
        if not allowed:
            logger.warning(
                "tenant rate limit exceeded",
                extra={"tenant_id": tenant_id, "utilization": bucket.utilization()},
            )
        return allowed

    def metrics(self) -> dict:
        """Snapshot of per-tenant utilization and throttling events."""
        with self._lock:
            tenants = set(self._buckets) | set(self._throttled) | set(self._allowed)
            return {
                "tenants": {
                    tid: {
                        "utilization": self._buckets[tid].utilization()
                        if tid in self._buckets
                        else 0.0,
                        "allowed": self._allowed.get(tid, 0),
                        "throttled": self._throttled.get(tid, 0),
                    }
                    for tid in tenants
                }
            }


_tenant_limiter = TenantRateLimiter(default_rpm=config.API_RATE_LIMIT_RPM)


def _tenant_id_from_key(api_key: str) -> str:
    """Derive a stable tenant identity from the authenticated API key.

    The raw key is never used as a bucket key; a short digest keeps limiter
    state keyed on authenticated identity without storing secrets.
    """
    import hashlib

    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _enforce_tenant_limit(api_key: str) -> None:
    """Apply the per-tenant limit, raising 429 when the bucket is empty."""
    tenant_id = _tenant_id_from_key(api_key)
    if not _tenant_limiter.check(tenant_id):
        raise HTTPException(
            status_code=429,
            detail="Per-tenant rate limit exceeded",
            headers={"Retry-After": "1"},
        )


# ---------------------------------------------------------------------------
# Idempotency-key support for write endpoints
# ---------------------------------------------------------------------------
# Mirrors the dedupe-within-a-TTL-window pattern used in
# ``pipeline/idempotency.py``: a key is remembered for a configurable window
# and a duplicate request replays the original stored response instead of
# reprocessing the write. State is scoped per authenticated tenant so one
# tenant's keys can never collide with or observe another tenant's writes.


class IdempotencyStore:
    """Thread-safe, TTL-bounded store of responses keyed by idempotency key."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = float(ttl_seconds)
        # (tenant_id, key) -> (expires_at, status_code, response_body)
        self._entries: dict[tuple[str, str], tuple[float, int, dict]] = {}
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        expired = [k for k, (exp, _, _) in self._entries.items() if exp <= now]
        for k in expired:
            self._entries.pop(k, None)

    def get(self, tenant_id: str, key: str) -> tuple[int, dict] | None:
        """Return the stored ``(status_code, body)`` for a live key, else None."""
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            entry = self._entries.get((tenant_id, key))
            if entry is None:
                return None
            _, status_code, body = entry
            return status_code, body

    def put(self, tenant_id: str, key: str, status_code: int, body: dict) -> None:
        """Store a response for ``key`` until the TTL window elapses."""
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            self._entries[(tenant_id, key)] = (now + self._ttl, status_code, body)


_idempotency_store = IdempotencyStore(
    ttl_seconds=getattr(config, "API_IDEMPOTENCY_TTL_SECONDS", 86400)
)


def _resolve_idempotency_key(request: Request, body: BaseModel | None) -> str | None:
    """Extract an idempotency key from the header or the request body field."""
    key = request.headers.get("Idempotency-Key")
    if not key and body is not None:
        key = getattr(body, "idempotency_key", None)
    if key is not None:
        key = str(key).strip()
        if not key:
            return None
    return key


def _replay_if_duplicate(api_key: str, key: str | None):
    """Return the stored response for a duplicate key, else None."""
    if not key:
        return None
    tenant_id = _tenant_id_from_key(api_key)
    stored = _idempotency_store.get(tenant_id, key)
    if stored is None:
        return None
    status_code, body = stored
    logger.info(
        "idempotent replay",
        extra={"tenant_id": tenant_id, "idempotency_key": key},
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content=body)


def _remember_response(api_key: str, key: str | None, status_code: int, body: dict) -> None:
    """Persist a response so a later duplicate key can replay it."""
    if not key:
        return
    tenant_id = _tenant_id_from_key(api_key)
    _idempotency_store.put(tenant_id, key, status_code, body)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail fast on a missing/misconfigured var (e.g. no API_KEYS, meaning
    # every request would 401 forever) instead of discovering it from the
    # first reque

/* … truncated 5791 chars — edit only what you need near the top … */
