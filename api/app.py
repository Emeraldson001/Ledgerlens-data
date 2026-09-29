"""FastAPI REST API exposing LedgerLens wallet risk scores.

Endpoints:
    GET /v1/wallets/{address}/scores   — paginated risk score history
    GET /v1/wallets/{address}/latest   — latest score + top-3 features
    GET /v1/health                     — liveness / readiness check
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail fast on a missing/misconfigured var (e.g. no API_KEYS, meaning
    # every request would 401 forever) instead of discovering it from the
    # first request that hits the affected code path.
    try:
        validate_mode("api")
    except OSError as exc:
        logger.error(str(exc))
        raise
    yield


app = FastAPI(
    title="LedgerLens Risk Score API",
    version="1.0.0",
    description="Wallet risk scores for Stellar DEX wash-trade detection.",
    lifespan=_lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class RiskScoreResponse(BaseModel):
    score_id: int
    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    propagated_risk: float | None = None
    ring_id: str | None = None
    updated_at: str


class PaginatedScoresResponse(BaseModel):
    items: list[RiskScoreResponse]
    next_cursor: int | None = Field(None, description="score_id cursor for next page")
    total: int


class LatestScoreResponse(BaseModel):
    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    top_features: list[dict]


class HealthResponse(BaseModel):
    status: str
    db: str
    model: str
    workers: dict[str, dict] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_store = RiskScoreStore()
_session_factory = get_session_factory()


def _record_to_response(r: RiskScoreRecord) -> RiskScoreResponse:
    return RiskScoreResponse(
        score_id=r.id,
        wallet=r.wallet,
        asset_pair=r.asset_pair,
        score=r.score,
        benford_flag=r.benford_flag,
        ml_flag=r.ml_flag,
        confidence=r.confidence,
        propagated_risk=r.propagated_risk,
        ring_id=r.ring_id,
        updated_at=r.updated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def health(request: Request):
    """Liveness and readiness probe."""
    db_status = "ok"
    try:
        with _session_factory() as session:
            session.execute(select(RiskScoreRecord).limit(1))
    except Exception:
        db_status = "unavailable"

    import os

    model_status = "ok" if os.path.isdir(config.MODEL_DIR) else "unavailable"

    registry = get_health_registry()
    worker_status, worker_report = registry.get_overall_status()

    if db_status != "ok" or worker_status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
        overall = "degraded" if db_status == "ok" else "unavailable"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        db=db_status,
        model=model_status,
        workers=worker_report.get("components"),
    )


@app.get("/v1/health/workers", tags=["ops"])
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def worker_health(request: Request):
    """Detailed health probe for all registered background worker processes."""
    registry = get_health_registry()
    overall_status, report = registry.get_overall_status()
    return {
        "status": overall_status.value,
        "report": report,
    }


@app.get("/v1/rate-limit/metrics", tags=["ops"])
async def rate_limit_metrics(request: Request, _key: str = Depends(_check_api_key)):
    """Per-tenant rate limit utilization and throttling event metrics."""
    return _tenant_limiter.metrics()


@app.get(
    "/v1/wallets/{address}/scores",
    response_model=PaginatedScoresResponse,
    tags=["scores"],
)
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def get_wallet_scores(
    request: Request,
    address: str,
    start_ts: int | None = Query(None, description="Unix timestamp lower bound"),
    end_ts: int | None = Query(None, description="Unix timestamp upper bound"),
    asset_pair: str | None = Query(None),
    min_score: int | None = Query(None, ge=0, le=100),
    cursor: int | None = Query(None, description="Cursor from previous page (score_id)"),
    limit: int = Query(50, ge=1, le=200),
    _key: str = Depends(_check_api_key),
):
    """Paginated risk score history for a wallet (cursor-based on score_id)."""
    _enforce_tenant_limit(_key)
    _validate_stellar_address(address)

    from datetime import UTC, datetime

    with _session_factory() as session:
        stmt = (
            select(RiskScoreRecord)
            .where(RiskScoreRecord.wallet == address)
            .order_by(RiskScoreRecord.id.desc())
        )
        if cursor is not None:
            stmt = stmt.where(RiskScoreRecord.id < cursor)
        if asset_pair is not None:
            stmt = stmt.where(RiskScoreRecord.asset_pair == asset_pair)
        if min_score is not None:
            stmt = stmt.where(RiskScoreRecord.score >= min_score)
        if start_ts is not None:
            stmt = stmt.where(
                RiskScoreRecord.updated_at >= datetime.fromtimestamp(start_ts, tz=UTC)
            )
        if end_ts is not None:
            stmt = stmt.where(RiskScoreRecord.updated_at <= datetime.fromtimestamp(end_ts, tz=UTC))

        rows = list(session.scalars(stmt.limit(limit + 1)))

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None

    return PaginatedScoresResponse(
        items=[_record_to_response(r) for r in items],
        next_cursor=next_cursor,
        total=len(items),
    )
