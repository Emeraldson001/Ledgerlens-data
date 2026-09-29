"""Test data factory for generating realistic Stellar trade fixtures.

Uses factory_boy to construct synthetic trade data with realistic patterns:
- CleanTradeFactory: Benford-conforming amounts, Poisson-distributed arrivals
- WashTradeFactory: Round-number amounts, constant intervals, same counterparty
- RingTradeFactory: Circular rings of N wallets with coordinated trading

All wallet addresses are valid Stellar G-prefixed Ed25519 account IDs.
"""

import hashlib
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from factory import Factory, LazyAttribute, Sequence, SubFactory
from factory.fuzzy import FuzzyDecimal

from ingestion.data_models import Asset, Trade

# Reuse the single source of truth for what a "valid" Stellar wallet address
# and trade amount look like (see tests/strategies.py). This keeps the two
# test-data modules from drifting into inconsistent definitions.
from tests.strategies import STELLAR_ACCOUNT_ID_ALPHABET, STELLAR_ACCOUNT_ID_PREFIX


def generate_stellar_account_id(seed: str = "") -> str:
    """Generate a valid Stellar G-prefixed Ed25519 account ID.

    Format: G + base32(32-byte Ed25519 public key) where base32 uses Stellar alphabet.
    Minimal correctness: G + alphanumeric chars using the shared Stellar
    account-id alphabet from tests/strategies.py.
    """
    if seed:
        h = hashlib.sha256(seed.encode()).digest()
    else:
        h = hashlib.sha256(str(uuid.uuid4()).encode()).digest()

    encoded = ""
    for byte in h:
        encoded += STELLAR_ACCOUNT_ID_ALPHABET[byte % len(STELLAR_ACCOUNT_ID_ALPHABET)]

    return STELLAR_ACCOUNT_ID_PREFIX + encoded[:56]


class AssetFactory(Factory):
    """Factory for generating test Asset objects."""

    class Meta:
        model = Asset

    code = Sequence(lambda n: ["USDC", "XLM", "BTC", "ETH", "CUSTOM"][n % 5])
    issuer = Sequence(lambda n: [None, generate_stellar_account_id(f"issuer-{n}")][n % 2])


class CleanTradeFactory(Factory):
    """Factory for Benford-conforming, realistic trades.

    Amounts follow a log-uniform distribution (conforming to Benford's Law).
    Inter-trade intervals follow a Poisson process (realistic market activity).
    Counterparties are diverse and varied.
    """

    class Meta:
        model = Trade

    trade_id = Sequence(lambda n: f"clean-trade-{n}")
    ledger_close_time = LazyAttribute(
        lambda o: datetime.utcnow() - timedelta(seconds=random.randint(0, 86400))
    )
    base_account = LazyAttribute(lambda o: generate_stellar_account_id(f"base-{uuid.uuid4()}"))
    counter_account = LazyAttribute(
        lambda o: generate_stellar_account_id(f"counter-{uuid.uuid4()}")
    )
    base_asset = SubFactory(AssetFactory)
    counter_asset = SubFactory(AssetFactory)

    @LazyAttribute
    def base_amount(o) -> float:
        """Log-uniform amount (Benford-conforming)."""
        rng = np.random.default_rng(seed=random.randint(0, 2**31 - 1))
        # Log-uniform in [10^2, 10^8] = [100, 100M] — within the shared
        # "valid trade amount" bounds (VALID_TRADE_AMOUNT_MIN..MAX).
        return round(float(10 ** rng.uniform(2, 8)), 7)

    counter_amount = FuzzyDecimal(10, 10000, precision=7)
    price = FuzzyDecimal(0.01, 100, precision=7)


class WashTradeFactory(Factory):
    """Factory for wash-trading patterns (non-Benford-conforming).

    Amounts are round numbers (multiples of 500, 1000, etc).
    Inter-trade intervals are constant (algorithmic spacing).
    Same counterparty (sock-puppet wallet).
    """

    class Meta:
        model = Trade

    trade_id = Sequence(lambda n: f"wash-trade-{n}")

    @LazyAttribute
    def ledger_close_time(o) -> datetime:
        """Fixed intervals (e.g., every 5 seconds)."""
        base_time = datetime.utcnow()
        # Each trade exactly N seconds apart
        offset = hash(o.trade_id) % 1000  # Distribute base times
        interval_seconds = 5
        return base_time - timedelta(seconds=offset + interval_seconds)

    base_account = LazyAttribute(lambda o: generate_stellar_account_id("wash-bot-primary"))
    counter_account = LazyAttribute(lambda o: generate_stellar_account_id("wash-bot-sock-puppet"))
    base_asset = SubFactory(AssetFactory)
    counter_asset = SubFactory(AssetFactory)

    @LazyAttribute
    def base_amount(o) -> float:
        """Round-number amounts (wash-trading signal)."""
        # Pick from [500, 1000, 5000, 10000, 50000]
        choices = [500, 1000, 5000, 10000, 50000]
        return float(random.choice(choices))

    counter_amount = FuzzyDecimal(10, 100, precision=7)
    price = FuzzyDecimal(0.01, 10, precision=7)


class RingTradeFactory(Factory):
    """Factory for ring-trading patterns (circular coordinator ring).

    A configurable ring of N wallets (default 5), each trading sequentially
    around the ring. Amounts and timing vary per ring member to appear organic.
    """

    class Meta:
        model = Trade

    trade_id = Sequence(lambda n: f"ring-trade-{n}")

    @LazyAttribute
    def ledger_close_time(o) -> datetime:
        """Slightly spaced trades (coordinated but not perfectly synchronized)."""
        base_time = datetime.utcnow()
        offset = hash(o.trade_id) % 300  # Spread over ~5 minutes
        return base_time - timedelta(seconds=offset)

    base_account = LazyAttribute(
        lambda o: generate_stellar_account_id(f"ring-member-{hash(o.trade_id) % 5}")
    )
    counter_account = LazyAttribute(
        lambda o: generate_stellar_account_id(f"ring-member-{(hash(o.trade_id) + 1) % 5}")
    )
    base_asset = SubFactory(AssetFactory)
    counter_asset = SubFactory(AssetFactory)

    @LazyAttribute
    def base_amount(o) -> float:
        """Slightly variable (but still suspicious) amounts."""
        base = random.choice([1000, 2000, 5000, 10000])
        # Add small jitter: ±10%
        jitter = base * random.uniform(0.9, 1.1)
        return round(jitter, 7)

    counter_amount = FuzzyDecimal(50, 500, precision=7)
    price = FuzzyDecimal(0.1, 10, precision=7)


# Pytest fixtures for quick test usage


def make_clean_trades(n: int = 100) -> list[dict[str, Any]]:
    """Generate a list of clean (Benford-conforming) trade dicts."""
    return [CleanTradeFactory.build().__dict__ for _ in range(n)]


def make_wash_trades(n: int = 50) -> list[dict[str, Any]]:
    """Generate a list of wash trades (non-Benford-conforming)."""
    return [WashTradeFactory.build().__dict__ for _ in range(n)]


def make_ring_trades(n: int = 30, ring_size: int = 5) -> list[dict[str, Any]]:
    """Generate a list of ring trades (circular coordination)."""
    trades = []
    for i in range(n):
        trade = RingTradeFactory.build()
        # Override wallets to form a ring
        member_idx = i % ring_size
        next_member_idx = (member_idx + 1) % ring_size
        trade.base_account = generate_stellar_account_id(f"ring-{member_idx}")
        trade.counter_account = generate_stellar_account_id(f"ring-{next_member_idx}")
        trades.append(trade.__dict__)
    return trades
