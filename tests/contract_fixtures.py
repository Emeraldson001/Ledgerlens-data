"""Contract-driven test fixture framework for complex ledger scenarios (Issue #465).

Enables declarative specification, compilation, execution, and invariant verification
of complex Stellar ledger attack/anomaly scenarios (e.g. wash trading rings, MEV sandwich attacks,
liquidity pool drains, and multi-hop cyclic transactions).
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd


class AccountRole(StrEnum):
    ATTACKER = "ATTACKER"
    VICTIM = "VICTIM"
    LIQUIDITY_PROVIDER = "LIQUIDITY_PROVIDER"
    BENIGN = "BENIGN"
    HUB = "HUB"


class OperationType(StrEnum):
    PAYMENT = "PAYMENT"
    TRADE = "TRADE"
    MANAGE_BUY_OFFER = "MANAGE_BUY_OFFER"
    MANAGE_SELL_OFFER = "MANAGE_SELL_OFFER"
    PATH_PAYMENT = "PATH_PAYMENT"
    LIQUIDITY_ADD = "LIQUIDITY_ADD"
    LIQUIDITY_REMOVE = "LIQUIDITY_REMOVE"


class ScenarioPatternType(StrEnum):
    WASH_TRADE_RING = "WASH_TRADE_RING"
    MEV_SANDWICH = "MEV_SANDWICH"
    FLASH_LIQUIDITY_DRAIN = "FLASH_LIQUIDITY_DRAIN"
    MULTI_HOP_PAYMENT_CYCLE = "MULTI_HOP_PAYMENT_CYCLE"
    HIGH_FREQUENCY_SPOOFING = "HIGH_FREQUENCY_SPOOFING"


@dataclass
class AccountSpec:
    """Specification of an account participating in a ledger scenario."""

    account_id: str
    role: AccountRole = AccountRole.BENIGN
    initial_balance: float = 10000.0
    asset_code: str = "XLM"
    flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value if isinstance(self.role, AccountRole) else self.role
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccountSpec:
        d = dict(data)
        d["role"] = AccountRole(d["role"])
        return cls(**d)


@dataclass
class TransactionSpec:
    """Specification of a single transaction in a scenario timeline."""

    tx_id: str
    ledger_seq: int
    offset_seconds: float  # Seconds relative to scenario start_time
    source_account: str
    destination_account: str
    operation: OperationType
    amount: float
    asset_code: str = "XLM"
    counter_asset_code: str | None = None
    price: float | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["operation"] = (
            self.operation.value if isinstance(self.operation, OperationType) else self.operation
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionSpec:
        d = dict(data)
        d["operation"] = OperationType(d["operation"])
        return cls(**d)


@dataclass
class InvariantExpectationSpec:
    """Post-scenario assertion invariant expectation."""

    expectation_id: str
    target_metric: (
        str  # e.g., "benford_chi_square_24h", "counterparty_concentration_ratio", "is_anomaly"
    )
    condition: str  # "GREATER_THAN", "LESS_THAN", "EQUALS", "IN_RANGE"
    expected_value: float | int | str | bool
    target_account: str | None = None
    tolerance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvariantExpectationSpec:
        return cls(**data)


@dataclass
class ScenarioContract:
    """Declarative contract for a complex ledger scenario."""

    scenario_id: str
    name: str
    pattern_type: ScenarioPatternType
    description: str
    accounts: list[AccountSpec]
    timeline: list[TransactionSpec]
    expectations: list[InvariantExpectationSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "pattern_type": (
                self.pattern_type.value
                if isinstance(self.pattern_type, ScenarioPatternType)
                else self.pattern_type
            ),
            "description": self.description,
            "accounts": [a.to_dict() for a in self.accounts],
            "timeline": [t.to_dict() for t in self.timeline],
            "expectations": [e.to_dict() for e in self.expectations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioContract:
        d = dict(data)
        d["pattern_type"] = ScenarioPatternType(d["pattern_type"])
        d["accounts"] = [AccountSpec.from_dict(a) for a in d["accounts"]]
        d["timeline"] = [TransactionSpec.from_dict(t) for t in d["timeline"]]
        d["expectations"] = [InvariantExpectationSpec.from_dict(e) for e in d["expectations"]]
        return cls(**d)

    def validate_schema(self) -> list[str]:
        """Validate internal consistency of the scenario contract."""
        errors: list[str] = []
        acct_ids = {a.account_id for a in self.accounts}

        if not self.accounts:
            errors.append("Scenario contract must define at least one account.")
        if not self.timeline:
            errors.append("Scenario contract must define at least one transaction in timeline.")

        for idx, tx in enumerate(self.timeline):
            if tx.source_account not in acct_ids:
                errors.append(
                    f"Transaction {idx} ({tx.tx_id}) source_account '{tx.source_account}' not in defined accounts."
                )
            if tx.destination_account not in acct_ids:
                errors.append(
                    f"Transaction {idx} ({tx.tx_id}) destination_account '{tx.destination_account}' not in defined accounts."
                )
            if tx.amount <= 0:
                errors.append(f"Transaction {idx} ({tx.tx_id}) amount must be strictly positive.")

        return errors


@dataclass
class ScenarioValidationResult:
    """Result of running invariant expectations against pipeline output."""

    scenario_id: str
    all_passed: bool
    passed_expectations: list[str]
    failed_expectations: list[dict[str, Any]]
    evaluated_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LedgerScenarioBuilder:
    """Builder and simulator for contract-driven ledger test scenarios."""

    def __init__(self, contract: ScenarioContract) -> None:
        errors = contract.validate_schema()
        if errors:
            raise ValueError(f"Invalid ScenarioContract '{contract.scenario_id}': {errors}")
        self.contract = contract

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerScenarioBuilder:
        contract = ScenarioContract.from_dict(data)
        return cls(contract)

    @classmethod
    def from_json(cls, filepath_or_str: str | Path) -> LedgerScenarioBuilder:
        path = Path(str(filepath_or_str))
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(filepath_or_str)
        data = json.loads(content)
        return cls.from_dict(data)

    def build_trades_dataframe(
        self,
        base_time: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """Compile scenario timeline into a standard trade DataFrame."""
        if base_time is None:
            base_time = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=2)

        records = []
        for tx in self.contract.timeline:
            ts = base_time + datetime.timedelta(seconds=tx.offset_seconds)
            records.append(
                {
                    "trade_id": tx.tx_id,
                    "ledger_sequence": tx.ledger_seq,
                    "ledger_close_time": ts.isoformat(),
                    "account": tx.source_account,
                    "counterparty": tx.destination_account,
                    "seller": tx.source_account,
                    "buyer": tx.destination_account,
                    "amount": tx.amount,
                    "asset_code": tx.asset_code,
                    "price": tx.price or 1.0,
                    "operation_type": tx.operation.value,
                }
            )
        return pd.DataFrame(records)

    def verify_expectations(
        self,
        metrics_by_account: dict[str, dict[str, Any]],
    ) -> ScenarioValidationResult:
        """Verify scenario expectation invariants against computed feature/pipeline metrics."""
        passed_exp = []
        failed_exp = []

        for exp in self.contract.expectations:
            target_acct = exp.target_account
            if target_acct is None:
                # Default to attacker account if not specified
                attackers = [
                    a.account_id for a in self.contract.accounts if a.role == AccountRole.ATTACKER
                ]
                target_acct = attackers[0] if attackers else self.contract.accounts[0].account_id

            acct_metrics = metrics_by_account.get(target_acct, {})
            actual_val = acct_metrics.get(exp.target_metric)

            passed = False
            if actual_val is not None:
                if exp.condition == "GREATER_THAN":
                    passed = float(actual_val) > float(exp.expected_value)
                elif exp.condition == "LESS_THAN":
                    passed = float(actual_val) < float(exp.expected_value)
                elif exp.condition == "EQUALS":
                    if isinstance(exp.expected_value, (int, float)):
                        passed = abs(float(actual_val) - float(exp.expected_value)) <= exp.tolerance
                    else:
                        passed = str(actual_val) == str(exp.expected_value)

            if passed:
                passed_exp.append(exp.expectation_id)
            else:
                failed_exp.append(
                    {
                        "expectation_id": exp.expectation_id,
                        "target_account": target_acct,
                        "metric": exp.target_metric,
                        "condition": exp.condition,
                        "expected": exp.expected_value,
                        "actual": actual_val,
                    }
                )

        return ScenarioValidationResult(
            scenario_id=self.contract.scenario_id,
            all_passed=len(failed_exp) == 0,
            passed_expectations=passed_exp,
            failed_expectations=failed_exp,
            evaluated_metrics=metrics_by_account,
        )


# ---------------------------------------------------------------------------
# Pre-built Declarative Scenario Templates
# ---------------------------------------------------------------------------


def make_wash_trade_ring_contract() -> ScenarioContract:
    """Create circular wash trade ring contract (A -> B -> C -> D -> A)."""
    accounts = [
        AccountSpec(account_id="GATTACKER_A", role=AccountRole.ATTACKER),
        AccountSpec(account_id="GATTACKER_B", role=AccountRole.ATTACKER),
        AccountSpec(account_id="GATTACKER_C", role=AccountRole.ATTACKER),
        AccountSpec(account_id="GATTACKER_D", role=AccountRole.ATTACKER),
    ]

    ring = ["GATTACKER_A", "GATTACKER_B", "GATTACKER_C", "GATTACKER_D"]
    timeline = []
    tx_idx = 1
    # Generate 40 repetitive circular trades with identical amounts
    for _cycle in range(10):
        for i in range(len(ring)):
            src = ring[i]
            dst = ring[(i + 1) % len(ring)]
            timeline.append(
                TransactionSpec(
                    tx_id=f"ring_tx_{tx_idx}",
                    ledger_seq=1000 + tx_idx,
                    offset_seconds=float(tx_idx * 15),
                    source_account=src,
                    destination_account=dst,
                    operation=OperationType.TRADE,
                    amount=5000.00,  # Constant volume spike
                    asset_code="XLM",
                )
            )
            tx_idx += 1

    expectations = [
        InvariantExpectationSpec(
            expectation_id="high_counterparty_concentration",
            target_metric="counterparty_concentration_ratio",
            condition="GREATER_THAN",
            expected_value=0.70,
            target_account="GATTACKER_A",
        ),
        InvariantExpectationSpec(
            expectation_id="perfect_counterparty_concentration",
            target_metric="counterparty_concentration_ratio",
            condition="EQUALS",
            expected_value=1.0,
            target_account="GATTACKER_A",
        ),
    ]

    return ScenarioContract(
        scenario_id="scenario_wash_ring_v1",
        name="Circular Wash Trade Ring",
        pattern_type=ScenarioPatternType.WASH_TRADE_RING,
        description="4 colluding accounts trading round-trip fixed amounts to inflate DEX volume.",
        accounts=accounts,
        timeline=timeline,
        expectations=expectations,
    )


def make_mev_sandwich_contract() -> ScenarioContract:
    """Create MEV Sandwich attack scenario contract (Frontrun -> Victim -> Backrun)."""
    accounts = [
        AccountSpec(account_id="GMEV_BOT", role=AccountRole.ATTACKER),
        AccountSpec(account_id="GVICTIM_TRADER", role=AccountRole.VICTIM),
        AccountSpec(account_id="GDEX_POOL", role=AccountRole.HUB),
    ]

    timeline = [
        # 1. Front-run buy
        TransactionSpec(
            tx_id="mev_frontrun",
            ledger_seq=2001,
            offset_seconds=0.0,
            source_account="GMEV_BOT",
            destination_account="GDEX_POOL",
            operation=OperationType.TRADE,
            amount=50000.0,
            price=1.00,
        ),
        # 2. Victim trade
        TransactionSpec(
            tx_id="victim_trade",
            ledger_seq=2001,
            offset_seconds=1.0,
            source_account="GVICTIM_TRADER",
            destination_account="GDEX_POOL",
            operation=OperationType.TRADE,
            amount=10000.0,
            price=1.05,  # Slippage suffered
        ),
        # 3. Back-run sell
        TransactionSpec(
            tx_id="mev_backrun",
            ledger_seq=2001,
            offset_seconds=2.0,
            source_account="GDEX_POOL",
            destination_account="GMEV_BOT",
            operation=OperationType.TRADE,
            amount=50000.0,
            price=1.08,
        ),
    ]

    expectations = [
        InvariantExpectationSpec(
            expectation_id="intra_minute_clustering",
            target_metric="intra_minute_clustering",
            condition="GREATER_THAN",
            expected_value=0.80,
            target_account="GMEV_BOT",
        ),
    ]

    return ScenarioContract(
        scenario_id="scenario_mev_sandwich_v1",
        name="MEV Sandwich Attack",
        pattern_type=ScenarioPatternType.MEV_SANDWICH,
        description="Front-running and back-running victim trades within single ledger block.",
        accounts=accounts,
        timeline=timeline,
        expectations=expectations,
    )
