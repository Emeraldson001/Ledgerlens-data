"""Tests for alerts.router (configurable alert routing)."""

import os

import pytest

from alerts.router import (
    Alert,
    AlertRouter,
    RouteDestination,
    RoutingConfigError,
    RoutingRule,
    load_routing_config,
)


def _alert(**overrides: object) -> Alert:
    """Create a test alert with optional field overrides."""
    base: Alert = {
        "wallet_address": "GWALLET",
        "asset_pair": "USDC:GISSUER/native",
        "detectors": ["benford_engine"],
        "risk_score": 50,
        "tenant": "default",
    }
    base.update(overrides)  # type: ignore[typeddict-unknown-key]
    return base


# ---------------------------------------------------------------------------
# RoutingRule.matches
# ---------------------------------------------------------------------------


def test_rule_with_no_predicates_matches_everything():
    rule = RoutingRule(name="catch-all", destinations=[RouteDestination("webhook", "url")])
    assert rule.matches(_alert()) is True


def test_rule_min_risk_score_filters():
    rule = RoutingRule(
        name="high-risk", destinations=[RouteDestination("pagerduty", "x")], min_risk_score=90
    )
    assert rule.matches(_alert(risk_score=50)) is False
    assert rule.matches(_alert(risk_score=95)) is True


def test_rule_detectors_requires_intersection():
    rule = RoutingRule(
        name="consensus",
        destinations=[RouteDestination("slack", "#c")],
        detectors={"consensus_escalator"},
    )
    assert rule.matches(_alert(detectors=["benford_engine"])) is False
    assert rule.matches(_alert(detectors=["benford_engine", "consensus_escalator"])) is True


def test_rule_asset_pair_glob_matching():
    rule = RoutingRule(
        name="compliance",
        destinations=[RouteDestination("webhook", "x")],
        asset_pair_patterns=["USDC:*/native"],
    )
    assert rule.matches(_alert(asset_pair="USDC:GISSUER/native")) is True
    assert rule.matches(_alert(asset_pair="EUR:GISSUER/native")) is False


def test_rule_tenant_exact_match():
    rule = RoutingRule(name="acme", destinations=[RouteDestination("slack", "x")], tenant="acme")
    assert rule.matches(_alert(tenant="acme")) is True
    assert rule.matches(_alert(tenant="other")) is False


# ---------------------------------------------------------------------------
# AlertRouter.route — union / stop_on_match / defaults
# ---------------------------------------------------------------------------


def test_route_unions_destinations_from_multiple_matching_rules():
    router = AlertRouter(
        rules=[
            RoutingRule(name="r1", destinations=[RouteDestination("webhook", "a")]),
            RoutingRule(name="r2", destinations=[RouteDestination("slack", "b")]),
        ]
    )
    destinations = router.route(_alert())
    assert {d.key() for d in destinations} == {("webhook", "a"), ("slack", "b")}


def test_route_dedupes_same_destination_from_multiple_rules():
    router = AlertRouter(
        rules=[
            RoutingRule(name="r1", destinations=[RouteDestination("webhook", "a")]),
            RoutingRule(name="r2", destinations=[RouteDestination("webhook", "a")]),
        ]
    )
    destinations = router.route(_alert())
    assert len(destinations) == 1


def test_alert_dict_structure_is_typed():
    """Verify that Alert TypedDict properly documents alert structure."""
    alert = _alert()

    # All expected fields should be present and have correct types
    assert isinstance(alert.get("wallet_address"), str)
    assert isinstance(alert.get("asset_pair"), str)
    assert isinstance(alert.get("detectors"), list)
    assert isinstance(alert.get("risk_score"), (int, float))
    assert isinstance(alert.get("tenant"), str)


def test_stop_on_match_short_circuits_and_owns_alert():
    router = AlertRouter(
        rules=[
            RoutingRule(
                name="critical",
                destinations=[RouteDestination("pagerduty", "x")],
                min_risk_score=90,
                stop_on_match=True,
            ),
            RoutingRule(name="catch-all", destinations=[RouteDestination("webhook", "y")]),
        ]
    )
    destinations = router.route(_alert(risk_score=95))
    assert [d.key() for d in destinations] == [("pagerduty", "x")]


def test_no_matching_rule_falls_back_to_defaults():
    router = AlertRouter(
        rules=[
            RoutingRule(
                name="high", destinations=[RouteDestination("pagerduty", "x")], min_risk_score=90
            )
        ],
        default_destinations=[RouteDestination("webhook", "default-url")],
    )
    destinations = router.route(_alert(risk_score=10))
    assert [d.key() for d in destinations] == [("webhook", "default-url")]


def test_no_matching_rule_and_no_defaults_returns_empty():
    router = AlertRouter(rules=[])
    assert router.route(_alert()) == []


def test_set_rules_hot_reload_replaces_active_rules():
    router = AlertRouter(
        rules=[RoutingRule(name="old", destinations=[RouteDestination("webhook", "old")])]
    )
    router.set_rules([RoutingRule(name="new", destinations=[RouteDestination("webhook", "new")])])
    destinations = router.route(_alert())
    assert [d.key() for d in destinations] == [("webhook", "new")]


# ---------------------------------------------------------------------------
# AlertRouter.explain
# ---------------------------------------------------------------------------


def test_explain_lists_matching_rule_names_in_order():
    router = AlertRouter(
        rules=[
            RoutingRule(name="r1", destinations=[RouteDestination("webhook", "a")]),
            RoutingRule(
                name="r2", destinations=[RouteDestination("slack", "b")], min_risk_score=90
            ),
        ]
    )
    assert router.explain(_alert(risk_score=10)) == ["r1"]
    assert router.explain(_alert(risk_score=95)) == ["r1", "r2"]


def test_explain_stops_after_stop_on_match_rule():
    router = AlertRouter(
        rules=[
            RoutingRule(
                name="critical",
                destinations=[RouteDestination("pagerduty", "x")],
                min_risk_score=90,
                stop_on_match=True,
            ),
            RoutingRule(name="catch-all", destinations=[RouteDestination("webhook", "y")]),
        ]
    )
    assert router.explain(_alert(risk_score=95)) == ["critical"]


# ---------------------------------------------------------------------------
# load_routing_config / from_yaml
# ---------------------------------------------------------------------------


def test_load_routing_config_parses_shipped_config():
    path = os.path.join(os.path.dirname(__file__), "..", "alerts", "routing_config.yaml")
    rules = load_routing_config(path)
    assert len(rules) >= 1
    assert rules[0].name == "critical-risk-pagerduty"
    assert rules[0].stop_on_match is True


def test_from_yaml_builds_working_router():
    path = os.path.join(os.path.dirname(__file__), "..", "alerts", "routing_config.yaml")
    router = AlertRouter.from_yaml(path)

    destinations = router.route(_alert(risk_score=95))
    assert any(d.channel == "pagerduty" for d in destinations)


def test_load_routing_config_rejects_rule_without_destinations(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("rules:\n  - name: broken\n    min_risk_score: 10\n")
    with pytest.raises(RoutingConfigError):
        load_routing_config(str(bad_config))


def test_load_routing_config_rejects_non_list_rules(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("rules: not-a-list\n")
    with pytest.raises(RoutingConfigError):
        load_routing_config(str(bad_config))


def test_load_routing_config_rejects_non_numeric_min_risk_score(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "rules:\n"
        "  - name: broken\n"
        "    min_risk_score: not-a-number\n"
        "    destinations:\n"
        "      - channel: webhook\n"
        "        target: x\n"
    )
    with pytest.raises(RoutingConfigError):
        load_routing_config(str(bad_config))
