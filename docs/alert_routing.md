# Configurable Alert Routing for Detection Outcomes

## Overview

`streaming/alert_dispatcher.py` delivers every alert above the risk
threshold to a single, process-wide channel chosen at construction time
(stdout, one webhook, or one websocket client). `alerts/deduplicator.py`
groups correlated alerts from multiple detectors into one enriched alert,
but still hands the result to whatever single destination the caller wired
up. Routing policy -- *which* channel(s) a given alert should go to based
on its severity, contributing detectors, asset pair, or tenant -- was not
expressible without writing new dispatch code.

`alerts/router.py` adds a declarative routing layer between alert
production and delivery:

- **`RoutingRule`** -- a match predicate (`min_risk_score`, `detectors`,
  `asset_pair_patterns` glob, `tenant`) plus the destinations to route to.
- **`RouteDestination`** -- one delivery target: `channel` (e.g.
  `"webhook"`, `"pagerduty"`, `"slack"`) and an opaque `target` (URL,
  channel id, PagerDuty service key).
- **`AlertRouter`** -- evaluates an alert dict against an ordered rule
  list, unions matched destinations (deduplicated), supports
  `stop_on_match` for exclusive routing (e.g. "critical alerts go to
  PagerDuty only"), and falls back to `default_destinations` when nothing
  matches.
- **`load_routing_config`** / **`AlertRouter.from_yaml`** -- parses the
  YAML shape in `alerts/routing_config.yaml`, raising
  `RoutingConfigError` (naming the offending rule) on a structurally
  invalid config at load time.
- **`AlertRouter.explain(alert)`** -- returns the ordered list of rule
  names that matched, for diagnosing "why did/didn't this alert reach
  channel X" without manually re-evaluating every rule by hand.

## Design tradeoffs

- **Routing decides; it does not deliver.** `AlertRouter.route()` returns
  `RouteDestination` objects, not delivered alerts. Actual delivery stays
  the job of `AlertDispatcher` (or a per-channel sender you write) --
  keeping the routing policy pure and independent of transport concerns
  (retry, backoff, auth) that already live in `alert_dispatcher.py`.
- **Union-by-default, exclusive on request.** Most operational policies
  want "send to every channel whose rule matches" (e.g. both Slack *and*
  a compliance webhook for bridge-anchor pairs). `stop_on_match` is opt-in
  per rule for the minority case where a match should be exclusive (a
  critical alert going to PagerDuty and *nothing else configured after
  it*), rather than making exclusivity the default and requiring every
  rule to declare non-exclusivity.
- **Config errors fail at load time, not at alert time.** A rule with no
  `destinations`, a non-list `detectors`/`asset_pair_patterns`, or a
  non-numeric `min_risk_score` raises `RoutingConfigError` immediately in
  `load_routing_config()`, rather than silently never matching once alerts
  start flowing.
- **No existing dispatcher code was changed.** `AlertRouter` is additive;
  `streaming/alert_dispatcher.py` and `alerts/deduplicator.py` are
  untouched. Wiring the router in front of the dispatcher is a follow-up
  (see below), kept out of this change to avoid touching the alert
  delivery hot path in the same change as the new routing primitive.

## Usage

```python
from alerts.router import AlertRouter

router = AlertRouter.from_yaml("alerts/routing_config.yaml")

for destination in router.route(alert):
    send(destination.channel, destination.target, alert)

# Diagnostics: why did this alert go where it went?
router.explain(alert)  # -> ["critical-risk-pagerduty"]
```

Config shape (`alerts/routing_config.yaml`):

```yaml
rules:
  - name: critical-risk-pagerduty
    min_risk_score: 90
    stop_on_match: true
    destinations:
      - channel: pagerduty
        target: "ledgerlens-critical"

default_destinations:
  - channel: webhook
    target: "${ALERT_WEBHOOK_URL}"
```

## Validation

```
pytest tests/test_alert_router.py -v
```

Covers: each `RoutingRule` predicate in isolation (risk score, detector
intersection, asset-pair glob, tenant); `AlertRouter.route()` union
semantics, deduplication, `stop_on_match` exclusivity, default-route
fallback, and empty-result behavior with no rules/no defaults; hot-reload
via `set_rules()`; `explain()` diagnostics including the stop-on-match
short-circuit; and `load_routing_config`/`from_yaml` against both the
shipped `alerts/routing_config.yaml` and deliberately malformed configs
(missing destinations, non-list fields, non-numeric threshold).

## Follow-up work

- Wire `AlertRouter` into `streaming/alert_dispatcher.py` so live alerts
  are routed automatically instead of requiring callers to invoke both
  separately.
- Support `${ENV_VAR}` interpolation in `target` values at load time (the
  shipped config documents the intent with `${AUDIT_WEBHOOK_URL}`-style
  placeholders; resolving them is not yet implemented).
- Add a `tenant_config.py`-style loader integration so per-tenant routing
  rules can live alongside `config/tenants.yaml`.
