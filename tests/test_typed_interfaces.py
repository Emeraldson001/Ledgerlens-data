import os
import unittest

from cli.diagnostics import EnvironmentHealthCheck, StreamingHealthCheck, run_diagnostics
from utils.interfaces import (
    HealthCheckable,
    MetricsProvider,
    ServiceRegistry,
)


class DummyMetricsService:
    def get_metrics(self):
        return {"requests_total": 100, "error_rate": 0.01}


class DummyIncompleteService:
    pass


class TestTypedInterfaces(unittest.TestCase):

    def test_protocol_conformance(self):
        env_check = EnvironmentHealthCheck()
        streaming_check = StreamingHealthCheck()
        metrics_service = DummyMetricsService()

        self.assertTrue(isinstance(env_check, HealthCheckable))
        self.assertTrue(isinstance(streaming_check, HealthCheckable))
        self.assertTrue(isinstance(metrics_service, MetricsProvider))
        self.assertFalse(isinstance(metrics_service, HealthCheckable))

    def test_service_registry(self):
        registry = ServiceRegistry()
        env_check = EnvironmentHealthCheck()
        metrics_service = DummyMetricsService()
        incomplete_service = DummyIncompleteService()

        registry.register("environment", env_check)
        registry.register("metrics", metrics_service)
        registry.register("incomplete", incomplete_service)

        health_results = registry.check_all_health()
        self.assertIn("environment", health_results)
        self.assertEqual(
            health_results["environment"].status,
            "FAIL" if "RISK_SCORE_DB_URL" not in os.environ else "PASS",
        )
        self.assertEqual(health_results["incomplete"].status, "UNKNOWN")

        collected_metrics = registry.collect_all_metrics()
        self.assertIn("metrics", collected_metrics)
        self.assertEqual(collected_metrics["metrics"]["requests_total"], 100)
        self.assertNotIn("environment", collected_metrics)

    def test_run_diagnostics_backwards_compatibility(self):
        os.environ["RISK_SCORE_DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        os.environ["HORIZON_URL"] = "https://horizon.stellar.org"

        diag = run_diagnostics()
        self.assertEqual(diag["overall_status"], "PASS")
        self.assertIn("environment", diag["checks"])
        self.assertIn("streaming", diag["checks"])
        self.assertNotIn("pass", diag["checks"]["environment"]["details"]["RISK_SCORE_DB_URL"])


if __name__ == "__main__":
    unittest.main()
