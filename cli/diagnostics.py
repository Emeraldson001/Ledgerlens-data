import os

from utils.interfaces import ServiceHealth
from utils.secrets import mask_secret, sanitize_url


class EnvironmentHealthCheck:
    def healthcheck(self) -> ServiceHealth:
        db_url = os.environ.get("RISK_SCORE_DB_URL", "")
        horizon_url = os.environ.get("HORIZON_URL", "")
        kafka_sasl_pass = os.environ.get("KAFKA_SASL_PASSWORD", "")

        details = {}
        missing = []

        if db_url:
            details["RISK_SCORE_DB_URL"] = sanitize_url(db_url)
        else:
            missing.append("RISK_SCORE_DB_URL")

        if horizon_url:
            details["HORIZON_URL"] = sanitize_url(horizon_url)
        else:
            missing.append("HORIZON_URL")

        if kafka_sasl_pass:
            details["KAFKA_SASL_PASSWORD"] = mask_secret(kafka_sasl_pass)

        status = "PASS" if not missing else "FAIL"
        return ServiceHealth(status=status, details=details, missing=missing)


class StreamingHealthCheck:
    def healthcheck(self) -> ServiceHealth:
        backend = os.environ.get("STREAMING_BACKEND", "stdout")
        return ServiceHealth(status="PASS", details={"backend": backend})


def run_diagnostics():
    env_service = EnvironmentHealthCheck()
    streaming_service = StreamingHealthCheck()

    env_health = env_service.healthcheck()
    streaming_health = streaming_service.healthcheck()

    overall = (
        "PASS" if env_health.status == "PASS" and streaming_health.status == "PASS" else "FAIL"
    )

    return {
        "overall_status": overall,
        "checks": {
            "environment": {
                "status": env_health.status,
                "details": env_health.details,
                "missing": env_health.missing,
            },
            "streaming": {
                "status": streaming_health.status,
                "backend": streaming_health.details.get("backend", "stdout"),
            },
        },
    }
