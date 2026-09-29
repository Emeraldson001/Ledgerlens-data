"""Worker health check management.

Provides an HTTP health endpoint to verify long-running processes (Kafka workers,
SSE streams) are active. Workers call `heartbeat()` periodically. If any worker
fails to heartbeat within `HEALTH_CHECK_TIMEOUT_SECONDS`, the `/health` endpoint
returns 503 instead of 200.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class WorkerHealthManager:
    """Tracks heartbeat timestamps for multiple workers."""

    def __init__(self, timeout_seconds: float = 120.0):
        self._workers: dict[str, float] = {}
        self._timeout = timeout_seconds
        self._lock = threading.Lock()

    def heartbeat(self, worker_id: str) -> None:
        """Register a heartbeat for the given worker."""
        with self._lock:
            self._workers[worker_id] = time.time()

    def is_healthy(self) -> tuple[bool, dict[str, Any]]:
        """Return True if all registered workers have heartbeated recently."""
        with self._lock:
            if not self._workers:
                # If no workers registered yet, assume healthy to avoid premature failure.
                return True, {"status": "ok", "message": "no workers registered"}

            now = time.time()
            details = {}
            all_healthy = True
            for wid, last_hb in self._workers.items():
                if now - last_hb > self._timeout:
                    details[wid] = "stalled"
                    all_healthy = False
                else:
                    details[wid] = "ok"

            return all_healthy, details


# Global instance
_health_manager = WorkerHealthManager(
    timeout_seconds=float(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "120.0"))
)


def heartbeat(worker_id: str) -> None:
    """Global convenience for registering a heartbeat."""
    _health_manager.heartbeat(worker_id)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler serving a /health endpoint."""

    def do_GET(self) -> None:
        if self.path == "/health":
            healthy, details = _health_manager.is_healthy()
            status_code = 200 if healthy else 503

            self.send_response(status_code)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response_body = {
                "status": "ok" if healthy else "unhealthy",
                "workers": details,
            }
            self.wfile.write(json.dumps(response_body).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logging."""
        pass


def start_health_server(port: int = 8080) -> None:
    """Start the health check HTTP server in a daemon thread."""

    def run_server() -> None:
        try:
            # Bind to all interfaces for Kubernetes / Docker checks
            server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            logger.info("Health check server started on port %d", port)
            server.serve_forever()
        except Exception as exc:
            logger.error("Failed to start health check server: %s", exc)

    thread = threading.Thread(target=run_server, daemon=True, name="health-check-server")
    thread.start()
