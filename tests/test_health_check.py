import time

from streaming.health_check import WorkerHealthManager


def test_worker_health_manager_empty():
    manager = WorkerHealthManager(timeout_seconds=1.0)
    healthy, details = manager.is_healthy()
    assert healthy is True
    assert details["message"] == "no workers registered"


def test_worker_health_manager_healthy():
    manager = WorkerHealthManager(timeout_seconds=1.0)
    manager.heartbeat("worker-1")
    manager.heartbeat("worker-2")

    healthy, details = manager.is_healthy()
    assert healthy is True
    assert details["worker-1"] == "ok"
    assert details["worker-2"] == "ok"


def test_worker_health_manager_timeout():
    manager = WorkerHealthManager(timeout_seconds=0.1)
    manager.heartbeat("worker-1")
    manager.heartbeat("worker-2")

    time.sleep(0.2)
    manager.heartbeat("worker-1")

    healthy, details = manager.is_healthy()
    assert healthy is False
    assert details["worker-1"] == "ok"
    assert details["worker-2"] == "stalled"


if __name__ == "__main__":
    test_worker_health_manager_empty()
    test_worker_health_manager_healthy()
    test_worker_health_manager_timeout()
    print("All tests passed.")
