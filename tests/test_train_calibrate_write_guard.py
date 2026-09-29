"""Tests for training/train.py (Grand 2 / issue #671 acceptance criterion):
a deliberately regressed/blocked candidate must not overwrite production
"regardless of whether submitted via retrain_if_drifted.py or
training/train.py".

``train_and_calibrate`` shells out to ``python -m detection.model_training``,
which itself calls ``detection.model_governance.guard_production_write``
before writing any artifact to disk (see
``tests/test_model_training.py::test_main_refuses_to_overwrite_already_promoted_model_dir``
for the guard's own behavior). This test verifies the *caller*,
``training/train.py``, correctly treats that subprocess's non-zero exit as a
hard failure and does not proceed to calibrate against files the blocked
write never touched.
"""

from unittest.mock import MagicMock, patch

import pytest

from training.train import train_and_calibrate


def test_train_and_calibrate_propagates_blocked_write_guard_failure(tmp_path):
    data_path = str(tmp_path / "data.parquet")
    model_dir = str(tmp_path / "models")

    fake_result = MagicMock(returncode=1)
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        with pytest.raises(SystemExit) as excinfo:
            train_and_calibrate(data_path, model_dir)

    assert excinfo.value.code == 1
    mock_run.assert_called_once()
    # No calibrator artifacts must have been written — calibration never ran.
    assert not list(tmp_path.glob("**/calibrator_*.pkl"))
