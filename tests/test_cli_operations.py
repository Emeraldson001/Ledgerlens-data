import json
import os
import shutil
import tempfile
import unittest

from cli.commands.validate_artifacts import validate_artifacts
from cli.diagnostics import run_diagnostics
from cli.main import main


class TestCliOperations(unittest.TestCase):

    def setUp(self):
        self.old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_diagnostics_missing_env(self):
        os.environ.pop("RISK_SCORE_DB_URL", None)
        os.environ.pop("HORIZON_URL", None)
        report = run_diagnostics()
        self.assertEqual(report["overall_status"], "FAIL")
        self.assertIn("RISK_SCORE_DB_URL", report["checks"]["environment"]["missing"])

    def test_diagnostics_pass_env(self):
        os.environ["RISK_SCORE_DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        os.environ["HORIZON_URL"] = "https://horizon.stellar.org"
        os.environ["STREAMING_BACKEND"] = "stdout"
        report = run_diagnostics()
        self.assertEqual(report["overall_status"], "PASS")

    def test_validate_artifacts_missing_dir(self):
        res = validate_artifacts("/nonexistent_path_12345")
        self.assertEqual(res["status"], "FAIL")

    def test_validate_artifacts_valid(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            art_dir = os.path.join(tmp_dir, "artifacts")
            os.makedirs(art_dir)
            meta_file = os.path.join(art_dir, "model_metadata.json")
            with open(meta_file, "w") as f:
                json.dump({"model_version": "1.0.0", "feature_schema_hash": "abc123hash"}, f)

            res = validate_artifacts(art_dir)
            self.assertEqual(res["status"], "PASS")
            self.assertEqual(res["version"], "1.0.0")
        finally:
            shutil.rmtree(tmp_dir)

    def test_cli_entrypoint_healthcheck(self):
        os.environ["RISK_SCORE_DB_URL"] = "postgresql://localhost"
        os.environ["HORIZON_URL"] = "https://horizon.stellar.org"
        exit_code = main(["healthcheck"])
        self.assertEqual(exit_code, 0)

    def test_cli_entrypoint_healthcheck_json(self):
        os.environ["RISK_SCORE_DB_URL"] = "postgresql://localhost"
        os.environ["HORIZON_URL"] = "https://horizon.stellar.org"
        exit_code = main(["healthcheck", "--json"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
