from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.calibration import calibrated_confidence, fit_calibration
from src.config import Settings
from src.datasets import DatasetRepository
from src.evaluation import EvaluationRunner, compare_reports, release_gate, retrieval_metrics
from src.index_state import sign_report, verify_report
from src.jobs import JobQueue
from src.observability import Monitor
from src.security import Principal, rate_limit
from src.store import Store
from tests.test_faq_datasets import case


class EvaluationOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "control.sqlite")
        self.settings = replace(Settings(), state_dir=self.root)
        self.principal = Principal("admin", "admin")

    def tearDown(self):
        self.temp.cleanup()

    def test_retrieval_metrics_and_gate_missing_semantics(self):
        scores = retrieval_metrics(["a", "b"], ["x", "b", "a"])
        self.assertEqual(scores["recall_at_1"], 0)
        self.assertEqual(scores["recall_at_3"], 1)
        self.assertEqual(scores["mrr"], .5)
        self.assertEqual(scores["context_precision"], 2 / 3)
        self.assertFalse(release_gate(scores, 1, [case()])[0])

    def test_evaluation_reports_and_regression_gate(self):
        dataset = DatasetRepository(self.root / "datasets", self.store)
        cases = [case(i, reviewed=True, reviewer="human") for i in range(12)]
        version = dataset.create(cases, self.principal)
        evaluator = EvaluationRunner(dataset, self.settings, self.root / "reports")

        def run(sample):
            return {"answer": sample.expected_answer, "answerable": True, "sources": [],
                    "retrieved_chunks": [{"id": "chunk", "text": sample.expected_answer, "metadata": {"document_id": "doc", "page_number": None}}]}

        report = evaluator.run(version, run, self.principal, variant="E")
        self.assertEqual(report["metrics"]["recall_at_5"], 1)
        self.assertIsNone(report["metrics"]["faithfulness"])
        self.assertFalse(report["release_gate"])
        self.assertTrue((self.root / "reports" / (report["run_id"] + ".csv")).exists())
        self.assertTrue(verify_report(self.store, report))
        report["metrics"]["recall_at_5"] = .4
        self.assertFalse(verify_report(self.store, report))
        second = evaluator.run(version, run, self.principal, variant="C")
        comparison = compare_reports([report, second])
        self.assertFalse(comparison["comparisons"][0]["regression_gate"])
        with self.assertRaises(ValueError):
            evaluator.run(version, run, self.principal, variant="invalid")

    def test_job_idempotency_retries_and_success(self):
        queue = JobQueue(self.store)
        first = queue.enqueue("ingest", {"path": "test"}, self.principal, "same")
        self.assertEqual(first, queue.enqueue("ingest", {"path": "test"}, self.principal, "same"))
        result = queue.run_once({"ingest": lambda payload: None})
        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertIsNone(queue.run_once({}))
        queue.enqueue("generate_faq", {}, self.principal)

        def fail(payload):
            raise RuntimeError("no model")

        self.assertEqual(queue.run_once({"generate_faq": fail})["status"], "QUEUED")
        self.assertIsNone(queue.claim())  # Backoff is respected.

    def test_trace_feedback_retention_redaction_and_ownership(self):
        monitor = Monitor(self.store, self.settings)
        query_id = monitor.trace("session-one", {"question": "Email me at private@example.com", "latency_seconds": 1})
        self.assertNotIn("private@example.com", self.store.rows("SELECT body FROM traces")[0]["body"])
        with self.assertRaises(PermissionError):
            monitor.feedback(query_id, "helpful", "", "session-two")
        monitor.feedback(query_id, "incorrect_answer", "Call +91 9876543210", "session-one")
        self.assertEqual(monitor.summary()["feedback"][0]["count"], 1)
        with self.store.connect() as conn:
            conn.execute("UPDATE traces SET created_at='2000-01-01T00:00:00+00:00'")
        monitor.prune()
        self.assertEqual(self.store.rows("SELECT * FROM traces"), [])
        self.assertEqual(self.store.rows("SELECT * FROM feedback"), [])

    def test_rate_limits_backup_and_calibration(self):
        rate_limit(self.store, "session", 1)
        with self.assertRaises(PermissionError):
            rate_limit(self.store, "session", 1)
        self.store.backup(self.root / "backup.sqlite")
        self.assertEqual(Store(self.root / "backup.sqlite").rows("SELECT * FROM rate_limits")[0]["count"], 1)
        signals = {"max_dense_similarity": .8, "evidence_coverage": 1, "verification_passed": True}
        calibration = fit_calibration([{"signals": signals, "correct": True, "reviewed": True}] * 20, "dev-v1")
        self.assertEqual(calibrated_confidence(signals, calibration), 1)
        with self.assertRaises(ValueError):
            fit_calibration([{"signals": signals, "correct": True}], "dev-v1")


if __name__ == "__main__":
    unittest.main()
