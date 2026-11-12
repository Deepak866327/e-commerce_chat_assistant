import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.evaluate_support_routes import run_cases
from src.config import ROOT
from src.schemas import QueryIntent


class SupportRouteBenchmarkTests(unittest.TestCase):
    def test_offline_report_does_not_count_unexecuted_model_routes_as_passed(self):
        cases = json.loads((ROOT / "data/benchmarks/support_routes.json").read_text(encoding="utf-8"))["cases"]
        report = run_cases(cases)
        self.assertEqual(report["total"], 25)
        self.assertEqual(report["executed"], 18)
        self.assertEqual(report["passed"], 18)
        self.assertEqual(report["skipped"], 7)
        self.assertEqual(report["model_backed_executed"], 0)

    def test_wrong_model_route_is_reported_as_failure(self):
        client = Mock()
        client.structured.return_value = QueryIntent(intent="OUT_OF_SCOPE")
        report = run_cases([{"id": "followup", "question": "And express?", "expected": "KNOWLEDGE",
                             "history": [{"role": "user", "content": "What is standard shipping time?"}]}],
                           client, SimpleNamespace(utility_model="utility"))
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["model_backed_executed"], 1)
        self.assertFalse(report["results"][0]["passed"])
