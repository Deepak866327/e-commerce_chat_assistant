from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_real_models import inspect_result, inspect_support_turn, main
from src.ollama_client import ModelUnavailable


class RealModelRunnerTests(unittest.TestCase):
    def test_conversation_contracts_detect_lost_context_and_reasked_known_facts(self):
        failures = inspect_support_turn({"facts": ["dispatch_status"], "never_ask": ["channel"],
            "search_contains": ["missing cancellation"], "ticket": False, "model_calls": 0},
            {"support_context": {"reported_facts": {}}, "resolution": {"asked": ["channel"]}, "ticket": {"reference": "test"}}, 1)
        self.assertEqual(len(failures), 5)
        self.assertIn("missing_fact:dispatch_status", failures)

    def test_policy_case_cannot_pass_without_verified_citations(self):
        case = {"answerable": True}
        result = {"answerable": True, "verification": {"passed": False}, "sources": []}
        self.assertEqual(inspect_result(case, result), ["grounding_verification_failed", "missing_citations"])

    def test_action_case_requires_handoff_and_no_retrieval(self):
        case = {"answerable": False, "route": "account_action"}
        result = {"answerable": False, "route": "account_action", "sources": [],
                  "needs_human_support": True, "retrieval_performed": False}
        self.assertEqual(inspect_result(case, result), [])
        self.assertIn("unexpected_retrieval_for_account_action", inspect_result(case, {**result, "retrieval_performed": True}))

    def test_unavailable_models_are_infrastructure_errors_not_successful_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with patch("scripts.evaluate_real_models.OllamaClient.model_digest", side_effect=ModelUnavailable("unavailable")), \
                 redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(["--report", str(path)]), 2)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "infrastructure_error")
            self.assertEqual(report["stage"], "model_preflight")
            self.assertNotIn("summary", report)

    def test_legacy_runtime_paths_are_not_used_for_evaluation(self):
        with patch("scripts.evaluate_real_models.run_checks", return_value=0) as run, \
             redirect_stdout(StringIO()):
            main(["--answer-model", "existing-finetuned:latest", "--ollama-url", "http://localhost:11434"])
        settings = run.call_args.args[0]
        self.assertEqual(settings.answer_model, "existing-finetuned:latest")
        self.assertFalse(settings.allow_model_download)
