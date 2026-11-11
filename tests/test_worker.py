from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import Settings
from src.jobs import JobQueue
from src.ollama_client import OllamaClient
from src.schemas import Scope
from src.security import Principal
from src.store import Store
from src.worker import handlers


class WorkerTests(unittest.TestCase):
    def test_queued_extraction_runs_without_loading_vector_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(Settings(), state_dir=root / "state")
            store = Store(settings.state_dir / "control.sqlite3")
            source = root / "policy.txt"
            source.write_text("Returns require an invoice.")
            queue = JobQueue(store)
            queue.enqueue("ingest", {"path": str(source), "extract_only": True}, Principal("admin", "admin"))
            with patch("src.worker.ChromaIndex", side_effect=AssertionError("No index required")), \
                 patch("src.worker.OllamaClient", side_effect=AssertionError("No inference required")):
                result = queue.run_once(handlers(settings, store))
            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(store.rows("SELECT status FROM document_versions")[0]["status"], "EXTRACTED")

    def test_structured_output_budget_supports_faq_batches(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {"content": '{"label":"IN_SCOPE"}'}}

        client = OllamaClient(Settings(), transport)
        client.structured("model", [], Scope, num_ctx=8192, max_output_tokens=2816)
        self.assertEqual(payloads[0]["options"]["num_predict"], 2816)
        self.assertEqual(payloads[0]["options"]["num_ctx"], 8192)
        with self.assertRaises(ValueError):
            client.structured("model", [], Scope, max_output_tokens=0)
