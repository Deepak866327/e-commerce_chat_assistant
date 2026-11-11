from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import Settings
from src.datasets import DatasetRepository, EvaluationCase, reproducible_splits
from src.faq import FAQService
from src.faq_catalog import published_suggestions
from src.ingestion import IngestionService
from src.ollama_client import InvalidModelOutput, OllamaClient
from src.schemas import FAQBatch, Scope
from src.security import Principal, authenticate, password_hash, redact
from src.store import Store
from tests.test_ingestion import FakeVector


class StubClient:
    def structured(self, model, messages, schema, **kwargs):
        return FAQBatch.model_validate({"faqs": [{"question": "When can I return an item?",
            "answer": "Returns are accepted within 30 days.", "category": "returns", "question_type": "policy",
            "supporting_text": "Returns are accepted within 30 days."}]})


class StubVerifier:
    def verify(self, *args):
        return {"passed": True}


def case(index=0, **overrides):
    data = {"id": f"E{index}", "question": f"What is policy number {index}?",
            "expected_answer": f"Policy {index} applies.", "expected_document_ids": ["doc"],
            "expected_chunk_ids": ["chunk"], "answerable": True, "difficulty": "easy",
            "language": "en", "category": "direct", "group_id": f"g{index}"}
    data.update(overrides)
    return EvaluationCase.model_validate(data)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = replace(Settings(), state_dir=self.root / "state")
        self.store = Store(self.root / "db.sqlite")
        vector = FakeVector(self.settings)
        self.store.set_setting("active_index", vector.id)
        self.ingestion = IngestionService(self.settings, self.store, vector)
        path = self.root / "policy.txt"
        path.write_text("Returns are accepted within 30 days.")
        self.document = self.ingestion.ingest(path)
        self.chunk = self.store.active_chunks()[0]
        self.admin = Principal("Alice", "admin")
        self.reviewer = Principal("Bob", "reviewer")
        self.faq = FAQService(self.store, self.settings, StubClient(), StubVerifier())

    def tearDown(self):
        self.temp.cleanup()

    def test_generate_review_publish_edit_and_outdated(self):
        result = self.faq.generate(self.chunk["id"], self.admin)
        faq_id = result["faq_ids"][0]
        self.assertEqual(self.faq.published(), [])
        with self.assertRaises(ValueError):
            self.faq.transition(faq_id, "PUBLISHED", self.admin)
        self.faq.transition(faq_id, "APPROVED", self.reviewer)
        with self.assertRaises(PermissionError):
            self.faq.transition(faq_id, "PUBLISHED", self.reviewer)
        self.faq.transition(faq_id, "PUBLISHED", self.admin)
        self.assertTrue(self.faq.published()[0]["verified"])
        self.faq.edit(faq_id, {"question": "What is the return window?"}, self.reviewer)
        self.assertEqual(self.faq.published(), [])
        with self.assertRaises(ValueError):
            self.faq.edit(faq_id, {"supporting_text": "Invented evidence"}, self.reviewer)
        self.ingestion.delete_document(self.document["document_id"], "Alice")
        with self.assertRaises(ValueError):
            self.faq.transition(faq_id, "APPROVED", self.reviewer)

    def test_duplicate_generation(self):
        self.faq.generate(self.chunk["id"], self.admin)
        result = self.faq.generate(self.chunk["id"], self.admin)
        self.assertEqual(result["faq_ids"], [])
        self.assertIn("Duplicate question", result["rejected"])

    def test_imported_local_faq_requires_evidence_review_and_publication(self):
        candidate = StubClient().structured(None, None, None).faqs[0].model_dump()
        faq_id = self.faq.import_candidate(self.chunk["id"], candidate, self.admin)
        row = self.store.rows("SELECT status,reviewer,body FROM faqs WHERE id=?", (faq_id,))[0]
        self.assertEqual(row["status"], "PENDING_REVIEW")
        self.assertIsNone(row["reviewer"])
        self.assertFalse(json.loads(row["body"])["verified"])
        self.assertEqual(published_suggestions(self.store), [])
        with self.assertRaises(ValueError):
            self.faq.transition(faq_id, "PUBLISHED", self.admin)
        self.faq.transition(faq_id, "APPROVED", self.reviewer)
        self.faq.transition(faq_id, "PUBLISHED", self.admin)
        self.assertEqual(published_suggestions(self.store)[0]["faq_id"], faq_id)
        with self.assertRaises(ValueError):
            self.faq.import_candidate(self.chunk["id"], candidate, self.admin)

    def test_local_faq_import_rejects_unsupported_claims_and_unauthorized_roles(self):
        candidate = StubClient().structured(None, None, None).faqs[0].model_dump()
        with self.assertRaises(PermissionError):
            self.faq.import_candidate(self.chunk["id"], candidate, self.reviewer)
        for changes in ({"supporting_text": "This is invented evidence."},
                        {"answer": "Returns are accepted within 99 days."}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.faq.import_candidate(self.chunk["id"], {**candidate, **changes}, self.admin)
        with patch.object(self.faq.verifier, "verify", return_value={"passed": False}):
            with self.assertRaises(ValueError):
                self.faq.import_candidate(self.chunk["id"], candidate, self.admin)
        self.assertFalse(self.store.rows("SELECT * FROM faqs"))

    def test_staged_faq_can_be_reviewed_but_not_published_or_suggested(self):
        with self.store.connect() as conn:
            conn.execute("UPDATE documents SET active_version=NULL")
        faq_id = self.faq.generate(self.chunk["id"], self.admin)["faq_ids"][0]
        self.faq.transition(faq_id, "APPROVED", self.reviewer)
        self.assertEqual(published_suggestions(self.store), [])
        with self.assertRaises(ValueError):
            self.faq.transition(faq_id, "PUBLISHED", self.admin)

    def test_only_published_current_verified_faqs_are_customer_suggestions(self):
        faq_id = self.faq.generate(self.chunk["id"], self.admin)["faq_ids"][0]
        self.assertEqual(published_suggestions(self.store), [])
        self.faq.transition(faq_id, "APPROVED", self.reviewer)
        self.assertEqual(published_suggestions(self.store), [])
        self.faq.transition(faq_id, "PUBLISHED", self.admin)
        self.assertEqual(published_suggestions(self.store), [{"faq_id": faq_id, "question": "When can I return an item?"}])
        self.faq.edit(faq_id, {"question": "What is the return window?"}, self.reviewer)
        self.assertEqual(published_suggestions(self.store), [])

    def test_worker_cannot_approve_or_publish_and_invalid_count_is_rejected(self):
        worker = Principal("background-worker", "worker")
        worker.require("ingest")
        for permission in ("review", "publish", "activate", "delete"):
            with self.assertRaises(PermissionError):
                worker.require(permission)
        with self.assertRaises(ValueError):
            self.faq.generate(self.chunk["id"], self.admin, count=0)

    def test_dataset_versioning_review_roundtrip_and_isolation(self):
        repository = DatasetRepository(self.root / "evaluation", self.store)
        sample = case(expected_document_ids=[self.chunk["document_id"]], expected_chunk_ids=[self.chunk["id"]])
        version = repository.create([sample], self.admin)
        self.assertFalse(repository.load(version)[0].reviewed)
        reviewed = repository.review(version, [sample.id], self.reviewer, self.store)
        self.assertNotEqual(version, reviewed)
        self.assertTrue(repository.load(reviewed)[0].reviewed)
        self.assertFalse(repository.load(version)[0].reviewed)
        for suffix in ("json", "csv"):
            path = self.root / f"export.{suffix}"
            repository.export(reviewed, path)
            imported = repository.import_file(path, self.admin)
            self.assertFalse(repository.load(imported)[0].reviewed)
        self.assertEqual(len(self.store.active_chunks()), 1)
        dataset_file = repository.root / version / "cases.json"
        dataset_file.write_text("[]")
        with self.assertRaises(ValueError):
            repository.load(version)

    def test_grouped_splits_and_duplicates(self):
        cases = [case(i) for i in range(12)]
        first = reproducible_splits(cases)
        self.assertEqual([c.split for c in first], [c.split for c in reproducible_splits(cases)])
        self.assertEqual({c.split for c in first}, {"development", "validation", "test"})
        with self.assertRaises(ValueError):
            reproducible_splits([case(1), case(2, question=case(1).question)])

    def test_ollama_schema_retry_and_failure(self):
        replies = iter([{"message": {"content": "invalid"}}, {"message": {"content": '{"label":"IN_SCOPE"}'}}])
        client = OllamaClient(self.settings, lambda _: next(replies))
        self.assertEqual(client.structured("model", [], Scope).label, "IN_SCOPE")
        self.assertEqual(len(client.calls), 2)
        invalid = OllamaClient(self.settings, lambda _: {"message": {"content": '{"label":"IN_SCOPE","secret":"bad"}'}})
        with self.assertRaises(InvalidModelOutput):
            invalid.structured("model", [], Scope)

    def test_auth_and_redaction(self):
        value = password_hash("a-long-test-password")
        with patch.dict(os.environ, {"SHOPASSIST_ADMIN_PASSWORD_HASH": value}):
            self.assertEqual(authenticate("admin", "a-long-test-password").role, "admin")
            with self.assertRaises(PermissionError):
                authenticate("admin", "wrong")
        redacted = redact("user@example.com order:ABC123456 +91 98765 43210 password=secret")
        self.assertNotIn("example.com", redacted)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("ABC123456", redacted)


if __name__ == "__main__":
    unittest.main()
