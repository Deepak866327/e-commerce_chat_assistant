"""FAQ withdrawal affects every customer surface, without rebuilding vectors."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.config import Settings
from src.faq import FAQService
from src.faq_catalog import customer_faqs, published_suggestions
from src.ingestion import IngestionService
from src.retrieval import Retriever
from src.security import Principal
from src.store import Store, encode
from tests.test_faq_datasets import StubClient, StubVerifier
from tests.test_ingestion import FakeVector


class FAQPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = replace(Settings(), state_dir=root / "state")
        self.store = Store(root / "control.sqlite3")
        vector = FakeVector(self.settings)
        self.store.set_setting("active_index", vector.id)
        self.ingestion = IngestionService(self.settings, self.store, vector)
        self.path = root / "returns.txt"
        self.path.write_text("Returns are accepted within 30 days.")
        self.document = self.ingestion.ingest(self.path)
        self.chunk = self.store.active_chunks()[0]
        self.admin, self.reviewer = Principal("admin", "admin"), Principal("reviewer", "reviewer")
        self.service = FAQService(self.store, self.settings, StubClient(), StubVerifier())
        self.faq_id = self.service.generate(self.chunk["id"], self.admin)["faq_ids"][0]
        # An alias with no source-word overlap makes FAQ-only retrieval observable.
        with self.store.connect() as conn:
            body = json.loads(conn.execute("SELECT body FROM faqs WHERE id=?", (self.faq_id,)).fetchone()[0])
            body["question"] = "Parcel beacon assistance?"
            conn.execute("UPDATE faqs SET body=? WHERE id=?", (encode(body), self.faq_id))
        self.retriever = Retriever(self.store, self.settings)

    def publish(self):
        self.service.transition(self.faq_id, "APPROVED", self.reviewer)
        self.service.transition(self.faq_id, "PUBLISHED", self.admin)

    def assert_visible(self, visible):
        self.assertEqual(bool(published_suggestions(self.store)), visible)
        self.assertEqual(bool(customer_faqs(self.store)), visible)
        self.assertEqual(bool(self.service.published()), visible)
        self.assertEqual(bool(self.retriever.search("beacon")["chunks"]), visible)

    def test_review_publish_and_withdraw_are_consistent(self):
        self.assert_visible(False)
        self.publish()
        self.assert_visible(True)
        self.service.transition(self.faq_id, "PENDING_REVIEW", self.reviewer)
        self.assert_visible(False)

    def test_customer_article_exposes_reviewed_answer_and_safe_category(self):
        self.publish()
        row = self.store.rows("SELECT body FROM faqs WHERE id=?", (self.faq_id,))[0]
        body = json.loads(row["body"])
        with self.store.connect() as conn:
            body["category"] = "  Returns  "
            conn.execute("UPDATE faqs SET body=? WHERE id=?", (encode(body), self.faq_id))
        self.assertEqual([{k: a[k] for k in ("faq_id", "question", "answer", "category")} for a in customer_faqs(self.store)], [{"faq_id": self.faq_id,
                         "question": body["question"], "answer": body["answer"], "category": "Returns"}])
        with self.store.connect() as conn:
            body["category"] = None
            conn.execute("UPDATE faqs SET body=? WHERE id=?", (encode(body), self.faq_id))
        self.assertEqual(customer_faqs(self.store)[0]["category"], "General")

    def test_edit_removes_published_alias_immediately(self):
        self.publish()
        self.assert_visible(True)
        self.service.edit(self.faq_id, {"question": "Where is the parcel beacon?"}, self.reviewer)
        self.assert_visible(False)

    def test_source_update_and_delete_remove_faqs(self):
        self.publish()
        self.path.write_text("Returns are accepted within 14 days.")
        self.ingestion.ingest(self.path)
        self.assert_visible(False)
        self.ingestion.delete_document(self.document["document_id"], "admin")
        self.assert_visible(False)

    def test_unreviewed_unverified_or_malformed_publications_are_hidden(self):
        self.publish()
        original = self.store.rows("SELECT body FROM faqs WHERE id=?", (self.faq_id,))[0]["body"]
        mutations = [(original, None), (original, ""), (encode({**json.loads(original), "verified": False}), "reviewer"),
                     ("invalid json", "reviewer"), ("[]", "reviewer"),
                     (encode({**json.loads(original), "answer": ""}), "reviewer")]
        for body, reviewer in mutations:
            with self.subTest(body=body, reviewer=reviewer):
                with self.store.connect() as conn:
                    conn.execute("UPDATE faqs SET body=?,reviewer=? WHERE id=?", (body, reviewer, self.faq_id))
                self.assert_visible(False)


if __name__ == "__main__":
    unittest.main()
