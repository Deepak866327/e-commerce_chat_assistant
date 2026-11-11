from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from src.chunking import chunk_sections, token_count
from src.config import Settings
from src.documents import ExtractionError, Section, extract
from src.ingestion import IngestionService
from src.store import Store


class FakeVector:
    def __init__(self, settings, fail=False):
        self.id = settings.index_identity()[0]
        self.collection_name = "test_" + self.id
        self.dimension = 3
        self.chunks = {}
        self.fail = fail

    def upsert(self, chunks):
        if self.fail:
            raise RuntimeError("insertion failed")
        self.chunks.update({c.id: c for c in chunks})


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = replace(Settings(), state_dir=self.root / "state", vector_dir=self.root / "vectors")
        self.store = Store(self.settings.state_dir / "control.sqlite3")
        self.vector = FakeVector(self.settings)
        self.service = IngestionService(self.settings, self.store, self.vector)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, text="Returns are accepted within 30 days.", name="policy.txt"):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_ingest_and_failed_update_keeps_current_version(self):
        path = self.source()
        result = self.service.ingest(path)
        with self.store.connect() as conn:
            conn.execute("UPDATE documents SET active_version=? WHERE id=?", (result["version_id"], result["document_id"]))
        self.store.set_setting("active_index", self.vector.id)
        self.assertEqual(self.service.ingest(path)["status"], "UNCHANGED")
        path.write_text("Changed policy text.", encoding="utf-8")
        self.vector.fail = True
        with self.assertRaises(RuntimeError):
            self.service.ingest(path)
        self.assertEqual(self.store.rows("SELECT active_version FROM documents")[0]["active_version"], result["version_id"])
        self.assertEqual(self.store.rows("SELECT status FROM document_versions ORDER BY version DESC")[0]["status"], "FAILED")

    def test_update_and_delete_filter_chunks(self):
        first = self.service.ingest(self.source())
        self.store.set_setting("active_index", self.vector.id)
        second = self.service.ingest(self.source("Returns are accepted within 45 days."))
        self.assertNotEqual(first["version_id"], second["version_id"])
        self.assertTrue(all(c["version_id"] == second["version_id"] for c in self.store.active_chunks()))
        self.service.delete_document(first["document_id"], "admin")
        self.assertEqual(self.store.active_chunks(), [])

    def test_duplicate_content_and_extract_only(self):
        self.store.set_setting("active_index", self.vector.id)
        self.service.ingest(self.source())
        other = self.service.ingest(self.source(name="copy.txt"))
        self.assertEqual(other["status"], "DUPLICATE")
        offline = IngestionService(self.settings, self.store)
        result = offline.ingest(self.source("Warranty covers defects.", "warranty.txt"))
        self.assertEqual(result["status"], "EXTRACTED")
        self.assertFalse(self.store.rows("SELECT active_version FROM documents WHERE id=?", (result["document_id"],))[0]["active_version"])
        self.service.index_extracted(result["version_id"])
        self.assertEqual(self.store.rows("SELECT status FROM document_versions WHERE id=?", (result["version_id"],))[0]["status"], "INDEXED")

    def test_chunk_metadata_and_unicode_bounds(self):
        sections = [Section("Refund policy " * 300, 12, "Refunds"), Section("Question: Reset?\nAnswer: Select reset.", kind="faq")]
        chunks = chunk_sections(sections, "doc", "ver", "policy.pdf", size=100, overlap=20)
        self.assertTrue(all(token_count(c.text) <= 100 for c in chunks))
        self.assertEqual(chunks[0].metadata["page_number"], 12)
        self.assertEqual(chunks[-1].metadata["kind"], "faq")
        self.assertEqual(chunks[-1].text, sections[-1].text)
        hindi = chunk_sections([Section("नमस्ते दुनिया " * 50)], "d", "v", "hi.txt", 100, 15)
        self.assertTrue(all(token_count(c.text) <= 100 for c in hindi))
        table = chunk_sections([Section("Plan | Days\n" + "standard | 5\n" * 20, kind="table")], "d", "v", "table.csv", 100, 15)
        self.assertTrue(all(c.text.startswith("Plan | Days\n") for c in table))

    def test_formats_and_eval_isolation(self):
        self.assertEqual(extract(self.source("# Shipping\nOrders arrive in five days."))[0].heading, "Shipping")
        faq = self.root / "faq.json"
        faq.write_text(json.dumps([{"question": "When?", "answer": "Five days", "verified": True, "status": "PUBLISHED"}]))
        self.assertEqual(extract(faq)[0].kind, "faq")
        faq.write_text(json.dumps([{"question": "When?", "expected_answer": "Five days"}]))
        with self.assertRaises(ExtractionError):
            extract(faq)
        self.assertEqual(extract(self.source("plan,days\nstandard,5\nexpress,2", "table.csv"))[0].kind, "table")
        docx = self.root / "test.docx"
        with ZipFile(docx, "w") as archive:
            archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Policy text</w:t></w:r></w:p></w:body></w:document>')
        self.assertEqual(extract(docx)[0].text, "Policy text")

    def test_validation_and_activation_gate(self):
        with self.assertRaises(ExtractionError):
            extract(self.source("bad", "script.exe"))
        with self.assertRaises(ExtractionError):
            extract(self.source("invalid PDF", "bad.pdf"))
        with self.assertRaises(ValueError):
            self.service.activate_index(self.vector.id, {}, "admin")
        with self.assertRaises(ValueError):
            self.service.ingest(self.source(), "../escape")

    def test_real_repository_documents(self):
        policies = Path(__file__).resolve().parents[1] / "data" / "policies"
        for path in sorted(policies.iterdir()):
            with self.subTest(path=path.name):
                sections = extract(path)
                self.assertTrue(sections)
                if path.suffix == ".pdf":
                    self.assertTrue(all(s.page_number >= 1 for s in sections))


if __name__ == "__main__":
    unittest.main()
