from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from zipfile import ZipFile

from src.config import Settings
from src.documents import ExtractionError, Section, extract
from src.index_state import corpus_fingerprint, sign_report
from src.ingestion import IngestionService
from src.jobs import JobQueue
from src.security import Principal
from src.speech_controls import speech_controls_html
from src.store import Store
from src.voice import SpeechInputError, validate_recording
from tests.test_ingestion import FakeVector


class AdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = replace(Settings(), state_dir=self.root / "state")
        self.store = Store(self.root / "db.sqlite")
        self.vector = FakeVector(self.settings)
        self.service = IngestionService(self.settings, self.store, self.vector)

    def tearDown(self):
        self.temp.cleanup()

    def test_index_activation_signed_snapshot_and_stale_report(self):
        path = self.root / "policy.txt"
        path.write_text("Returns within seven days.")
        first = self.service.ingest(path)
        report = {"index_id": self.vector.id, "release_gate": True, "run_id": "test-fixture",
                  "corpus_fingerprint": corpus_fingerprint(self.store, self.vector.id)}
        # Synthetic report only tests the cryptographic/state guard, not metrics.
        report["signature"] = sign_report(self.store, report)
        self.service.activate_index(self.vector.id, report, "test-admin")
        self.assertEqual(self.store.setting("active_index"), self.vector.id)
        self.assertEqual(self.store.active_chunks()[0]["version_id"], first["version_id"])
        path.write_text("Returns within eight days.")
        self.service.ingest(path)
        with self.assertRaises(ValueError):
            self.service.activate_index(self.vector.id, report, "test-admin")

    def test_failed_extraction_then_retry_and_staged_idempotency(self):
        path = self.root / "policy.txt"
        path.write_text("A valid policy source.")
        with patch("src.ingestion.extract", side_effect=ExtractionError("bad extraction")):
            with self.assertRaises(ExtractionError):
                self.service.ingest(path)
        result = self.service.ingest(path)
        repeat = self.service.ingest(path)
        self.assertEqual(repeat["status"], "UNCHANGED")
        self.assertEqual(repeat["version_id"], result["version_id"])

    def test_ocr_fallback_and_persisted_sections_for_reindex(self):
        from pypdf import PdfWriter
        path = self.root / "scanned.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with path.open("wb") as stream:
            writer.write(stream)
        with self.assertRaises(ExtractionError):
            extract(path, self.settings)
        sections = extract(path, self.settings, ocr=lambda *args: "Scanned policy: returns within seven days.")
        self.assertEqual(sections[0].extraction, "ocr")
        with patch("src.ingestion.tesseract_ocr", return_value="Scanned policy: returns within seven days."):
            result = self.service.ingest(path, use_ocr=True)
        with patch("src.ingestion.extract", side_effect=AssertionError("must reuse original extraction")):
            self.service.index_extracted(result["version_id"])

    def test_archive_expansion_and_size_limits(self):
        path = self.root / "archive.docx"
        with ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<!DOCTYPE x [<!ENTITY x 'unsafe'>]><x/>")
        with self.assertRaises(ExtractionError):
            extract(path, self.settings)
        path = self.root / "large.txt"
        path.write_text("x" * 100)
        with self.assertRaises(ExtractionError):
            extract(path, replace(self.settings, max_upload_bytes=10))

    def test_job_reclaims_expired_lease_and_exhausts_retries(self):
        queue = JobQueue(self.store)
        principal = Principal("admin", "admin")
        job = queue.enqueue("ingest", {}, principal)
        original = queue.claim()
        with self.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until='2000-01-01' WHERE id=?", (job,))
        claimed = queue.claim()
        self.assertNotEqual(original["lease_token"], claimed["lease_token"])
        self.assertEqual(claimed["attempts"], 2)
        with self.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until='2000-01-01',attempts=3 WHERE id=?", (job,))
        self.assertIsNone(queue.claim())
        self.assertEqual(self.store.rows("SELECT status FROM jobs WHERE id=?", (job,))[0]["status"], "FAILED")

    def test_preserved_voice_validation_and_html_escaping(self):
        with self.assertRaises(SpeechInputError):
            validate_recording(b"not a WAV")
        output = BytesIO()
        with wave.open(output, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(16000)
            recording.writeframes(b"\0\0" * 1600)
        self.assertAlmostEqual(validate_recording(output.getvalue()), .1)
        html = speech_controls_html('"><script>alert(1)</script>', "test")
        self.assertNotIn('data-speech=""><script>', html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
