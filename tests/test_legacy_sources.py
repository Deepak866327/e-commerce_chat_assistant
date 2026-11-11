from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.config import ROOT, Settings
from src.legacy_sources import adopt_legacy_sources, read_pdf_pages
from src.security import Principal
from src.store import Store


class Collection:
    def __init__(self):
        self.documents = read_pdf_pages(ROOT / "data/policies/shipping.pdf")
        self.calls = 0

    def get(self, **kwargs):
        self.calls += 1
        return {"ids": [str(i) for i in range(len(self.documents))], "documents": self.documents,
                "metadatas": [{"source": "shipping.pdf", "page": i} for i in range(len(self.documents))]}


class LegacySourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = replace(Settings(), state_dir=Path(self.temp.name), pipeline="legacy")
        self.store = Store(self.settings.state_dir / "control.sqlite3")
        self.admin = Principal("admin", "admin")

    def test_adopt_existing_index_is_idempotent_and_does_not_publish_or_activate_index(self):
        collection = Collection()
        result = adopt_legacy_sources(self.settings, self.store, self.admin, collection)
        self.assertFalse(result["new_index_activated"])
        self.assertEqual(result["faqs_published"], 0)
        self.assertEqual(len(self.store.active_chunks()) > 0, True)
        self.assertIsNone(self.store.setting("active_index"))
        self.assertFalse(self.store.rows("SELECT * FROM faqs"))
        repeated = adopt_legacy_sources(self.settings, self.store, self.admin, collection)
        self.assertEqual(result["sources"], repeated["sources"])
        self.assertEqual(len(self.store.rows("SELECT * FROM document_versions")), 1)

    def test_mismatched_or_incomplete_index_cannot_adopt(self):
        for text in ("Unrelated invented policy text.", "Shipping"):
            collection = Collection()
            collection.documents = [text]
            with self.assertRaises(ValueError):
                adopt_legacy_sources(self.settings, self.store, self.admin, collection)
            self.assertFalse(self.store.rows("SELECT * FROM documents"))

    def test_adoption_requires_admin_and_cannot_override_grounded_index(self):
        with self.assertRaises(PermissionError):
            adopt_legacy_sources(self.settings, self.store, Principal("worker", "worker"), Collection())
        self.store.set_setting("active_index", "existing-grounded-index")
        with self.assertRaises(ValueError):
            adopt_legacy_sources(self.settings, self.store, self.admin, Collection())
        self.assertEqual(self.store.setting("active_index"), "existing-grounded-index")


if __name__ == "__main__":
    unittest.main()
