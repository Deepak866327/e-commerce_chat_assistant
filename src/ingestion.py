"""Idempotent document versions; preserve the working index on any failure."""

from dataclasses import asdict
from hashlib import sha256
import json
from uuid import uuid4

from src.chunking import chunk_sections, token_count
from src.config import Settings
from src.documents import Section, content_hash, extract, tesseract_ocr, validate_file
from src.store import Store, encode, now
from src.index_state import corpus_fingerprint, verify_report
from src.evaluation import compare_reports


class IngestionService:
    def __init__(self, settings=None, store=None, vector=None):
        self.settings = settings or Settings.from_env()
        self.store = store or Store(self.settings.state_dir / "control.sqlite3")
        self.vector = vector

    def ingest(self, path, source_key=None, actor="local", use_ocr=False):
        path = validate_file(path, self.settings)
        source_key = source_key or path.name
        if not source_key.strip() or len(source_key) > 200 or any(c in source_key for c in ("/", "\\", "\x00")):
            raise ValueError("Source key must be a short filename, without path separators")
        with path.open("rb") as stream:
            upload = stream.read(self.settings.max_upload_bytes + 1)
        if not 0 < len(upload) <= self.settings.max_upload_bytes:
            raise ValueError("File changed size during upload or exceeds the limit")
        file_hash = sha256(upload).hexdigest()
        document_id = "DOC_" + sha256(source_key.casefold().encode()).hexdigest()[:24]
        blob_dir = self.settings.state_dir / "documents"
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob = blob_dir / (file_hash + path.suffix.lower())
        if not blob.exists():
            temporary = blob_dir / (uuid4().hex + ".tmp")
            temporary.write_bytes(upload)
            temporary.replace(blob)
        elif sha256(blob.read_bytes()).hexdigest() != file_hash:
            raise ValueError("Stored source blob failed its content hash check")
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO documents(id,source_key,created_at) VALUES(?,?,?)",
                         (document_id, source_key.casefold(), now()))
            current = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            staged = conn.execute("SELECT * FROM document_versions WHERE document_id=? AND file_hash=? AND status IN ('EXTRACTED','INDEXED') ORDER BY version DESC LIMIT 1", (document_id, file_hash)).fetchone()
            if staged and not current["deleted"] and not current["active_version"]:
                if not self.vector or conn.execute("SELECT 1 FROM index_members m JOIN chunks c ON c.id=m.chunk_id WHERE c.version_id=? AND m.index_id=?", (staged["id"], self.vector.id)).fetchone():
                    return {"document_id": document_id, "version_id": staged["id"], "status": "UNCHANGED"}
            if current["active_version"] and not current["deleted"]:
                previous = conn.execute("SELECT * FROM document_versions WHERE id=?", (current["active_version"],)).fetchone()
                if previous["file_hash"] == file_hash:
                    if not self.vector or conn.execute("SELECT 1 FROM index_members m JOIN chunks c ON c.id=m.chunk_id WHERE c.version_id=? AND m.index_id=?", (previous["id"], self.vector.id)).fetchone():
                        return {"document_id": document_id, "version_id": previous["id"], "status": "UNCHANGED"}
            if conn.execute("SELECT id FROM document_versions WHERE document_id=? AND status='PROCESSING'", (document_id,)).fetchone():
                raise ValueError("Document ingestion already in progress")
            version = conn.execute("SELECT COALESCE(MAX(version),0)+1 FROM document_versions WHERE document_id=?", (document_id,)).fetchone()[0]
            version_id = "VER_" + uuid4().hex
            conn.execute("INSERT INTO document_versions VALUES(?,?,?,?,?,?,?,?,?)",
                         (version_id, document_id, version, file_hash, None, "PROCESSING", str(blob), None, now()))
        try:
            sections = extract(blob, self.settings, tesseract_ocr if use_ocr else None)
            digest = content_hash(sections)
            duplicate = self.store.rows("""SELECT v.document_id FROM document_versions v JOIN documents d
                 ON d.active_version=v.id WHERE v.content_hash=? AND d.deleted=0 AND d.id<>?""", (digest, document_id))
            if duplicate:
                with self.store.connect() as conn:
                    conn.execute("UPDATE document_versions SET status='DUPLICATE',content_hash=? WHERE id=?", (digest, version_id))
                    if not current["active_version"]:
                        conn.execute("UPDATE documents SET deleted=1 WHERE id=?", (document_id,))
                    self.store.audit(actor, "document.duplicate", document_id, {"duplicate_of": duplicate[0]["document_id"]}, conn)
                return {"document_id": document_id, "version_id": version_id, "status": "DUPLICATE", "duplicate_of": duplicate[0]["document_id"]}
            counter = getattr(self.vector, "count_tokens", token_count)
            chunks = chunk_sections(sections, document_id, version_id, source_key,
                                    self.settings.chunk_tokens, self.settings.chunk_overlap, counter)
            if self.vector:
                self.vector.upsert(chunks)
            with self.store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for ordinal, section in enumerate(sections):
                    conn.execute("INSERT INTO extracted_sections VALUES(?,?,?)", (version_id, ordinal, encode(asdict(section))))
                for chunk in chunks:
                    conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?)", (chunk.id, version_id, document_id, chunk.text, encode(chunk.metadata)))
                conn.execute("UPDATE document_versions SET status=?,content_hash=? WHERE id=?",
                             ("INDEXED" if self.vector else "EXTRACTED", digest, version_id))
                if self.vector:
                    index_id, spec = self.settings.index_identity()
                    if self.vector.id != index_id:
                        raise ValueError("Wrong vector index identity")
                    conn.execute("INSERT OR IGNORE INTO indexes VALUES(?,?,?,?,?,?)",
                                 (index_id, self.vector.collection_name, encode(spec), "CANDIDATE", self.vector.dimension, now()))
                    conn.executemany("INSERT OR IGNORE INTO index_members VALUES(?,?)", [(index_id, c.id) for c in chunks])
                    if self.store.setting("active_index") == index_id:
                        conn.execute("UPDATE documents SET active_version=?,deleted=0 WHERE id=?", (version_id, document_id))
                        conn.execute("UPDATE faqs SET status='OUTDATED',updated_at=? WHERE document_id=? AND version_id<>?", (now(), document_id, version_id))
                self.store.audit(actor, "document.ingest", document_id,
                                 {"version_id": version_id, "chunks": len(chunks), "indexed": bool(self.vector)}, conn)
            return {"document_id": document_id, "version_id": version_id, "status": "INDEXED" if self.vector else "EXTRACTED", "chunks": len(chunks)}
        except Exception as exc:
            with self.store.connect() as conn:
                conn.execute("UPDATE document_versions SET status='FAILED',error=? WHERE id=?", (type(exc).__name__ + ": " + str(exc)[:300], version_id))
                self.store.audit(actor, "document.failed", document_id, {"error_type": type(exc).__name__}, conn)
            raise

    def index_extracted(self, version_id, use_ocr=False):
        if not self.vector:
            raise ValueError("A vector backend is required")
        versions = self.store.rows("SELECT v.*,d.source_key FROM document_versions v JOIN documents d ON d.id=v.document_id WHERE v.id=?", (version_id,))
        if not versions or versions[0]["status"] not in {"EXTRACTED", "INDEXED"}:
            raise ValueError("No complete extracted version")
        version = versions[0]
        saved = self.store.rows("SELECT body FROM extracted_sections WHERE version_id=? ORDER BY ordinal", (version_id,))
        sections = [Section(**json.loads(r["body"])) for r in saved] if saved else extract(version["blob_path"], self.settings, tesseract_ocr if use_ocr else None)
        counter = getattr(self.vector, "count_tokens", token_count)
        chunks = chunk_sections(sections, version["document_id"], version_id, version["source_key"],
                                self.settings.chunk_tokens, self.settings.chunk_overlap, counter)
        self.vector.upsert(chunks)
        identity, spec = self.settings.index_identity()
        if identity != self.vector.id:
            raise ValueError("Wrong vector index identity")
        with self.store.connect() as conn:
            for chunk in chunks:
                conn.execute("INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?)", (chunk.id, version_id, version["document_id"], chunk.text, encode(chunk.metadata)))
            conn.execute("INSERT OR IGNORE INTO indexes VALUES(?,?,?,?,?,?)", (identity, self.vector.collection_name, encode(spec), "CANDIDATE", self.vector.dimension, now()))
            conn.executemany("INSERT OR IGNORE INTO index_members VALUES(?,?)", [(identity, c.id) for c in chunks])
            conn.execute("UPDATE document_versions SET status='INDEXED' WHERE id=? AND status='EXTRACTED'", (version_id,))

    def activate_index(self, index_id, validation, actor, baseline=None):
        if validation.get("release_gate") is not True or validation.get("index_id") != index_id:
            raise ValueError("A passing held-out evaluation for this index is required")
        if not verify_report(self.store, validation) or validation.get("corpus_fingerprint") != corpus_fingerprint(self.store, index_id):
            raise ValueError("Report is unsigned, modified, or predates a corpus change; rerun evaluation")
        previous_index = self.store.setting("active_index")
        if previous_index and previous_index != index_id:
            if not baseline or not verify_report(self.store, baseline) or baseline.get("index_id") != previous_index:
                raise ValueError("Changing an active index requires its signed baseline evaluation")
            comparison = compare_reports([baseline, validation])["comparisons"][0]
            if not comparison["regression_gate"]:
                raise ValueError("Candidate failed independent regression checks")
            if not any((comparison["deltas"].get(k) or 0) > 0 for k in ("recall_at_5", "mrr", "context_precision", "context_recall")):
                raise ValueError("An embedding/index replacement requires measurable retrieval improvement")
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Validate again under the writer lock so a concurrent ingestion or
            # deletion cannot change the evaluated corpus during activation.
            if validation.get("corpus_fingerprint") != corpus_fingerprint(self.store, index_id):
                raise ValueError("Corpus changed during activation")
            if not conn.execute("SELECT 1 FROM indexes WHERE id=?", (index_id,)).fetchone():
                raise ValueError("Unknown index")
            incomplete = conn.execute("""SELECT d.id FROM documents d WHERE d.deleted=0 AND NOT EXISTS(
                SELECT 1 FROM document_versions v JOIN chunks c ON c.version_id=v.id JOIN index_members m ON m.chunk_id=c.id
                WHERE v.document_id=d.id AND v.status='INDEXED' AND m.index_id=?
                AND v.version=(SELECT MAX(v2.version) FROM document_versions v2 WHERE v2.document_id=d.id))""", (index_id,)).fetchall()
            if incomplete:
                raise ValueError("Re-embed every non-deleted document before activation")
            previous = self.store.setting("active_index")
            if previous:
                self.store.set_setting("previous_index", previous, conn)
                self.store.set_setting("previous_run_id", self.store.setting("active_run_id", ""), conn)
                conn.execute("UPDATE indexes SET status='RETIRED' WHERE id=?", (previous,))
            old_versions = {r["id"]: r["active_version"] for r in conn.execute("SELECT id,active_version FROM documents")}
            self.store.set_setting("rollback_documents", encode(old_versions), conn)
            conn.execute("UPDATE indexes SET status='ACTIVE' WHERE id=?", (index_id,))
            self.store.set_setting("active_index", index_id, conn)
            self.store.set_setting("active_run_id", validation.get("run_id", ""), conn)
            for document in conn.execute("SELECT id FROM documents WHERE deleted=0").fetchall():
                latest = conn.execute("""SELECT DISTINCT v.id FROM document_versions v JOIN chunks c ON c.version_id=v.id
                    JOIN index_members m ON m.chunk_id=c.id WHERE v.document_id=? AND m.index_id=? AND v.status='INDEXED'
                    ORDER BY v.version DESC LIMIT 1""", (document["id"], index_id)).fetchone()
                conn.execute("UPDATE documents SET active_version=? WHERE id=?", (latest["id"], document["id"]))
            conn.execute("""UPDATE faqs SET status='OUTDATED',updated_at=? WHERE NOT EXISTS(
                SELECT 1 FROM documents d WHERE d.id=faqs.document_id AND d.active_version=faqs.version_id AND d.deleted=0)""", (now(),))
            self.store.audit(actor, "index.activate", index_id, {"previous": previous, "run_id": validation.get("run_id")}, conn)

    def rollback_index(self, actor):
        previous = self.store.setting("previous_index")
        pointers = json.loads(self.store.setting("rollback_documents", "{}"))
        if not previous or not pointers:
            raise ValueError("No previous index snapshot")
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE indexes SET status='RETIRED' WHERE status='ACTIVE'")
            conn.execute("UPDATE indexes SET status='ACTIVE' WHERE id=?", (previous,))
            self.store.set_setting("active_index", previous, conn)
            self.store.set_setting("active_run_id", self.store.setting("previous_run_id", ""), conn)
            for document, version in pointers.items():
                conn.execute("UPDATE documents SET active_version=? WHERE id=? AND deleted=0", (version, document))
            for row in conn.execute("SELECT id FROM documents WHERE deleted=0").fetchall():
                if row["id"] not in pointers:
                    conn.execute("UPDATE documents SET active_version=NULL WHERE id=?", (row["id"],))
            conn.execute("""UPDATE faqs SET status='OUTDATED',updated_at=? WHERE NOT EXISTS(
                SELECT 1 FROM documents d WHERE d.id=faqs.document_id AND d.active_version=faqs.version_id AND d.deleted=0)""", (now(),))
            self.store.audit(actor, "index.rollback", previous, conn=conn)

    def delete_document(self, document_id, actor):
        with self.store.connect() as conn:
            if not conn.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
                raise ValueError("Unknown document")
            conn.execute("UPDATE documents SET deleted=1 WHERE id=?", (document_id,))
            conn.execute("UPDATE faqs SET status='OUTDATED',updated_at=? WHERE document_id=?", (now(), document_id))
            self.store.audit(actor, "document.delete", document_id, conn=conn)

    def recover_ingestion(self, version_id, actor):
        """Explicit operator recovery after stopping the crashed worker."""
        with self.store.connect() as conn:
            row = conn.execute("SELECT status FROM document_versions WHERE id=?", (version_id,)).fetchone()
            if not row or row["status"] != "PROCESSING":
                raise ValueError("Only a stuck PROCESSING version can be recovered")
            conn.execute("UPDATE document_versions SET status='FAILED',error='Operator recovered interrupted ingestion' WHERE id=?", (version_id,))
            self.store.audit(actor, "document.recover", version_id, conn=conn)
