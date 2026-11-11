"""Record versions of documents already served by the unchanged legacy index.

This is metadata adoption, not activation of an unvalidated new vector index.
Every existing vector text must match the current PDF page, with sufficient page
coverage. No FAQ is approved or published by this migration.
"""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from src.config import ROOT
from src.documents import clean_text
from src.ingestion import IngestionService
from src.store import encode, now


def read_pdf_pages(path):
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def normalized(text):
    return " ".join(clean_text(text).casefold().split())


def snapshot(collection):
    records = collection.get(include=["documents", "metadatas"])
    items = list(zip(records["ids"], records["documents"], records["metadatas"]))
    return items, sha256(encode(sorted(items, key=lambda item: item[0])).encode()).hexdigest()


def adopt_legacy_sources(settings, store, principal, collection=None, policy_dir=None):
    principal.require("activate")
    if settings.pipeline != "legacy" or store.setting("active_index"):
        raise ValueError("Source adoption is only for legacy deployments without an active upgraded index")
    if collection is None:
        import chromadb
        collection = chromadb.PersistentClient(path=str(settings.vector_dir)).get_collection("shopassist_policies", embedding_function=None)
    policy_dir = Path(policy_dir or ROOT / "data" / "policies").resolve()
    records, fingerprint = snapshot(collection)
    if not records:
        raise ValueError("Legacy production index is empty")
    grouped = defaultdict(list)
    for chunk_id, text, metadata in records:
        name = str(metadata.get("source", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if not name or Path(name).suffix.lower() != ".pdf":
            raise ValueError("Legacy source must identify a local policy PDF")
        grouped[name].append((text, metadata))
    verified = []
    for name, chunks in grouped.items():
        path = policy_dir / name
        if path.is_symlink() or path.resolve().parent != policy_dir or not path.is_file():
            raise ValueError("Production source file is missing or outside the policy directory")
        file_hash = sha256(path.read_bytes()).hexdigest()
        pages = [normalized(text) for text in read_pdf_pages(path)]
        intervals = defaultdict(list)
        for text, metadata in chunks:
            page = metadata.get("page")
            if not isinstance(page, int) or not 0 <= page < len(pages):
                raise ValueError("Production chunk has invalid page metadata")
            excerpt = normalized(text)
            start = pages[page].find(excerpt)
            if not excerpt or start < 0:
                raise ValueError(f"{name}: indexed content does not match the current PDF; review and reindex separately")
            intervals[page].append((start, start + len(excerpt)))
        for page, text in enumerate(pages):
            if not text:
                continue
            covered, end = 0, 0
            for start, stop in sorted(intervals[page]):
                covered += max(0, stop - max(start, end))
                end = max(end, stop)
            if covered / len(text) < .8:
                raise ValueError(f"{name}: production index does not cover page {page + 1} sufficiently")
        verified.append({"path": path, "file_hash": file_hash, "matched_legacy_chunks": len(chunks)})
    service = IngestionService(settings, store)
    adopted = []
    for source in verified:
        result = service.ingest(source["path"], actor=principal.name)
        if result["status"] == "DUPLICATE":
            raise ValueError("Review duplicate source identities before adoption")
        adopted.append({**source, **result})
    if snapshot(collection)[1] != fingerprint:
        raise ValueError("Legacy index changed during source adoption")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if store.setting("active_index"):
            raise ValueError("An upgraded index was activated during source adoption")
        for source in adopted:
            version = conn.execute("SELECT file_hash FROM document_versions WHERE id=?", (source["version_id"],)).fetchone()
            if sha256(source["path"].read_bytes()).hexdigest() != source["file_hash"] or version[0] != source["file_hash"]:
                raise ValueError("Source changed during metadata adoption")
            conn.execute("UPDATE documents SET active_version=? WHERE id=? AND deleted=0", (source["version_id"], source["document_id"]))
            conn.execute("UPDATE faqs SET status='OUTDATED',updated_at=? WHERE document_id=? AND version_id<>?", (now(), source["document_id"], source["version_id"]))
        details = [{"source": s["path"].name, "file_hash": s["file_hash"], "document_id": s["document_id"],
                    "version_id": s["version_id"], "matched_legacy_chunks": s["matched_legacy_chunks"]} for s in adopted]
        store.set_setting("legacy_source_snapshot", encode({"fingerprint": fingerprint, "sources": details}), conn)
        store.audit(principal.name, "sources.adopt_existing_production", "shopassist_policies", {"sources": details}, conn)
    return {"sources": details, "new_index_activated": False, "faqs_published": 0}
