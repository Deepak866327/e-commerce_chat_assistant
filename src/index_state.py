"""Content-bound index snapshots and locally authenticated release reports."""

from hashlib import sha256
import hmac
import secrets

from src.store import encode


def corpus_fingerprint(store, index_id):
    with store.connect() as conn:
        rows = [dict(r) for r in conn.execute("""SELECT c.id,c.version_id,c.text,d.deleted,v.status FROM chunks c
            JOIN index_members m ON m.chunk_id=c.id JOIN documents d ON d.id=c.document_id
            JOIN document_versions v ON v.id=c.version_id WHERE m.index_id=? ORDER BY c.id""", (index_id,))]
        # Include pending/new/failed versions too. A newer document arriving
        # after evaluation must invalidate a release, even before its vectors exist.
        documents = [dict(r) for r in conn.execute("SELECT id,deleted FROM documents ORDER BY id")]
        versions = [dict(r) for r in conn.execute("SELECT id,document_id,version,file_hash,status FROM document_versions ORDER BY id")]
    return sha256(encode({"chunks": rows, "documents": documents, "versions": versions}).encode()).hexdigest()


def sign_report(store, report):
    key = store.setting("report_signing_key")
    if not key:
        key = store.get_or_create_setting("report_signing_key", secrets.token_hex(32))
    body = {k: v for k, v in report.items() if k != "signature"}
    return hmac.new(bytes.fromhex(key), encode(body).encode(), "sha256").hexdigest()


def verify_report(store, report):
    key = store.setting("report_signing_key")
    if not key or not report.get("signature"):
        return False
    return hmac.compare_digest(sign_report(store, report), report["signature"])
