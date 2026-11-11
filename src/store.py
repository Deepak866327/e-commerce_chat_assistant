"""Transactional control plane; Chroma remains the dense vector database."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL, active_version TEXT,
 deleted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_versions(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
 version INTEGER NOT NULL, file_hash TEXT NOT NULL, content_hash TEXT,
 status TEXT NOT NULL, blob_path TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
 UNIQUE(document_id,version));
CREATE TABLE IF NOT EXISTS chunks(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES document_versions(id),
 document_id TEXT NOT NULL REFERENCES documents(id), text TEXT NOT NULL,
 metadata TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_version ON chunks(version_id);
CREATE TABLE IF NOT EXISTS extracted_sections(
 version_id TEXT NOT NULL REFERENCES document_versions(id), ordinal INTEGER NOT NULL,
 body TEXT NOT NULL, PRIMARY KEY(version_id,ordinal));
CREATE TABLE IF NOT EXISTS indexes(
 id TEXT PRIMARY KEY, collection TEXT UNIQUE NOT NULL, spec TEXT NOT NULL,
 status TEXT NOT NULL, dimension INTEGER, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS index_members(
 index_id TEXT NOT NULL REFERENCES indexes(id), chunk_id TEXT NOT NULL REFERENCES chunks(id),
 PRIMARY KEY(index_id,chunk_id));
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS faqs(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version_id TEXT NOT NULL,
 chunk_id TEXT NOT NULL REFERENCES chunks(id), status TEXT NOT NULL,
 body TEXT NOT NULL, reviewer TEXT, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS faq_history(
 id INTEGER PRIMARY KEY, faq_id TEXT NOT NULL, action TEXT NOT NULL,
 actor TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS generation_runs(
 id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL, model TEXT NOT NULL,
 status TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
 available_at TEXT NOT NULL, lease_until TEXT, error TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces(
 id TEXT PRIMARY KEY, session_hash TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(
 id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, comment TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log(
 id INTEGER PRIMARY KEY, actor TEXT NOT NULL, action TEXT NOT NULL,
 target TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rate_limits(
 subject TEXT NOT NULL, minute INTEGER NOT NULL, count INTEGER NOT NULL,
 PRIMARY KEY(subject,minute));
CREATE TABLE IF NOT EXISTS tickets(
 id TEXT PRIMARY KEY, reference TEXT UNIQUE NOT NULL, session_hash TEXT NOT NULL,
 query_id TEXT REFERENCES traces(id) ON DELETE SET NULL, status TEXT NOT NULL,
 reason TEXT NOT NULL, payload TEXT NOT NULL, contact_ciphertext TEXT,
 assignee TEXT, dedupe_key TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL, closed_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS active_ticket_dedupe ON tickets(session_hash,dedupe_key) WHERE closed_at IS NULL;
CREATE TABLE IF NOT EXISTS ticket_events(
 id INTEGER PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
 actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ticket_messages(
 id INTEGER PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
 author TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            # Schema is additive; never drop existing data or migrate Chroma.
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise RuntimeError("Database is newer than this application")
            conn.executescript(SCHEMA)
            conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES(1,?)", (now(),))
            conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES(2,?)", (now(),))
            conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES(3,?)", (now(),))
            conn.execute("PRAGMA user_version=3")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def rows(self, sql, args=()):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, args)]

    def setting(self, key, default=None):
        rows = self.rows("SELECT value FROM settings WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_setting(self, key, value, conn=None):
        if conn is None:
            with self.connect() as target:
                self.set_setting(key, value, target)
        else:
            conn.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, value))

    def get_or_create_setting(self, key, candidate):
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO settings VALUES(?,?)", (key, candidate))
            return conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()[0]

    def audit(self, actor, action, target, detail=None, conn=None):
        if conn is None:
            with self.connect() as target_conn:
                self.audit(actor, action, target, detail, target_conn)
        else:
            conn.execute("INSERT INTO audit_log(actor,action,target,detail,created_at) VALUES(?,?,?,?,?)",
                         (actor, action, target, encode(detail or {}), now()))

    def active_chunks(self, index_id=None):
        sql = """SELECT c.* FROM chunks c JOIN documents d ON d.id=c.document_id
                 WHERE d.deleted=0 AND d.active_version=c.version_id"""
        args = ()
        if index_id:
            sql += " AND EXISTS(SELECT 1 FROM index_members m WHERE m.chunk_id=c.id AND m.index_id=?)"
            args = (index_id,)
        return self.rows(sql, args)

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists() or destination.resolve() == self.path.resolve():
            raise ValueError("Backup destination must be new")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn, closing(sqlite3.connect(destination)) as target:
            conn.backup(target)
