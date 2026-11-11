"""Internal local observability; no third-party prompt or conversation exports."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import secrets
from uuid import uuid4

from src.security import redact, redact_value
from src.store import encode, now

FEEDBACK_KINDS = {"helpful", "not_helpful", "incorrect_answer", "missing_information", "outdated_information"}


class Monitor:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        # Per-install salt stops externally matching session identifiers.
        self.salt = store.get_or_create_setting("trace_salt", secrets.token_hex(32))

    def trace(self, session, body, query_id=None):
        query_id = query_id or "Q_" + uuid4().hex
        digest = sha256((self.salt + session).encode()).hexdigest()
        # Prompts and full conversation histories are deliberately excluded.
        safe = redact_value(body)
        with self.store.connect() as conn:
            conn.execute("INSERT INTO traces VALUES(?,?,?,?)", (query_id, digest, encode(safe), now()))
        self.prune()
        return query_id

    def feedback(self, query_id, kind, comment, session):
        if kind not in FEEDBACK_KINDS or len(comment) > 2000:
            raise ValueError("Invalid feedback kind or comment length")
        rows = self.store.rows("SELECT session_hash FROM traces WHERE id=?", (query_id,))
        digest = sha256((self.salt + session).encode()).hexdigest()
        if not rows or rows[0]["session_hash"] != digest:
            raise PermissionError("Feedback must belong to the current session")
        feedback_id = "FB_" + uuid4().hex
        with self.store.connect() as conn:
            conn.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", (feedback_id, query_id, kind, redact(comment), now()))
        return feedback_id

    def prune(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.retention_days)).isoformat()
        with self.store.connect() as conn:
            conn.execute("DELETE FROM traces WHERE created_at<?", (cutoff,))
            conn.execute("DELETE FROM traces WHERE id IN (SELECT id FROM traces ORDER BY created_at DESC LIMIT -1 OFFSET ?)", (self.settings.max_traces,))

    def summary(self):
        rows = self.store.rows("SELECT id,body FROM traces")
        traces = [json.loads(r["body"]) for r in rows]
        faq_ids = {r["id"] for r, trace in zip(rows, traces) if trace.get("event_type") == "faq_feedback"}
        faq_feedback = self.store.rows("SELECT trace_id,kind FROM feedback")
        traces = [trace for trace in traces if trace.get("event_type") != "faq_feedback"]
        total = len(traces)
        return {"queries": total,
                "escalations": sum(bool(t.get("needs_human_support")) for t in traces),
                "verification_failures": sum(t.get("verification", {}).get("passed") is False for t in traces),
                "errors": sum(bool(t.get("error")) for t in traces),
                "average_latency_seconds": sum(t.get("latency_seconds", 0) for t in traces) / total if total else None,
                "feedback": self.store.rows("SELECT kind,COUNT(*) AS count FROM feedback WHERE trace_id IN "
                                             "(SELECT id FROM traces WHERE json_extract(body,'$.event_type') IS NOT 'faq_feedback') GROUP BY kind"),
                "faq_feedback": {"helpful": sum(f["kind"] == "helpful" for f in faq_feedback if f["trace_id"] in faq_ids),
                                 "not_helpful": sum(f["kind"] == "not_helpful" for f in faq_feedback if f["trace_id"] in faq_ids)}}
