"""Customer-owned support tickets and an authenticated human response queue."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
import secrets
from uuid import uuid4

from src.security import rate_limit, redact, redact_value
from src.store import encode, now

STATUSES = {"OPEN", "ASSIGNED", "IN_PROGRESS", "WAITING_ON_CUSTOMER", "RESOLVED", "CLOSED"}
TERMINAL = {"RESOLVED", "CLOSED"}
ESCALATION_REASONS = {"explicit_handoff", "resolution_attempts_exhausted", "account_access_required", "no_validated_knowledge",
    "insufficient_retrieval", "insufficient_evidence", "verification_failed", "invalid_model_output",
    "intent_filter_unavailable", "model_unavailable", "service_unavailable", "index_unavailable",
    "source_changed_during_generation", "generation_failed", "user_requested", "legacy_unanswered"}


def requires_ticket(result):
    reason = result.get("reason")
    if reason in {"out_of_scope", "unsafe_request", "rate_limited", "clarification_needed"}:
        return False
    if result.get("needs_human_support") or reason in ESCALATION_REASONS:
        return True
    if result.get("route") == "knowledge" and not result.get("sources"):
        return True
    text = " ".join(result.get("answer", "").casefold().replace("’", "'").split())
    # Legacy generation is plain text. Recognize its explicit inability to help,
    # not policy prohibitions such as 'you cannot return a used product'.
    if any(phrase in text for phrase in ("verified policy information nahi", "i don't know",
        "i do not know", "i don't have access", "i do not have access", "i don't have enough information",
        "i do not have enough information", "mere paas account access nahi", "mere paas access nahi",
        "verified information nahi", "policy information available nahi", "जानकारी नहीं है")):
        # A live-access disclaimer alone does not invalidate a sourced policy
        # answer. Explicit abstention/uncertainty must still be escalated.
        policy_unknown = any(phrase in text for phrase in ("verified policy information nahi", "i don't know",
            "i do not know", "i don't have enough information", "i do not have enough information",
            "verified information nahi", "policy information available nahi", "जानकारी नहीं है"))
        guidance = re.search(r"\byou (?:can|may|should)\b|\bfollow (?:the|these)\b|\bbefore dispatch\b|"
                             r"aap.*(?:kar sakte|follow)|dispatch.*pehle|आप.*(?:कर सकते|करें)", text)
        return policy_unknown or not (result.get("route") == "knowledge" and result.get("sources") and guidance)
    return False


class TicketService:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.salt = store.get_or_create_setting("trace_salt", secrets.token_hex(32))

    def _session(self, session):
        if not isinstance(session, str) or not session:
            raise PermissionError("A conversation is required")
        return sha256((self.salt + session).encode()).hexdigest()

    def _owned(self, reference, session):
        rows = self.store.rows("SELECT * FROM tickets WHERE reference=? AND session_hash=?", (reference, self._session(session)))
        if not rows:
            raise PermissionError("Ticket is unavailable for this conversation")
        return rows[0]

    @staticmethod
    def summary(row):
        return {"reference": row["reference"], "status": row["status"], "created_at": row["created_at"]}

    def open(self, question, result, history, session, reason="user_requested"):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Describe the issue in 1–2000 characters")
        session_hash = self._session(session)
        from src.support_context import clean_context
        context = clean_context(result.get("support_context"))
        if context["ticket_reference"] and result.get("route") in {"account_action", "human_support", "handoff"}:
            previous = self._owned(context["ticket_reference"], session)
            if previous["status"] not in TERMINAL:
                self.reply(previous["reference"], question, session=session)
                return self.summary(previous)
        issue_id = result.get("resolution", {}).get("issue_id")
        dedupe_text = "resolution:" + issue_id if isinstance(issue_id, str) and re.fullmatch(r"ISSUE_[a-f0-9]{32}", issue_id) else " ".join(question.casefold().split())
        dedupe = sha256(dedupe_text.encode()).hexdigest()
        existing = self.store.rows("SELECT * FROM tickets WHERE session_hash=? AND dedupe_key=? AND closed_at IS NULL", (session_hash, dedupe))
        if existing:
            return self.summary(existing[0])
        rate_limit(self.store, "ticket-create:" + session, 5)
        timestamp = now()
        ticket_id, reference = "TKT_" + uuid4().hex, "SA-" + secrets.token_hex(8).upper()
        payload = redact_value({"question": question, "last_answer": result.get("answer", ""),
            "history": history[-8:], "sources": result.get("citations", result.get("sources", [])),
            "route": result.get("route"), "verification": result.get("verification"),
            "customer_reported_context": context, "resolution_summary": result.get("resolution_summary", {})})
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM tickets WHERE session_hash=? AND dedupe_key=? AND closed_at IS NULL", (session_hash, dedupe)).fetchone()
            if existing:
                return self.summary(existing)
            query_id = result.get("query_id")
            if query_id and not conn.execute("SELECT id FROM traces WHERE id=? AND session_hash=?", (query_id, session_hash)).fetchone():
                query_id = None
            conn.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (ticket_id, reference, session_hash,
                query_id, "OPEN", reason, encode(payload), None, None, dedupe, timestamp, timestamp, None))
            conn.execute("INSERT INTO ticket_events(ticket_id,actor,action,detail,created_at) VALUES(?,?,?,?,?)",
                         (ticket_id, "customer", "OPENED", encode({"reason": reason}), timestamp))
            self.store.audit("customer", "ticket.open", reference, {"query_id": query_id}, conn)
            if query_id:
                trace = conn.execute("SELECT body FROM traces WHERE id=?", (query_id,)).fetchone()
                body = json.loads(trace[0])
                body.update(needs_human_support=True, ticket={"reference": reference, "status": "OPEN"})
                conn.execute("UPDATE traces SET body=? WHERE id=?", (encode(body), query_id))
        return {"reference": reference, "status": "OPEN", "created_at": timestamp}

    def contact(self, reference, value, session):
        ticket = self._owned(reference, session)
        value = value.strip()
        if not value or len(value) > 254 or not (re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value)
                or re.fullmatch(r"\+?[\d ()-]{7,25}", value)):
            raise ValueError("Enter a valid email address or phone number")
        if not self.settings.ticket_encryption_key:
            raise RuntimeError("Contact collection is not configured")
        from cryptography.fernet import Fernet
        encrypted = Fernet(self.settings.ticket_encryption_key.encode()).encrypt(value.encode()).decode()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT status FROM tickets WHERE id=?", (ticket["id"],)).fetchone()
            if not current or current["status"] in TERMINAL:
                raise ValueError("Ticket is closed; open a new support request")
            conn.execute("UPDATE tickets SET contact_ciphertext=?,updated_at=? WHERE id=?", (encrypted, now(), ticket["id"]))
            self.store.audit("customer", "ticket.contact_updated", reference, conn=conn)

    def customer_view(self, reference, session):
        ticket = self._owned(reference, session)
        messages = self.store.rows("SELECT author,body,created_at FROM ticket_messages WHERE ticket_id=? ORDER BY id", (ticket["id"],))
        return {**self.summary(ticket), "messages": messages, "has_contact": bool(ticket["contact_ciphertext"])}

    def reply(self, reference, message, *, session=None, principal=None):
        if not isinstance(message, str) or not 0 < len(message.strip()) <= 4000:
            raise ValueError("Messages must contain 1–4000 characters")
        if principal is None:
            ticket = self._owned(reference, session)
            author = "CUSTOMER"
            rate_limit(self.store, "ticket-reply:" + session, 10)
        else:
            principal.require("ticket_manage")
            rows = self.store.rows("SELECT * FROM tickets WHERE reference=?", (reference,))
            if not rows:
                raise ValueError("Unknown ticket")
            ticket, author = rows[0], "SUPPORT"
        if ticket["status"] in TERMINAL:
            raise ValueError("Ticket is closed; open a new support request")
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT status FROM tickets WHERE id=?", (ticket["id"],)).fetchone()
            if not current or current["status"] in TERMINAL:
                raise ValueError("Ticket is closed; open a new support request")
            timestamp = now()
            conn.execute("INSERT INTO ticket_messages(ticket_id,author,body,created_at) VALUES(?,?,?,?)",
                         (ticket["id"], author, redact(message.strip()), timestamp))
            conn.execute("UPDATE tickets SET updated_at=? WHERE id=?", (timestamp, ticket["id"]))
            conn.execute("INSERT INTO ticket_events(ticket_id,actor,action,detail,created_at) VALUES(?,?,?,?,?)",
                         (ticket["id"], principal.name if principal else "customer", "REPLIED", encode({"author": author}), timestamp))
            self.store.audit(principal.name if principal else "customer", "ticket.reply", reference, conn=conn)

    def list(self, principal, include_closed=False):
        principal.require("tickets")
        sql = "SELECT reference,status,assignee,reason,created_at,updated_at,payload FROM tickets"
        if not include_closed:
            sql += " WHERE closed_at IS NULL"
        return self.store.rows(sql + " ORDER BY created_at DESC LIMIT 200")

    def detail(self, reference, principal):
        principal.require("tickets")
        rows = self.store.rows("SELECT * FROM tickets WHERE reference=?", (reference,))
        if not rows:
            raise ValueError("Unknown ticket")
        ticket = rows[0]
        result = {**self.summary(ticket), "assignee": ticket["assignee"], "reason": ticket["reason"],
                  "payload": json.loads(ticket["payload"]), "contact": None,
                  "events": self.store.rows("SELECT actor,action,detail,created_at FROM ticket_events WHERE ticket_id=? ORDER BY id", (ticket["id"],)),
                  "messages": self.store.rows("SELECT author,body,created_at FROM ticket_messages WHERE ticket_id=? ORDER BY id", (ticket["id"],))}
        if ticket["contact_ciphertext"]:
            principal.require("ticket_manage")
            from cryptography.fernet import Fernet
            result["contact"] = Fernet(self.settings.ticket_encryption_key.encode()).decrypt(ticket["contact_ciphertext"].encode()).decode()
            self.store.audit(principal.name, "ticket.contact_viewed", reference)
        return result

    def update(self, reference, status, assignee, principal):
        principal.require("ticket_manage")
        if status not in STATUSES or len(assignee or "") > 100:
            raise ValueError("Invalid ticket status or assignee")
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM tickets WHERE reference=?", (reference,)).fetchone()
            if not row:
                raise ValueError("Unknown ticket")
            if row["status"] in TERMINAL and status not in TERMINAL:
                raise ValueError("Closed tickets cannot be reopened; create a new request")
            closed_at = (row["closed_at"] or now()) if status in TERMINAL else None
            conn.execute("UPDATE tickets SET status=?,assignee=?,updated_at=?,closed_at=? WHERE id=?",
                         (status, assignee or None, now(), closed_at, row["id"]))
            conn.execute("INSERT INTO ticket_events(ticket_id,actor,action,detail,created_at) VALUES(?,?,?,?,?)",
                         (row["id"], principal.name, "UPDATED", encode({"status": status, "assignee": assignee or None}), now()))
            self.store.audit(principal.name, "ticket.update", reference, {"status": status, "assignee": assignee or None}, conn)

    def prune(self, principal):
        principal.require("ticket_manage")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.ticket_retention_days)).isoformat()
        with self.store.connect() as conn:
            conn.execute("DELETE FROM tickets WHERE closed_at IS NOT NULL AND closed_at<?", (cutoff,))
