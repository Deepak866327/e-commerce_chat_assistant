"""Versioned customer FAQ access, feedback and untrusted reading history."""

from hashlib import sha256

from src.default_help import starter_faqs
from src.faq_catalog import customer_faqs
from src.observability import Monitor
from src.store import encode, now


class CustomerFAQs:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    def articles(self):
        return customer_faqs(self.store) or starter_faqs()

    def resolve(self, reference):
        if not isinstance(reference, dict):
            return None
        return next((a for a in self.articles() if a["faq_id"] == reference.get("faq_id")
                     and a["version"] == reference.get("version")), None)

    def context(self, reference):
        article = self.resolve(reference)
        if not article:
            return None
        # Browsed content is memory of what was shown, never policy evidence.
        return {**article, "outcome": "not_helpful"}

    def feedback(self, reference, outcome, session):
        if outcome not in {"helpful", "not_helpful"} or not isinstance(session, str) or not session:
            raise ValueError("Invalid FAQ feedback")
        from src.security import rate_limit
        rate_limit(self.store, "faq-feedback:" + session, self.settings.requests_per_minute)
        article = self.resolve(reference)
        if not article:
            raise ValueError("This article has changed. Please reopen it and try again.")
        monitor = Monitor(self.store, self.settings)
        owner = sha256((monitor.salt + session).encode()).hexdigest()
        identity = sha256((owner + article["faq_id"] + article["version"]).encode()).hexdigest()
        query_id, feedback_id = "FAQ_" + identity, "FB_" + identity
        # One current vote per session/article revision, retained by normal trace pruning.
        with self.store.connect() as conn:
            conn.execute("INSERT INTO traces VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET created_at=excluded.created_at",
                         (query_id, owner, encode({"event_type": "faq_feedback", "faq_id": article["faq_id"],
                          "faq_version": article["version"], "kind": article["kind"]}), now()))
            conn.execute("INSERT INTO feedback VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,created_at=excluded.created_at",
                         (feedback_id, query_id, outcome, "", now()))
        monitor.prune()
        return article

    def report(self, principal):
        principal.require("review")
        # Retained votes and support outcomes grouped by article revision.
        rows = self.store.rows("SELECT t.body,f.kind FROM feedback f JOIN traces t ON t.id=f.trace_id")
        import json
        grouped = {}
        for row in rows:
            trace = json.loads(row["body"])
            if trace.get("event_type") != "faq_feedback":
                continue
            key = (trace["faq_id"], trace["faq_version"])
            record = grouped.setdefault(key, {"faq_id": key[0], "version": key[1], "helpful": 0, "not_helpful": 0})
            if row["kind"] in {"helpful", "not_helpful"}:
                record[row["kind"]] += 1
        issues = {}
        for row in self.store.rows("SELECT session_hash,body FROM traces ORDER BY created_at,id"):
            trace = json.loads(row["body"])
            resolution = trace.get("resolution", {})
            faq = trace.get("resolution_summary", {}).get("faq")
            if not faq or not resolution.get("issue_id"):
                continue
            key = (faq["faq_id"], faq["version"])
            grouped.setdefault(key, {"faq_id": key[0], "version": key[1], "helpful": 0, "not_helpful": 0})
            issue_key = (row["session_hash"], resolution["issue_id"], key)
            previous = issues.get(issue_key, {})
            issues[issue_key] = {"phase": resolution["phase"], "repeated": previous.get("repeated", False) or trace.get("repeated_guidance", False)}
        for key, record in grouped.items():
            outcomes = [value for (_, _, faq_key), value in issues.items() if faq_key == key]
            record["confirmed_resolutions"] = sum(value["phase"] == "resolved" for value in outcomes)
            record["escalated_issues"] = sum(value["phase"] == "handed_off" for value in outcomes)
            record["issues_with_repeated_guidance"] = sum(value["repeated"] for value in outcomes)
        return list(grouped.values())
