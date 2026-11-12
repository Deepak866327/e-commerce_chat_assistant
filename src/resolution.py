"""Persistent, per-issue support attempts before human handoff.

Customer-facing attempts are separate from model retries. Policy facts still
come from retrieval; the questions below collect customer-reported information.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
import secrets
from difflib import SequenceMatcher
from uuid import uuid4

from src.conversation import conversational_reply
from src.language import message
from src.intent_rules import explicit_handoff
from src.security import redact, redact_value, suspicious
from src.store import encode, now
from src.support_context import clean_context, remember_customer

PLANS = {
    "cancellation": ["dispatch_status", "blocker", "steps_attempted"],
    "refund": ["refund_approval", "approval_date", "steps_attempted"],
    "payment": ["payment_error", "charged_status", "steps_attempted"],
    "delivery": ["tracking_update", "expected_delivery", "steps_attempted"],
    "returns": ["item_problem", "delivery_date", "steps_attempted"],
    "warranty": ["product_problem", "purchase_date", "steps_attempted"],
    None: ["issue_details", "steps_attempted", "blocker"],
}
SEARCH = {"cancellation": "order cancellation eligibility and procedure",
          "refund": "refund approval processing timeline and delayed refund procedure",
          "payment": "payment failure and charged payment troubleshooting",
          "delivery": "delivery tracking and delayed shipment procedure",
          "returns": "product return eligibility and procedure",
          "warranty": "product warranty eligibility and claim procedure"}
NO_HELP = re.compile(r"\b(?:still|not working|did not help|didn't help|not resolved|cannot|can't|unable|"
                     r"no luck|nahi hua|nahi ho raha|nahi mila|nahi samajh)\b|नहीं हुआ|नहीं हो रहा|नहीं मिला", re.I)
YES = {"yes", "yes it has", "yes it is", "haan", "ha", "ji haan", "हाँ", "हां"}
NO = {"no", "no it hasn't", "no it has not", "no it is not", "not yet", "nahi", "nahin", "नहीं"}
NON_ATTEMPTS = {"greeting", "small_talk", "thanks", "farewell", "capabilities", "conversation", "out_of_scope", "blocked"}


def contextual_search(context):
    return context.get("search_question") if isinstance(context, dict) and context.get("force_knowledge") else None


class ResolutionService:
    def __init__(self, store, settings, session):
        if not isinstance(session, str) or not session:
            raise ValueError("A unique support session is required")
        self.store, self.settings = store, settings
        salt = store.get_or_create_setting("trace_salt", secrets.token_hex(32))
        self.owner = sha256((salt + session).encode()).hexdigest()

    def reset(self):
        with self.store.connect() as conn:
            conn.execute("DELETE FROM support_sessions WHERE session_hash=?", (self.owner,))

    def read_faq(self, article):
        """Remember a saved answer shown in chat without using a support attempt."""
        prepared = self.prepare(article["question"])
        with self.store.connect() as conn:
            row = conn.execute("SELECT body FROM support_sessions WHERE session_hash=? AND version=?",
                               (self.owner, prepared["_resolution_version"])).fetchone()
            if not row:
                raise ValueError("The conversation changed. Please select the question again.")
            state = json.loads(row["body"])
            state["context"]["faq"] = {**article, "outcome": "shown"}
            if not state["attempts"] and state["phase"] not in {"handed_off", "handoff_pending"}:
                state.update(phase="awaiting_outcome", pending_slot=None)
            updated = conn.execute("UPDATE support_sessions SET body=?,version=version+1,updated_at=? WHERE session_hash=? AND version=?",
                                   (encode(state), now(), self.owner, prepared["_resolution_version"]))
            if updated.rowcount != 1:
                raise ValueError("The conversation changed. Please select the question again.")
        return clean_context(state["context"])

    @staticmethod
    def fresh(context):
        return {"issue_id": "ISSUE_" + uuid4().hex, "context": context, "attempts": 0, "asked": [], "guidance": [],
                "failures": [], "phase": "new", "pending_slot": None, "handoff_declined": False}

    def prepare(self, question, supplied=None):
        normalized = " ".join(question.casefold().replace("’", "'").split()).rstrip(".!?।？！")
        conversation = conversational_reply(question)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM support_sessions WHERE updated_at<?", (cutoff,))
            row = conn.execute("SELECT * FROM support_sessions WHERE session_hash=?", (self.owner,)).fetchone()
            state = json.loads(row["body"]) if row else self.fresh(clean_context({}))
            previous = state["context"]
            context = remember_customer(previous, question)
            if context["steps_tried"]:
                context["reported_facts"]["steps_attempted"] = context["steps_tried"][-1]
            # Report-memory resets signal an explicit new issue, even with the same topic.
            reset = (context["issue_type"] != previous["issue_type"] or
                     previous["customer_reported"] and previous["customer_reported"][0] not in context["customer_reported"])
            if reset:
                state = self.fresh(context)
            else:
                state["context"] = context
            # Only accept an ID/revision resolved against the current server catalog.
            # Answers supplied by clients are never adopted as policy evidence.
            from src.customer_faq import CustomerFAQs
            faq_reference = supplied.get("faq_reference") if isinstance(supplied, dict) else None
            faq = CustomerFAQs(self.store, self.settings).context(faq_reference) if faq_reference else None
            previous_faq = context.get("faq") or {}
            faq_followup = bool(faq and ((previous_faq.get("faq_id"), previous_faq.get("version"), previous_faq.get("outcome"))
                                        != (faq["faq_id"], faq["version"], "not_helpful")))
            if faq:
                context["faq"] = faq
            if context["ticket_reference"]:
                ticket = conn.execute("SELECT status FROM tickets WHERE reference=? AND session_hash=?", (context["ticket_reference"], self.owner)).fetchone()
                if not ticket or ticket["status"] in {"RESOLVED", "CLOSED"}:
                    context["ticket_reference"] = None
                    state = self.fresh(context)
            # Adopt an already-created ticket only after server-side ownership validation.
            reference = clean_context(supplied).get("ticket_reference")
            if reference and not row:
                owned = conn.execute("SELECT status FROM tickets WHERE reference=? AND session_hash=?", (reference, self.owner)).fetchone()
                if owned and owned["status"] not in {"RESOLVED", "CLOSED"}:
                    context["ticket_reference"] = reference
                    state["phase"] = "handed_off"
            force_knowledge = False
            explicit_resolution = normalized in {"it worked", "issue resolved", "it is resolved", "problem solved",
                "ho gaya", "problem solve ho gayi", "समस्या हल हो गई", "ठीक हो गया"}
            explicit_resolution = explicit_resolution or bool(re.fullmatch(
                r"(?:yes|thanks|thank you)[, ]+(?:it worked|issue resolved|problem solved)", normalized))
            consent = "approved_handoff" if state["phase"] == "awaiting_handoff_consent" and normalized in YES else \
                      "declined_handoff" if state["phase"] == "awaiting_handoff_consent" and normalized in NO else None
            human_mention = bool(re.search(r"\b(?:human|agent|support team)\b|एजेंट|मानव", normalized))
            if not conversation and not suspicious(question) and context["issue_type"] is None:
                report = redact(question)[:300]
                if report not in context["customer_reported"]:
                    context["customer_reported"] = (context["customer_reported"][:1] + context["customer_reported"][-4:] + [report])[-6:]
            if "?" not in question and not re.search(r"\b(?:if|what if|suppose)\b", normalized):
                if re.search(r"(?:not|hasn't) (?:yet )?(?:been )?(?:dispatched|shipped)", normalized):
                    context["reported_facts"]["dispatch_status"] = "not dispatched; customer said: " + redact(question)[:250]
                elif re.search(r"(?:has been|already) (?:dispatched|shipped)", normalized):
                    context["reported_facts"]["dispatch_status"] = "dispatched; customer said: " + redact(question)[:250]
                if re.search(r"refund.*(?:not|hasn't been|isn't) approved", normalized):
                    context["reported_facts"]["refund_approval"] = "not approved; customer said: " + redact(question)[:250]
                elif re.search(r"refund.*(?:was|is|has been) approved", normalized):
                    context["reported_facts"]["refund_approval"] = "approved; customer said: " + redact(question)[:250]
                    if re.search(r"\d+\s*(?:days|weeks)|\d{4}-\d{2}-\d{2}", normalized):
                        context["reported_facts"]["approval_date"] = redact(question)[:300]
                if context["issue_type"] == "payment":
                    if re.search(r"error|declined|insufficient funds", normalized):
                        context["reported_facts"]["payment_error"] = redact(question)[:300]
                    if re.search(r"debited|charged|paise kate|पैसे कट", normalized):
                        context["reported_facts"]["charged_status"] = redact(question)[:300]
            if not conversation and not suspicious(question) and not human_mention:
                slot = state["pending_slot"]
                if state["phase"] == "awaiting_detail" and slot:
                    # 'No' to dispatch/approval is data, not failed resolution.
                    value = None
                    if slot == "dispatch_status":
                        if normalized in NO or re.search(r"not (?:yet )?(?:dispatched|shipped)|dispatch nahi", normalized):
                            value = "not dispatched; customer said: " + question
                        elif normalized in YES or re.search(r"(?:has been|already) (?:dispatched|shipped)", normalized):
                            value = "dispatched; customer said: " + question
                    elif slot == "refund_approval":
                        if normalized in NO or re.search(r"not approved|approve nahi", normalized):
                            value = "not approved; customer said: " + question
                        elif normalized in YES or re.search(r"approved|approval confirmation", normalized):
                            value = "approved; customer said: " + question
                    elif normalized not in NO and not NO_HELP.search(normalized):
                        value = question
                    if value:
                        context["reported_facts"][slot] = redact(value)[:300]
                    force_knowledge = True
                elif state["phase"] == "awaiting_detail":
                    force_knowledge = True
                elif state["phase"] == "awaiting_outcome" and normalized in YES:
                    explicit_resolution = True
                elif (state["attempts"] or context.get("faq")) and (normalized in NO or NO_HELP.search(normalized)):
                    force_knowledge = True
            if re.search(r"\b(?:python|javascript|programming|capital of|politics|sports)\b", normalized):
                force_knowledge = False
            if explicit_resolution:
                state.update(phase="resolved", pending_slot=None)
                context["resolution"] = "customer_reported_resolved"
            # A negative answer to a requested detail is never a failed-outcome signal.
            detail_is_boolean = state["phase"] == "awaiting_detail" and state["pending_slot"] in {"dispatch_status", "refund_approval"}
            unresolved = not detail_is_boolean and (normalized in NO or bool(NO_HELP.search(normalized)))
            if unresolved and context.get("faq") and context["faq"].get("outcome") == "shown":
                context["faq"]["outcome"] = "not_helpful"
            threshold = bool(unresolved and state["attempts"] >= self.settings.support_max_attempts
                             and not conversation and not suspicious(question) and not human_mention)
            version = (row["version"] if row else 0) + 1
            conn.execute("INSERT INTO support_sessions VALUES(?,?,?,?) ON CONFLICT(session_hash) DO UPDATE "
                         "SET body=excluded.body,version=excluded.version,updated_at=excluded.updated_at",
                         (self.owner, encode(state), version, now()))
        query = SEARCH.get(context["issue_type"], "")
        if force_knowledge and not query:
            query = "approved customer support guidance for this reported issue: " + (context["customer_reported"][0] if context["customer_reported"] else question)
        if context["reported_facts"].get("refund_approval", "").startswith("not approved"):
            query = "refund request eligibility and initiation procedure"
        dispatch = context["reported_facts"].get("dispatch_status", "")
        if dispatch:
            query += " before dispatch" if dispatch.startswith("not dispatched") else " after dispatch"
        requested = explicit_handoff(question) or bool(conversation and conversation.get("reason") == "explicit_handoff")
        return {**context, "_resolution_version": version, "_resolution_direct": "requested_handoff" if requested else consent or ("resolved" if explicit_resolution else
                "exhausted" if threshold else "faq_followup" if faq_followup else None), "force_knowledge": force_knowledge,
                "search_question": "What is the " + query + "?" if query else ""}

    def finish(self, result, prepared, language):
        from src.tickets import requires_ticket
        result = dict(result)
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM support_sessions WHERE session_hash=?", (self.owner,)).fetchone()
            if not row or row["version"] != prepared.get("_resolution_version"):
                # An older concurrent reply must not overwrite a newer turn or open a ticket.
                return {**result, "needs_human_support": False, "stale_turn": True}, None
            state = json.loads(row["body"])
            context = state["context"]
            if prepared.get("_faq_shown"):
                context["faq"] = {**prepared["_faq_shown"], "outcome": "shown"}
            reason = result.get("reason")
            explicit = reason in {"explicit_handoff", "user_requested"} or result.get("route") == "human_support"
            direct = prepared.get("_resolution_direct")
            explicit = explicit or direct in {"approved_handoff", "requested_handoff"}
            handoff = False
            if direct == "resolved":
                result = self.plain(result, message("resolution_done", language), "resolved", None)
            elif reason == "handoff_declined" or direct == "declined_handoff":
                state.update(handoff_declined=True, phase="awaiting_detail", pending_slot=None)
                result = self.plain(result, message("declined_handoff", language), "clarification", "handoff_declined")
            elif explicit or direct == "exhausted":
                handoff = True
            elif result.get("route") in NON_ATTEMPTS or reason in {"out_of_scope", "unsafe_request", "rate_limited"}:
                result["needs_human_support"] = False
                if reason == "rate_limited":
                    result = self.plain(result, message("resolution_rate_limited", language), "rate_limited", reason)
            elif state["phase"] == "handed_off" and not requires_ticket(result):
                pass
            else:
                needs_help = requires_ticket(result) or direct == "faq_followup"
                fingerprint = sha256(" ".join(result.get("answer", "").casefold().split()).encode()).hexdigest()
                repeated = fingerprint in [item["fingerprint"] for item in state["guidance"]]
                faq = context.get("faq")
                if faq and faq.get("outcome") == "not_helpful":
                    repeated = repeated or SequenceMatcher(None, " ".join(result.get("answer", "").casefold().split()),
                                                           " ".join(faq["answer"].casefold().split())).ratio() >= .8
                result["repeated_guidance"] = repeated
                if needs_help:
                    result["failure_reason"] = reason or "unanswered"
                    state["failures"] = (state.get("failures", []) + [result["failure_reason"]])[-3:]
                if (needs_help or repeated) and (state["attempts"] >= self.settings.support_max_attempts or state["phase"] == "handed_off"):
                    handoff = True
                else:
                    if needs_help or repeated:
                        slots = list(dict.fromkeys(PLANS.get(context["issue_type"], PLANS[None]) + ["blocker", "issue_details", "steps_attempted"]))
                        if not context["reported_facts"].get("refund_approval", "").startswith("approved"):
                            slots = [slot for slot in slots if slot != "approval_date"]
                        slot = next((slot for slot in slots if slot not in state["asked"] and slot not in context["reported_facts"]), None)
                        if slot is None:
                            handoff = True
                        else:
                            state["asked"].append(slot)
                            state.update(phase="awaiting_detail", pending_slot=slot)
                            result = self.plain(result, message("followup_" + slot, language), "support_followup", "clarification_needed")
                            state["attempts"] = min(self.settings.support_max_attempts, state["attempts"] + 1)
                    else:
                        state["guidance"] = (state["guidance"] + [{"fingerprint": fingerprint, "answer": redact(result["answer"])[:500],
                            "sources": result.get("citations", result.get("sources", []))}])[-3:]
                        state["attempts"] = min(self.settings.support_max_attempts, state["attempts"] + 1)
                        if "?" in result["answer"] or "？" in result["answer"]:
                            slot = "dispatch_status" if re.search(r"dispatch|shipped", result["answer"], re.I) and "dispatch_status" not in context["reported_facts"] and "dispatch_status" not in state["asked"] else None
                            state.update(phase="awaiting_detail", pending_slot=slot)
                            if slot and slot not in state["asked"]:
                                state["asked"].append(slot)
                        else:
                            # Dispatch changes the applicable policy; ask this before an outcome check.
                            slot = "dispatch_status" if context["issue_type"] == "cancellation" and "dispatch_status" not in context["reported_facts"] and "dispatch_status" not in state["asked"] else None
                            result["answer"] += "\n\n" + message("followup_" + slot if slot else "resolution_check", language)
                            state.update(phase="awaiting_detail" if slot else "awaiting_outcome", pending_slot=slot)
                            if slot:
                                state["asked"].append(slot)
                            result["followup_added"] = True
                        result["needs_human_support"] = False
            if handoff:
                if state.get("handoff_declined") and not explicit:
                    state.update(phase="awaiting_handoff_consent", pending_slot=None)
                    result = self.plain(result, message("resolution_handoff_offer", language), "handoff_offer", "handoff_offered")
                else:
                    state.update(phase="handoff_pending", pending_slot=None)
                    reason = "explicit_handoff" if explicit else "existing_handoff" if context["ticket_reference"] else "resolution_attempts_exhausted"
                    result = self.plain(result, message("resolution_handoff_pending", language), "human_support" if explicit else "handoff", reason)
                    result["needs_human_support"] = True
            context["assistant_attempts"] = [item["answer"] for item in state["guidance"]]
            version = row["version"] + 1
            conn.execute("UPDATE support_sessions SET body=?,version=?,updated_at=? WHERE session_hash=? AND version=?",
                         (encode(state), version, now(), self.owner, row["version"]))
        result["support_context"] = clean_context(context)
        result["resolution"] = {"phase": state["phase"], "attempts": state["attempts"],
                                "max_attempts": self.settings.support_max_attempts, "asked": state["asked"], "issue_id": state["issue_id"]}
        result["resolution_summary"] = {"customer_reported": context["customer_reported"], "facts": context["reported_facts"],
                                        "attempts": state["attempts"], "asked": state["asked"], "guidance": state["guidance"],
                                        "failures": state.get("failures", []), "faq": context.get("faq")}
        return result, version

    @staticmethod
    def plain(original, answer, route, reason):
        return {**original, "answer": answer, "answerable": False, "needs_human_support": False, "sources": [],
                "citations": [], "route": route, "reason": reason,
                "verification": {"passed": None, "reason": "non_factual_support_flow"}, "confidence": None}

    def attach_ticket(self, result, version):
        with self.store.connect() as conn:
            row = conn.execute("SELECT body FROM support_sessions WHERE session_hash=? AND version=?", (self.owner, version)).fetchone()
            if row:
                state = json.loads(row["body"])
                state["context"]["ticket_reference"] = result["ticket"]["reference"]
                state["phase"] = "handed_off"
                conn.execute("UPDATE support_sessions SET body=?,updated_at=? WHERE session_hash=? AND version=?",
                             (encode(state), now(), self.owner, version))
                result["support_context"] = clean_context(state["context"])
                result["resolution"]["phase"] = "handed_off"

    def update_trace(self, result):
        if not result.get("query_id"):
            return
        with self.store.connect() as conn:
            row = conn.execute("SELECT body FROM traces WHERE id=? AND session_hash=?", (result["query_id"], self.owner)).fetchone()
            if row:
                body = json.loads(row["body"])
                body.update({key: result[key] for key in ("answer", "answerable", "sources", "needs_human_support", "reason",
                    "route", "resolution", "ticket", "ticket_error", "verification", "resolution_summary", "repeated_guidance") if key in result})
                conn.execute("UPDATE traces SET body=? WHERE id=? AND session_hash=?",
                             (encode(redact_value(body)), result["query_id"], self.owner))
