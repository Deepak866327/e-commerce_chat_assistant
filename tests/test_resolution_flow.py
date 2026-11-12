from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.config import Settings
from src.observability import Monitor
from src.rag_memory import ask_shopassist, ask_structured, request_human_support, reset_support_resolution, _escalate
from src.resolution import ResolutionService
from src.schemas import QueryIntent
from src.security import Principal
from src.store import Store
from src.tickets import TicketService


def unavailable(reason="insufficient_evidence", route="knowledge"):
    return {"answer": "Verified information is unavailable.", "answerable": False, "sources": [],
            "needs_human_support": True, "reason": reason, "route": route,
            "retrieval_performed": route == "knowledge", "query_id": None,
            "verification": {"passed": False}, "retrieved_chunks": []}


def guidance(text="The approved policy permits cancellation before dispatch."):
    return {"answer": text, "answerable": True, "sources": [{"document_id": "doc", "chunk_id": "chunk", "page_number": 1}],
            "needs_human_support": False, "reason": None, "route": "knowledge", "query_id": None,
            "retrieval_performed": True, "verification": {"passed": True},
            "retrieved_chunks": [{"id": "chunk", "metadata": {"document_id": "doc", "source": "policy.pdf"}}]}


class ResolutionFlowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = replace(Settings(), state_dir=Path(temporary.name), requests_per_minute=100)
        self.store = Store(self.settings.state_dir / "control.sqlite3")
        self.model = Mock(return_value=unavailable())
        self.settings_patch = patch("src.rag_memory.Settings.from_env", side_effect=lambda: self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        runtime_patch = patch("src.rag_memory.runtime", return_value=SimpleNamespace(ask=self.model))
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def ask(self, question, session="customer", **kwargs):
        return ask_shopassist(question, [], session, response_language="en", **kwargs)

    def test_three_attempts_then_one_ticket_with_full_issue_context(self):
        replies = [self.ask(question) for question in
                   ("Cancel my order", "No", "The cancellation step is failing", "Still not working")]
        self.assertEqual([r["resolution"]["attempts"] for r in replies], [1, 2, 3, 3])
        self.assertTrue(all(not r["needs_human_support"] and "ticket" not in r for r in replies[:3]))
        self.assertTrue(replies[3]["needs_human_support"])
        self.assertEqual(replies[3]["reason"], "resolution_attempts_exhausted")
        self.assertEqual(self.model.call_count, 3)  # No fourth model call just to decide handoff.
        self.assertEqual(len(self.store.rows("SELECT * FROM tickets")), 1)
        ticket = self.store.rows("SELECT * FROM tickets")[0]
        payload = json.loads(ticket["payload"])
        self.assertIn("Cancel my order", payload["question"])
        self.assertIn("not dispatched", payload["resolution_summary"]["facts"]["dispatch_status"])
        self.assertEqual(payload["resolution_summary"]["attempts"], 3)
        self.assertEqual(len(set(replies[3]["resolution"]["asked"])), 3)
        self.assertEqual(len(payload["resolution_summary"]["failures"]), 3)

    def test_detail_no_forces_policy_retrieval_and_can_resolve_without_handoff(self):
        self.model.side_effect = [unavailable(route="account_action"), guidance("Use the approved cancellation procedure for an order that has not been dispatched.")]
        first = self.ask("Cancel my order")
        self.assertIn("dispatched", first["answer"])
        second = self.ask("No")
        context = self.model.call_args.kwargs["support_context"]
        self.assertTrue(context["force_knowledge"])
        self.assertIn("before dispatch", context["search_question"])
        self.assertIn("not dispatched", context["reported_facts"]["dispatch_status"])
        self.assertEqual(second["resolution"]["phase"], "awaiting_outcome")
        resolved = self.ask("Yes")
        self.assertEqual(resolved["resolution"]["phase"], "resolved")
        self.assertFalse(resolved["needs_human_support"])
        self.assertEqual(self.model.call_count, 2)
        self.assertEqual(self.store.rows("SELECT * FROM tickets"), [])

    def test_explicit_human_request_bypasses_remaining_attempts(self):
        self.ask("My refund is missing")
        result = self.ask("talk to a human")
        self.assertIn("ticket", result)
        self.assertEqual(result["resolution"]["attempts"], 1)
        self.assertEqual(self.model.call_count, 1)

    def test_explicit_handoff_in_a_longer_request_does_not_need_model_approval(self):
        from src.intent_rules import explicit_handoff
        self.assertFalse(explicit_handoff("What does 'talk to a human' mean?"))
        self.assertFalse(explicit_handoff("How does the create a ticket button work?"))
        for index, question in enumerate(("Please connect me to a human about my refund",
                                          "Please escalate this to a human", "I need a human to help with my refund")):
            with self.subTest(question=question):
                result = self.ask(question, "explicit-" + str(index))
                self.assertIn("ticket", result)
                self.assertEqual(result["resolution"]["attempts"], 0)
        self.model.assert_not_called()

    def test_grounded_engine_uses_contextual_query_for_a_short_detail_reply(self):
        from tests.test_chat_retrieval import RetrievalChatTests, Model, Verdict
        from src.chat_service import ChatService
        fixture = RetrievalChatTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.ask("Cancel my order")
        model = Model(fixture.chunk)
        original = model.structured
        def generate(name, messages, schema, **kwargs):
            self.assertIsNot(schema, QueryIntent)  # 'No' must not be misclassified again.
            return original(name, messages, schema, **kwargs)
        model.structured = Mock(side_effect=generate)
        fixture.retriever.search = Mock(wraps=fixture.retriever.search)
        service = ChatService(fixture.store, fixture.settings, model, fixture.retriever, Verdict())
        with patch("src.rag_memory.runtime", return_value=service):
            result = self.ask("No")
        self.assertNotIn("ticket", result)
        self.assertEqual(fixture.retriever.search.call_count, 1)
        self.assertIn("before dispatch", fixture.retriever.search.call_args.args[0])

    def test_legacy_engine_receives_contextual_search_after_followup(self):
        from types import ModuleType
        self.ask("Cancel my order")
        legacy = ModuleType("src.legacy_rag")
        legacy.ask_shopassist = Mock(return_value={"answer": "Approved policy guidance before dispatch.", "sources": ["policy.pdf, page 1"]})
        messages = ModuleType("langchain_core.messages")
        messages.AIMessage = messages.HumanMessage = lambda content: SimpleNamespace(content=content)
        self.settings = replace(self.settings, pipeline="legacy")
        client = Mock()
        with patch("src.rag_memory.OllamaClient", return_value=client), \
             patch.dict("sys.modules", {"src.legacy_rag": legacy, "langchain_core.messages": messages}):
            result = self.ask("No")
        self.assertNotIn("ticket", result)
        client.structured.assert_not_called()
        self.assertIn("before dispatch", legacy.ask_shopassist.call_args.kwargs["search_question"])

    def test_manual_handoff_form_bypasses_attempt_budget(self):
        ticket = request_human_support("Please investigate my refund", [], "manual")
        self.assertTrue(ticket["reference"].startswith("SA-"))
        self.model.assert_not_called()

    def test_limit_two_is_configurable(self):
        self.settings = replace(self.settings, support_max_attempts=2)
        replies = [self.ask(q) for q in ("My payment failed", "No", "Still not working")]
        self.assertTrue(all("ticket" not in r for r in replies[:2]))
        self.assertEqual(replies[-1]["resolution"]["attempts"], 2)
        self.assertIn("ticket", replies[-1])
        with self.assertRaises(ValueError):
            replace(self.settings, support_max_attempts=0)

    def test_greeting_and_thanks_do_not_consume_attempts_or_claim_resolution(self):
        first = self.ask("My refund is missing")
        self.assertEqual(first["resolution"]["attempts"], 1)
        for question in ("hi", "thanks", "who are you"):
            result = self.ask(question)
            self.assertEqual(result["resolution"]["attempts"], 1)
            self.assertNotEqual(result["resolution"]["phase"], "resolved")
            self.assertNotIn("ticket", result)

    def test_new_issue_and_clear_conversation_reset_the_budget(self):
        first = self.ask("My refund is missing")
        second = self.ask("My payment failed on a new purchase")
        self.assertEqual(second["resolution"]["attempts"], 1)
        self.assertNotEqual(first["resolution"]["issue_id"], second["resolution"]["issue_id"])
        reset_support_resolution("customer")
        third = self.ask("My payment failed")
        self.assertEqual(third["resolution"]["attempts"], 1)
        self.assertNotEqual(second["resolution"]["issue_id"], third["resolution"]["issue_id"])

    def test_sessions_are_isolated_and_client_cannot_reset_attempts(self):
        first = self.ask("My refund is missing", "one")
        second = self.ask("My refund is missing", "two")
        self.assertNotEqual(first["resolution"]["issue_id"], second["resolution"]["issue_id"])
        continued = self.ask("No", "one", support_context={"attempts": 0, "resolution_attempts": 999})
        self.assertEqual(continued["resolution"]["attempts"], 2)
        self.assertNotIn("ticket", continued)

    def test_known_customer_details_are_not_asked_again(self):
        result = self.ask("My refund was approved 20 days ago and has not arrived")
        self.assertIn("already tried", result["answer"])
        self.assertNotIn("confirmation", result["answer"])
        self.assertIn("approval_date", result["support_context"]["reported_facts"])
        other = self.ask("My payment failed with error declined", "payment")
        self.assertIn("charged", other["answer"])
        tried = self.ask("My refund was approved 20 days ago and I already checked my bank", "already-tried")
        self.assertNotIn("already tried", tried["answer"])
        self.assertIn("steps_attempted", tried["support_context"]["reported_facts"])

    def test_repeated_guidance_changes_to_a_new_question(self):
        self.model.return_value = guidance("The approved refund policy applies after inspection.")
        first = self.ask("What is the refund policy?")
        second = self.ask("No")
        self.assertNotEqual(first["answer"], second["answer"])
        self.assertNotIn("approved refund policy", second["answer"])
        self.assertEqual(second["resolution"]["phase"], "awaiting_detail")

    def test_resolution_only_follows_customer_confirmation(self):
        self.model.return_value = guidance()
        self.ask("How can I cancel my order?")
        self.ask("No")  # This answers dispatch, not resolution.
        result = self.ask("Thanks, it worked")
        self.assertEqual(result["resolution"]["phase"], "resolved")
        self.assertEqual(self.store.rows("SELECT * FROM tickets"), [])

    def test_stale_concurrent_reply_cannot_consume_attempt_or_create_ticket(self):
        flow = ResolutionService(self.store, self.settings, "customer")
        old = flow.prepare("My refund is missing")
        new = flow.prepare("My refund is missing")
        result = _escalate("talk to a human", [], "customer", {**unavailable(), "reason": "explicit_handoff"}, support_context=old)
        self.assertTrue(result["stale_turn"])
        self.assertNotIn("ticket", result)
        current, _ = flow.finish(unavailable(), new, "en")
        self.assertEqual(current["resolution"]["attempts"], 1)

    def test_state_expires_and_does_not_reuse_an_old_budget(self):
        self.ask("My refund is missing")
        expired = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        with self.store.connect() as conn:
            conn.execute("UPDATE support_sessions SET updated_at=?", (expired,))
        self.assertEqual(self.ask("My refund is missing")["resolution"]["attempts"], 1)

    def test_deferral_updates_the_owned_trace_and_redacts_private_details(self):
        trace = Monitor(self.store, self.settings).trace("customer", {"needs_human_support": True})
        self.model.return_value = {**unavailable(), "query_id": trace}
        result = self.ask("My refund is missing, email person@example.com password=secret")
        body = json.loads(self.store.rows("SELECT body FROM traces WHERE id=?", (trace,))[0]["body"])
        self.assertFalse(body["needs_human_support"])
        self.assertEqual(body["resolution"]["attempts"], 1)
        rows = self.store.rows("SELECT body FROM support_sessions")
        self.assertNotIn("person@example.com", str(rows))
        self.assertNotIn("password=secret", str(rows))

    def test_blocked_and_rate_limited_messages_never_open_tickets_or_consume_attempts(self):
        for reason in ("unsafe_request", "rate_limited", "out_of_scope"):
            with self.subTest(reason=reason):
                self.model.return_value = unavailable(reason=reason, route="blocked")
                result = self.ask("A customer message", reason)
                self.assertFalse(result["needs_human_support"])
                self.assertEqual(result["resolution"]["attempts"], 0)
                self.assertNotIn("ticket", result)
        with patch("src.security.rate_limit", side_effect=PermissionError("limited")):
            limited = self.ask("hi", "limited")
        self.assertEqual(limited["reason"], "rate_limited")
        self.assertFalse(limited["needs_human_support"])
        self.assertEqual(limited["resolution"]["attempts"], 0)

    def test_same_generic_failure_on_different_issues_creates_separate_tickets(self):
        for issue in ("My refund is missing", "My payment failed"):
            for question in (issue, "No", "No", "No"):
                self.ask(question)
        tickets = self.store.rows("SELECT * FROM tickets")
        self.assertEqual(len(tickets), 2)
        self.assertNotEqual(tickets[0]["dedupe_key"], tickets[1]["dedupe_key"])

    def test_both_public_entrypoints_apply_the_same_retry_policy(self):
        a = self.ask("My refund is missing", "normal")
        b = ask_structured("My refund is missing", [], "structured", response_language="en")
        self.assertEqual(a["resolution"]["attempts"], b["resolution"]["attempts"])
        self.assertNotIn("ticket", b)

    def test_declined_handoff_is_offered_again_after_limit_not_forced(self):
        self.ask("My refund is missing")
        self.ask("I do not want to talk to a human")
        self.ask("Still not working")
        self.ask("Still not working")
        offer = self.ask("Still not working")
        self.assertEqual(offer["resolution"]["phase"], "awaiting_handoff_consent")
        self.assertNotIn("ticket", offer)
        accepted = self.ask("Yes")
        self.assertIn("ticket", accepted)
