from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from cryptography.fernet import Fernet

from src.config import Settings
from src.observability import Monitor
from src.security import Principal
from src.store import Store
from src.tickets import TicketService, requires_ticket


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = replace(Settings(), state_dir=Path(self.temp.name), requests_per_minute=100,
                                ticket_encryption_key=Fernet.generate_key().decode())
        self.store = Store(self.settings.state_dir / "control.sqlite3")
        self.tickets = TicketService(self.store, self.settings)
        self.staff = Principal("Sam", "support")

    def open(self, **kwargs):
        return self.tickets.open(kwargs.get("question", "Please investigate my missing refund"),
                                 kwargs.get("result", {}), kwargs.get("history", []),
                                 kwargs.get("session", "owner"))

    def test_dedupe_and_customer_ownership(self):
        ticket = self.open()
        self.assertEqual(self.open()["reference"], ticket["reference"])
        self.assertNotEqual(self.open(session="other")["reference"], ticket["reference"])
        for action in (lambda: self.tickets.customer_view(ticket["reference"], "other"),
                       lambda: self.tickets.reply(ticket["reference"], "Hi", session="other"),
                       lambda: self.tickets.contact(ticket["reference"], "a@example.com", "other")):
            with self.assertRaises(PermissionError):
                action()

    def test_support_reply_resolution_and_role_boundaries(self):
        ticket = self.open()
        reference = ticket["reference"]
        reviewer = Principal("reviewer", "reviewer")
        with self.assertRaises(PermissionError):
            self.tickets.detail(reference, reviewer)
        with self.assertRaises(PermissionError):
            self.tickets.update(reference, "CLOSED", "", reviewer)
        with self.assertRaises(PermissionError):
            self.staff.require("publish")
        self.tickets.update(reference, "ASSIGNED", "Sam", self.staff)
        self.tickets.reply(reference, "We are checking this with the payments team.", principal=self.staff)
        self.tickets.reply(reference, "Thank you. Contact me at a@example.com", session="owner")
        view = self.tickets.customer_view(reference, "owner")
        self.assertEqual([m["author"] for m in view["messages"]], ["SUPPORT", "CUSTOMER"])
        self.assertIn("[EMAIL]", view["messages"][1]["body"])
        self.assertNotIn("payload", view)
        self.assertEqual(len(self.tickets.detail(reference, self.staff)["events"]), 4)
        self.tickets.update(reference, "RESOLVED", "Sam", self.staff)
        self.assertEqual(self.tickets.list(self.staff), [])
        with self.assertRaises(ValueError):
            self.tickets.reply(reference, "More details", session="owner")
        with self.assertRaises(ValueError):
            self.tickets.update(reference, "OPEN", "Sam", self.staff)
        self.assertNotEqual(self.open()["reference"], reference)

    def test_pii_is_redacted_and_contact_is_encrypted(self):
        ticket = self.open(question="I need help, email a@example.com, order: ABC123",
                           history=[{"role": "user", "content": "password=topsecret"}])
        self.tickets.contact(ticket["reference"], "a@example.com", "owner")
        row = self.store.rows("SELECT * FROM tickets")[0]
        self.assertNotIn("a@example.com", row["payload"] + row["contact_ciphertext"])
        self.assertNotIn("ABC123", row["payload"])
        self.assertNotIn("topsecret", row["payload"])
        self.assertEqual(self.tickets.detail(ticket["reference"], self.staff)["contact"], "a@example.com")
        self.assertTrue(self.store.rows("SELECT id FROM audit_log WHERE action='ticket.contact_viewed'"))
        without_key = TicketService(self.store, replace(self.settings, ticket_encryption_key=""))
        with self.assertRaises(RuntimeError):
            without_key.contact(ticket["reference"], "a@example.com", "owner")

    def test_trace_link_ownership_and_retention(self):
        monitor = Monitor(self.store, self.settings)
        query_id = monitor.trace("owner", {"answer": "Cannot check your account."})
        ticket = self.open(result={"query_id": query_id})
        body = json.loads(self.store.rows("SELECT body FROM traces WHERE id=?", (query_id,))[0]["body"])
        self.assertEqual(body["ticket"]["reference"], ticket["reference"])
        foreign_id = monitor.trace("other", {})
        self.open(question="A second issue", result={"query_id": foreign_id})
        self.assertIsNone(self.store.rows("SELECT query_id FROM tickets WHERE reference<>?", (ticket["reference"],))[0]["query_id"])
        with self.store.connect() as conn:
            conn.execute("DELETE FROM traces WHERE id=?", (query_id,))
        self.assertEqual(self.tickets.customer_view(ticket["reference"], "owner")["status"], "OPEN")
        self.tickets.update(ticket["reference"], "CLOSED", "", self.staff)
        self.tickets.reply(self.tickets.list(self.staff)[0]["reference"], "Active ticket", session="owner")
        past = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        with self.store.connect() as conn:
            conn.execute("UPDATE tickets SET closed_at=? WHERE reference=?", (past, ticket["reference"]))
        self.tickets.prune(self.staff)
        self.assertEqual(len(self.tickets.list(self.staff, True)), 1)
        self.assertFalse(self.store.rows("SELECT * FROM ticket_events WHERE ticket_id NOT IN (SELECT id FROM tickets)"))

    def test_escalation_rules_avoid_greetings_and_policy_prohibitions(self):
        for result in ({"answer": "Hello", "route": "greeting"},
                       {"answer": "You cannot return a used product.", "sources": ["policy.pdf"]},
                       {"needs_human_support": True, "reason": "unsafe_request"},
                       {"needs_human_support": True, "reason": "out_of_scope"}):
            self.assertFalse(requires_ticket(result))
        for result in ({"needs_human_support": True}, {"reason": "verification_failed"},
                       {"route": "knowledge", "sources": []},
                       {"answer": "Mere paas iski verified policy information nahi hai."},
                       {"answer": "I don’t have enough information to help."}):
            self.assertTrue(requires_ticket(result))

    def test_explicit_handoff_bypasses_model_and_creates_one_ticket(self):
        from src.rag_memory import ask_shopassist
        with patch("src.rag_memory.Settings.from_env", return_value=self.settings), \
             patch("src.rag_memory.runtime", side_effect=AssertionError("No retrieval")), \
             patch("src.rag_memory.OllamaClient", side_effect=AssertionError("No model")):
            result = ask_shopassist("talk to a human", [], "owner")
            repeated = ask_shopassist("talk to a human", [], "owner")
        self.assertEqual(result["ticket"]["reference"], repeated["ticket"]["reference"])
        self.assertIn(result["ticket"]["reference"], result["answer"])
        self.assertFalse(result["retrieval_performed"])

    def test_unavailable_grounded_index_automatically_creates_ticket(self):
        from src.rag_memory import ask_shopassist
        from src.vector_index import IndexUnavailable
        with patch("src.rag_memory.Settings.from_env", return_value=self.settings), \
             patch("src.rag_memory.runtime", side_effect=IndexUnavailable("unavailable")):
            result = ask_shopassist("What is the return policy?", [], "owner")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["ticket"]["status"], "OPEN")

    def test_ticket_failure_does_not_claim_a_successful_handoff(self):
        from src.rag_memory import ask_shopassist
        with patch("src.rag_memory.Settings.from_env", return_value=self.settings), \
             patch("src.tickets.TicketService.open", side_effect=OSError("unavailable")):
            result = ask_shopassist("talk to a human", [], "owner")
        self.assertTrue(result["ticket_error"])
        self.assertNotIn("ticket", result)
        self.assertNotIn("SA-", result["answer"])

    def test_account_action_and_legacy_failure_raise_tickets(self):
        from src.rag_memory import ask_shopassist
        legacy = ModuleType("src.legacy_rag")
        legacy.ask_shopassist = Mock(side_effect=RuntimeError("unavailable"))
        with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline="legacy")), \
             patch("src.rag_memory.classify_query", return_value="ACCOUNT_ACTION"), \
             patch.dict("sys.modules", {"src.legacy_rag": legacy}):
            result = ask_shopassist("Please check my order status", [], "owner")
        self.assertFalse(result["retrieval_performed"])
        self.assertIn("ticket", result)
        legacy.ask_shopassist.assert_not_called()
        with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline="legacy")), \
             patch("src.rag_memory.classify_query", return_value="KNOWLEDGE"), \
             patch.dict("sys.modules", {"src.legacy_rag": legacy}):
            result = ask_shopassist("What is the return policy?", [], "owner")
        self.assertEqual(result["reason"], "service_unavailable")
        self.assertIn("ticket", result)

    def test_ticket_creation_rate_is_bounded_and_invalid_questions_rejected(self):
        for index in range(5):
            self.open(question=f"Issue {index}")
        with self.assertRaises(PermissionError):
            self.open(question="One more issue")
        self.open(question="Issue 0")  # Existing tickets do not consume the quota.
        for question in ("", " ", "x" * 2001, None):
            with self.assertRaises(ValueError):
                self.open(question=question)


if __name__ == "__main__":
    unittest.main()
