"""Support relevance and agent handoffs using only customer-reported records."""

import json
import unittest

from src.knowledge_quality import knowledge_gaps
from src.observability import Monitor
from src.security import Principal
from src.support_diagnostics import reported_details, manual_action_requested, claims_backend_action
from tests import test_resolution_flow as flow_fixture


class NoCRMSupportTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = flow_fixture.ResolutionFlowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.ask, self.model, self.store = fixture.ask, fixture.model, fixture.store

    def test_missing_cancel_button_collects_only_missing_details_then_searches_the_blocker(self):
        first = self.ask("My cancellation option is missing")
        self.assertEqual(first["resolution"]["asked"], ["dispatch_status"])
        self.assertEqual(first["resolution_summary"]["failures"], [])
        self.model.assert_not_called()
        second = self.ask("No")
        self.assertIn("website", second["answer"])
        self.assertEqual(second["resolution"]["asked"], ["dispatch_status", "channel"])
        self.model.assert_not_called()
        self.model.return_value = flow_fixture.guidance("The approved procedure applies to the reported missing cancellation option.")
        self.ask("Website")
        context = self.model.call_args.kwargs["support_context"]
        self.assertIn("before dispatch", context["search_question"])
        self.assertIn("missing", context["search_question"])
        self.assertIn("website", context["search_question"])
        self.assertTrue(context["force_knowledge"])

    def test_payment_confirmation_and_return_reason_are_not_asked_again(self):
        result = self.ask("My payment was deducted but I have no order confirmation")
        facts = result["support_context"]["reported_facts"]
        self.assertTrue(facts["charged_status"].startswith("charged"))
        self.assertTrue(facts["order_confirmation"].startswith("not received"))
        self.assertIn("already tried", result["answer"])
        context = self.model.call_args.kwargs["support_context"]
        self.assertIn("confirmation missing", context["search_question"])
        returned = self.ask("I want to return an item that arrived damaged", "return-customer")
        self.assertNotIn("item_problem", returned["resolution"]["asked"])
        self.assertIn("damaged", self.model.call_args.kwargs["support_context"]["search_question"])

    def test_general_payment_check_is_actionable_and_not_repeated_after_customer_tried_it(self):
        first = self.ask("My payment was deducted but I have no order confirmation")
        self.assertIn("duplicate payment", first["answer"])
        second = self.ask("I already checked my order history and email")
        self.assertNotIn("Before retrying", second["answer"])
        self.assertIn("steps_attempted", second["support_context"]["reported_facts"])

    def test_real_backend_request_hands_off_without_performing_or_inventing_an_action(self):
        result = self.ask("Cancel my order for me")
        self.assertEqual(result["reason"], "backend_action_required")
        self.assertIn("don't have", result["answer"])
        self.assertEqual(result["resolution"]["attempts"], 0)
        self.model.assert_not_called()
        payload = json.loads(self.store.rows("SELECT payload FROM tickets")[0]["payload"])
        brief = payload["agent_brief"]
        self.assertFalse(brief["backend_access"])
        self.assertEqual(brief["completed_backend_actions"], [])
        self.assertTrue(brief["facts_are_customer_reported"])
        self.assertEqual(brief["latest_customer_update"], "Cancel my order for me")

    def test_guidance_requests_do_not_take_the_backend_shortcut(self):
        for question in ("How can I cancel my order?", "I don't want you to process my refund now"):
            result = self.ask(question, question)
            self.assertNotEqual(result.get("reason"), "backend_action_required")
            self.assertNotIn("ticket", result)
        self.assertEqual(self.model.call_count, 2)

    def test_human_receives_new_reported_details_on_existing_ticket_without_more_model_calls(self):
        first = self.ask("Cancel my order for me")
        updated = self.ask("My order has not been dispatched and I already checked the website")
        self.assertEqual(updated["ticket"]["reference"], first["ticket"]["reference"])
        self.model.assert_not_called()
        payload = json.loads(self.store.rows("SELECT payload FROM tickets")[0]["payload"])
        self.assertIn("not dispatched", payload["agent_brief"]["reported_facts"]["dispatch_status"])
        self.assertIn("checked the website", payload["agent_brief"]["latest_customer_update"])
        self.assertEqual(len(self.store.rows("SELECT * FROM tickets")), 1)
        self.assertEqual(len(self.store.rows("SELECT * FROM ticket_messages")), 1)

    def test_false_backend_completion_is_rejected_even_if_a_generator_reports_it_as_supported(self):
        self.model.return_value = flow_fixture.guidance("I have cancelled your order and processed your refund.")
        result = self.ask("How can I cancel my order?")
        self.assertNotIn("I have cancelled", result["answer"])
        self.assertEqual(result["failure_reason"], "backend_claim_rejected")
        self.assertFalse(result["verification"]["passed"] is True)

    def test_knowledge_gaps_require_review_permission_and_exclude_infrastructure_failures(self):
        monitor = Monitor(self.store, self.fixture.settings)
        for reason in ("insufficient_retrieval", "model_unavailable", "insufficient_retrieval"):
            monitor.trace("one", {"question": "Cancellation option missing for user@example.com", "reason": "clarification_needed",
                                  "failure_reason": reason, "search_query": "missing cancellation procedure"})
        rows = knowledge_gaps(self.store, Principal("reviewer", "reviewer"))
        self.assertEqual(len(rows), 1)
        self.assertNotIn("example.com", str(rows))
        with self.assertRaises(PermissionError):
            knowledge_gaps(self.store, Principal("support", "support"))


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_hypothetical_status_and_negative_charge_are_not_misread_as_verified_records(self):
        self.assertEqual(reported_details("What if my order has been dispatched?", "cancellation"), {})
        self.assertTrue(reported_details("My payment was not charged", "payment")["charged_status"].startswith("not charged"))
        self.assertTrue(reported_details("My order has not been dispatched?", "cancellation")["dispatch_status"].startswith("not dispatched"))
        self.assertFalse(manual_action_requested("Write Python code and do it for me"))

    def test_completion_guard_accepts_guidance_and_disclaimers(self):
        for answer in ("I cannot cancel your order directly.", "You said the order was cancelled.",
                       "Use the approved refund procedure.", "Maine aapka order cancel nahi kiya."):
            self.assertFalse(claims_backend_action(answer), answer)
        self.assertTrue(claims_backend_action("Your order has been cancelled."))
        self.assertTrue(claims_backend_action("Maine aapka order cancel kar diya."))


if __name__ == "__main__":
    unittest.main()
