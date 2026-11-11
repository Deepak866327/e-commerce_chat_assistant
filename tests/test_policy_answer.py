from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.policy_answer import generate_policy_answer, supported_numbers


class PolicyAnswerTests(unittest.TestCase):
    evidence = ["Approved refunds are generally processed within 5–7 business days after inspection."]

    def test_live_ten_day_claim_is_rejected_without_treating_list_markers_as_facts(self):
        self.assertFalse(supported_numbers("Refunds are issued within 10 business days.", self.evidence))
        self.assertTrue(supported_numbers("1. Approved refunds take 5–7 business days after inspection.", self.evidence))
        self.assertTrue(supported_numbers("निरीक्षण के बाद मंजूर रिफंड में ५–७ कार्य दिवस लगते हैं।", self.evidence))
        self.assertFalse(supported_numbers("5.7 days", self.evidence))

    def test_one_repair_can_return_supported_guidance(self):
        chain = Mock()
        chain.invoke.side_effect = [SimpleNamespace(content="Refunds take 10 business days."),
                                   SimpleNamespace(content=self.evidence[0])]
        payload = {"question": "When is a refund processed?", "context": self.evidence[0], "chat_history": []}
        self.assertEqual(generate_policy_answer(chain, payload, self.evidence), self.evidence[0])
        self.assertEqual(chain.invoke.call_count, 2)
        self.assertNotIn("Re-answer", payload["question"])
        self.assertIn("Re-answer", chain.invoke.call_args.args[0]["question"])

    def test_persistent_unsupported_claim_does_not_escape(self):
        chain = Mock()
        chain.invoke.return_value = SimpleNamespace(content="Refunds are issued within 10 business days.")
        self.assertIsNone(generate_policy_answer(chain, {"question": "Refund timing?"}, self.evidence))
        self.assertEqual(chain.invoke.call_count, 2)
