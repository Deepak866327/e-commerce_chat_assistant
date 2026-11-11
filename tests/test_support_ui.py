from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.config import Settings
from src.security import Principal, password_hash
from src.store import Store
from src.tickets import TicketService


APP = str(Path(__file__).resolve().parents[1] / "app.py")


class SupportUITests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.settings = replace(Settings(), state_dir=root)
        self.store = Store(root / "control.sqlite3")
        self.tickets = TicketService(self.store, self.settings)
        pipeline = ModuleType("src.rag_memory")
        pipeline.MAX_HISTORY_MESSAGES = 8
        self.answer = pipeline.ask_shopassist = Mock(return_value={"answer": "Hello", "sources": []})
        pipeline.submit_feedback = Mock()
        pipeline.reset_support_resolution = Mock()
        pipeline.faq_articles = Mock(return_value=[])
        pipeline.request_human_support = Mock(side_effect=lambda q, h, s: self.tickets.open(q, {}, h, s))
        self.stack.enter_context(patch.dict("sys.modules", {"src.rag_memory": pipeline}))
        self.stack.enter_context(patch("src.support_ui.services", return_value=(self.settings, self.store, self.tickets)))
        self.stack.enter_context(patch("streamlit.audio_input", return_value=None))
        self.app = AppTest.from_file(APP).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def text_input(self, label):
        return next(field for field in self.app.text_input if field.label == label)

    def create_ticket(self):
        self.text_input("Describe your issue").set_value("My refund has not arrived")
        self.button("Create support ticket").click().run()
        self.assertFalse(self.app.exception)
        return self.app.session_state["support_tickets"][0]

    def test_explicit_ticket_and_staff_reply_appear_for_customer(self):
        reference = self.create_ticket()
        self.assertTrue(any(reference in expander.label for expander in self.app.expander))
        self.tickets.reply(reference, "A support team member is checking your refund.", principal=Principal("Sam", "support"))
        self.button("Refresh ticket").click().run()
        self.assertTrue(any("checking your refund" in element.value for element in self.app.markdown))
        self.text_input("Add details for the support team").set_value("It is still pending.")
        self.button("Send to support").click().run()
        self.assertFalse(self.app.exception)
        messages = self.tickets.customer_view(reference, self.app.session_state["support_session"])["messages"]
        self.assertEqual([m["author"] for m in messages], ["SUPPORT", "CUSTOMER"])
        self.answer.assert_not_called()

    def test_automatic_ticket_is_remembered_without_resending_chat(self):
        session = self.app.session_state["support_session"]
        ticket = self.tickets.open("Check my order", {}, [], session)
        self.answer.return_value = {"answer": "The support team can check this.", "sources": [],
                                    "needs_human_support": True, "ticket": ticket}
        self.app.chat_input[0].set_value("Check my order").run()
        self.assertFalse(self.app.exception)
        self.assertIn(ticket["reference"], self.app.session_state["support_tickets"])
        self.assertTrue(any(ticket["reference"] in expander.label for expander in self.app.expander))
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.answer.assert_called_once()

    def test_staff_queue_requires_signin_and_can_reply_and_resolve(self):
        reference = self.create_ticket()
        next(toggle for toggle in self.app.toggle if toggle.label == "Staff workspace").set_value(True).run()
        self.assertFalse(any(header.value == "Human support queue" for header in self.app.subheader))
        self.text_input("Password").set_value("test-support-password")
        with patch.dict("os.environ", {"SHOPASSIST_SUPPORT_PASSWORD_HASH": password_hash("test-support-password")}):
            self.button("Sign in to staff workspace").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any(header.value == "Human support queue" for header in self.app.subheader))
        next(area for area in self.app.text_area if area.label == "Reply to customer").set_value("We have resolved this request.")
        next(select for select in self.app.selectbox if select.label == "Status").select("RESOLVED")
        self.button("Save ticket and send reply").click().run()
        self.assertFalse(self.app.exception)
        view = self.tickets.customer_view(reference, self.app.session_state["support_session"])
        self.assertEqual(view["status"], "RESOLVED")
        self.assertEqual(view["messages"][0]["body"], "We have resolved this request.")
        self.assertFalse(any(header.value == "FAQ generation and review" for header in self.app.subheader))


if __name__ == "__main__":
    unittest.main()
