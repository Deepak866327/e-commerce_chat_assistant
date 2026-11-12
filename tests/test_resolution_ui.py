from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.config import Settings
from src.store import Store
from src.tickets import TicketService
from tests.test_resolution_flow import unavailable


class ResolutionUITests(unittest.TestCase):
    def test_chat_asks_three_followups_then_shows_one_ticket_and_clear_resets_budget(self):
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            settings = replace(Settings(), state_dir=root, requests_per_minute=100)
            store = Store(root / "control.sqlite3")
            tickets = TicketService(store, settings)
            model = Mock(return_value=unavailable())
            stack.enter_context(patch("src.rag_memory.Settings.from_env", return_value=settings))
            stack.enter_context(patch("src.rag_memory.runtime", return_value=SimpleNamespace(ask=model)))
            stack.enter_context(patch("src.support_ui.services", return_value=(settings, store, tickets)))
            stack.enter_context(patch("streamlit.audio_input", return_value=None))
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            for question in ("Cancel my order", "No", "The step is failing"):
                app.chat_input[0].set_value(question).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["support_tickets"], [])
            app.chat_input[0].set_value("Still not working").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["support_tickets"]), 1)
            self.assertEqual(model.call_count, 3)
            reference = app.session_state["support_tickets"][0]
            self.assertTrue(any(reference in item.value for item in app.markdown))
            next(button for button in app.button if button.label == "Clear conversation").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(store.rows("SELECT * FROM support_sessions"), [])
            app.chat_input[0].set_value("Cancel my order").run()
            self.assertFalse(app.exception)
            self.assertEqual(json.loads(store.rows("SELECT body FROM support_sessions")[0]["body"])["attempts"], 1)
            self.assertEqual(len(store.rows("SELECT * FROM tickets")), 1)
