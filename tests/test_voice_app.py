from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.voice import SpeechInputError, Transcription
from src.faq_catalog import versioned_article
from src.default_help import starter_faqs


APP = str(Path(__file__).resolve().parents[1] / "app.py")


class VoiceAppTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        pipeline = ModuleType("src.rag_memory")
        pipeline.MAX_HISTORY_MESSAGES = 10
        self.answer = Mock(return_value={
            "answer": "Standard shipping takes 5 to 7 business days.",
            "sources": ["shipping.pdf, page 1"],
        })
        pipeline.ask_shopassist = self.answer
        pipeline.submit_feedback = Mock()
        pipeline.reset_support_resolution = Mock()
        self.suggestions = Mock(return_value=[])
        pipeline.faq_articles = lambda: [versioned_article(a) for a in self.suggestions()]
        self.faq_feedback = pipeline.submit_faq_feedback = Mock(side_effect=lambda ref, outcome, session: next(
            a for a in (pipeline.faq_articles() or starter_faqs()) if a["faq_id"] == ref["faq_id"] and a["version"] == ref["version"]))
        def open_article(reference, session):
            article = next(a for a in (pipeline.faq_articles() or starter_faqs()) if a["faq_id"] == reference["faq_id"] and a["version"] == reference["version"])
            return {"answer": article["answer"], "sources": [], "faq": article,
                    "support_context": {"faq": {**article, "outcome": "shown"}}}
        self.open_article = pipeline.open_faq = Mock(side_effect=open_article)
        self.stack.enter_context(patch.dict("sys.modules", {"src.rag_memory": pipeline}))
        self.transcribe = self.stack.enter_context(patch(
            "src.voice.transcribe_recording",
            return_value=Transcription("What is shipping time?", "en", 3),
        ))
        self.recording = self.stack.enter_context(patch("streamlit.audio_input", return_value=None))
        self.app = AppTest.from_file(APP).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def transcribe_question(self):
        self.recording.return_value = BytesIO(b"recorded audio")
        self.app.run()
        self.button("Transcribe recording").click().run()
        self.assertFalse(self.app.exception)

    def test_text_chat_does_not_load_speech_and_still_saves_history(self):
        self.app.chat_input[0].set_value("When will my package arrive?").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(len(self.app.session_state["chat_history"]), 2)
        self.transcribe.assert_not_called()

    def test_reviewed_voice_question_is_sent_only_once(self):
        self.transcribe_question()
        self.answer.assert_not_called()
        self.app.text_area[0].set_value("What is express shipping time?")
        self.button("Send voice question").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.answer.call_args.args[0], "What is express shipping time?")
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(self.app.session_state["voice_transcript"], "")
        self.app.run()
        self.answer.assert_called_once()
        self.transcribe.assert_called_once()

    def test_new_recording_discards_the_previous_transcript(self):
        self.transcribe_question()
        self.recording.return_value = BytesIO(b"a different recording")
        self.app.run()
        self.assertEqual(self.app.session_state["voice_transcript"], "")
        self.assertTrue(self.button("Send voice question").disabled)

    def test_clear_conversation_resets_voice_and_history(self):
        self.transcribe_question()
        previous_generation = self.app.session_state["voice_generation"]
        self.recording.return_value = None
        self.button("Clear conversation").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["chat_history"], [])
        self.assertEqual(self.app.session_state["voice_transcript"], "")
        self.assertGreater(self.app.session_state["voice_generation"], previous_generation)
        self.assertEqual(self.app.session_state["support_context"], {})
        self.answer.assert_not_called()

    def test_silence_warns_without_sending_a_question(self):
        self.transcribe.side_effect = SpeechInputError("No speech was detected.")
        self.transcribe_question()
        self.assertTrue(self.app.warning)
        self.assertTrue(self.button("Send voice question").disabled)
        self.answer.assert_not_called()
        self.app.chat_input[0].set_value("What is your return policy?").run()
        self.assertFalse(self.app.exception)
        self.answer.assert_called_once()

    def test_hinglish_selection_reaches_transcription(self):
        next(field for field in self.app.selectbox if field.label == "Speech language").select("Hindi / Hinglish").run()
        self.transcribe_question()
        self.assertEqual(self.transcribe.call_args.kwargs["language"], "hi")

    def test_published_faq_button_sends_question_once_through_normal_chat(self):
        self.suggestions.return_value = [{"faq_id": "FAQ_one", "question": "How do I request a return?", "answer": "Open your orders and select Return.", "category": "Returns"}]
        self.app.run()
        self.answer.assert_not_called()
        self.button("How do I request a return?").click().run()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.answer.assert_not_called()
        self.button("I still need help").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("How do I request a return?", self.answer.call_args.args[0])
        self.assertIn("still need help", self.answer.call_args.args[0])
        self.assertEqual(self.answer.call_args.kwargs["support_context"]["faq_reference"]["faq_id"], "FAQ_one")
        self.assertEqual(len(self.app.session_state["messages"]), 4)
        self.app.run()
        self.answer.assert_called_once()

    def test_fresh_deployment_has_searchable_faq_answers_without_model_calls(self):
        self.assertFalse(any("being prepared" in item.value for item in self.app.info))
        self.assertEqual(sum((button.key or "").startswith("faq-select-") for button in self.app.button), 6)
        self.assertTrue(any(item.label == "How can I cancel my order?" for item in self.app.button))
        self.assertFalse(any("cancellation option" in item.value for item in self.app.markdown))
        next(field for field in self.app.text_input if field.label == "Search help articles").set_value("refund").run()
        self.assertTrue(any(item.label == "My refund has not arrived. What should I check?" for item in self.app.button))
        self.assertEqual(self.app.session_state["messages"], [])
        self.answer.assert_not_called()

    def test_helpful_faq_records_feedback_without_chat_or_resolution_attempts(self):
        self.button("How can I cancel my order?").click().run()
        self.button("This helped").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("glad this article helped" in item.value for item in self.app.success))
        self.faq_feedback.assert_called_once()
        self.assertEqual(self.faq_feedback.call_args.args[1], "helpful")
        self.answer.assert_not_called()
        self.assertEqual(self.app.session_state["support_context"]["faq"]["outcome"], "shown")
        self.app.run()
        self.faq_feedback.assert_called_once()

    def test_changed_faq_does_not_send_stale_context(self):
        self.faq_feedback.side_effect = ValueError("This article has changed.")
        self.button("How can I cancel my order?").click().run()
        self.button("I still need help").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("article has changed" in item.value for item in self.app.warning))
        self.answer.assert_not_called()

    def test_starter_faq_continues_in_chat_once(self):
        self.button("How can I cancel my order?").click().run()
        self.button("I still need help").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("How can I cancel my order?", self.answer.call_args.args[0])
        self.app.run()
        self.answer.assert_called_once()

    def test_faq_click_adds_question_and_saved_answer_once_to_chat(self):
        self.assertFalse(self.app.session_state["messages"])
        self.button("How can I cancel my order?").click().run()
        self.assertFalse(self.app.exception)
        messages = self.app.session_state["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "How can I cancel my order?")
        self.assertEqual(messages[1]["content"], starter_faqs()[0]["answer"])
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertEqual(self.app.session_state["chat_history"][1]["content"], messages[1]["content"])
        self.app.run()
        self.open_article.assert_called_once()
        self.answer.assert_not_called()
        self.assertEqual(len(self.app.session_state["messages"]), 2)

    def test_stale_faq_click_does_not_show_a_cached_answer(self):
        self.open_article.side_effect = ValueError("This FAQ has changed.")
        self.button("How can I cancel my order?").click().run()
        self.assertTrue(any("FAQ has changed" in item.value for item in self.app.warning))
        self.assertEqual(self.app.session_state["messages"], [])
        self.answer.assert_not_called()

    def test_faq_search_topics_and_publication_refresh_do_not_send_chat(self):
        self.suggestions.return_value = [
            {"faq_id": "return", "question": "How do I return?", "answer": "Use the returns page.", "category": "Returns"},
            {"faq_id": "ship", "question": "When does it arrive?", "answer": "Express delivery takes two days.", "category": "Shipping"},
        ]
        self.app.run()
        self.assertFalse(any(item.value == "Use the returns page." for item in self.app.markdown))
        next(field for field in self.app.text_input if field.label == "Search help articles").set_value("EXPRESS days").run()
        self.assertEqual([item.label for item in self.app.button if item.label in {"How do I return?", "When does it arrive?"}], ["When does it arrive?"])
        next(field for field in self.app.selectbox if field.label == "Browse by topic").select("Returns").run()
        self.assertTrue(any("No matching" in item.value for item in self.app.info))
        # Withdrawn articles disappear on the next rerun, including a stale topic selection.
        self.suggestions.return_value = [self.suggestions.return_value[1]]
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["faq_category"], "All topics")
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.app.session_state["support_context"], {})
        self.answer.assert_not_called()

    def test_faq_load_failure_leaves_chat_available(self):
        self.suggestions.side_effect = RuntimeError("Unavailable")
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("temporarily unavailable" in item.value for item in self.app.info))
        self.assertTrue(any(item.label == "How can I track my order?" for item in self.app.button))
        self.app.chat_input[0].set_value("Help with my order").run()
        self.answer.assert_called_once()

    def test_chat_language_is_independent_of_speech_language(self):
        next(select for select in self.app.selectbox if select.label == "Chat language").select("English").run()
        self.app.chat_input[0].set_value("What is your refund policy?").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.answer.call_args.kwargs["response_language"], "en")
        self.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
