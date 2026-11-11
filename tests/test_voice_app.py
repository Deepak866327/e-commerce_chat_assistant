from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.voice import SpeechInputError, Transcription


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
        pipeline.faq_suggestions = self.suggestions
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
        self.app.selectbox[0].select("Hindi / Hinglish").run()
        self.transcribe_question()
        self.assertEqual(self.transcribe.call_args.kwargs["language"], "hi")

    def test_published_faq_button_sends_question_once_through_normal_chat(self):
        self.suggestions.return_value = [{"faq_id": "FAQ_one", "question": "How do I request a return?"}]
        self.app.run()
        self.button("How do I request a return?").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.answer.call_args.args[0], "How do I request a return?")
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.app.run()
        self.answer.assert_called_once()

    def test_empty_published_catalog_does_not_show_faq_buttons(self):
        self.assertFalse(any(header.value == "Frequently asked questions" for header in self.app.subheader))

    def test_chat_language_is_independent_of_speech_language(self):
        next(select for select in self.app.selectbox if select.label == "Chat language").select("English").run()
        self.app.chat_input[0].set_value("What is your refund policy?").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.answer.call_args.kwargs["response_language"], "en")
        self.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
