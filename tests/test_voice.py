from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from src import voice
from src.speech_controls import speech_controls_html


def make_recording(seconds=1):
    output = BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(voice.SAMPLE_RATE)
        recording.writeframes(b"\0\0" * int(voice.SAMPLE_RATE * seconds))
    return output.getvalue()


class RecordingTests(unittest.TestCase):
    def test_accepts_the_thirty_second_boundary(self):
        self.assertEqual(voice.validate_recording(make_recording(30)), 30)

    def test_invalid_recordings_are_rejected_before_loading_a_model(self):
        recordings = [
            b"",
            b"not a WAV",
            make_recording(0),
            make_recording(31),
            make_recording()[:-10],
            b"x" * (voice.MAX_AUDIO_BYTES + 1),
        ]
        with patch.object(voice, "get_transcriber") as load:
            for recording in recordings:
                with self.subTest(size=len(recording)):
                    with self.assertRaises(voice.SpeechInputError):
                        voice.transcribe_recording(recording)
            load.assert_not_called()

    def test_silence_does_not_become_a_customer_question(self):
        transcriber = object.__new__(voice.SpeechTranscriber)
        from threading import Lock
        transcriber._lock = Lock()
        transcriber._model = Mock()
        transcriber._model.transcribe.return_value = (
            iter([]), SimpleNamespace(language="en", duration=1)
        )
        with self.assertRaisesRegex(voice.SpeechInputError, "No speech"):
            transcriber.transcribe(make_recording(), "en")

    def test_only_one_model_is_initialized_for_simultaneous_requests(self):
        voice._cached_transcriber.cache_clear()
        try:
            with patch.object(voice, "SpeechTranscriber") as constructor:
                with ThreadPoolExecutor(max_workers=4) as workers:
                    models = list(workers.map(lambda _: voice.get_transcriber(), range(8)))
                constructor.assert_called_once_with("tiny")
                self.assertTrue(all(model is models[0] for model in models))
        finally:
            voice._cached_transcriber.cache_clear()


class SpeechMarkupTests(unittest.TestCase):
    def test_assistant_output_cannot_inject_markup_or_scripts(self):
        text = '\" ><script>alert("customer")</script> & हिन्दी __ELEMENT_ID__'

        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.scripts = 0
                self.speech = None

            def handle_starttag(self, tag, attrs):
                if tag == "script":
                    self.scripts += 1
                if "data-speech" in dict(attrs):
                    self.speech = dict(attrs)["data-speech"]

        parser = Parser()
        parser.feed(speech_controls_html(text, "reply-1"))
        self.assertEqual(parser.scripts, 1)
        self.assertEqual(parser.speech, text)


if __name__ == "__main__":
    unittest.main()
