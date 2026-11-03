"""Small, local speech recognition for short customer questions."""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
import wave


SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MODEL_CACHE = Path(__file__).resolve().parents[1] / "models" / "speech"
_MODEL_LOAD_LOCK = Lock()


class SpeechInputError(ValueError):
    """The recording cannot be used as a customer question."""


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    duration_seconds: float


def validate_recording(audio: bytes) -> float:
    """Reject empty, malformed, or long recordings before loading a model."""
    if not audio:
        raise SpeechInputError("Record a question first.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechInputError("The recording is too large. Record up to 30 seconds.")

    try:
        with wave.open(BytesIO(audio), "rb") as recording:
            frames = recording.getnframes()
            duration = frames / recording.getframerate()
            if not frames:
                raise SpeechInputError("The recording is empty. Please record again.")
            if duration > MAX_AUDIO_SECONDS:
                raise SpeechInputError("Keep your recording to 30 seconds or less.")
            expected_bytes = (
                frames * recording.getnchannels() * recording.getsampwidth()
            )
            if len(recording.readframes(frames)) != expected_bytes:
                raise SpeechInputError("The recording is incomplete. Please record again.")
    except (wave.Error, EOFError, ZeroDivisionError) as error:
        raise SpeechInputError("Could not read this recording. Please record again.") from error

    return duration


class SpeechTranscriber:
    """Keep one CPU model in memory and serialize access across sessions."""

    def __init__(self, model_size: str = "tiny"):
        if model_size not in {"tiny", "base"}:
            raise ValueError("Choose the multilingual tiny or base speech model.")

        # Import and download only when voice input is actually requested.
        from faster_whisper import WhisperModel
        from huggingface_hub.errors import LocalEntryNotFoundError

        settings = dict(
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
            download_root=str(MODEL_CACHE),
        )
        try:
            self._model = WhisperModel(model_size, local_files_only=True, **settings)
        except LocalEntryNotFoundError:
            self._model = WhisperModel(model_size, **settings)
        self._lock = Lock()

    def transcribe(self, audio: bytes, language: str | None) -> Transcription:
        with self._lock:
            segments, info = self._model.transcribe(
                BytesIO(audio),
                language=language,
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            # Faster Whisper performs inference while this generator is consumed.
            text = " ".join(segment.text.strip() for segment in segments).strip()

        if not text:
            raise SpeechInputError("No speech was detected. Try speaking closer to the microphone.")
        return Transcription(text, info.language, info.duration)


@lru_cache(maxsize=1)
def _cached_transcriber(model_size: str) -> SpeechTranscriber:
    return SpeechTranscriber(model_size)


def get_transcriber(model_size: str = "tiny") -> SpeechTranscriber:
    # lru_cache alone allows duplicate initialization on concurrent first calls.
    with _MODEL_LOAD_LOCK:
        return _cached_transcriber(model_size)


def transcribe_recording(
    audio: bytes, language: str | None = None, model_size: str = "tiny"
) -> Transcription:
    validate_recording(audio)
    if language not in {None, "en", "hi"}:
        raise SpeechInputError("Choose auto detection, English, or Hindi / Hinglish.")
    return get_transcriber(model_size).transcribe(audio, language)
