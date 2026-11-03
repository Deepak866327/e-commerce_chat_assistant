"""Browser speech playback without a hosted text-to-speech API."""

from hashlib import sha256
from html import escape
from pathlib import Path


_TEMPLATE = Path(__file__).with_name("read_aloud.html").read_text(encoding="utf-8")


def speech_controls_html(text: str, key: str) -> str:
    element_id = "shopassist-speech-" + sha256(key.encode()).hexdigest()[:16]
    # Assistant output is data in an escaped attribute, never executable HTML/JS.
    return _TEMPLATE.replace("__ELEMENT_ID__", element_id).replace(
        "__SPEECH_TEXT__", escape(text, quote=True)
    )
