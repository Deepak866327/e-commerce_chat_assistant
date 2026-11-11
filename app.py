from hashlib import sha256
import logging
from uuid import uuid4

import streamlit as st
from src.config import Settings
from src.support_context import remember_customer, remember_result
from src.rag_memory import MAX_HISTORY_MESSAGES, ask_shopassist, submit_feedback, faq_suggestions
from src.speech_controls import speech_controls_html
from src.voice import MAX_AUDIO_SECONDS, SAMPLE_RATE, SpeechInputError, transcribe_recording
from src.support_ui import remember_ticket, render_customer_support, render_staff_workspace


logger = logging.getLogger(__name__)


def reset_voice_input():
    st.session_state.voice_generation += 1
    st.session_state.voice_transcript = ""
    st.session_state.voice_recording_digest = None
    st.session_state.voice_feedback = None


def clear_conversation():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.support_context = {}
    st.session_state.pop("pending_voice_question", None)
    st.session_state.pop("pending_faq_question", None)
    reset_voice_input()


def send_voice_question():
    question = st.session_state.voice_transcript.strip()
    if question:
        st.session_state.pending_voice_question = question
        reset_voice_input()


def send_faq_question(question):
    st.session_state.pending_faq_question = question


def render_message(message, index):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.write(source)
        if message["role"] == "assistant":
            st.html(
                speech_controls_html(message["content"], f"reply-{index}"),
                unsafe_allow_javascript=True,
            )
            if message.get("needs_human_support"):
                if message.get("ticket"):
                    st.info(f"Support ticket {message['ticket']['reference']} is in the human support queue. Check replies under Your support tickets.")
                else:
                    st.caption("Use Talk to a human to create a support ticket.")
            if message.get("query_id"):
                with st.expander("Rate this answer"):
                    with st.form(f"feedback-{message['query_id']}"):
                        kind = st.selectbox("Feedback", ["helpful", "not_helpful", "incorrect_answer",
                                                          "missing_information", "outdated_information"],
                                            format_func=lambda value: value.replace("_", " ").capitalize())
                        comment = st.text_input("Details (optional)", max_chars=2000)
                        if st.form_submit_button("Submit feedback"):
                            try:
                                submit_feedback(message["query_id"], kind, comment, st.session_state.support_session)
                                st.success("Feedback recorded for review.")
                            except (ValueError, PermissionError):
                                st.warning("Feedback could not be saved for this conversation.")


def render_voice_input(language):
    with st.expander("Ask with your voice", expanded=False):
        st.caption(
            f"Record up to {MAX_AUDIO_SECONDS} seconds, then transcribe and review your question."
        )
        recording = st.audio_input(
            "Record a question",
            sample_rate=SAMPLE_RATE,
            key=f"voice_recording_{st.session_state.voice_generation}",
        )
        audio = recording.getvalue() if recording is not None else b""
        digest = sha256(audio).hexdigest() if audio else None
        if digest != st.session_state.voice_recording_digest:
            st.session_state.voice_recording_digest = digest
            st.session_state.voice_transcript = ""
            st.session_state.voice_feedback = None

        if st.button("Transcribe recording", disabled=not audio):
            st.session_state.voice_transcript = ""
            st.session_state.voice_feedback = None
            try:
                with st.spinner("Transcribing… The first use may download the speech model."):
                    result = transcribe_recording(audio, language=language)
                st.session_state.voice_transcript = result.text
                st.session_state.voice_feedback = (
                    "caption",
                    f"Detected language: {result.language}. Check the transcript before sending.",
                )
            except SpeechInputError as error:
                st.session_state.voice_feedback = ("warning", str(error))
            except ModuleNotFoundError:
                st.session_state.voice_feedback = (
                    "error",
                    "Speech dependencies are missing. Install requirements.txt and restart the app.",
                )
            except Exception:
                logger.exception("Voice transcription failed")
                st.session_state.voice_feedback = (
                    "error",
                    "Could not transcribe the recording. On first use, an internet connection "
                    "is needed to download the model. Try again or use text chat.",
                )

        feedback = st.session_state.voice_feedback
        if feedback:
            getattr(st, feedback[0])(feedback[1])

        with st.form("voice_question_review"):
            st.text_area("Review your question", key="voice_transcript", height=100)
            st.form_submit_button(
                "Send voice question",
                on_click=send_voice_question,
                disabled=not st.session_state.voice_transcript.strip(),
            )


def main():
    st.set_page_config(page_title="ShopAssist AI", page_icon="🛍️", layout="centered")
    if Settings.from_env().require_login and not st.user.is_logged_in:
        st.title("ShopAssist AI")
        st.write("Sign in to contact support.")
        if st.button("Sign in"):
            st.login()
        st.stop()
    st.title("ShopAssist AI")
    st.caption("Your AI-powered shopping support assistant")

    defaults = {
        "messages": [],
        "chat_history": [],
        "voice_generation": 0,
        "voice_transcript": "",
        "voice_recording_digest": None,
        "voice_feedback": None,
        "support_session": uuid4().hex,
        "support_tickets": [],
        "support_context": {},
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)

    with st.sidebar:
        st.header("ShopAssist AI")
        st.write("Powered by Qwen")
        st.button("Clear conversation", on_click=clear_conversation)
        voice_enabled = st.toggle("Voice input", value=True)
        speech_language = st.selectbox(
            "Speech language",
            options=["Auto detect", "Hindi / Hinglish", "English"],
            disabled=not voice_enabled,
            help="Try Hindi / Hinglish if automatic detection struggles with mixed speech.",
        )
        chat_language = st.selectbox("Chat language", ["Hinglish", "English", "Hindi"])
        response_language = {"Hinglish": "hinglish", "English": "en", "Hindi": "hi"}[chat_language]
        st.caption("Voice recordings are transcribed locally using a small multilingual model.")
        st.caption("Read-aloud voices depend on your browser and device.")
        try:
            suggestions = faq_suggestions()
        except Exception:
            logger.error("FAQ suggestions are temporarily unavailable")
            suggestions = []
        if suggestions:
            st.subheader("Frequently asked questions")
            for suggestion in suggestions:
                st.button(suggestion["question"], key="faq-" + suggestion["faq_id"],
                          on_click=send_faq_question, args=(suggestion["question"],))
        render_customer_support()
        staff_mode = st.toggle("Staff workspace", value=False)

    if staff_mode:
        render_staff_workspace()
        return

    for index, message in enumerate(st.session_state.messages):
        render_message(message, index)

    if voice_enabled:
        language = {"Auto detect": None, "Hindi / Hinglish": "hi", "English": "en"}
        render_voice_input(language[speech_language])

    typed_question = st.chat_input("Ask about shipping, returns, refunds...")
    question = (st.session_state.pop("pending_voice_question", None)
                or st.session_state.pop("pending_faq_question", None) or typed_question)
    if not question or not question.strip():
        return
    question = question.strip()

    pending_context = remember_customer(st.session_state.support_context, question)
    user_message = {"role": "user", "content": question}
    render_message(user_message, len(st.session_state.messages))
    try:
        with st.spinner("ShopAssist is thinking..."):
            result = ask_shopassist(question, st.session_state.chat_history, st.session_state.support_session,
                                    response_language=response_language, support_context=pending_context)
    except Exception:
        logger.error("Chat request failed")
        st.error("Support is temporarily unavailable. Please try again shortly.")
        return

    st.session_state.support_context = remember_result(pending_context, result)
    assistant_message = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "query_id": result.get("query_id"),
        "needs_human_support": result.get("needs_human_support", False),
        "ticket": result.get("ticket"),
    }
    if result.get("ticket"):
        remember_ticket(result["ticket"])
    render_message(assistant_message, len(st.session_state.messages) + 1)
    st.session_state.messages.extend([user_message, assistant_message])
    st.session_state.chat_history.extend(
        [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]
    )
    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]
    if result.get("ticket"):
        st.rerun()


if __name__ == "__main__":
    main()
