"""Stable chat entry points. Model/vector loading happens only on a request."""

from functools import lru_cache
import logging
import time

from src.chat_service import ChatService, MAX_HISTORY_MESSAGES
from src.config import Settings
from src.language import message, language_instruction
from src.conversation import conversational_reply
from src.observability import Monitor
from src.ollama_client import OllamaClient
from src.query_routing import classify_query, response_without_retrieval
from src.retrieval import BGEReranker, Retriever
from src.store import Store
from src.vector_index import ChromaIndex, IndexUnavailable
from src.verification import GroundingVerifier

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _vector(settings, index_id):
    return ChromaIndex(settings, index_id)


def runtime():
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    monitor = Monitor(store, settings)
    active_index = store.setting("active_index")
    client = OllamaClient(settings)
    retriever = Retriever(store, settings,
        vector_factory=(lambda: _vector(settings, active_index)) if active_index else None,
        reranker_factory=(lambda: BGEReranker(settings)) if settings.use_reranker and active_index else None)
    return ChatService(store, settings, client, retriever,
                       GroundingVerifier(client, settings.verifier_model), monitor)


def _escalate(question, history, session, result, response_language="hinglish"):
    from src.chat_service import history_messages
    from src.tickets import TicketService, requires_ticket
    if result.get("ticket") or not requires_ticket(result):
        return result
    result["needs_human_support"] = True
    reason = result.get("reason") or "legacy_unanswered"
    try:
        settings = Settings.from_env()
        store = Store(settings.state_dir / "control.sqlite3")
        result["ticket"] = TicketService(store, settings).open(question, result, history_messages(history), session, reason)
        if reason == "explicit_handoff":
            result["answer"] = message("support_queued", response_language, reference=result["ticket"]["reference"])
    except Exception as exc:
        logger.error("Support ticket creation failed: %s", type(exc).__name__)
        result["ticket_error"] = True
    return result


def _conversation(question, session, history=None, response_language="hinglish"):
    started = time.perf_counter()
    reply = conversational_reply(question, response_language)
    if reply is None:
        return None
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    from src.security import rate_limit
    rate_limit(store, session, settings.requests_per_minute)
    reply["latency_seconds"] = time.perf_counter() - started
    try:
        reply["query_id"] = Monitor(store, settings).trace(session, {"question": question, **reply})
    except Exception as exc:
        logger.error("Conversation trace failed: %s", type(exc).__name__)
    return _escalate(question, history, session, reply, response_language)


def ask_structured(question, chat_history=None, session="local", *, response_language="hinglish"):
    """Public structured contract. Citation sources are verified objects."""
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 2000:
        raise ValueError("Question must contain 1–2000 characters")
    language_instruction(response_language)
    question = question.strip()
    reply = _conversation(question, session, chat_history, response_language)
    if reply is not None:
        return reply
    try:
        return _escalate(question, chat_history, session, runtime().ask(
            question, chat_history, session, response_language=response_language), response_language)
    except IndexUnavailable:
        # Keep the UI operational when models/indexes need installation. Do not
        # serve a different model/dimension behind the active collection name.
        return _escalate(question, chat_history, session, {"answer": message("abstention", response_language), "answerable": False, "sources": [],
                "needs_human_support": True, "reason": "index_unavailable", "search_query": question,
                "query_id": None, "verification": {"passed": False}, "confidence": None}, response_language)


def ask_shopassist(question, chat_history, session="local", *, response_language="hinglish"):
    """Preserve answer/search_query/display-source keys used by existing callers."""
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 2000:
        raise ValueError("Question must contain 1–2000 characters")
    language_instruction(response_language)
    question = question.strip()
    reply = _conversation(question, session, chat_history, response_language)
    if reply is not None:
        return reply
    settings = Settings.from_env()
    if settings.pipeline == "legacy":
        # Explicit migration mode lets existing deployments retain their working
        # collection while new indexes and human-reviewed evaluations are staged.
        # This mode makes no upgraded-grounding or verified-citation claim.
        from src.chat_service import history_messages
        from src.security import rate_limit, redact, suspicious
        history = history_messages(chat_history)
        store = Store(settings.state_dir / "control.sqlite3")
        monitor = Monitor(store, settings)
        rate_limit(store, session, settings.requests_per_minute)
        started = time.perf_counter()
        client = OllamaClient(settings)
        if suspicious(question):
            answer = ChatService.abstain("unsafe_request", response_language)
            intent = "BLOCKED"
        else:
            from src.ollama_client import InvalidModelOutput, ModelUnavailable
            try:
                intent = classify_query(client, settings.utility_model, redact(question), history)
                answer = response_without_retrieval(intent, question, response_language)
            except (InvalidModelOutput, ModelUnavailable):
                # An unavailable filter must not trigger blind retrieval.
                intent = "UNAVAILABLE"
                answer = ChatService.abstain("intent_filter_unavailable", response_language)
        if answer is not None:
            result = {**answer.model_dump(), "search_query": "", "sources": [], "route": intent.lower(),
                      "retrieval_required": False, "retrieval_performed": False,
                      "verification": {"passed": None, "reason": "no_policy_answer_generated"}}
        else:
            try:
                from src.legacy_rag import ask_shopassist as legacy_ask
                from langchain_core.messages import AIMessage, HumanMessage
                legacy_history = [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]) for m in history]
                result = legacy_ask(question, legacy_history, intent_checked=True, response_language=response_language)
                result.update(route="knowledge", retrieval_required=True, retrieval_performed=True,
                              needs_human_support=False,
                              verification={"passed": False, "reason": "legacy_pipeline_not_verified"})
            except Exception as exc:
                logger.error("Legacy answer failed: %s", type(exc).__name__)
                result = {**ChatService.abstain("service_unavailable", response_language).model_dump(),
                          "route": "knowledge", "search_query": question,
                          "retrieval_required": True, "retrieval_performed": None}
        result.update(pipeline="legacy", confidence=None, latency_seconds=time.perf_counter() - started)
        result["query_id"] = monitor.trace(session, {"question": question, **result})
        return _escalate(question, chat_history, session, result, response_language)
    result = ask_structured(question, chat_history, session, response_language=response_language)
    citations = result["sources"]
    filenames = {c["id"]: c["metadata"].get("source", c["metadata"]["document_id"])
                 for c in result.get("retrieved_chunks", [])}
    displayed = [f"{filenames.get(s['chunk_id'], s['document_id'])}, page {s['page_number']}"
                 if s["page_number"] is not None else filenames.get(s["chunk_id"], s["document_id"]) for s in citations]
    return {**result, "sources": displayed, "citations": citations}


def submit_feedback(query_id, kind, comment="", session="local"):
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    return Monitor(store, settings).feedback(query_id, kind, comment, session)


def faq_suggestions(limit=5):
    from src.faq_catalog import published_suggestions
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    return published_suggestions(store, limit)


def request_human_support(question, history=None, session="local"):
    from src.chat_service import history_messages
    from src.tickets import TicketService
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    return TicketService(store, settings).open(question, {}, history_messages(history), session, "user_requested")
