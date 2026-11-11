"""Stable chat entry points. Model/vector loading happens only on a request."""

from functools import lru_cache
import logging

from src.chat_service import ChatService, MAX_HISTORY_MESSAGES
from src.config import Settings
from src.observability import Monitor
from src.ollama_client import OllamaClient
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
    vector = _vector(settings, active_index) if active_index else None
    reranker = BGEReranker(settings) if settings.use_reranker and active_index else None
    client = OllamaClient(settings)
    retriever = Retriever(store, settings, vector, reranker)
    return ChatService(store, settings, client, retriever,
                       GroundingVerifier(client, settings.verifier_model), monitor)


def ask_structured(question, chat_history=None, session="local"):
    """Public structured contract. Citation sources are verified objects."""
    try:
        return runtime().ask(question, chat_history, session)
    except IndexUnavailable:
        # Keep the UI operational when models/indexes need installation. Do not
        # serve a different model/dimension behind the active collection name.
        from src.chat_service import ABSTENTION
        return {"answer": ABSTENTION, "answerable": False, "sources": [],
                "needs_human_support": True, "reason": "index_unavailable", "search_query": question,
                "query_id": None, "verification": {"passed": False}, "confidence": None}


def ask_shopassist(question, chat_history, session="local"):
    """Preserve answer/search_query/display-source keys used by existing callers."""
    settings = Settings.from_env()
    if settings.pipeline == "legacy":
        # Explicit migration mode lets existing deployments retain their working
        # collection while new indexes and human-reviewed evaluations are staged.
        # This mode makes no upgraded-grounding or verified-citation claim.
        from src.legacy_rag import ask_shopassist as legacy_ask
        from langchain_core.messages import AIMessage, HumanMessage
        from src.chat_service import history_messages
        history = [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
                   for m in history_messages(chat_history)]
        store = Store(settings.state_dir / "control.sqlite3")
        monitor = Monitor(store, settings)
        from src.security import rate_limit
        rate_limit(store, session, settings.requests_per_minute)
        result = legacy_ask(question, history)
        result.update(pipeline="legacy", confidence=None, needs_human_support=False,
                      verification={"passed": False, "reason": "legacy_pipeline_not_verified"})
        result["query_id"] = monitor.trace(session, {"question": question, **result})
        return result
    result = ask_structured(question, chat_history, session)
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
