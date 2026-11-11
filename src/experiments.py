"""Run the preserved baseline and upgraded variants on identical reviewed cases."""

from hashlib import sha256
import json
import time

from src.chat_service import ChatService
from src.ollama_client import OllamaClient
from src.retrieval import BGEReranker, Retriever
from src.vector_index import ChromaIndex
from src.verification import GroundingVerifier


def upgraded_runner(store, settings, variant, index_id):
    if variant not in {"B", "C", "D", "E"}:
        raise ValueError("Use legacy_runner for variant A")
    vector = ChromaIndex(settings, index_id)
    reranker = BGEReranker(settings) if variant in {"D", "E"} else None

    def run(case):
        client = OllamaClient(settings)
        service = ChatService(store, settings, client,
            Retriever(store, settings, vector, reranker, candidate_index=index_id),
            GroundingVerifier(client, settings.verifier_model))
        return service.ask(case.question, case.history, "evaluation", hybrid=variant in {"C", "D", "E"},
                           rerank=variant in {"D", "E"}, verify=variant == "E", record=False)
    run.model_digest = OllamaClient(settings).model_digest(settings.answer_model)
    return run


def legacy_runner(store):
    # Ground truth has v2 chunk IDs. Exact text containment maps old chunks to
    # their v2 reference IDs. Unmappable chunks keep distinct IDs and count as
    # misses; no document-wide shortcut inflates baseline recall.
    from src.legacy_rag import _initialize, ask_shopassist
    from langchain_core.messages import AIMessage, HumanMessage
    db, _, _, _, _ = _initialize()
    reference = store.rows("SELECT id,text,metadata FROM chunks")

    def run(case):
        history = [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"]) for m in case.history]
        started = time.perf_counter()
        result = ask_shopassist(case.question, history)
        documents = db.similarity_search(result["search_query"], k=3)
        chunks = []
        for document in documents:
            text = " ".join(document.page_content.split())
            matches = [r for r in reference if " ".join(r["text"].split()) in text]
            if matches:
                for row in matches:
                    chunks.append({"id": row["id"], "text": row["text"], "metadata": json.loads(row["metadata"])})
            else:
                chunk_id = "LEGACY_" + sha256(text.encode()).hexdigest()[:24]
                chunks.append({"id": chunk_id, "text": text, "metadata": {
                    "document_id": "legacy", "page_number": document.metadata.get("page", 0) + 1}})
        return {"answer": result["answer"], "answerable": bool(result["sources"]), "sources": [],
                "retrieved_chunks": chunks, "retrieval_candidates": chunks, "error": None,
                "latency_seconds": time.perf_counter() - started,
                "citation_note": "Legacy labels are retrieved pages, not validated answer citations"}
    return run


def chunk_benchmark(sections, settings, counter=None):
    """Offline chunk statistics only. Retrieval quality requires reviewed runs."""
    from src.chunking import chunk_sections, token_count
    results = []
    counter = counter or token_count
    for size in (400, 500, 700):
        started = time.perf_counter()
        chunks = chunk_sections(sections, "BENCH", "BENCH", "benchmark", size, min(75, size // 5), counter)
        results.append({"chunk_budget": size, "tokenizer": "embedding-model" if counter is not token_count else "utf8-byte-conservative",
                        "overlap": min(75, size // 5), "chunks": len(chunks),
                        "average_chunk_tokens": sum(counter(c.text) for c in chunks) / len(chunks) if chunks else 0,
                        "chunking_seconds": time.perf_counter() - started,
                        "retrieval_quality": None, "quality_note": "Not a retrieval benchmark; requires reviewed ground truth"})
    return results
