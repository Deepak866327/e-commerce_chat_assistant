"""Dense/BM25 fusion over active versions and published, reviewed FAQs only."""

from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
import math
import re
import time

from src.chunking import token_count
from src.faq_catalog import published_faqs
from src.security import suspicious


def terms(text):
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def bm25(query, rows, k=20):
    tokenized = [terms(r["text"]) for r in rows]
    if not tokenized:
        return []
    frequencies = [Counter(tokens) for tokens in tokenized]
    lengths = [len(tokens) for tokens in tokenized]
    average = sum(lengths) / len(lengths) or 1
    document_frequency = Counter(t for f in frequencies for t in f)
    scores = []
    for row, frequency, length in zip(rows, frequencies, lengths):
        score = 0.0
        for term in set(terms(query)):
            tf = frequency.get(term, 0)
            df = document_frequency.get(term, 0)
            inverse = math.log(1 + (len(rows) - df + .5) / (df + .5))
            score += inverse * (tf * 2.5) / (tf + 1.5 * (.25 + .75 * length / average))
        if score > 0:
            scores.append((row["id"], score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:k]


def rrf(rankings, constant=60):
    fused = Counter()
    for ranking in rankings:
        seen = set()
        for rank, (chunk_id, _) in enumerate(ranking, 1):
            if chunk_id not in seen:
                fused[chunk_id] += 1 / (constant + rank)
                seen.add(chunk_id)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


@lru_cache(maxsize=1)
def _cross_encoder(model, allow_download):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model, device="cpu", local_files_only=not allow_download, max_length=512)


class BGEReranker:
    def __init__(self, settings):
        self.model = _cross_encoder(settings.reranker_model, settings.allow_model_download)
        self.version = settings.reranker_model

    def rank(self, query, candidates):
        scores = self.model.predict([(query, c["text"]) for c in candidates], batch_size=8, show_progress_bar=False)
        return [(c["id"], float(score)) for c, score in zip(candidates, scores)]


class Retriever:
    def __init__(self, store, settings, vector=None, reranker=None, candidate_index=None,
                 vector_factory=None, reranker_factory=None):
        self.store, self.settings = store, settings
        self.vector, self.reranker = vector, reranker
        self.candidate_index = candidate_index
        self.vector_factory, self.reranker_factory = vector_factory, reranker_factory

    def corpus(self):
        index_id = self.candidate_index or self.store.setting("active_index")
        if not index_id:
            return []
        if self.candidate_index:
            # Candidate evaluation uses latest complete versions without making
            # them available to customers. Exactly one version per document.
            rows = self.store.rows("""SELECT c.* FROM chunks c JOIN index_members m ON m.chunk_id=c.id
                JOIN document_versions v ON v.id=c.version_id JOIN documents d ON d.id=c.document_id
                WHERE m.index_id=? AND v.status='INDEXED' AND d.deleted=0 AND v.version=(
                  SELECT MAX(v2.version) FROM document_versions v2 JOIN chunks c2 ON c2.version_id=v2.id
                  JOIN index_members m2 ON m2.chunk_id=c2.id WHERE v2.document_id=d.id
                  AND v2.status='INDEXED' AND m2.index_id=?)""", (index_id, index_id))
        else:
            rows = self.store.active_chunks(index_id)
        for row in rows:
            row["metadata"] = json.loads(row["metadata"])
        return [r for r in rows if not suspicious(r["text"])]

    def search(self, query, hybrid=True, rerank=True, metadata_filter=None):
        started = time.perf_counter()
        rows = self.corpus()
        if metadata_filter:
            if not set(metadata_filter).issubset({"document_id", "kind", "source", "heading", "page_number"}):
                raise ValueError("Unsupported metadata filter")
            rows = [r for r in rows if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())]
        by_id = {r["id"]: r for r in rows}
        if by_id and self.vector is None and self.vector_factory:
            self.vector = self.vector_factory()
        dense = self.vector.search(query, self.settings.dense_k, set(by_id)) if self.vector and by_id else []
        dense = [(cid, score) for cid, score in dense if cid in by_id and score >= self.settings.min_dense_similarity]
        keyword = bm25(query, rows, self.settings.keyword_k) if hybrid else []
        # FAQ question variants improve lexical recall but always resolve back
        # to the official chunk. Draft/rejected/outdated FAQs never enter search.
        if hybrid and rows:
            faq_search = [{"id": r["chunk_id"], "text": r["body"]["question"] + " " + r["body"]["answer"]}
                          for r in published_faqs(self.store) if r["chunk_id"] in by_id]
            keyword = rrf([keyword, bm25(query, faq_search, self.settings.keyword_k)], self.settings.rrf_k)[:self.settings.keyword_k]
        fused = rrf([dense, keyword], self.settings.rrf_k) if hybrid else dense
        dense_scores, keyword_scores = dict(dense), dict(keyword)
        candidates, fingerprints = [], set()
        for chunk_id, score in fused:
            row = by_id[chunk_id]
            fingerprint = sha256(" ".join(row["text"].split()).encode()).hexdigest()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            candidates.append({**row, "retrieval_score": score, "dense_score": dense_scores.get(chunk_id),
                               "keyword_score": keyword_scores.get(chunk_id), "reranker_score": None})
            if len(candidates) >= self.settings.fusion_k:
                break
        retrieval_seconds = time.perf_counter() - started
        rerank_start = time.perf_counter()
        if rerank and candidates and self.reranker is None and self.reranker_factory:
            self.reranker = self.reranker_factory()
        if rerank and self.reranker and candidates:
            scores = dict(self.reranker.rank(query, candidates))
            for candidate in candidates:
                candidate["reranker_score"] = scores.get(candidate["id"], float("-inf"))
            candidates.sort(key=lambda c: c["reranker_score"], reverse=True)
        rerank_seconds = time.perf_counter() - rerank_start
        context, used = [], 0
        for candidate in candidates:
            cost = token_count(candidate["text"])
            if used + cost <= self.settings.context_tokens:
                context.append(candidate)
                used += cost
            if len(context) == self.settings.context_k:
                break
        return {"chunks": context, "candidates": candidates, "retrieval_seconds": retrieval_seconds,
                "reranking_seconds": rerank_seconds, "dense_available": self.vector is not None,
                "index_id": self.candidate_index or self.store.setting("active_index")}
