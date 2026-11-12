"""Rank current FAQs; semantic scores are similarities, not probabilities."""

from functools import lru_cache
import math
import re
import unicodedata

from src.config import Settings
from src.security import suspicious


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


@lru_cache(maxsize=2)
def encoder_for(settings):
    try:
        from src.vector_index import WindowEmbeddings
        return WindowEmbeddings(settings)
    except Exception:
        # Keep FAQ reading/search operational without inference dependencies.
        # Missing models are retried after application restart.
        return None


@lru_cache(maxsize=8)
def article_vectors(settings, records):
    encoder = encoder_for(settings)
    return encoder.encode([question for _, _, question in records]) if encoder else None


def ranked_articles(articles, query, *, settings=None, encoder=None):
    query = query.strip()[:500]
    if not query:
        return [(article, 0.0, "browse") for article in articles]
    if suspicious(query):
        return []
    settings = settings or Settings.from_env()
    normalized = normalize(query)
    if not normalized:
        return []
    lexical = {a["faq_id"] for a in articles if all(
        word in normalize(a["question"] + " " + a["answer"] + " " + a["category"])
        for word in normalized.split())}
    vectors = None
    query_vector = None
    try:
        active_encoder = encoder or encoder_for(settings)
        if active_encoder and articles:
            records = tuple((a["faq_id"], a.get("version", ""), a["question"]) for a in articles)
            vectors = active_encoder.encode([a["question"] for a in articles]) if encoder else article_vectors(settings, records)
            query_vector = active_encoder.encode([query])[0]
    except Exception:
        pass
    ranked = []
    for index, article in enumerate(articles):
        score, method = 0.0, "keyword"
        if normalize(article["question"]) == normalized:
            score, method = 1.0, "exact"
        elif vectors is not None and query_vector is not None:
            vector = vectors[index]
            norm = math.sqrt(sum(float(v) ** 2 for v in vector) * sum(float(v) ** 2 for v in query_vector))
            score = sum(float(a) * float(b) for a, b in zip(vector, query_vector)) / norm if norm else 0.0
            method = "semantic"
        if method == "exact" or article["faq_id"] in lexical or score >= settings.faq_search_min_similarity:
            ranked.append((article, score, method))
    return sorted(ranked, key=lambda item: (item[2] == "exact", item[1]), reverse=True)


def clear_match(articles, query, *, settings=None, encoder=None):
    settings = settings or Settings.from_env()
    if suspicious(query):
        return None
    exact = [a for a in articles if normalize(a["question"]) == normalize(query)]
    if exact:
        return exact[0] if len(exact) == 1 else None
    if not settings.faq_semantic_answers:
        return None
    # Personal qualifiers and failures require contextual policy retrieval, even
    # when a generic FAQ has a high embedding similarity.
    if re.search(r"\b(?:not|no|cannot|can t|after|before|already|failed|fails|but|missing|delayed|charged)\b|\d", normalize(query)):
        return None
    ranked = ranked_articles(articles, query, settings=settings, encoder=encoder)
    if not ranked:
        return None
    article, score, method = ranked[0]
    if method == "exact":
        # Duplicate canonical questions are ambiguous even with an exact match.
        return article if sum(normalize(a["question"]) == normalize(query) for a in articles) == 1 else None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if article.get("kind") == "published" and score >= settings.faq_answer_min_similarity and score - runner_up >= settings.faq_answer_margin:
        return article
    return None
