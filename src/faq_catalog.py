"""One publication gate for customer suggestions, retrieval and FAQ exports."""

import json

from src.security import suspicious


def published_faqs(store):
    rows = store.rows("""SELECT f.id,f.chunk_id,f.body FROM faqs f
        JOIN documents d ON d.id=f.document_id
        JOIN chunks c ON c.id=f.chunk_id AND c.document_id=f.document_id AND c.version_id=f.version_id
        WHERE f.status='PUBLISHED' AND f.reviewer IS NOT NULL AND TRIM(f.reviewer)<>'' AND d.deleted=0
        AND d.active_version=f.version_id ORDER BY f.updated_at DESC,f.id""")
    published = []
    for row in rows:
        try:
            body = json.loads(row["body"])
        except (ValueError, TypeError):
            continue
        if not isinstance(body, dict) or body.get("verified") is not True:
            continue
        question, answer = body.get("question"), body.get("answer")
        if not all(isinstance(text, str) and text.strip() and not suspicious(text)
                   for text in (question, answer)):
            continue
        published.append({"id": row["id"], "chunk_id": row["chunk_id"], "body": body})
    return published


def published_suggestions(store, limit=5):
    if not 1 <= limit <= 10:
        raise ValueError("Choose 1–10 FAQ suggestions")
    return [{"faq_id": row["id"], "question": row["body"]["question"]}
            for row in published_faqs(store)[:limit]]


def customer_faqs(store):
    """Public help articles share the same current-source gate as retrieval."""
    articles = []
    for row in published_faqs(store):
        body = row["body"]
        category = body.get("category")
        if not isinstance(category, str) or not category.strip() or suspicious(category):
            category = "General"
        articles.append({"faq_id": row["id"], "question": body["question"],
                         "answer": body["answer"], "category": category.strip()})
    return articles
