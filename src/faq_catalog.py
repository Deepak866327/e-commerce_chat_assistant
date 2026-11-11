"""Public suggestions contain only questions from reviewed, published FAQs."""

import json

from src.security import suspicious


def published_suggestions(store, limit=5):
    if not 1 <= limit <= 10:
        raise ValueError("Choose 1–10 FAQ suggestions")
    rows = store.rows("""SELECT f.id,f.body FROM faqs f JOIN documents d ON d.id=f.document_id
        WHERE f.status='PUBLISHED' AND f.reviewer IS NOT NULL AND d.deleted=0
        AND d.active_version=f.version_id ORDER BY f.updated_at DESC,f.id LIMIT 100""")
    suggestions = []
    for row in rows:
        body = json.loads(row["body"])
        question = body.get("question")
        if body.get("verified") is True and isinstance(question, str) and question.strip() and not suspicious(question):
            suggestions.append({"faq_id": row["id"], "question": question})
        if len(suggestions) == limit:
            break
    return suggestions
