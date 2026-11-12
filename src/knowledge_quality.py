"""Review queues for missing evidence; never turn failed answers into policy."""

import json

from src.security import redact_value


def knowledge_gaps(store, principal, limit=30):
    principal.require("review")
    missing = {"insufficient_evidence", "no_relevant_evidence", "unsupported_policy_numbers", "verification_failed",
               "citation_validation_failed", "grounding_failed", "no_validated_knowledge", "insufficient_retrieval"}
    gaps, seen = [], set()
    for row in store.rows("SELECT body FROM traces ORDER BY created_at DESC LIMIT 1000"):
        body = json.loads(row["body"])
        reason = body.get("failure_reason") or body.get("reason")
        if reason not in missing:
            continue
        summary = body.get("resolution_summary") or {}
        reports = summary.get("customer_reported") or []
        question = reports[0] if reports else body.get("question", "")
        key = (question.casefold(), reason)
        if key in seen:
            continue
        seen.add(key)
        gaps.append(redact_value({"customer_question": question, "blocker": summary.get("facts", {}).get("blocker", ""),
                                  "reason": reason, "retrieval_query": body.get("search_query", "")}))
        if len(gaps) >= limit:
            break
    return gaps
