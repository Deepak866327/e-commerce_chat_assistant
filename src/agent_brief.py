"""A factual handoff brief: reported facts are never presented as verified records."""

from src.security import redact_value
from src.support_context import clean_context


def agent_brief(question, result):
    context = clean_context(result.get("support_context"))
    summary = result.get("resolution_summary") or {}
    guidance = summary.get("guidance", [])
    sources = list(result.get("citations", result.get("sources", [])))
    for item in guidance[-3:]:
        for source in item.get("sources", []) if isinstance(item, dict) else []:
            if source not in sources:
                sources.append(source)
    return redact_value({
        "customer_goal": (context["customer_reported"] or [question])[0][:300],
        "issue_type": context["issue_type"] or "unspecified",
        "reported_facts": context["reported_facts"],
        "facts_are_customer_reported": True,
        "customer_steps_tried": context["steps_tried"],
        "guidance_shown": [item.get("answer", "")[:500] for item in guidance[-3:] if isinstance(item, dict)],
        "unresolved_blocker": context["reported_facts"].get("blocker") or
                              (context["customer_reported"][-1] if context["customer_reported"] else question)[:300],
        "faq_already_shown": context.get("faq"),
        "handoff_reason": result.get("reason") or "customer_requested",
        "latest_customer_update": summary.get("latest_customer_update") or question[:500],
        "policy_sources": sources[:8],
        "backend_access": False,
        "completed_backend_actions": [],
    })
