"""Bounded customer-reported issue context, separate from verified policy evidence."""

import re

from src.security import redact, suspicious

TOPICS = {
    "refund": r"\brefund\b|रिफंड|पैसे वापस",
    "cancellation": r"\bcancel(?:lation)?\b|कैंसल|रद्द",
    "payment": r"\b(?:payment|transaction|debited)\b|भुगतान|पेमेंट",
    "returns": r"\b(?:return|exchange|replacement)\b|वापसी|बदलना",
    "warranty": r"\bwarranty\b|वारंटी",
    "delivery": r"\b(?:order|package|parcel|delivery|tracking)\b|ऑर्डर|डिलीवरी|पार्सल",
}


def clean_context(context):
    context = context if isinstance(context, dict) else {}
    def reports(key):
        values = context.get(key, [])
        if not isinstance(values, list):
            return []
        return [redact(value)[:300] for value in values[-6:]
                if isinstance(value, str) and value.strip() and not suspicious(value)]
    reference = context.get("ticket_reference")
    topic = context.get("issue_type")
    facts = context.get("reported_facts", {})
    facts = {key: redact(value)[:300] for key, value in facts.items()
             if key in {"dispatch_status", "refund_approval", "approval_date", "payment_error", "charged_status",
                        "tracking_update", "expected_delivery", "item_problem", "delivery_date", "product_problem",
                        "purchase_date", "steps_attempted", "blocker", "issue_details", "channel", "order_confirmation"}
             and isinstance(value, str) and not suspicious(value)} if isinstance(facts, dict) else {}
    faq = context.get("faq")
    if isinstance(faq, dict):
        faq = {key: redact(faq[key])[:limit] for key, limit in
               {"faq_id": 100, "version": 64, "question": 500, "answer": 2000, "category": 100,
                "kind": 30, "source_version": 100, "chunk_id": 100, "outcome": 20}.items()
               if isinstance(faq.get(key), str) and not suspicious(faq[key])}
        if not all(faq.get(key) for key in ("faq_id", "version", "question", "answer")):
            faq = None
    else:
        faq = None
    return {"issue_type": topic if isinstance(topic, str) and topic in TOPICS else None,
            "customer_reported": reports("customer_reported"), "steps_tried": reports("steps_tried"),
            "resolution": "customer_reported_resolved" if context.get("resolution") == "customer_reported_resolved" else "unknown",
            "ticket_reference": reference if isinstance(reference, str) and re.fullmatch(r"SA-[A-F0-9]{16}", reference) else None,
            "reported_facts": facts, "assistant_attempts": reports("assistant_attempts")[-3:], "faq": faq}


def remember_customer(context, question):
    state = clean_context(context)
    if not isinstance(question, str) or suspicious(question):
        return state
    normalized = " ".join(question.casefold().split()).rstrip(".!")
    topic = next((name for name, pattern in TOPICS.items() if re.search(pattern, normalized)), None)
    new_issue = bool(re.search(r"\b(?:new (?:issue|problem|order|purchase)|another (?:refund|order|issue)|different (?:order|issue))\b", normalized))
    # General order mentions must not overwrite a more specific active issue.
    if topic == "delivery" and state["issue_type"] and not new_issue and not re.search(r"delivery|tracking|package|parcel|डिलीवरी|पार्सल", normalized):
        topic = state["issue_type"]
    if topic and (topic != state["issue_type"] or new_issue or state["resolution"] == "customer_reported_resolved"):
        state = clean_context({"issue_type": topic})
    if state["issue_type"] and normalized:
        report = redact(question)[:300]
        if report not in state["customer_reported"]:
            reports = state["customer_reported"] + [report]
            state["customer_reported"] = reports[:2] + reports[-4:] if len(reports) > 6 else reports
        if re.search(r"\b(?:tried|already|attempted)\b|कोशिश|पहले ही", normalized):
            state["steps_tried"] = (state["steps_tried"] + [report])[-6:]
    if normalized in {"it worked", "issue resolved", "it is resolved", "problem solved",
                      "ho gaya", "problem solve ho gayi", "समस्या हल हो गई", "ठीक हो गया"}:
        state["resolution"] = "customer_reported_resolved"
    elif topic:
        state["resolution"] = "unknown"
    return state


def remember_result(context, result):
    state = clean_context(result.get("support_context") or context)
    ticket = result.get("ticket") or {}
    if ticket.get("reference"):
        state["ticket_reference"] = ticket["reference"]
    return clean_context(state)
