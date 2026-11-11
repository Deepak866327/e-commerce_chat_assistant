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
        return [redact(value[:300]) for value in values[-6:]
                if isinstance(value, str) and value.strip() and not suspicious(value)]
    reference = context.get("ticket_reference")
    topic = context.get("issue_type")
    return {"issue_type": topic if isinstance(topic, str) and topic in TOPICS else None,
            "customer_reported": reports("customer_reported"), "steps_tried": reports("steps_tried"),
            "resolution": "customer_reported_resolved" if context.get("resolution") == "customer_reported_resolved" else "unknown",
            "ticket_reference": reference if isinstance(reference, str) and re.fullmatch(r"SA-[A-F0-9]{16}", reference) else None}


def remember_customer(context, question):
    state = clean_context(context)
    if not isinstance(question, str) or suspicious(question):
        return state
    normalized = " ".join(question.casefold().split()).rstrip(".!")
    topic = next((name for name, pattern in TOPICS.items() if re.search(pattern, normalized)), None)
    # General order mentions must not overwrite a more specific active issue.
    if topic == "delivery" and state["issue_type"] and not re.search(r"delivery|tracking|package|parcel|डिलीवरी|पार्सल", normalized):
        topic = state["issue_type"]
    if topic and topic != state["issue_type"]:
        state = clean_context({"issue_type": topic})
    if state["issue_type"] and normalized:
        report = redact(question[:300])
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
    state = clean_context(context)
    ticket = result.get("ticket") or {}
    if ticket.get("reference"):
        state["ticket_reference"] = ticket["reference"]
    return clean_context(state)
