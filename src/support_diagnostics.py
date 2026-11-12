"""Deterministic customer-reported facts and no-CRM capability boundaries.

This module contains no eligibility, refund timeline or store procedure claims.
"""

import re

from src.security import redact, suspicious


def reported_details(question, topic):
    text = " ".join(question.casefold().replace("’", "'").split())
    if suspicious(question) or re.search(r"\b(?:what if|suppose|assuming|if my|if the)\b", text):
        return {}
    report = redact(question)[:250]
    facts = {}
    if re.search(r"(?:not|hasn't|has not|isn't|is not) (?:yet )?(?:been )?(?:dispatched|shipped)|(?:dispatch|ship) nahi", text):
        facts["dispatch_status"] = "not dispatched; customer said: " + report
    elif re.search(r"(?:has been|already|was|is) (?:dispatched|shipped)", text):
        facts["dispatch_status"] = "dispatched; customer said: " + report
    if topic == "refund":
        if re.search(r"refund.*(?:not|hasn't been|has not been|isn't) approved", text):
            facts["refund_approval"] = "not approved; customer said: " + report
        elif re.search(r"refund.*(?:was|is|has been|got) approved", text):
            facts["refund_approval"] = "approved; customer said: " + report
            if re.search(r"\d+\s*(?:business )?(?:days|weeks)|\d{4}-\d{2}-\d{2}|yesterday|today|last (?:week|month)", text):
                facts["approval_date"] = report
    if topic == "payment":
        if re.search(r"(?:not|wasn't|was not|isn't) (?:been )?(?:charged|debited)|no (?:charge|deduction)", text):
            facts["charged_status"] = "not charged; customer said: " + report
        elif re.search(r"(?:was|been|got|is|were) (?:charged|debited|deducted)|money.*(?:deducted|debited)|paise kate|पैसे कट", text):
            facts["charged_status"] = "charged; customer said: " + report
        if re.search(r"(?:no|without) (?:an? )?order confirmation|order.*(?:not confirmed|unconfirmed)|confirmation.*(?:missing|not received)", text):
            facts["order_confirmation"] = "not received; customer said: " + report
        elif re.search(r"(?:received|got|have) (?:an? |the )?order confirmation", text):
            facts["order_confirmation"] = "received; customer said: " + report
        if re.search(r"error|declined|insufficient funds", text):
            facts["payment_error"] = report
    if topic == "cancellation" and re.search(r"cancel(?:lation)? (?:button|option|link).*(?:missing|not (?:there|visible|available)|can't|cannot)|(?:no|missing) cancel(?:lation)? (?:button|option|link)", text):
        facts["blocker"] = "cancellation option missing; customer said: " + report
    if re.search(r"\b(?:website|browser|desktop site|web site)\b", text):
        facts["channel"] = "website; customer said: " + report
    elif re.search(r"\b(?:mobile app|android app|iphone app|in the app|on the app)\b", text):
        facts["channel"] = "mobile app; customer said: " + report
    if topic == "returns" and re.search(r"damaged|broken|wrong item|wrong product|defective", text):
        facts["item_problem"] = report
    if topic == "delivery" and re.search(r"tracking (?:says|shows)|(?:latest|last) (?:courier |tracking )?update", text):
        facts["tracking_update"] = report
    return facts


def manual_action_requested(question, topic=None):
    text = " ".join(question.casefold().replace("’", "'").split())
    if suspicious(question) or re.search(r"\b(?:don't|do not|no need|not asking|what if|suppose|python|javascript|programming)\b", text):
        return False
    if re.match(r"^(?:how can i|how do i|how to)\b", text) and not re.search(r"[?;.!]\s*(?:please (?:cancel|process|issue)|do it for me)", text):
        return False
    if not topic and not re.search(r"\b(?:order|refund|cancel|payment|transaction|address|shipping|delivery|expedite)\b", text):
        return False
    return bool(re.search(
        r"\b(?:cancel|refund|process|issue|change|expedite)\b.*\bfor me\b|"
        r"\b(?:please )?(?:process|issue) (?:my |the |a )?refund (?:now|for me)\b|"
        r"\b(?:check|verify|look up)\b.*\b(?:in your system|in the system|live order status|bank transaction)\b|"
        r"\bdo it for me\b", text))


def next_slots(context, defaults):
    facts = context["reported_facts"]
    topic = context["issue_type"]
    if topic == "cancellation" and "cancellation option missing" in facts.get("blocker", ""):
        return ["dispatch_status", "channel", "steps_attempted"]
    if topic == "payment" and facts.get("charged_status", "").startswith("charged"):
        return ["order_confirmation", "steps_attempted", "payment_error"]
    return defaults


def diagnostic_question(context):
    facts = context["reported_facts"]
    if context["issue_type"] == "cancellation" and "cancellation option missing" in facts.get("blocker", ""):
        if "dispatch_status" not in facts:
            return "dispatch_status"
        if facts["dispatch_status"].startswith("not dispatched") and "channel" not in facts:
            return "channel"
    if context["issue_type"] == "payment" and facts.get("charged_status", "").startswith("charged") and "order_confirmation" not in facts:
        return "order_confirmation"
    return None


def situation_query(context, base):
    facts = context["reported_facts"]
    query = base
    if facts.get("refund_approval", "").startswith("not approved"):
        query = "refund request eligibility and initiation procedure"
    if "dispatch_status" in facts:
        query += " before dispatch" if facts["dispatch_status"].startswith("not dispatched") else " after dispatch"
    if "cancellation option missing" in facts.get("blocker", ""):
        query += " cancellation button option missing troubleshooting"
    if facts.get("charged_status", "").startswith("charged"):
        query += " payment deducted"
    if facts.get("order_confirmation", "").startswith("not received"):
        query += " order confirmation missing avoid duplicate payment"
    if facts.get("channel", "").startswith("website"):
        query += " website"
    elif facts.get("channel", "").startswith("mobile app"):
        query += " mobile app"
    # Retain the actual blocker without identifiers or the entire conversation.
    for key in ("approval_date", "item_problem", "tracking_update", "payment_error"):
        if facts.get(key):
            query += " " + redact(facts[key])[:150]
    return query[:900]


def general_check(context, already_shown):
    """Non-policy checks; never invent a store UI or a processing promise."""
    facts = context["reported_facts"]
    candidate = None
    if context["issue_type"] == "payment" and facts.get("charged_status", "").startswith("charged") and facts.get("order_confirmation", "").startswith("not received"):
        candidate = "general_payment_confirmation"
    elif context["issue_type"] == "cancellation" and "cancellation option missing" in facts.get("blocker", "") and "channel" in facts and facts.get("dispatch_status", "").startswith("not dispatched"):
        candidate = "general_cancellation_instructions"
    if not candidate or candidate in already_shown:
        return None
    prior = " ".join(context["steps_tried"] + context["assistant_attempts"] + [(context.get("faq") or {}).get("answer", "")]).casefold()
    if re.search(r"order (?:history|confirmation)|confirmation email|checked.*email", prior):
        return None
    return candidate


def claims_backend_action(answer):
    """Reject explicit system-access/completion claims; not a semantic proof."""
    text = answer.casefold().replace("’", "'")
    english = (r"\bi(?:'ve| have)? (?:successfully )?(?:cancelled|canceled|refunded|processed (?:your|the) refund|"
               r"checked your order|verified your payment|accessed your account)\b|"
               r"\bi can (?:check|access|verify|cancel|refund) your (?:order|account|payment)\b|"
               r"(?:^|[.!\n])\s*your (?:order|refund|payment) (?:has been|is now) (?:cancelled|canceled|processed|verified)\b")
    if re.search(english, text):
        return True
    for sentence in re.split(r"[.!?।\n]", text):
        if re.search(r"nahi|nahin|नहीं", sentence):
            continue
        if re.search(r"maine.*(?:order|refund|payment).*(?:cancel|process|verify).*(?:kar diya|kar di)|"
                     r"मैंने.*(?:ऑर्डर|रिफंड|भुगतान).*(?:रद्द|प्रोसेस|सत्यापित).*दिया", sentence):
            return True
    return False
