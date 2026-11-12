"""Conservative fast routes for explicit policy/procedure questions.

This decides whether to retrieve evidence; it never supplies policy facts.
Ambiguous, combined action requests and human handoffs still use the model.
"""

import re
import unicodedata

from src.security import suspicious


def explicit_handoff(question):
    text = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if suspicious(text) or re.search(r"\b(?:not|don't|don’t|no|never|nahi|nahin)\b|नहीं", text):
        return False
    return bool(re.search(r"(?:^|[.!?;]\s*)(?:please )?(?:(?:can|could|would|will) (?:you|i) )?"
                          r"(?:connect me (?:to|with)|(?:talk|speak) to|i (?:want|need|would like) (?:a|an)|i (?:want|need|would like) to (?:talk|speak) to) "
                          r"(?:a |an |the )?(?:human|(?:support |customer support )?agent|support team)\b|"
                          r"(?:^|[.!?;]\s*)(?:please )?(?:raise|create|open) (?:a |the )?(?:support )?ticket\b|"
                          r"(?:^|[.!?;]\s*)(?:please )?escalate(?: (?:this|my issue|my request))? to (?:a |an |the )?(?:human|agent|support team)\b|"
                          r"^(?:i need|i want|get me) (?:a |an )?(?:human|support agent)\b|"
                          r"mujhe.*(?:agent|customer care|human).*baat karni|"
                          r"मुझे.*(?:एजेंट|मानव|कस्टमर केयर).*बात करनी", text))


def deterministic_intent(question):
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 2000 or suspicious(question):
        return None
    text = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if explicit_handoff(question):
        return "HUMAN_SUPPORT"
    text = re.sub(r"^(?:hi|hello|hey|namaste)[,!:.\s]+", "", text)
    if re.search(r"\b(?:python|javascript|programming|api|function)\b", text):
        return None
    # A declined handoff followed by an actual policy question is still knowledge.
    text = re.sub(r"^(?:i (?:do not|don't|don’t) want (?:an? )?(?:human|agent)|no agent please)[,;:.]\s*", "", text)
    if re.search(r"\b(?:human|agent|representative|support team|customer care)\b|एजेंट|मानव|कस्टमर केयर", text):
        return None
    # Do not swallow a second request such as 'How do I cancel? Do it for me'.
    if re.search(r"\b(?:can|could|would|will) you (?:please )?(?:cancel|refund|process|issue|check|track|change|expedite)\b|"
                 r"\bplease (?:cancel|refund|process|issue|check|track|change|expedite)\b|"
                 r"[.!?;]\s*(?:cancel|refund|process|issue|check|track|change|expedite)\b|"
                 r"\b(?:do it for me|cancel it (?:now|for me)|expedite)\b|कर दो|कर दीजिए", text):
        return None
    topic = r"\b(?:cancel(?:lation)?|return(?:s)?|refund(?:s)?|shipping|delivery|warranty|exchange(?:s)?|tracking)\b"
    if re.search(topic, text):
        procedural = (re.match(r"^(?:how (?:can|do|should|could) i|how to|can i|am i (?:allowed|eligible) to)\b", text)
                      and re.search(r"\b(?:order|purchase|item|product|refund|shipping|delivery|warranty)\b", text))
        policy = (re.search(r"\b(?:policy|policies|procedure|eligibility|return window|shipping charges|shipping fees)\b", text)
                  and re.match(r"^(?:what|which|how|when|is|are|do|does|explain|tell me|can i)\b", text))
        timing = re.match(r"^how (?:long|much)\b", text) and re.search(r"\b(?:take|cost|fee|charge|period|duration)\b", text)
        if procedural or policy or timing:
            return "KNOWLEDGE"
    hindi_topic = re.search(r"ऑर्डर|आर्डर|रिफंड|वापसी|वारंटी|डिलीवरी|रद्द|कैंसल", text)
    if hindi_topic and (re.search(r"कैसे|नीति|कितने|कितना|प्रक्रिया", text)
                        or text.startswith("क्या ") and re.search(r"सकता|सकती|सकते", text)):
        return "KNOWLEDGE"
    if re.search(topic, text) and re.search(r"\b(?:kaise|policy kya|kitne din|kya main)\b", text):
        return "KNOWLEDGE"
    return None
