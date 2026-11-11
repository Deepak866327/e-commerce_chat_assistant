"""Conservative whole-message routing for turns that need no policy evidence.

Never match a greeting prefix alone: 'hi, how do I get a refund?' must continue
to the knowledge pipeline. Ambiguous short follow-ups such as 'yes', 'and then?'
or 'what about it?' also continue so conversation context can resolve them.
"""

import re
import unicodedata

from src.language import message, language_instruction


def conversational_reply(question, response_language="hinglish"):
    language_instruction(response_language)
    reply_key = None
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 2000:
        raise ValueError("Question must contain 1–2000 characters")
    normalized = unicodedata.normalize("NFKC", question).casefold().strip()
    # Keep combining marks in Hindi and other scripts; Python's \w excludes
    # vowel signs and viramas, which would break whole-word greeting matches.
    normalized = "".join(character if character.isspace() or unicodedata.category(character)[0] in {"L", "M", "N"}
                         else " " for character in normalized)
    normalized = " ".join(normalized.split())
    if re.fullmatch(r"(?:hi|hii|hiii|hello|hey|namaste|namaskar|नमस्ते|नमस्कार|"
                    r"good morning|good afternoon|good evening)(?: (?:there|bot|shopassist|ji))?", normalized):
        route = "greeting"
    elif normalized in {"how are you", "hi how are you", "hello how are you", "hey how are you",
                        "kaise ho", "aap kaise hain"}:
        route = "small_talk"
    elif normalized in {"thanks", "thank you", "thanks a lot", "thank you very much",
                        "thank you for your help", "thanks for your help", "ok thanks", "okay thanks",
                        "dhanyavaad", "dhanyavad", "shukriya", "धन्यवाद", "शुक्रिया"}:
        route = "thanks"
    elif normalized in {"bye", "goodbye", "good bye", "see you", "alvida", "अलविदा"}:
        route = "farewell"
    elif normalized in {"who are you", "what can you do", "how can you help me",
                        "what do you do", "what can you help with", "tum kaun ho", "aap kaun ho"}:
        route = "capabilities"
    elif normalized in {"human support", "talk to a human", "talk to human", "human agent", "speak to a human",
                        "connect me to a human", "create a ticket", "raise a ticket", "raise ticket", "agent please",
                        "talk to an agent", "speak to an agent", "real person", "customer support agent",
                        "connect me to support", "connect me with support", "speak to support", "talk to support",
                        "contact support team", "live agent", "customer care executive",
                        "mujhe customer care se baat karni hai", "mujhe agent se baat karni hai",
                        "mujhe human se baat karni hai", "support se baat karni hai",
                        "मुझे कस्टमर केयर से बात करनी है", "मुझे एजेंट से बात करनी है"}:
        route = "human_support"
    elif normalized in {"i do not want to talk to a human", "i don t want to talk to a human",
                        "i do not want a human agent", "no human agent", "no agent please",
                        "mujhe agent se baat nahi karni", "मुझे एजेंट से बात नहीं करनी"}:
        route = "clarification"
        reply_key = "declined_handoff"
    elif normalized in {"help", "help me", "i need help", "can you help", "can you help me",
                        "i need support", "mujhe help chahiye", "mujhe madad chahiye"}:
        route = "clarification"
    else:
        return None
    key = reply_key or route
    answer = message(key, response_language)
    return {"answer": answer, "answerable": route != "human_support", "sources": [], "citations": [],
            "needs_human_support": route == "human_support",
            "reason": "explicit_handoff" if route == "human_support" else None, "route": route,
            "retrieval_required": False, "retrieval_performed": False,
            "search_query": "", "query_id": None, "confidence": None,
            "verification": {"passed": None, "reason": "non_factual_conversation"},
            "retrieved_chunks": [], "retrieval_candidates": [],
            "retrieval_seconds": 0.0, "reranking_seconds": 0.0, "inference_seconds": 0.0,
            "error": None, "prompt_version": "conversation-routes-v2", "model_version": None}
