"""Conservative whole-message routing for turns that need no policy evidence.

Never match a greeting prefix alone: 'hi, how do I get a refund?' must continue
to the knowledge pipeline. Ambiguous short follow-ups such as 'yes', 'and then?'
or 'what about it?' also continue so conversation context can resolve them.
"""

import re
import unicodedata


def conversational_reply(question):
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
        answer = "Namaste! Main ShopAssist hoon. Aapko kis cheez mein madad chahiye?"
    elif normalized in {"how are you", "hi how are you", "hello how are you", "hey how are you",
                        "kaise ho", "aap kaise hain"}:
        route = "small_talk"
        answer = "Main madad ke liye taiyaar hoon! Bataiye, aapko kis cheez mein help chahiye?"
    elif normalized in {"thanks", "thank you", "thanks a lot", "thank you very much",
                        "thank you for your help", "thanks for your help", "ok thanks", "okay thanks",
                        "dhanyavaad", "dhanyavad", "shukriya", "धन्यवाद", "शुक्रिया"}:
        route = "thanks"
        answer = "Aapka swagat hai! Aur koi sawaal ho toh bataiye."
    elif normalized in {"bye", "goodbye", "good bye", "see you", "alvida", "अलविदा"}:
        route = "farewell"
        answer = "Alvida! Jab bhi shopping support chahiye, aap pooch sakte hain."
    elif normalized in {"who are you", "what can you do", "how can you help me",
                        "what do you do", "what can you help with", "tum kaun ho", "aap kaun ho"}:
        route = "capabilities"
        answer = ("Main ShopAssist customer support assistant hoon. Main shipping, returns, "
                  "refunds, cancellations aur warranty ke sawaalon mein madad kar sakta hoon.")
    elif normalized in {"human support", "talk to a human", "talk to human", "human agent", "speak to a human",
                        "connect me to a human", "create a ticket", "raise a ticket", "raise ticket", "agent please",
                        "talk to an agent", "speak to an agent", "real person", "customer support agent",
                        "connect me to support", "connect me with support", "speak to support", "talk to support",
                        "contact support team", "live agent", "customer care executive"}:
        route = "human_support"
        answer = "Main aapki request ko human support tak pahunchane mein madad karunga."
    elif normalized in {"help", "help me", "i need help", "can you help", "can you help me",
                        "i need support", "mujhe help chahiye", "mujhe madad chahiye"}:
        route = "clarification"
        answer = "Zaroor. Aap apna sawaal ya problem bataiye, main madad karunga."
    else:
        return None
    return {"answer": answer, "answerable": route != "human_support", "sources": [], "citations": [],
            "needs_human_support": route == "human_support",
            "reason": "explicit_handoff" if route == "human_support" else None, "route": route,
            "retrieval_required": False, "retrieval_performed": False,
            "search_query": "", "query_id": None, "confidence": None,
            "verification": {"passed": None, "reason": "non_factual_conversation"},
            "retrieved_chunks": [], "retrieval_candidates": [],
            "retrieval_seconds": 0.0, "reranking_seconds": 0.0, "inference_seconds": 0.0,
            "error": None, "prompt_version": "conversation-routes-v1", "model_version": None}
