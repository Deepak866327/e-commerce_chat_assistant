# Frozen pre-upgrade pipeline for controlled baseline evaluation only.
from pathlib import Path
from src.language import language_instruction, message
from src.policy_answer import generate_policy_answer

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from functools import lru_cache


MAX_HISTORY_MESSAGES = 8

@lru_cache(maxsize=1)
def _initialize():
    # STEP 1: INITIALIZE EMBEDDING MODEL
    # ==========================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ==========================================
    # STEP 2: CONNECT CHROMADB
    # ==========================================

    db = Chroma(
        collection_name="shopassist_policies",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).resolve().parents[1] / "vector_store")
    )

    stored_chunks = db._collection.count()

    if stored_chunks == 0:
        raise RuntimeError(
            "ChromaDB is empty. Run python src/ingest.py first."
        )

    # ==========================================
    # STEP 3: INITIALIZE OLLAMA MODELS
    # ==========================================

    rewrite_llm = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0,
        num_ctx=2048
    )

    answer_llm = ChatOllama(
        model="shopassist-finetuned",
        temperature=0,
        num_ctx=4096
    )
    # ==========================================
    # STEP 4: QUERY REWRITING PROMPT
    # ==========================================

    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a search query rewriting assistant.

            Your ONLY task is to rewrite the latest
            customer message into a standalone search question.

            STRICT RULES:
            1. Return exactly one question.
            2. Never answer the question.
            3. Never provide policy facts or numbers
               unless explicitly present in the question.
            4. Use chat history only to understand references.
            5. Keep the rewritten question in English.
            6. Return only the rewritten question.
            7. End your output with a question mark.

            EXAMPLE 1:
            Previous question: What is standard shipping time?
            Latest question: And express?
            Output: What is the express shipping time?

            EXAMPLE 2:
            Previous question: What is the return policy?
            Latest question: What about damaged items?
            Output: What is the return policy for damaged items?

            EXAMPLE 3:
            Previous question: What is the warranty period?
            Latest question: Does it cover water damage?
            Output: Does the warranty cover water damage?
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Rewrite this as a standalone search question: {question}"
        )
    ])

    rewrite_chain = rewrite_prompt | rewrite_llm

    # ==========================================
    # STEP 5: FINAL ANSWER PROMPT
    # ==========================================

    SYSTEM_RULES = """
    You are ShopAssist AI, a customer support assistant.

    STRICT INSTRUCTIONS:

    1. Follow the requested response language.
    2. For company policy facts, use ONLY the retrieved
       policy context.
    3. Do not add unrelated policy information.
    4. Do not invent timelines, charges, eligibility
       conditions, or company contact details.
    5. Never claim to have checked an order, processed
       a refund, expedited a refund, or cancelled an order.
    6. If the customer asks you to perform an action
       requiring backend access, clearly explain that
       you cannot perform it directly.
    7. If verified policy information is unavailable,
       say:
       "Mere paas iski verified policy information nahi hai."
    8. Answer the customer's specific question directly.
    9. Keep answers concise, preferably 2-4 sentences.
    10. Do not treat chat history as verified policy facts.
    11. Never contradict an explicit statement in the retrieved policy context.
    12. If the policy allows only domestic deliveries within India, clearly
        state that international shipping is unavailable.
    13. Distinguish refunds, returns, exchanges and order cancellations.
    14. For refund questions, use refund evidence. Cancellation rules alone
        do not establish refund eligibility, procedure or timing.
    15. Use customer-reported details to address the original issue, including short
        follow-up replies. Do not repeat guidance already tried. Do not ask questions
        or claim a support ticket was created; the support flow manages the next step.
    16. If the context does not answer the question, say verified policy
        information is unavailable instead of inferring an answer.
    17. An faq record in history is an article previously shown to the customer,
        not current policy evidence. If its outcome is not_helpful, address the
        blocker using retrieved evidence instead of repeating that article.
    18. Lead with the next step relevant to the reported blocker, not a generic
        greeting or the whole policy. Mention missing system access only when
        the task requires it. Never invent app menus, buttons or procedures.
        A related eligibility rule does not explain how to fix a missing button.

    EXAMPLE:

    Customer: My refund is delayed. Can you expedite it?

    Assistant: Main directly refund expedite nahi kar sakta,
    kyunki mere paas refund processing system ka access nahi
    hai. Agar refund expected timeline se zyada delayed hai,
    toh aap support team se refund status verify karwa sakte
    hain.
    """

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_RULES
            + "\nRESPONSE LANGUAGE: {response_language}"
            + "\n\nVERIFIED POLICY CONTEXT:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    answer_chain = answer_prompt | answer_llm

    # ==========================================
    # OUT-OF-SCOPE GUARDRAIL
    # ==========================================

    scope_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You classify customer messages for an
    e-commerce customer support chatbot.

    Return ONLY one label:
    IN_SCOPE or OUT_OF_SCOPE.

    IN_SCOPE includes:
    - Orders and order tracking
    - Shipping and delivery
    - Returns, refunds and exchanges
    - Order cancellation
    - Product warranty
    - Payments and shopping support
    - Greetings and clarification related to support

    OUT_OF_SCOPE includes:
    - General knowledge
    - Coding and programming
    - Mathematics
    - Politics and entertainment
    - Questions unrelated to customer support

    Use conversation history to interpret follow-ups.

    Return only the label."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    scope_chain = scope_prompt | rewrite_llm

    OUT_OF_SCOPE_REPLY = (
        "Main ShopAssist customer support assistant hoon. "
        "Main sirf orders, shipping, returns, refunds, "
        "cancellations aur warranty se related questions "
        "mein help kar sakta hoon."
    )

    # ==========================================
    # STEP 6: CONVERSATION MEMORY
    # ==========================================

    # Last 8 messages = approximately 4 turns
    MAX_HISTORY_MESSAGES = 8


    # ==========================================
    # STEP 7: ANSWER CUSTOMER QUESTIONS
    # ==========================================

    return db, scope_chain, rewrite_chain, answer_chain, OUT_OF_SCOPE_REPLY

def ask_shopassist(question, chat_history, *, intent_checked=False, response_language="hinglish", search_question=None):
    language_instruction(response_language)
    db, scope_chain, rewrite_chain, answer_chain, OUT_OF_SCOPE_REPLY = _initialize()
    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]


    question = question.strip()

    # CHECK SCOPE USING BASE QWEN
    # The deployed wrapper performs the richer retrieval-intent decision first.
    # Controlled A/B baseline calls retain the original scope classifier.
    scope_result = "IN_SCOPE" if intent_checked else scope_chain.invoke({
        "question": question,
        "chat_history": chat_history
    }).content.strip().upper()

    if scope_result != "IN_SCOPE":
        return {
            "answer": message("out_of_scope", response_language),
            "search_query": question,
            "sources": []
        }

    # A. REWRITE FOLLOW-UP QUESTION

    if chat_history:
        rewritten_question = rewrite_chain.invoke({
            "question": question,
            "chat_history": chat_history
        }).content.strip()

        if (
            not rewritten_question
            or not rewritten_question.endswith("?")
            or len(rewritten_question) > 300
        ):
            rewritten_question = question
    else:
        rewritten_question = question

    # B. RETRIEVE RELEVANT PDF CHUNKS

    if intent_checked and search_question:
        rewritten_question = search_question

    docs = db.similarity_search(
        rewritten_question,
        k=3
    )

    # C. PREPARE CONTEXT FOR LLM

    context_parts = []
    unique_sources = set()

    for doc in docs:
        filename = Path(
            doc.metadata.get("source", "unknown")
        ).name

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: {filename}, Page: {page}\n"
            f"{doc.page_content}"
        )

        unique_sources.add((filename, page))

    context = "\n\n".join(context_parts)

    if not context:
        context = "No relevant policy information found."

    # D. CALL FINE-TUNED QWEN

    payload = {
        "question": question,
        "context": context,
        "response_language": language_instruction(response_language),
        "chat_history": chat_history
    }
    if intent_checked:
        answer = generate_policy_answer(answer_chain, payload, [doc.page_content for doc in docs])
        if answer is None:
            return {"answer": message("abstention", response_language),
                    "sources": [], "search_query": rewritten_question,
                    "answerable": False, "needs_human_support": True, "reason": "unsupported_policy_numbers"}
    else:
        # Keep the controlled pre-upgrade baseline unmodified.
        answer = answer_chain.invoke(payload).content

    return {
        "answer": answer,
        "search_query": rewritten_question,
        "sources": [
            f"{filename}, page {page}"
            for filename, page in sorted(unique_sources)
        ]
    }
