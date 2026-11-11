
from pathlib import Path
from src.support_db import create_ticket

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)


# ==========================================
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
    persist_directory="./vector_store"
)

stored_chunks = db._collection.count()

if stored_chunks == 0:
    raise RuntimeError(
        "ChromaDB is empty. Run python src/ingest.py first."
    )


# ==========================================
# CONNECT FAQ COLLECTION
# ==========================================

faq_db = Chroma(
    collection_name="shopassist_faqs",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

faq_count = faq_db._collection.count()

if faq_count == 0:
    raise RuntimeError(
        "FAQ collection is empty. Run python src/ingest_faqs.py first."
    )

FAQ_DISTANCE_THRESHOLD = 1.45


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

Your ONLY task is to rewrite the latest customer
message into a standalone search question.

STRICT RULES:

1. Return exactly one question.
2. Never answer the question.
3. Never provide policy facts or numbers unless
   explicitly present in the question or history.
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

EXAMPLE 4:
Previous question: How can I return my order?
Latest question: What about the refund?
Output: What is the refund process for my order?
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

1. Answer in natural Hinglish (Hindi + English).

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

11. Never contradict an explicit statement in the
    retrieved policy context.

12. If the policy says domestic shipping within India
    only, clearly state that international shipping
    is not available.

13. If the retrieved context does not contain the
    answer, do not guess or infer. Say:
    "Mere paas iski verified policy information nahi hai."

14. Treat retrieved policy context as the source of
    truth, not your general knowledge.

15. Carefully distinguish between:
    - Refund
    - Return
    - Exchange
    - Order cancellation

16. If the customer asks about a refund, prioritize
    information specifically related to refunds.

17. Do not answer a refund question using only
    cancellation information.

18. If refund procedure or eligibility information
    is unavailable in the retrieved context, clearly
    say verified refund information is unavailable.

19. If the customer asks how to initiate a refund,
    explain only the verified steps available in
    the retrieved policy.

20. If the retrieved context contains information
    about cancellation but not refunds, do not
    present cancellation rules as refund rules.


EXAMPLE:

Customer: My refund is delayed. Can you expedite it?

Assistant: Main directly refund expedite nahi kar
sakta, kyunki mere paas refund processing system ka
access nahi hai. Agar refund expected timeline se
zyada delayed hai, toh aap support team se refund
status verify karwa sakte hain.
"""

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_RULES
        + "\n\nVERIFIED POLICY CONTEXT:\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

answer_chain = answer_prompt | answer_llm


# ==========================================
# STEP 6: OUT-OF-SCOPE GUARDRAIL
# ==========================================

scope_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an INTENT CLASSIFICATION system
for an e-commerce customer support chatbot.

Your task is to determine whether the customer's
message is related to e-commerce customer support.

Return EXACTLY one label:
IN_SCOPE
or
OUT_OF_SCOPE

==========================================
IN_SCOPE INTENTS
==========================================

Classify as IN_SCOPE when the customer asks about:

1. ORDERS:
   - Order details
   - Order status
   - Order tracking
   - Missing orders
   - Order not received

2. SHIPPING AND DELIVERY:
   - Shipping charges
   - Delivery time
   - Delayed delivery
   - Express shipping
   - Delivery address

3. REFUNDS:
   - How to get a refund
   - Refund eligibility
   - Refund process
   - Refund status
   - Refund not received
   - Money back
   - Delayed refund

4. RETURNS AND EXCHANGES:
   - Return policy
   - How to return a product
   - Exchange requests
   - Damaged or defective products
   - Replacement requests

5. CANCELLATION:
   - Cancel an order
   - Cancellation eligibility
   - Cancellation after dispatch

6. OTHER SHOPPING SUPPORT:
   - Product warranty
   - Payment problems
   - Failed transactions
   - Shopping-related questions
   - Customer support requests

7. CONVERSATION:
   - Greetings such as hi, hello, hey, namaste
   - Follow-up questions related to a previous
     customer support conversation
   - Clarification questions related to support

==========================================
OUT_OF_SCOPE INTENTS
==========================================

Classify as OUT_OF_SCOPE when the customer asks
about unrelated topics such as:

- General knowledge
- Coding and programming
- Mathematics
- Politics
- Entertainment
- Sports
- Unrelated personal questions
- Questions unrelated to shopping or customer support

==========================================
IMPORTANT CLASSIFICATION EXAMPLES
==========================================

Customer: hii
Label: IN_SCOPE

Customer: How can I refund my order?
Label: IN_SCOPE

Customer: I want refund on my order.
Label: IN_SCOPE

Customer: My refund has not arrived.
Label: IN_SCOPE

Customer: How can I return a damaged product?
Label: IN_SCOPE

Customer: I want to cancel my order.
Label: IN_SCOPE

Customer: Where is my order?
Label: IN_SCOPE

Customer: What is the delivery time?
Label: IN_SCOPE

Customer: What about the refund?
Previous conversation: Customer asked about returning an order.
Label: IN_SCOPE

Customer: Explain Python programming.
Label: OUT_OF_SCOPE

Customer: What is the capital of France?
Label: OUT_OF_SCOPE

==========================================
FINAL RULES
==========================================

1. Understand the INTENT, not just the exact words.
2. Use conversation history to understand follow-ups.
3. A short or grammatically incorrect customer
   message can still be IN_SCOPE.
4. If a customer mentions refund, return, order,
   shipping or another shopping support topic,
   classify according to that intent.
5. Do not answer the customer's question.
6. Never explain your classification.
7. Return ONLY IN_SCOPE or OUT_OF_SCOPE.
"""
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
# GREETING DETECTION
# ==========================================

GREETINGS = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "hlo",
    "namaste",
    "namaskar",
    "good morning",
    "good afternoon",
    "good evening"
}


def is_greeting(question):

    normalized_question = (
        question.lower()
        .strip()
        .rstrip("!.,? ")
    )

    return normalized_question in GREETINGS


# ==========================================
# STEP 7: CONVERSATION MEMORY
# ==========================================

# Last 8 messages = approximately 4 turns

MAX_HISTORY_MESSAGES = 8


# ==========================================
# HUMAN SUPPORT REQUEST DETECTION
# ==========================================

HUMAN_SUPPORT_PHRASES = [
    "talk to a human",
    "speak to a human",
    "human agent",
    "talk to an agent",
    "speak to an agent",
    "real person",
    "customer support agent",
    "connect me to support",
    "connect me with support",
    "speak to support",
    "talk to support",
    "contact support team",
    "human support",
    "live agent",
    "customer care executive",
]


def is_human_support_request(question):

    normalized_question = question.lower().strip()

    return any(
        phrase in normalized_question
        for phrase in HUMAN_SUPPORT_PHRASES
    )


def prepare_ticket_conversation(chat_history, question):
    """Convert LangChain history into database-friendly messages."""

    conversation = []

    for message in chat_history:

        role = getattr(message, "type", None)

        if role == "human":
            role = "user"

        elif role == "ai":
            role = "assistant"

        if role in {"user", "assistant"}:

            conversation.append({
                "role": role,
                "content": str(message.content)
            })

    conversation.append({
        "role": "user",
        "content": question
    })

    return conversation


# ==========================================
# STEP 8: FAQ RETRIEVAL
# ==========================================

def search_faq(question):

    results = faq_db.similarity_search_with_score(
        question,
        k=1
    )

    if not results:
        return None

    doc, distance = results[0]

    print(
        f"FAQ Match: {doc.metadata.get('faq_id')} | "
        f"Distance: {distance:.4f}"
    )

    # Reject weak matches

    if distance > FAQ_DISTANCE_THRESHOLD:
        return None

    # Extract verified FAQ answer

    content = doc.page_content

    if "Answer:" not in content:
        return None

    answer = content.split("Answer:", 1)[1].split(
        "Keywords:", 1
    )[0].strip()

    if not answer:
        return None

    return {
        "answer": answer,
        "faq_id": doc.metadata.get("faq_id", "Unknown"),
        "category": doc.metadata.get("category", "Unknown"),
        "distance": float(distance)
    }


# ==========================================
# STEP 9: ANSWER CUSTOMER QUESTIONS
# ==========================================

def ask_shopassist(question, chat_history):

    question = question.strip()

    # --------------------------------------
    # A. GREETING CHECK
    # --------------------------------------

    # Greetings should not trigger:
    # - Intent classification
    # - FAQ retrieval
    # - PDF retrieval

    if is_greeting(question):

        return {
            "answer": (
                "Hii! Welcome to ShopAssist. "
                "How can I help you with your order, "
                "shipping, returns or refunds?"
            ),
            "search_query": question,
            "sources": []
        }


    # --------------------------------------
    # B. HUMAN SUPPORT REQUEST
    # --------------------------------------

    if is_human_support_request(question):

        ticket_conversation = prepare_ticket_conversation(
            chat_history,
            question
        )

        ticket_id = create_ticket(
            subject="Human support requested",
            customer_message=question,
            conversation=ticket_conversation,
            priority="Normal"
        )

        return {
            "answer": (
                f"Bilkul! Aapka support ticket successfully "
                f"create ho gaya hai. "
                f"Aapka ticket ID hai: **{ticket_id}**. "
                f"Support team is ticket ko review kar sakti hai. "
                f"Please future communication ke liye "
                f"is ticket ID ko save rakhein."
            ),
            "search_query": question,
            "sources": [f"Support Ticket: {ticket_id}"],
            "ticket_id": ticket_id
        }


    # --------------------------------------
    # C. INTENT CLASSIFICATION USING QWEN
    # --------------------------------------

    scope_response = scope_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    scope_result = scope_response.content.strip().upper()

    print(f"Intent Classifier Output: {scope_result}")

    # Handle output containing extra text

    if "OUT_OF_SCOPE" in scope_result:

        return {
            "answer": OUT_OF_SCOPE_REPLY,
            "search_query": question,
            "sources": []
        }

    elif "IN_SCOPE" not in scope_result:

        print(
            "Warning: Invalid intent classifier output. "
            "Treating query as OUT_OF_SCOPE."
        )

        return {
            "answer": OUT_OF_SCOPE_REPLY,
            "search_query": question,
            "sources": []
        }


    # --------------------------------------
    # D. REWRITE FOLLOW-UP QUESTION
    # --------------------------------------

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

    print(f"Rewritten Search Query: {rewritten_question}")


    # --------------------------------------
    # E. SEARCH FAQ COLLECTION FIRST
    # --------------------------------------

    faq_result = search_faq(rewritten_question)

    if faq_result:

        return {
            "answer": faq_result["answer"],
            "search_query": rewritten_question,
            "sources": [
                f"{faq_result['faq_id']} "
                f"({faq_result['category']}), faqs.json"
            ]
        }


    # --------------------------------------
    # F. RETRIEVE RELEVANT PDF CHUNKS
    # --------------------------------------

    docs = db.similarity_search(
        rewritten_question,
        k=3
    )


    # --------------------------------------
    # G. PREPARE CONTEXT FOR LLM
    # --------------------------------------

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


    # --------------------------------------
    # H. GENERATE FINAL ANSWER
    # --------------------------------------

    response = answer_chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })

    answer = response.content

    return {
        "answer": answer,
        "search_query": rewritten_question,
        "sources": [
            f"{filename}, page {page}"
            for filename, page in sorted(unique_sources)
        ]
    }
