
import csv
import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------
# 1. Load test cases
# -----------------------------------------

with open(
    "data/evaluation/test_cases.json",
    encoding="utf-8"
) as file:
    test_cases = json.load(file)

# -----------------------------------------
# 2. Load embedding model and ChromaDB
# -----------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="shopassist_policies",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

if db._collection.count() == 0:
    raise RuntimeError("ChromaDB is empty. Run ingest.py.")

# -----------------------------------------
# 3. Load local Qwen model
# -----------------------------------------

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ShopAssist AI.

        Answer using only the provided company policy.
        Respond clearly and concisely.

        If the policy does not contain the answer,
        say you cannot verify the information.

        Do not invent order status, stock counts,
        shipping fees or policy conditions.

        Policy context:
        {context}
        """
    ),
    ("human", "{question}")
])

chain = prompt | llm

# -----------------------------------------
# 4. Normalization for keyword checking
# -----------------------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def contains_expected_terms(answer, expected_terms):
    normalized_answer = normalize(answer)

    return all(
        normalize(term) in normalized_answer
        for term in expected_terms
    )

# -----------------------------------------
# 5. Run evaluation
# -----------------------------------------

results = []

for index, test in enumerate(test_cases, start=1):

    question = test["question"]

    print(f"\n[{index}/{len(test_cases)}] {question}")

    # Retrieve top 3 policy chunks
    docs = db.similarity_search(question, k=3)

    retrieved_sources = {
        Path(doc.metadata.get("source", "")).name
        for doc in docs
    }

    expected_source = test["source"]

    # Only meaningful for answerable questions
    retrieval_hit = (
        expected_source in retrieved_sources
        if expected_source is not None
        else None
    )

    # Build LLM context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Generate answer
    try:
        answer = chain.invoke({
            "question": question,
            "context": context
        }).content

        error = ""

    except Exception as exc:
        answer = ""
        error = str(exc)

    # Approximate keyword-based check
    if test["answerable"] and not error:
        fact_match = contains_expected_terms(
            answer,
            test["expected_terms"]
        )
    else:
        fact_match = None

    results.append({
        "question": question,
        "answerable": test["answerable"],
        "expected_source": expected_source or "",
        "retrieved_sources": "; ".join(
            sorted(retrieved_sources)
        ),
        "retrieval_hit": retrieval_hit,
        "fact_match": fact_match,
        "answer": answer,
        "error": error,
        "manual_review": ""
    })

    print("Retrieved:", retrieved_sources)
    print("Answer:", answer)
    print("Retrieval hit:", retrieval_hit)
    print("Fact match:", fact_match)

# -----------------------------------------
# 6. Calculate metrics
# -----------------------------------------

answerable_results = [
    row for row in results
    if row["answerable"] and not row["error"]
]

retrieval_hits = sum(
    row["retrieval_hit"]
    for row in answerable_results
)

fact_matches = sum(
    row["fact_match"]
    for row in answerable_results
)

total = len(answerable_results)

retrieval_rate = (
    retrieval_hits / total * 100
    if total else 0
)

fact_rate = (
    fact_matches / total * 100
    if total else 0
)

# -----------------------------------------
# 7. Save CSV report
# -----------------------------------------

report_path = Path("data/evaluation/baseline_results.csv")

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

# -----------------------------------------
# 8. Print summary
# -----------------------------------------

print("\n" + "=" * 50)
print("SHOPASSIST RAG BASELINE EVALUATION")
print("=" * 50)

print("Total questions:", len(test_cases))
print("Answerable questions evaluated:", total)
print("Unanswerable questions:", sum(
    not row["answerable"] for row in results
))

print(f"Document Retrieval Hit@3: {retrieval_rate:.2f}%")
print(f"Keyword Fact Match: {fact_rate:.2f}%")

print("\nReport saved:", report_path)
print("Review unsupported answers manually in the CSV.")
