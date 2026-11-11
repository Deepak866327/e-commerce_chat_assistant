
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


FAQ_FILE = Path("data/faqs/faqs.json")
VECTOR_DB_PATH = "./vector_store"
COLLECTION_NAME = "shopassist_faqs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not FAQ_FILE.exists():
        raise FileNotFoundError(f"FAQ file not found: {FAQ_FILE}")

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    faqs = data.get("faqs", [])

    # Only approved FAQs with non-empty answers
    active_faqs = [
        faq for faq in faqs
        if faq.get("status") == "active"
        and faq.get("question", "").strip()
        and faq.get("answer", "").strip()
    ]

    if not active_faqs:
        print("No active FAQs found. Nothing to ingest.")
        print("Add verified answers and set their status to 'active'.")
        return

    documents = []
    ids = []

    for faq in active_faqs:
        faq_id = faq["faq_id"]

        content = (
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}\n"
            f"Keywords: {', '.join(faq.get('keywords', []))}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "faq_id": faq_id,
                    "category": faq.get("category", "general"),
                    "source": "faqs.json"
                }
            )
        )
        ids.append(faq_id)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    faq_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    # Replace only these FAQ records; policy collection is untouched.
    faq_db.delete(ids=ids)
    faq_db.add_documents(documents, ids=ids)

    print("\nFAQ ingestion completed!")
    print("FAQs indexed:", len(documents))
    print("Collection:", COLLECTION_NAME)
    print("Collection count:", faq_db._collection.count())


if __name__ == "__main__":
    main()
