
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to FAQ collection
faq_db = Chroma(
    collection_name="shopassist_faqs",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

print("FAQ collection count:", faq_db._collection.count())

# 3. Test customer questions
test_questions = [
    "Can I cancel my order?",
    "Where is my package?",
    "My refund has not been credited",
    "How long does standard delivery take?",
    "What is the warranty period?",
    "Do you ship products internationally?",
    "How can I learn Python programming?"
]

# 4. Search FAQs
for question in test_questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    results = faq_db.similarity_search_with_score(
        question,
        k=3
    )

    for rank, (doc, score) in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("Distance Score:", round(score, 4))
        print("Metadata:", doc.metadata)
        print("FAQ Content:", doc.page_content)
