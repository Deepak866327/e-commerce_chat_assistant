
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Same embedding model use karna mandatory hai
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Existing vector database load karo
db = Chroma(
    collection_name="shopassist_policies",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

print("Vector database loaded.")
print("Type 'exit' to quit.")

while True:
    query = input("\nSearch query: ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    # Similarity search
    results = db.similarity_search_with_score(
        query,
        k=3
    )

    for index, (doc, distance) in enumerate(results, 1):
        filename = Path(
            doc.metadata.get("source", "unknown")
        ).name

        page = doc.metadata.get("page", 0) + 1

        print(f"\n--- Result {index} ---")
        print("File:", filename)
        print("Page:", page)
        print("Distance:", round(distance, 4))
        print("Content:")
        print(doc.page_content)
