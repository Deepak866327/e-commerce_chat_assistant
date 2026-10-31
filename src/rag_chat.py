
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
db = Chroma(
    collection_name="shopassist_policies",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

# Local Qwen LLM
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

# RAG prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ShopAssist AI, an e-commerce assistant.

        Answer company policy questions using only the
        retrieved policy context below.

        Respond in friendly Hinglish.

        If the information is missing, clearly say
        that the company policy does not provide it.

        Never invent refund periods, warranty terms,
        shipping fees or order statuses.

        Retrieved policy context:
        {context}
        """
    ),
    ("human", "{question}")
])

chain = prompt | llm

print("ShopAssist RAG Assistant started!")
print("Type 'exit' to quit.")

while True:
    question = input("\nCustomer: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=3)

    context_parts = []

    for doc in docs:
        source = Path(
            doc.metadata.get("source", "unknown")
        ).name
        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Generate answer using retrieved context
    response = chain.invoke({
        "question": question,
        "context": context
    })

    print("\nShopAssist:", response.content)

    print("\nRetrieved sources:")
    for doc in docs:
        source = Path(
            doc.metadata.get("source", "unknown")
        ).name
        page = doc.metadata.get("page", 0) + 1

        print(f"- {source}, page {page}")
