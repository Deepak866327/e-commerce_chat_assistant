from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)


class ShopAssistBot:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            collection_name="shopassist_policies",
            embedding_function=self.embeddings,
            persist_directory="./vector_store"
        )

        if self.db._collection.count() == 0:
            raise RuntimeError("ChromaDB is empty.")

        self.rewrite_llm = ChatOllama(
            model="qwen2.5:1.5b",
            temperature=0
        )

        self.answer_llm = ChatOllama(
            model="shopassist-finetuned",
            temperature=0,
            num_ctx=4096
        )

        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Rewrite the latest customer question into a
                standalone English search question.

                Use chat history to resolve references.
                Do not answer the question.
                Return only one question ending with ?"""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are ShopAssist AI.

                Answer in friendly Hinglish.

                Use ONLY retrieved policy context for
                company-specific facts.

                Never invent policy details or live order status.

                Never claim to have processed or expedited
                refunds, or cancelled orders.

                If information is missing, say you cannot
                verify it.

                Keep responses concise.

                POLICY CONTEXT:
                {context}"""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        self.rewrite_chain = (
            self.rewrite_prompt | self.rewrite_llm
        )

        self.answer_chain = (
            self.answer_prompt | self.answer_llm
        )

    def ask(self, question, chat_history):

        rewritten = question

        if chat_history:
            candidate = self.rewrite_chain.invoke({
                "question": question,
                "chat_history": chat_history
            }).content.strip()

            if (
                candidate.endswith("?")
                and len(candidate) <= 300
            ):
                rewritten = candidate

        docs = self.db.similarity_search(
            rewritten,
            k=3
        )

        context_parts = []
        sources = []

        for doc in docs:

            filename = Path(
                doc.metadata.get("source", "unknown")
            ).name

            page = doc.metadata.get("page", 0) + 1

            context_parts.append(
                f"Source: {filename}, Page: {page}\n"
                f"{doc.page_content}"
            )

            source = f"{filename}, page {page}"

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        if not context:
            context = "No relevant policy information found."

        answer = self.answer_chain.invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history
        }).content

        return {
            "answer": answer,
            "sources": sources,
            "search_query": rewritten
        }