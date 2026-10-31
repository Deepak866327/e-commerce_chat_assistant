
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Local Qwen model initialize karo
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.2
)

# 2. Assistant ke instructions
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are ShopAssist AI, an e-commerce customer
        support assistant.

        Rules:
        - Respond in friendly Hinglish.
        - Help with products, shipping, refunds,
          returns and warranties.
        - Never invent company policies.
        - Never invent order details.
        - If information is unavailable,
          clearly say that it is unavailable.
        """
    ),
    ("human", "{question}")
])

# 3. LangChain pipeline
chain = prompt | llm

print("ShopAssist AI started! Type 'exit' to quit.")

while True:
    question = input("\nCustomer: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:
        response = chain.invoke({"question": question})
        print("\nShopAssist:", response.content)
    except Exception as error:
        print("Error:", error)
