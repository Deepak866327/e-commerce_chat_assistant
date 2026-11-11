
from src.rag_memory import ask_shopassist

test_questions = [
    "Where is my package?",
    "My refund has not been credited",
    "What is the warranty period?",
    "Do you ship products internationally?",
    "How can I learn Python programming?"
]

for question in test_questions:

    print("\n" + "=" * 70)
    print("CUSTOMER:", question)

    result = ask_shopassist(question, [])

    print("ANSWER:", result["answer"])
    print("SEARCH QUERY:", result["search_query"])
    print("SOURCES:", result["sources"])
