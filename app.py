
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

# Import existing RAG function with scope guardrail
from src.rag_memory import ask_shopassist, MAX_HISTORY_MESSAGES


# ==========================================
# STEP 1: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ShopAssist AI",
    page_icon="Home",
    layout="centered"
)


# ==========================================
# STEP 2: PAGE HEADER
# ==========================================

st.title("ShopAssist AI")
st.caption("Your AI-powered shopping support assistant")


# ==========================================
# STEP 3: CONVERSATION MEMORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# STEP 4: SIDEBAR
# ==========================================

with st.sidebar:

    st.header("ShopAssist AI")

    st.write("Powered by:")
    st.write("• Fine-tuned Qwen2.5")
    st.write("• LangChain")
    st.write("• ChromaDB RAG")

    if st.button("Clear conversation"):

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()


# ==========================================
# STEP 5: DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Retrieved sources"):

                for source in message["sources"]:
                    st.write(source)


# ==========================================
# STEP 6: CUSTOMER INPUT
# ==========================================

question = st.chat_input(
    "Ask about shipping, returns, refunds..."
)


# ==========================================
# STEP 7: PROCESS CUSTOMER QUESTION
# ==========================================

if question:

    # Save customer message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display customer message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("ShopAssist is thinking..."):

            try:

                # IMPORTANT:
                # Calls src/rag_memory.py
                # Includes scope guardrail + RAG
                result = ask_shopassist(
                    question,
                    st.session_state.chat_history
                )

                answer = result["answer"]
                sources = result["sources"]

                # Display answer
                st.markdown(answer)

                # Show retrieved sources only if available
                if sources:

                    with st.expander("Retrieved sources"):

                        for source in sources:
                            st.write(source)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                # Update LangChain conversation memory
                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

                # Keep only recent messages
                st.session_state.chat_history = (
                    st.session_state.chat_history[
                        -MAX_HISTORY_MESSAGES:
                    ]
                )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )

                # Remove failed customer message
                st.session_state.messages.pop()
