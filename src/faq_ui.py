"""Customer self-service help. Browsing never submits a chat turn."""

import logging

import streamlit as st


logger = logging.getLogger(__name__)


def render_faq_help(load_articles, send_question):
    with st.expander("Help center · Frequently asked questions", expanded=not st.session_state.messages):
        st.caption("Find a quick answer, or continue in chat for help with your situation.")
        try:
            articles = load_articles()
        except Exception:
            logger.error("FAQ help center is temporarily unavailable")
            st.info("Help articles are temporarily unavailable. You can still ask us in chat below.")
            return
        if not articles:
            st.info("Help articles are being prepared. Ask your question in chat below and we’ll help you.")
            return

        search = st.text_input("Search help articles", placeholder="Try cancellation, delivery or refund",
                               key="faq_search", max_chars=200).strip().casefold()
        categories = sorted({article["category"] for article in articles}, key=str.casefold)
        # A publication change may remove the customer's previously selected category.
        options = ["All topics", *[category for category in categories if category != "All topics"]]
        if st.session_state.get("faq_category") not in options:
            st.session_state["faq_category"] = "All topics"
        category = st.selectbox("Browse by topic", options, key="faq_category")
        matches = [article for article in articles
                   if (category == "All topics" or article["category"] == category)
                   and all(term in " ".join((article["question"], article["answer"], article["category"])).casefold()
                           for term in search.split())]
        st.caption(f"{len(matches)} help article{'s' if len(matches) != 1 else ''}")
        if not matches:
            st.info("No matching articles. Try another search or describe your issue in chat below.")
        for article in matches:
            with st.expander(article["question"]):
                st.caption(article["category"])
                st.markdown(article["answer"])
                st.button("I still need help", key="faq-help-" + article["faq_id"],
                          on_click=send_question,
                          args=(f"{article['question']} I have read the FAQ and still need help with my situation.",))
        st.caption("Need personal help? Describe what happened in the chat below. Never share passwords or payment details.")
