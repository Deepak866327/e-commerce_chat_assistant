"""Customer self-service help. Browsing never submits a chat turn."""

import logging

import streamlit as st
from src.default_help import starter_faqs
from src.faq_search import ranked_articles


logger = logging.getLogger(__name__)


def render_faq_feedback(article, record_feedback, key_suffix):
    st.caption("Was this answer helpful?")
    outcome = st.session_state.get("faq_votes", {}).get(article["faq_id"] + article["version"])
    if outcome == "helpful":
        st.success("Thank you. We’re glad this article helped.")
    elif outcome == "not_helpful":
        st.caption("Continue below in chat. We’ll help with your specific issue.")
    left, right = st.columns(2)
    left.button("This helped", key="faq-helpful-" + article["faq_id"] + key_suffix,
                on_click=record_feedback, args=(article, "helpful"), disabled=outcome == "helpful")
    right.button("I still need help", key="faq-help-" + article["faq_id"] + key_suffix,
                 on_click=record_feedback, args=(article, "not_helpful"))


def render_faq_help(load_articles, select_article):
    with st.expander("Help center · Frequently asked questions", expanded=not st.session_state.messages):
        st.caption("Select a question to get its answer in chat.")
        try:
            articles = load_articles()
        except Exception:
            logger.error("FAQ help center is temporarily unavailable")
            st.info("Store-specific articles are temporarily unavailable. You can browse general help below or ask us in chat.")
            articles = []
        if not articles:
            articles = starter_faqs()
        if all(article.get("kind") == "general" for article in articles):
            st.caption("General help · Eligibility and timeframes depend on the store’s current policy.")
        notice = st.session_state.pop("faq_feedback_notice", None)
        if notice:
            st.warning(notice)

        search = st.text_input("Search help articles", placeholder="Try cancellation, delivery or refund",
                               key="faq_search", max_chars=200).strip().casefold()
        categories = sorted({article["category"] for article in articles}, key=str.casefold)
        # A publication change may remove the customer's previously selected category.
        options = ["All topics", *[category for category in categories if category != "All topics"]]
        if st.session_state.get("faq_category") not in options:
            st.session_state["faq_category"] = "All topics"
        category = st.selectbox("Browse by topic", options, key="faq_category")
        matches = [article for article, _, _ in ranked_articles(articles, search)
                   if category == "All topics" or article["category"] == category]
        st.caption(f"{len(matches)} help article{'s' if len(matches) != 1 else ''}")
        if not matches:
            st.info("No matching articles. Try another search or describe your issue in chat below.")
        for article in matches:
            st.button(article["question"], key="faq-select-" + article["faq_id"],
                      on_click=select_article, args=(article,), width="stretch")
        st.caption("Need personal help? Describe what happened in the chat below. Never share passwords or payment details.")
