from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.chat_service import ChatService
from src.config import Settings
from src.conversation import conversational_reply
from src.ollama_client import ModelUnavailable
from src.query_routing import classify_query
from src.schemas import QueryIntent
from src.store import Store


class QueryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = replace(Settings(), state_dir=Path(self.temp.name), requests_per_minute=100)
        self.store = Store(self.settings.state_dir / "control.sqlite3")

    def test_whole_message_shortcuts_do_not_swallow_policy_or_followups(self):
        for question in ("hi", "Hello!", "hi bot", "नमस्ते", "thanks", "goodbye", "how are you?", "help me"):
            with self.subTest(question=question):
                reply = conversational_reply(question)
                self.assertIsNotNone(reply)
                self.assertEqual(reply["sources"], [])
                self.assertFalse(reply["retrieval_required"])
        for question in ("Hi, what is the return policy?", "Thanks, but how long is the refund?",
                         "And express?", "yes", "Does it cover water damage?", "hello where is my order?"):
            with self.subTest(question=question):
                self.assertIsNone(conversational_reply(question))

    def test_greeting_bypasses_model_index_and_grounded_runtime_in_both_modes(self):
        from src.rag_memory import ask_shopassist, ask_structured
        for pipeline in ("legacy", "grounded"):
            with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline=pipeline)), \
                 patch("src.rag_memory.runtime", side_effect=AssertionError("must not load runtime")), \
                 patch("src.rag_memory.OllamaClient", side_effect=AssertionError("must not call model")):
                for call in (ask_shopassist, ask_structured):
                    result = call("hi", [{"role": "user", "content": "refund policy?"}], "routing-tests")
                    self.assertEqual(result["sources"], [])
                    self.assertEqual(result["route"], "greeting")
                    self.assertFalse(result["retrieval_performed"])
                    self.assertIsNotNone(result["query_id"])
        rows = self.store.rows("SELECT body FROM traces")
        self.assertTrue(all(not json.loads(r["body"])["retrieval_performed"] for r in rows))

    def test_grounded_nonknowledge_routes_do_not_read_corpus_or_search(self):
        client, retriever, verifier = Mock(calls=[]), Mock(), Mock()
        service = ChatService(self.store, self.settings, client, retriever, verifier)
        for intent in ("CONVERSATION", "CLARIFICATION", "ACCOUNT_ACTION", "HUMAN_SUPPORT", "OUT_OF_SCOPE"):
            client.structured.return_value = QueryIntent(intent=intent)
            with self.subTest(intent=intent):
                result = service.ask("An arbitrary routed customer message", record=False)
                self.assertEqual(result["route"], intent.lower())
                self.assertEqual(result["sources"], [])
                self.assertFalse(result["retrieval_performed"])
                self.assertFalse(result["retrieval_required"])
        retriever.corpus.assert_not_called()
        retriever.search.assert_not_called()
        verifier.verify.assert_not_called()

    def test_direct_service_greeting_skips_all_model_and_retrieval_calls(self):
        client, retriever, verifier = Mock(calls=[]), Mock(), Mock()
        result = ChatService(self.store, self.settings, client, retriever, verifier).ask("Hello!", record=False)
        self.assertEqual(result["route"], "greeting")
        client.structured.assert_not_called()
        retriever.corpus.assert_not_called()
        retriever.search.assert_not_called()

    def test_local_handoff_phrases_use_existing_session_owned_tickets(self):
        from src.rag_memory import ask_shopassist
        for question in ("Talk to an agent!", "Speak to support", "Live agent", "Customer care executive"):
            with self.subTest(question=question), \
                 patch("src.rag_memory.Settings.from_env", return_value=self.settings), \
                 patch("src.rag_memory.OllamaClient", side_effect=AssertionError("No model needed")):
                result = ask_shopassist(question, [], "handoff-test")
                self.assertEqual(result["route"], "human_support")
                self.assertTrue(result["ticket"]["reference"].startswith("SA-"))
                self.assertFalse(result["retrieval_performed"])
        self.assertIsNone(conversational_reply("Can a live agent explain the refund policy?"))

    def test_legacy_nonknowledge_never_initializes_legacy_rag(self):
        from src.rag_memory import ask_shopassist
        legacy = ModuleType("src.legacy_rag")
        legacy.ask_shopassist = Mock(side_effect=AssertionError("No model/index initialization"))
        for intent in ("CONVERSATION", "CLARIFICATION", "ACCOUNT_ACTION", "HUMAN_SUPPORT", "OUT_OF_SCOPE"):
            with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline="legacy")), \
                 patch("src.rag_memory.classify_query", return_value=intent), \
                 patch.dict("sys.modules", {"src.legacy_rag": legacy}):
                result = ask_shopassist("An arbitrary routed customer message", [], "routing-tests")
                self.assertEqual(result["route"], "support_followup" if intent == "ACCOUNT_ACTION" else intent.lower())
                self.assertFalse(result["retrieval_performed"])
                self.assertEqual(result["sources"], [])
                legacy.ask_shopassist.assert_not_called()

    def test_legacy_policy_uses_filter_once_and_preserves_followup_history(self):
        from src.rag_memory import ask_shopassist
        history = [{"role": "user", "content": "What is standard shipping time?"}]
        legacy = ModuleType("src.legacy_rag")
        legacy.ask_shopassist = Mock(return_value={"answer": "Policy answer", "search_query": "express shipping?", "sources": ["shipping.pdf, page 1"]})
        messages = ModuleType("langchain_core.messages")
        messages.AIMessage = messages.HumanMessage = lambda content: SimpleNamespace(content=content)
        with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline="legacy")), \
             patch("src.rag_memory.classify_query", return_value="KNOWLEDGE") as classify, \
             patch.dict("sys.modules", {"src.legacy_rag": legacy, "langchain_core.messages": messages}):
            result = ask_shopassist("And express?", history, "routing-tests")
            self.assertTrue(result["retrieval_performed"])
            self.assertTrue(result["sources"])
            self.assertEqual(classify.call_args.args[-1][-len(history):], history)
            self.assertTrue(legacy.ask_shopassist.call_args.kwargs["intent_checked"])
            self.assertEqual(legacy.ask_shopassist.call_args.args[1][-1].content, history[0]["content"])

    def test_filter_failure_does_not_trigger_retrieval(self):
        from src.rag_memory import ask_shopassist
        legacy = ModuleType("src.legacy_rag")
        legacy.ask_shopassist = Mock()
        with patch("src.rag_memory.Settings.from_env", return_value=replace(self.settings, pipeline="legacy")), \
             patch("src.rag_memory.classify_query", side_effect=ModelUnavailable("offline")), \
             patch.dict("sys.modules", {"src.legacy_rag": legacy}):
            result = ask_shopassist("A question that needs routing", [], "routing-tests")
            self.assertEqual(result["failure_reason"], "intent_filter_unavailable")
            self.assertNotIn("ticket", result)
            self.assertFalse(result["retrieval_performed"])
            legacy.ask_shopassist.assert_not_called()

    def test_classifier_keeps_latest_query_and_history_separate_from_rules(self):
        client = Mock()
        client.structured.return_value = QueryIntent(intent="KNOWLEDGE")
        history = [{"role": "user", "content": "Shipping?"}, {"role": "assistant", "content": "Which option?"}]
        self.assertEqual(classify_query(client, "utility", "And express?", history), "KNOWLEDGE")
        messages = client.structured.call_args.args[1]
        self.assertEqual(messages[1:-1], history)
        self.assertEqual(messages[-1], {"role": "user", "content": "And express?"})

    def test_hindi_handoff_and_negation_work_without_a_model(self):
        from src.rag_memory import ask_shopassist
        for question in ("mujhe customer care se baat karni hai", "मुझे एजेंट से बात करनी है"):
            with self.subTest(question=question), \
                 patch("src.rag_memory.Settings.from_env", return_value=self.settings), \
                 patch("src.rag_memory.runtime", side_effect=AssertionError("No model required")):
                result = ask_shopassist(question, [], "hindi-handoff")
                self.assertEqual(result["route"], "human_support")
                self.assertIn("ticket", result)
        for question in ("I do not want to talk to a human", "I don't want to talk to a human",
                         "mujhe agent se baat nahi karni", "मुझे एजेंट से बात नहीं करनी"):
            with self.subTest(question=question), patch("src.rag_memory.Settings.from_env", return_value=self.settings):
                result = ask_shopassist(question, [], "declined-handoff")
                self.assertFalse(result["needs_human_support"])
                self.assertNotIn("ticket", result)
        self.assertIsNone(conversational_reply("No agent please; what is your refund policy?"))

    def test_action_requests_ask_one_relevant_question_without_policy_claims(self):
        from src.query_routing import response_without_retrieval
        for question, expected in (("My refund is overdue. Expedite it.", "refund approval"),
                                   ("Cancel my order", "dispatch"),
                                   ("My payment failed", "error message"),
                                   ("Where is my package?", "tracking update")):
            with self.subTest(question=question):
                answer = response_without_retrieval("ACCOUNT_ACTION", question)
                self.assertIn(expected, answer.answer)
                self.assertTrue(answer.needs_human_support)
                self.assertEqual(answer.sources, [])
                self.assertEqual(answer.answer.count("?"), 1)
