"""Optional Ragas and DeepEval judges, explicitly configured to local Ollama.

Missing packages/models are errors, never silently substituted by proxy scores or
cloud providers. Library-dependent integration is a separately executed live gate.
"""

import os


class RagasJudge:
    def __init__(self, settings):
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        self.llm = LangchainLLMWrapper(ChatOllama(model=settings.verifier_model,
                     base_url=settings.ollama_url, temperature=0, client_kwargs={"timeout": settings.model_timeout}))
        self.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model,
                     model_kwargs={"local_files_only": not settings.allow_model_download, "device": "cpu"}))
        self.timeout = settings.model_timeout
        self.version = "ragas/local-ollama/" + settings.verifier_model

    def measure(self, case, result):
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
        from ragas.run_config import RunConfig
        sample = SingleTurnSample(user_input=case.question, response=result["answer"],
                                  reference=case.expected_answer,
                                  retrieved_contexts=[c["text"] for c in result.get("retrieved_chunks", [])])
        score = evaluate(EvaluationDataset(samples=[sample]), metrics=[faithfulness, answer_relevancy, answer_correctness],
                         llm=self.llm, embeddings=self.embeddings, show_progress=False, raise_exceptions=True,
                         run_config=RunConfig(timeout=self.timeout, max_retries=0, max_workers=1))
        row = score.to_pandas().iloc[0]
        return {"faithfulness": float(row["faithfulness"]), "answer_relevancy": float(row["answer_relevancy"]),
                "answer_correctness": float(row["answer_correctness"])}


class DeepEvalJudge:
    def __init__(self, settings, client):
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
        from deepeval.models import DeepEvalBaseLLM
        from src.schemas import Contract

        class Text(Contract):
            text: str

        class LocalJudge(DeepEvalBaseLLM):
            def load_model(self):
                return client

            def generate(self, prompt, schema=None):
                contract = schema or Text
                value = client.structured(settings.verifier_model, [{"role": "user", "content": prompt}], contract, retries=1, num_ctx=8192)
                return value if schema else value.text

            async def a_generate(self, prompt, schema=None):
                import asyncio
                return await asyncio.to_thread(self.generate, prompt, schema)

            def get_model_name(self):
                return settings.verifier_model

        self.model = LocalJudge()
        self.version = "deepeval/local-ollama/" + settings.verifier_model

    def measure(self, case, result):
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        test = LLMTestCase(input=case.question, actual_output=result["answer"], expected_output=case.expected_answer,
                           retrieval_context=[c["text"] for c in result.get("retrieved_chunks", [])])
        metrics = {"faithfulness": FaithfulnessMetric(model=self.model, async_mode=False),
                   "answer_relevancy": AnswerRelevancyMetric(model=self.model, async_mode=False),
                   "answer_correctness": GEval(name="Answer correctness", model=self.model, async_mode=False,
                     criteria="Assess factual correctness and completeness against expected_output; preserve all policy conditions and numbers.",
                     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])}
        values = {}
        for name, metric in metrics.items():
            metric.measure(test)
            values[name] = float(metric.score)
        return values


class CombinedJudge:
    def __init__(self, judges, citation_judge):
        self.judges = judges
        self.citation_judge = citation_judge
        self.version = "+".join(j.version for j in judges) + "+independent-citation-audit"

    def measure(self, case, result):
        from src.schemas import Answer
        measurements = [judge.measure(case, result) for judge in self.judges]
        # Conservative combination: a high score in one judge cannot hide a low
        # score in another. Preserve per-framework output in a future report schema.
        values = {key: min(row[key] for row in measurements) for key in measurements[0]}
        sources = result.get("sources", [])
        chunk_map = {c["id"]: c for c in result.get("retrieved_chunks", [])}
        valid = 0
        if sources:
            answer = Answer(answer=result["answer"], answerable=True, sources=sources, needs_human_support=False, reason=None)
            verdict = self.citation_judge.verify(case.question, answer, list(chunk_map.values()))
            used = {claim["chunk_id"] for claim in verdict.get("claims", []) if claim["supported"]} if verdict["passed"] else set()
            valid = sum(citation["chunk_id"] in used for citation in sources)
        values["citation_accuracy"] = valid / len(sources) if sources else 0.0
        return values
