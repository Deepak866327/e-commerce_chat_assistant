"""Real Ollama routing + fine-tuned RAG smoke checks in an isolated candidate index.

No mocks, production writes, index activation or automatic model downloads.
These development checks are not human-reviewed factual acceptance metrics.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_support_routes import run_cases
from src.config import ROOT, Settings
from src.ollama_client import OllamaClient


ANSWER_CASES = [
    {"id": "standard-en", "question": "How long does standard shipping take?", "language": "en", "answerable": True},
    {"id": "express-hinglish", "question": "Express delivery mein kitne din lagte hain?", "language": "hinglish", "answerable": True},
    {"id": "returns-hi", "question": "सामान मिलने के कितने दिनों के अंदर वापसी का अनुरोध कर सकते हैं?", "language": "hi", "answerable": True},
    {"id": "refund-conditions", "question": "When does the refund processing timeline start?", "language": "en", "answerable": True},
    {"id": "warranty", "question": "What does the limited warranty cover?", "language": "en", "answerable": True},
    {"id": "shipping-followup", "question": "And express?", "language": "en", "answerable": True,
     "history": [{"role": "user", "content": "How long does standard shipping take?"}]},
    {"id": "unsupported-fee", "question": "What is ShopAssist's premium membership annual fee?", "language": "en", "answerable": False},
    {"id": "refund-action", "question": "My refund is 20 days overdue. Please expedite it.", "language": "en", "answerable": False, "route": "account_action"},
    {"id": "cancel-action", "question": "Cancel my order for me", "language": "en", "answerable": False, "route": "account_action"},
    {"id": "cancel-procedure", "question": "How can I cancel my order?", "language": "en", "answerable": True, "route": "knowledge"},
    {"id": "declined-agent-policy", "question": "I do not want an agent; explain your return policy", "language": "en", "answerable": True},
]


def inspect_result(case, result):
    """Check execution/citation contracts; deliberately do not score semantic truth."""
    failures = []
    if result.get("error"):
        failures.append("execution_error")
    if bool(result.get("answerable")) != case["answerable"]:
        failures.append("unexpected_answerability")
    if case.get("route") and result.get("route") != case["route"]:
        failures.append("unexpected_route")
    if case["answerable"]:
        if result.get("verification", {}).get("passed") is not True:
            failures.append("grounding_verification_failed")
        if not result.get("sources"):
            failures.append("missing_citations")
    elif result.get("sources"):
        failures.append("citations_on_abstention")
    if case.get("route") == "account_action":
        if result.get("retrieval_performed") is not False:
            failures.append("unexpected_retrieval_for_account_action")
        if result.get("needs_human_support") is not True:
            failures.append("missing_handoff")
    return failures


def run_checks(settings, policies, report, checkpoint=lambda: None):
    client = OllamaClient(settings)
    report.update(stage="model_preflight", status="running")
    models = {"answer": settings.answer_model, "utility": settings.utility_model, "verifier": settings.verifier_model}
    report["models"] = {}
    for role, name in models.items():
        report["checking_model"] = {"role": role, "name": name}
        report["models"][role] = {"name": name, "digest": client.model_digest(name)}
    report.pop("checking_model", None)
    report["different_answer_and_verifier_weights"] = report["models"]["answer"]["digest"] != report["models"]["verifier"]["digest"]

    report["stage"] = "routing"
    print("Running real utility-model routing checks...", file=sys.stderr, flush=True)
    cases = json.loads((ROOT / "data/benchmarks/support_routes.json").read_text(encoding="utf-8"))["cases"]
    report["routing"] = run_cases(cases, client, settings)
    checkpoint()

    report["stage"] = "isolated_index"
    # Imports are lazy so preflight works before embedding dependencies are loaded.
    from src.chat_service import ChatService
    from src.ingestion import IngestionService
    from src.retrieval import BGEReranker, Retriever
    from src.store import Store
    from src.vector_index import ChromaIndex
    from src.verification import GroundingVerifier

    files = sorted(path for path in policies.glob("*.pdf") if path.is_file())
    if not files:
        raise ValueError("No policy PDFs found in the selected policy directory")
    report["documents"] = [{"name": path.name, "sha256": sha256(path.read_bytes()).hexdigest()} for path in files]
    with tempfile.TemporaryDirectory(prefix="shopassist-real-models-") as temporary:
        root = Path(temporary)
        isolated = replace(settings, state_dir=root / "state", vector_dir=root / "vectors")
        store = Store(isolated.state_dir / "control.sqlite3")
        vector = ChromaIndex(isolated)
        ingestion = IngestionService(isolated, store, vector)
        report["ingestion"] = [ingestion.ingest(path, actor="isolated-development-evaluation") for path in files]
        retriever = Retriever(store, isolated, vector=vector, candidate_index=vector.id,
                              reranker_factory=(lambda: BGEReranker(isolated)) if isolated.use_reranker else None)
        report["index"] = {"identity": vector.id, "embedding": settings.embedding_model,
                           "chunks": len(retriever.corpus()), "temporary": True, "activated": False}
        service = ChatService(store, isolated, client, retriever, GroundingVerifier(client, settings.verifier_model))
        report["stage"] = "fine_tuned_answers"
        report["answers"] = []
        for index, case in enumerate(ANSWER_CASES, 1):
            print(f"Answer check {index}/{len(ANSWER_CASES)}: {case['id']}", file=sys.stderr, flush=True)
            started = time.perf_counter()
            call_start = len(client.calls)
            result = service.ask(case["question"], case.get("history", []), session="isolated-evaluation",
                                 record=False, response_language=case["language"])
            failures = inspect_result(case, result)
            report["answers"].append({"id": case["id"], "question": case["question"], "language": case["language"],
                "answer": result["answer"], "answerable": result["answerable"], "route": result["route"],
                "sources": result["sources"], "verification": result["verification"], "reason": result.get("reason"),
                "retrieval_performed": result["retrieval_performed"], "search_query": result["search_query"],
                "retrieved_evidence": [{"id": chunk["id"], "text": chunk["text"], "metadata": chunk["metadata"]}
                                       for chunk in result.get("retrieved_chunks", [])],
                "model_calls": client.calls[call_start:], "seconds": time.perf_counter() - started,
                "contract_passed": not failures, "failures": failures})
            checkpoint()
        # No active index, live tickets or existing workflow state may be changed.
        if store.setting("active_index") is not None:
            raise RuntimeError("Isolated evaluation unexpectedly activated an index")

    report["stage"] = "final_model_identity"
    report["model_identity_unchanged"] = all(client.model_digest(models[role]) == record["digest"]
                                              for role, record in report["models"].items())
    report["summary"] = {"routing_passed": report["routing"]["passed"], "routing_total": report["routing"]["total"],
                         "routing_model_cases": report["routing"]["model_backed_executed"],
                         "answer_contracts_passed": sum(row["contract_passed"] for row in report["answers"]),
                         "answer_cases": len(report["answers"]), "human_reviewed_factual_accuracy": None}
    passed = (report["routing"]["passed"] == report["routing"]["total"]
              and all(row["contract_passed"] for row in report["answers"]) and report["model_identity_unchanged"])
    report["stage"] = "completed"
    report["status"] = "checks_passed" if passed else "checks_failed"
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", help="Reachable Ollama URL, or use SHOPASSIST_OLLAMA_URL")
    parser.add_argument("--answer-model", help="Exact registered fine-tuned Ollama model name")
    parser.add_argument("--utility-model", help="Utility model used for routing/rewrite")
    parser.add_argument("--verifier-model", help="Separate verifier model")
    parser.add_argument("--policies-dir", type=Path, default=ROOT / "data/policies")
    parser.add_argument("--report", type=Path, help="Save actual answers, evidence, timings and failures as JSON")
    args = parser.parse_args(argv)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "not_started", "stage": "settings",
              "scope": "Real-model development smoke test; not human-reviewed factual accuracy or release acceptance",
              "production_writes": False, "human_review_required": True}
    started = time.perf_counter()
    def save_report():
        report["seconds"] = time.perf_counter() - started
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_name(args.report.name + ".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.report)
    try:
        settings = Settings.from_env()
        overrides = {name: getattr(args, name) for name in ("ollama_url", "answer_model", "utility_model", "verifier_model") if getattr(args, name)}
        # Reuse cached embeddings only. Provisioning is separate from evaluation.
        settings = replace(settings, **overrides, allow_model_download=False)
        code = run_checks(settings, args.policies_dir, report, checkpoint=save_report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        code = 130
    except Exception as exc:
        report.update(status="infrastructure_error", error_type=type(exc).__name__)
        print(f"Cannot complete {report['stage']}: {type(exc).__name__}. Check Ollama access, installed models, runtime dependencies and cached embeddings.", file=sys.stderr)
        code = 2
    save_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
