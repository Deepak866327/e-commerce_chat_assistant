"""Reproducible evaluation, honest missing metrics and independent release gates."""

from datetime import datetime, timezone
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time
import tracemalloc
from uuid import uuid4

from src.store import encode
from src.index_state import corpus_fingerprint, sign_report
from src.verification import normalize

TARGETS = {"recall_at_5": .95, "mrr": .85, "context_precision": .85, "context_recall": .90,
           "faithfulness": .95, "answer_correctness": .90, "citation_accuracy": .98, "abstention_accuracy": .95}
VARIANTS = {"A": "existing_rag", "B": "improved_chunking_dense", "C": "hybrid",
            "D": "hybrid_reranking", "E": "hybrid_reranking_verification"}


def retrieval_metrics(expected_ids, retrieved_ids):
    expected = set(expected_ids)
    if not expected:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None,
                "mrr": None, "context_precision": None, "context_recall": None}
    ranked = list(dict.fromkeys(retrieved_ids))
    hits = [i + 1 for i, cid in enumerate(ranked) if cid in expected]
    result = {f"recall_at_{k}": len(expected & set(ranked[:k])) / len(expected) for k in (1, 3, 5)}
    result.update(mrr=1 / hits[0] if hits else 0.0,
                  context_precision=len(expected & set(ranked)) / len(ranked) if ranked else 0.0,
                  context_recall=len(expected & set(ranked)) / len(expected))
    return result


def deterministic_metrics(case, result):
    def coverage(chunks):
        identities = []
        for chunk in chunks:
            matches = {cid for cid, excerpt in case.expected_evidence.items()
                       if chunk.get("metadata", {}).get("document_id") in case.expected_document_ids
                       and normalize(excerpt) in normalize(chunk["text"])}
            if chunk["id"] in case.expected_chunk_ids:
                matches.add(chunk["id"])
            identities.append(matches)
        return identities

    expected = set(case.expected_chunk_ids)
    context = coverage(result.get("retrieved_chunks", []))
    candidates = coverage(result.get("retrieval_candidates", result.get("retrieved_chunks", [])))
    metrics = {f"recall_at_{k}": len(set().union(*candidates[:k])) / len(expected) if expected else None for k in (1, 3, 5)}
    first = next((rank for rank, matches in enumerate(candidates, 1) if matches), None)
    metrics["mrr"] = (1 / first if first else 0.0) if expected else None
    metrics["context_precision"] = sum(bool(matches) for matches in context) / len(context) if context and expected else (0.0 if expected else None)
    metrics["context_recall"] = len(set().union(*context)) / len(expected) if expected else None
    # Exact match is intentionally separate from semantic answer correctness.
    metrics["answer_exact_match"] = float(normalize(result.get("answer", "")) == normalize(case.expected_answer)) if case.answerable else None
    metrics["abstention_accuracy"] = float(bool(result.get("answerable")) == case.answerable)
    sources = result.get("sources", [])
    allowed = {c["id"]: c for c in result.get("retrieved_chunks", [])}
    valid = 0
    for citation in sources:
        chunk = allowed.get(citation.get("chunk_id"))
        if chunk:
            meta = chunk["metadata"]
            valid += int(citation.get("document_id") == meta.get("document_id") and citation.get("page_number") == meta.get("page_number"))
    metrics["citation_id_accuracy"] = valid / len(sources) if sources else (0.0 if result.get("answerable") else None)
    # A real source ID alone does not establish that it supports the claim.
    metrics.update(faithfulness=None, answer_relevancy=None, answer_correctness=None,
                   citation_accuracy=None, hallucination_rate=None)
    return metrics


def average(values):
    measured = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(measured) if measured else None


def percentile95(values):
    return sorted(values)[max(0, math.ceil(len(values) * .95) - 1)] if values else None


def release_gate(metrics, count, cases, errors=0):
    failures = [name for name, target in TARGETS.items() if metrics.get(name) is None or metrics[name] < target]
    if count < 30:
        failures.append("insufficient_held_out_cases_minimum_30")
    if not cases or not all(c.reviewed and c.reviewer and c.split == "test" for c in cases):
        failures.append("human_reviewed_held_out_test_required")
    if not any(c.answerable for c in cases) or not any(not c.answerable for c in cases):
        failures.append("answerable_and_unanswerable_cases_required")
    if errors:
        failures.append("execution_errors")
    return not failures, failures


class EvaluationRunner:
    def __init__(self, datasets, settings, output_dir, judge=None):
        self.datasets, self.settings, self.judge = datasets, settings, judge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, version, runner, principal, *, variant="E", split="test", index_id=None):
        principal.require("evaluate")
        if variant not in VARIANTS:
            raise ValueError("Unknown A/B variant")
        cases = self.datasets.load(version, split)
        if not cases or not all(c.reviewed and c.reviewer for c in cases):
            raise ValueError("All selected evaluation cases require human review")
        run_id = "RUN_" + uuid4().hex
        evaluated_index = index_id or self.settings.index_identity()[0]
        starting_fingerprint = corpus_fingerprint(self.datasets.store, evaluated_index) if self.datasets.store else None
        results, latencies = [], []
        for case in cases:
            started = time.perf_counter()
            tracemalloc.start()
            try:
                result = runner(case)
                response_seconds = time.perf_counter() - started
                metrics = deterministic_metrics(case, result)
                if self.judge and case.answerable:
                    measured = self.judge.measure(case, result)
                    metrics.update(measured)
                    faithful = metrics.get("faithfulness")
                    metrics["hallucination_rate"] = 1.0 - faithful if faithful is not None else None
                    # Citation entailment is judged per cited chunk, independently
                    # from referential integrity and the generation verifier.
                error = result.get("error")
            except Exception as exc:
                result, metrics, error = {}, {}, type(exc).__name__
                response_seconds = time.perf_counter() - started
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            total_seconds = time.perf_counter() - started
            latency = response_seconds
            latencies.append(latency)
            results.append({"case_id": case.id, "category": case.category, "metrics": metrics,
                            "latency_seconds": latency, "peak_python_bytes": peak, "error": error,
                            "evaluation_total_seconds": total_seconds,
                            "answer": result.get("answer"), "answerable": result.get("answerable"),
                            "retrieved_chunk_ids": [c["id"] for c in result.get("retrieved_chunks", [])],
                            "sources": result.get("sources", []),
                            "retrieval_seconds": result.get("retrieval_seconds"),
                            "reranking_seconds": result.get("reranking_seconds"),
                            "inference_seconds": result.get("inference_seconds")})
        names = sorted({key for row in results for key in row["metrics"]})
        aggregated = {name: average([r["metrics"].get(name) for r in results]) for name in names}
        errors = sum(bool(r["error"]) for r in results)
        # Errors count as failed outcomes instead of disappearing from the mean.
        if errors:
            for name in TARGETS:
                if aggregated.get(name) is not None:
                    measured_count = sum(r["metrics"].get(name) is not None for r in results)
                    aggregated[name] *= measured_count / (measured_count + errors)
        gate, failures = release_gate(aggregated, len(cases), cases, errors)
        ending_fingerprint = corpus_fingerprint(self.datasets.store, evaluated_index) if self.datasets.store else None
        if starting_fingerprint != ending_fingerprint:
            gate = False
            failures.append("corpus_changed_during_evaluation")
        if variant != "E":
            gate = False
            failures.append("verified_variant_E_required")
        if not getattr(runner, "model_digest", None):
            gate = False
            failures.append("model_artifact_identity_unverified")
        identity, spec = self.settings.index_identity()
        report = {"run_id": run_id, "dataset_version": version, "split": split, "variant": variant,
                  "variant_description": VARIANTS[variant], "index_id": index_id or identity,
                  "model_version": self.settings.answer_model, "embedding_model": self.settings.embedding_model,
                  "model_digest": getattr(runner, "model_digest", None),
                  "embedding_revision": self.settings.embedding_revision, "prompt_version": "shopassist-grounded-v2",
                  "chunking_configuration": spec, "retrieval_configuration": {
                      k: getattr(self.settings, k) for k in ("dense_k", "keyword_k", "fusion_k", "context_k", "rrf_k", "min_dense_similarity")},
                  "reranker_version": self.settings.reranker_model if variant in {"D", "E"} else None,
                  "judge": getattr(self.judge, "version", None), "timestamp": datetime.now(timezone.utc).isoformat(),
                  "count": len(cases), "metrics": aggregated, "targets": TARGETS,
                  "performance": {"average_latency_seconds": average(latencies), "p95_latency_seconds": percentile95(latencies),
                      "average_retrieval_seconds": average([r["retrieval_seconds"] for r in results]),
                      "average_reranking_seconds": average([r["reranking_seconds"] for r in results]),
                      "average_inference_seconds": average([r["inference_seconds"] for r in results]),
                      "error_rate": errors / len(cases), "peak_python_bytes": max(r["peak_python_bytes"] for r in results),
                      "resource_note": "Python allocation peak only; native model RAM/VRAM requires host monitoring"},
                  "release_gate": gate, "gate_failures": failures, "results": results,
                  "failure_examples": [r for r in results if r["error"] or any(v is not None and v < TARGETS[k] for k, v in r["metrics"].items() if k in TARGETS)]}
        if self.datasets.store:
            report["corpus_fingerprint"] = ending_fingerprint
            report["signature"] = sign_report(self.datasets.store, report)
        self.save(report)
        return report

    def save(self, report):
        path = self.output_dir / (report["run_id"] + ".json")
        if path.exists():
            raise ValueError("Evaluation reports are immutable")
        path.write_text(encode(report), encoding="utf-8")
        with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "category", "error", "latency_seconds", *sorted(report["metrics"])])
            writer.writeheader()
            for row in report["results"]:
                writer.writerow({k: row.get(k) for k in ("case_id", "category", "error", "latency_seconds")} | row["metrics"])


def compare_reports(reports, tolerances=None):
    if len(reports) < 2:
        raise ValueError("At least two completed runs are required")
    baseline = reports[0]
    required_equal = ("dataset_version", "split")
    if any(any(r[k] != baseline[k] for k in required_equal) for r in reports[1:]):
        raise ValueError("A/B tests must use the same dataset version and split")
    if any({r["case_id"] for r in report["results"]} != {r["case_id"] for r in baseline["results"]} for report in reports[1:]):
        raise ValueError("A/B tests must evaluate the same cases")
    tolerances = tolerances or {}
    comparisons = []
    for report in reports[1:]:
        deltas, regressions = {}, []
        for metric in TARGETS:
            old, new = baseline["metrics"].get(metric), report["metrics"].get(metric)
            deltas[metric] = new - old if old is not None and new is not None else None
            if deltas[metric] is None or deltas[metric] < -tolerances.get(metric, .02):
                regressions.append(metric)
        old_performance, new_performance = baseline.get("performance", {}), report.get("performance", {})
        for key in ("error_rate", "p95_latency_seconds"):
            old, new = old_performance.get(key), new_performance.get(key)
            deltas[key] = new - old if old is not None and new is not None else None
            if old is not None and new is not None:
                if key == "error_rate" and new > old + .01:
                    regressions.append(key)
                if key == "p95_latency_seconds" and new > max(old * 1.25, old + .25):
                    regressions.append(key)
        comparisons.append({"baseline_run": baseline["run_id"], "candidate_run": report["run_id"],
                            "variant": report["variant"], "deltas": deltas, "regressions": regressions,
                            "regression_gate": not regressions and report["release_gate"]})
    return {"dataset_version": baseline["dataset_version"], "comparisons": comparisons}
