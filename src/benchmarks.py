"""Retrieval-only chunk/model/reranker experiments; never switch the active index."""

from dataclasses import replace
import json
from pathlib import Path
import time
from uuid import uuid4

from src.evaluation import average, deterministic_metrics, percentile95
from src.ingestion import IngestionService
from src.retrieval import BGEReranker, Retriever
from src.store import encode, now
from src.vector_index import ChromaIndex


def retrieval_grid(store, settings, datasets, version, principal, models, sizes=(400, 500, 700), split="development", rerank=False):
    principal.require("evaluate")
    cases = datasets.load(version, split)
    if not cases or any(not c.reviewed for c in cases):
        raise ValueError("Retrieval benchmarks require reviewed ground truth")
    if any(c.answerable and set(c.expected_evidence) != set(c.expected_chunk_ids) for c in cases):
        raise ValueError("Cross-chunking experiments require exact source evidence for every expected chunk")
    versions = store.rows("""SELECT v.id FROM document_versions v JOIN documents d ON d.id=v.document_id
        WHERE d.deleted=0 AND v.status IN ('INDEXED','EXTRACTED') AND v.version=(SELECT MAX(v2.version)
        FROM document_versions v2 WHERE v2.document_id=d.id AND v2.status IN ('INDEXED','EXTRACTED'))""")
    experiments = []
    for model in models:
        for size in sizes:
            configuration = replace(settings, embedding_model=model, chunk_tokens=size, chunk_overlap=75)
            identity, spec = configuration.index_identity()
            started = time.perf_counter()
            try:
                vector = ChromaIndex(configuration)
                service = IngestionService(configuration, store, vector)
                for source in versions:
                    service.index_extracted(source["id"])
                build_seconds = time.perf_counter() - started
                reranker = BGEReranker(configuration) if rerank else None
                retriever = Retriever(store, configuration, vector, reranker, candidate_index=identity)
                rows = []
                for case in cases:
                    response_start = time.perf_counter()
                    found = retriever.search(case.question, hybrid=True, rerank=rerank)
                    metrics = deterministic_metrics(case, {"retrieved_chunks": found["chunks"],
                        "retrieval_candidates": found["candidates"], "sources": [], "answerable": False})
                    rows.append({"case_id": case.id, "metrics": {k: metrics[k] for k in
                        ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "context_precision", "context_recall")},
                        "latency_seconds": time.perf_counter() - response_start})
                experiments.append({"index_id": identity, "spec": spec, "dimension": vector.dimension,
                    "reranker": configuration.reranker_model if rerank else None,
                    "build_seconds": build_seconds, "metrics": {k: average([r["metrics"][k] for r in rows]) for k in rows[0]["metrics"]},
                    "p95_latency_seconds": percentile95([r["latency_seconds"] for r in rows]), "results": rows, "error": None})
            except Exception as exc:
                experiments.append({"index_id": identity, "spec": spec, "metrics": None, "error": type(exc).__name__})
    report = {"benchmark_id": "GRID_" + uuid4().hex, "dataset_version": version, "split": split,
              "timestamp": now(), "experiments": experiments, "active_index_changed": False,
              "release_gate": False, "note": "Retrieval experiments alone cannot authorize a production release"}
    output = settings.state_dir / "evaluation" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    (output / (report["benchmark_id"] + ".json")).write_text(encode(report), encoding="utf-8")
    return report
