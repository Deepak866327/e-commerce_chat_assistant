"""Execute real document extraction and generate unreviewed benchmark artifacts."""

from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_seed import seed_cases
from src.config import ROOT, Settings
from src.datasets import DatasetRepository
from src.documents import extract
from src.experiments import chunk_benchmark
from src.ingestion import IngestionService
from src.security import Principal
from src.store import encode, now


def main():
    settings = Settings.from_env()
    service = IngestionService(settings)
    evidence = []
    chunk_reports = []
    for path in sorted((ROOT / "data" / "policies").iterdir()):
        started = time.perf_counter()
        result = service.ingest(path, actor="offline-audit")
        sections = extract(path, settings)
        evidence.append({"filename": path.name, **result, "sections": len(sections),
                         "pages": sorted({s.page_number for s in sections if s.page_number}),
                         "elapsed_seconds": time.perf_counter() - started})
        chunk_reports.append({"filename": path.name, "experiments": chunk_benchmark(sections, settings)})
    rows = service.store.rows("SELECT * FROM chunks ORDER BY version_id,id")
    for row in rows:
        row["metadata"] = json.loads(row["metadata"])
    cases, missing = seed_cases(rows)
    principal = Principal("offline-draft-author", "admin")
    datasets = DatasetRepository(settings.state_dir / "evaluation" / "datasets", service.store)
    version = datasets.create(cases, principal)
    destination = ROOT / "docs" / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ingestion_smoke.json").write_text(encode({"timestamp": now(), "documents": evidence,
         "vector_indexing_executed": False, "live_inference_executed": False}), encoding="utf-8")
    (destination / "chunk_structure_benchmark.json").write_text(encode(chunk_reports), encoding="utf-8")
    summary = {"dataset_version": version, "count": len(cases), "reviewed": 0,
               "categories": dict(Counter(c.category for c in cases)), "missing_source_groups": missing,
               "quality_metrics": None, "release_gate": False,
               "reason": "Drafts require human ground-truth review; live fine-tuned model is absent"}
    (destination / "benchmark_draft_summary.json").write_text(encode(summary), encoding="utf-8")
    export_dir = ROOT / "data" / "benchmarks"
    export_dir.mkdir(parents=True, exist_ok=True)
    # A draft example is reviewable in Git. Regenerate IDs with seed-benchmark
    # against each deployment's actual index; do not ingest this as knowledge.
    (export_dir / "policy_seed_drafts.json").write_text(encode({"production_knowledge": False,
        "review_required": True, "dataset_version": version, "cases": [c.model_dump() for c in datasets.load(version)]}), encoding="utf-8")
    print(encode(summary))


if __name__ == "__main__":
    main()
