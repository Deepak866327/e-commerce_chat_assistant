"""Authenticated local operations; no duplicate admin frontend or public endpoints."""

import argparse
from getpass import getpass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import threading

from src.config import Settings
from src.datasets import DatasetRepository
from src.documents import extract, validate_file
from src.evaluation import EvaluationRunner, compare_reports
from src.experiments import chunk_benchmark, legacy_runner, upgraded_runner
from src.faq import FAQService
from src.ingestion import IngestionService
from src.jobs import JobQueue
from src.observability import Monitor
from src.ollama_client import OllamaClient
from src.security import authenticate, password_hash
from src.store import Store, encode
from src.vector_index import ChromaIndex, WindowEmbeddings
from src.verification import GroundingVerifier


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--role", choices=["admin", "reviewer", "evaluator", "support"], default="admin")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("hash-password", help="Create a role password hash using a hidden prompt")
    commands.add_parser("status")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--extract-only", action="store_true")
    ingest.add_argument("--ocr", action="store_true")
    ingest.add_argument("--enqueue", action="store_true")
    index = commands.add_parser("index-version")
    index.add_argument("version_id")
    delete = commands.add_parser("delete-document")
    delete.add_argument("document_id")
    recover = commands.add_parser("recover-ingestion")
    recover.add_argument("version_id")
    faq = commands.add_parser("generate-faq")
    faq.add_argument("chunk_id")
    faq.add_argument("--count", type=int, default=None)
    faq.add_argument("--enqueue", action="store_true")
    commands.add_parser("list-faqs")
    import_faq = commands.add_parser("import-faq")
    import_faq.add_argument("path", type=Path)
    import_faq.add_argument("faq_id")
    import_faq.add_argument("chunk_id")
    commands.add_parser("adopt-legacy-sources")
    tickets = commands.add_parser("tickets")
    tickets.add_argument("--include-closed", action="store_true")
    detail = commands.add_parser("ticket-detail")
    detail.add_argument("reference")
    update = commands.add_parser("ticket-update")
    update.add_argument("reference")
    update.add_argument("status", choices=["OPEN", "ASSIGNED", "IN_PROGRESS", "WAITING_ON_CUSTOMER", "RESOLVED", "CLOSED"])
    update.add_argument("--assignee", default="")
    reply = commands.add_parser("ticket-reply")
    reply.add_argument("reference")
    reply.add_argument("message", type=Path)
    commands.add_parser("prune-tickets")
    seed = commands.add_parser("seed-benchmark")
    seed.add_argument("--index-id")
    rebuild = commands.add_parser("rebuild-index")
    rebuild.add_argument("--ocr", action="store_true")
    edit = commands.add_parser("edit-faq")
    edit.add_argument("faq_id")
    edit.add_argument("changes", type=Path)
    transition = commands.add_parser("faq-transition")
    transition.add_argument("faq_id")
    transition.add_argument("status", choices=["PENDING_REVIEW", "APPROVED", "REJECTED", "PUBLISHED"])
    import_data = commands.add_parser("dataset-import")
    import_data.add_argument("path", type=Path)
    import_data.add_argument("--seed", type=int, default=42)
    review = commands.add_parser("dataset-review")
    review.add_argument("version")
    review.add_argument("case_ids", nargs="+")
    export = commands.add_parser("dataset-export")
    export.add_argument("version")
    export.add_argument("destination", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset_version")
    evaluate.add_argument("--variant", choices=["A", "B", "C", "D", "E", "all"], default="E")
    evaluate.add_argument("--split", choices=["development", "validation", "test"], default="test")
    evaluate.add_argument("--judge", choices=["none", "ragas", "deepeval", "both"], default="none")
    evaluate.add_argument("--index-id")
    compare = commands.add_parser("compare")
    compare.add_argument("reports", nargs="+", type=Path)
    activate = commands.add_parser("activate-index")
    activate.add_argument("index_id")
    activate.add_argument("report", type=Path)
    activate.add_argument("--baseline-report", type=Path)
    commands.add_parser("rollback-index")
    worker = commands.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    commands.add_parser("monitor")
    commands.add_parser("feedback")
    correction = commands.add_parser("feedback-correction")
    correction.add_argument("feedback_id")
    correction.add_argument("chunk_id")
    correction.add_argument("candidate", type=Path)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("dataset_version")
    calibration.add_argument("labels", type=Path)
    commands.add_parser("prune-traces")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    benchmark = commands.add_parser("chunk-benchmark")
    benchmark.add_argument("path", type=Path)
    grid = commands.add_parser("retrieval-grid")
    grid.add_argument("dataset_version")
    grid.add_argument("--models", nargs="+", default=None)
    grid.add_argument("--split", choices=["development", "validation", "test"], default="development")
    grid.add_argument("--rerank", action="store_true")
    return root


def execute(args, principal, settings):
    store = Store(settings.state_dir / "control.sqlite3")
    datasets = DatasetRepository(settings.state_dir / "evaluation" / "datasets", store)
    client = OllamaClient(settings)
    verifier = GroundingVerifier(client, settings.verifier_model)

    def faq_service():
        return FAQService(store, settings, client, verifier, WindowEmbeddings(settings))

    def ingestion(index=True):
        return IngestionService(settings, store, ChromaIndex(settings) if index else None)

    command = args.command
    if command in {"tickets", "ticket-detail", "ticket-update", "ticket-reply", "prune-tickets"}:
        from src.tickets import TicketService
        tickets = TicketService(store, settings)
        if command == "tickets":
            return tickets.list(principal, args.include_closed)
        if command == "ticket-detail":
            return tickets.detail(args.reference, principal)
        if command == "ticket-update":
            tickets.update(args.reference, args.status, args.assignee, principal)
            return {"reference": args.reference, "status": args.status}
        if command == "ticket-reply":
            tickets.reply(args.reference, args.message.read_text(encoding="utf-8"), principal=principal)
            return {"reference": args.reference, "reply_saved": True}
        tickets.prune(principal)
        return {"retention_applied": True}
    if command == "adopt-legacy-sources":
        from src.legacy_sources import adopt_legacy_sources
        return adopt_legacy_sources(settings, store, principal)
    if command == "status":
        principal.require("read")
        return {"active_index": store.setting("active_index"),
                "documents": store.rows("SELECT * FROM documents"),
                "versions": store.rows("SELECT id,document_id,version,status,error FROM document_versions"),
                "indexes": store.rows("SELECT * FROM indexes"), "jobs": store.rows("SELECT id,kind,status,attempts,error FROM jobs")}
    if command == "ingest":
        principal.require("ingest")
        if args.enqueue:
            source = validate_file(args.path, settings)
            digest = sha256(source.read_bytes()).hexdigest()
            folder = settings.state_dir / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            snapshot = folder / (digest + source.suffix.lower())
            if not snapshot.exists():
                shutil.copyfile(source, snapshot)
            return {"job_id": JobQueue(store).enqueue("ingest", {"path": str(snapshot), "source_key": source.name,
                    "extract_only": args.extract_only, "ocr": args.ocr}, principal, source.name.casefold() + digest + str(args.extract_only))}
        return ingestion(not args.extract_only).ingest(args.path, actor=principal.name, use_ocr=args.ocr)
    if command == "index-version":
        principal.require("ingest")
        ingestion().index_extracted(args.version_id)
        return {"version_id": args.version_id, "status": "INDEXED"}
    if command == "delete-document":
        principal.require("delete")
        ingestion(False).delete_document(args.document_id, principal.name)
        return {"deleted": args.document_id}
    if command == "recover-ingestion":
        principal.require("ingest")
        ingestion(False).recover_ingestion(args.version_id, principal.name)
        return {"version_id": args.version_id, "status": "FAILED", "next_step": "retry the original ingestion after stopping the failed worker"}
    if command == "generate-faq":
        principal.require("ingest")
        if args.enqueue:
            return {"job_id": JobQueue(store).enqueue("generate_faq", {"chunk_id": args.chunk_id, "count": args.count}, principal)}
        return faq_service().generate(args.chunk_id, principal, args.count)
    if command == "list-faqs":
        principal.require("read")
        return store.rows("SELECT * FROM faqs")
    if command == "import-faq":
        principal.require("ingest")
        data = json.loads(args.path.read_text(encoding="utf-8"))
        matches = [faq for faq in data.get("faqs", [])
                   if faq.get("faq_id") == args.faq_id and faq.get("status") == "active"]
        if len(matches) != 1:
            raise ValueError("Choose one active FAQ ID from the local catalog")
        service = faq_service()
        source = service._chunk(args.chunk_id)
        faq = matches[0]
        candidate = {"question": faq["question"], "answer": faq["answer"],
                     "category": faq.get("category", "general"), "question_type": "direct",
                     "supporting_text": source["text"]}
        return {"faq_id": service.import_candidate(args.chunk_id, candidate, principal),
                "status": "PENDING_REVIEW"}
    if command == "seed-benchmark":
        principal.require("dataset")
        from src.benchmark_seed import seed_cases
        from src.retrieval import Retriever
        if args.index_id:
            chunks = Retriever(store, settings, candidate_index=args.index_id).corpus()
        else:
            chunks = store.rows("SELECT * FROM chunks ORDER BY version_id,id")
            for chunk in chunks:
                chunk["metadata"] = json.loads(chunk["metadata"])
        cases, missing = seed_cases(chunks)
        return {"dataset_version": datasets.create(cases, principal), "count": len(cases),
                "missing_source_groups": missing, "reviewed": False}
    if command == "rebuild-index":
        principal.require("ingest")
        service = ingestion()
        versions = store.rows("""SELECT v.id FROM document_versions v JOIN documents d ON d.id=v.document_id
            WHERE d.deleted=0 AND v.status IN ('EXTRACTED','INDEXED') AND v.version=(SELECT MAX(v2.version)
            FROM document_versions v2 WHERE v2.document_id=d.id AND v2.status IN ('EXTRACTED','INDEXED'))""")
        for version in versions:
            service.index_extracted(version["id"], args.ocr)
        return {"candidate_index": settings.index_identity()[0], "versions_indexed": len(versions)}
    if command == "edit-faq":
        principal.require("review")
        faq_service().edit(args.faq_id, json.loads(args.changes.read_text(encoding="utf-8")), principal)
        return {"faq_id": args.faq_id, "status": "PENDING_REVIEW"}
    if command == "faq-transition":
        faq_service().transition(args.faq_id, args.status, principal)
        return {"faq_id": args.faq_id, "status": args.status}
    if command == "dataset-import":
        return {"dataset_version": datasets.import_file(args.path, principal, args.seed)}
    if command == "dataset-review":
        return {"dataset_version": datasets.review(args.version, args.case_ids, principal, store)}
    if command == "dataset-export":
        principal.require("read")
        datasets.export(args.version, args.destination)
        return {"exported": str(args.destination)}
    if command == "evaluate":
        principal.require("evaluate")
        judge = None
        if args.judge != "none":
            from src.evaluation_judges import CombinedJudge, DeepEvalJudge, RagasJudge
            judges = []
            if args.judge in {"ragas", "both"}:
                judges.append(RagasJudge(settings))
            if args.judge in {"deepeval", "both"}:
                judges.append(DeepEvalJudge(settings, client))
            judge = CombinedJudge(judges, verifier)
        evaluator = EvaluationRunner(datasets, settings, settings.state_dir / "evaluation" / "reports", judge)
        index_id = args.index_id or settings.index_identity()[0]
        reports = []
        for variant in (["A", "B", "C", "D", "E"] if args.variant == "all" else [args.variant]):
            runner = legacy_runner(store) if variant == "A" else upgraded_runner(store, settings, variant, index_id)
            reports.append(evaluator.run(args.dataset_version, runner, principal, variant=variant, split=args.split, index_id=index_id))
        return {"reports": [{"run_id": r["run_id"], "metrics": r["metrics"], "release_gate": r["release_gate"], "gate_failures": r["gate_failures"]} for r in reports],
                "comparison": compare_reports(reports) if len(reports) > 1 else None}
    if command == "compare":
        principal.require("evaluate")
        return compare_reports([json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
    if command == "activate-index":
        principal.require("activate")
        report_root = (settings.state_dir / "evaluation" / "reports").resolve()
        if args.report.resolve().parent != report_root:
            raise ValueError("Use an immutable evaluation report from this installation")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report.get("chunking_configuration") != settings.index_identity()[1] or report.get("model_version") != settings.answer_model:
            raise ValueError("Report does not match the configured model/index")
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8")) if args.baseline_report else None
        ingestion(False).activate_index(args.index_id, report, principal.name, baseline)
        return {"active_index": args.index_id}
    if command == "rollback-index":
        principal.require("activate")
        ingestion(False).rollback_index(principal.name)
        return {"active_index": store.setting("active_index")}
    if command == "worker":
        principal.require("ingest")
        queue = JobQueue(store)
        handlers = {"ingest": lambda p: ingestion(not p["extract_only"]).ingest(p["path"], p["source_key"], principal.name, p["ocr"]),
                    "generate_faq": lambda p: faq_service().generate(p["chunk_id"], principal, p["count"])}
        if args.once:
            return queue.run_once(handlers)
        stop = threading.Event()
        while not stop.is_set():
            result = queue.run_once(handlers)
            if result:
                print(encode(result), flush=True)
            stop.wait(2)
    if command in {"monitor", "feedback", "prune-traces"}:
        principal.require("read")
        monitor = Monitor(store, settings)
        if command == "feedback":
            return store.rows("SELECT f.*,t.body FROM feedback f JOIN traces t ON t.id=f.trace_id ORDER BY f.created_at DESC LIMIT 200")
        if command == "prune-traces":
            principal.require("delete")
            monitor.prune()
        return monitor.summary()
    if command == "feedback-correction":
        principal.require("review")
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        return {"faq_id": faq_service().correction_from_feedback(args.feedback_id, args.chunk_id, candidate, principal), "status": "PENDING_REVIEW"}
    if command == "calibrate":
        principal.require("review")
        from src.calibration import fit_calibration
        cases = datasets.load(args.dataset_version)
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        allowed = {c.id for c in cases if c.split == "development" and c.reviewed}
        if any(r.get("case_id") not in allowed for r in labels):
            raise ValueError("Use reviewed development examples; held-out cases cannot fit calibration")
        calibration = fit_calibration(labels, args.dataset_version)
        calibration["index_id"] = store.setting("active_index")
        calibration["model_version"] = settings.answer_model
        store.set_setting("confidence_calibration", encode(calibration))
        store.audit(principal.name, "confidence.calibrate", args.dataset_version)
        return calibration
    if command == "backup":
        principal.require("backup")
        if args.destination.exists():
            raise ValueError("Backup destination must be new")
        # Stop workers/customer app before this operation: SQLite backup itself
        # is consistent, but a coordinated vector/blob snapshot needs quiescence.
        args.destination.mkdir(parents=True)
        store.backup(args.destination / "control.sqlite3")
        for name in ("documents", "evaluation"):
            source = settings.state_dir / name
            if source.exists():
                shutil.copytree(source, args.destination / name)
        if settings.vector_dir.exists():
            shutil.copytree(settings.vector_dir, args.destination / "vector_store")
        return {"backup": str(args.destination), "requires_quiesced_writers": True}
    if command == "chunk-benchmark":
        principal.require("evaluate")
        return chunk_benchmark(extract(args.path, settings), settings)
    if command == "retrieval-grid":
        principal.require("evaluate")
        from src.benchmarks import retrieval_grid
        return retrieval_grid(store, settings, datasets, args.dataset_version, principal,
                              args.models or [settings.embedding_model], split=args.split, rerank=args.rerank)
    raise ValueError("Unknown operation")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "hash-password":
        print(password_hash(getpass("New role password: ")))
        return 0
    try:
        principal = authenticate(args.role, getpass(f"{args.role.capitalize()} password: "))
        print(encode(execute(args, principal, Settings.from_env())))
        return 0
    except (ValueError, PermissionError, FileNotFoundError, RuntimeError) as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
