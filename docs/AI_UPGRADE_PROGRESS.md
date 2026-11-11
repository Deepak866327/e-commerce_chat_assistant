# AI upgrade progress

## Baseline complete

- Inspected all 15 tracked files and hidden repository configuration.
- Audit and dependency-ordered plan recorded in adjacent documents.
- Existing tests: none. Existing compilation passed. Ingestion/chat fail because
  LangChain runtime packages are missing. Ollama and weights are unavailable.
- No existing user edits were present. No vectors, documents or model weights
  have been deleted or overwritten.

## Implemented and tested locally

- A: PDF/DOCX/TXT/CSV/approved-FAQ extraction, pages/headings/tables, opt-in OCR,
  bounds, hashes, duplicate detection, immutable blobs, extraction snapshots,
  idempotent versions, status/errors, safe reindex/delete and recovery.
- B: section/FAQ/table-aware chunks, parent metadata, configurable token budgets
  and overlap, full-window embedding pooling, versioned Chroma identities and
  dimension guards. Structure benchmarks at 400/500/700 are recorded; they are
  not retrieval-quality measurements.
- C: schema-constrained Qwen FAQ generation, retries, evidence/duplicate checks,
  lifecycle/history/edit/review/publish, feedback corrections and source expiry.
- D: evaluation-only immutable datasets, JSON/CSV import/export, review, grouped
  seeded splits and duplicate/leakage checks. 54 actual-policy drafts generated;
  all remain unreviewed. No unsupported cases were added to reach 300.
- E: active-version dense retrieval, BM25, RRF, deduplication, metadata filters,
  optional CPU BGE reranker, relevance threshold, bounded context and rewriting.
- F/G: structured answers, source-ID validation, separate claim/evidence checks,
  one correction, abstention, escalation and empirical confidence calibration.
- H/I: deterministic metrics, optional local Ragas/DeepEval adapters, preserved
  baseline and A–E runners, immutable JSON/CSV reports, signed corpus/model-bound
  release gates, regression checks and retrieval/chunk/model experiment commands.
- J/L: redacted bounded traces/retention, feedback ownership, role-authenticated
  CLI, audit, rate limits, retry queue/leases, backups, index rollback, safe
  customer errors and production boot checks.
- K: no admin frontend existed. Extended existing Python architecture with an
  authenticated CLI; preserved Streamlit and voice, added feedback and optional
  customer OIDC. Added explicit legacy rollout mode for the running v1 collection.
- Deployment: Docker/Compose, offline CI and operating runbook added. They are
  source deliverables; this entry does not claim Docker or production acceptance.

## Executed validation

- Local `.venv` now has the original pinned runtime packages, pinned Pydantic and
  PDF/OCR extras. It is isolated from unrelated global packages.
- `.venv/Scripts/python.exe -m unittest discover -v`: **31 passed**.
- After preserving/adapting server voice tests: **42 passed** locally; compilation
  and `pip check` passed again.
- Compilation of app/source/scripts/tests: passed.
- `.venv/Scripts/python.exe -m pip check`: passed.
- Preserved the server-only voice unit/UI tests in the repository. Initial server
  staging ran 42 tests and found six UI mock import failures (the old mock lacked
  the new `submit_feedback` API). Added that mock without weakening assertions;
  live service was not changed by the failed preflight.
- Real extraction of all three policy PDFs and the DOCX: passed.
- Streamlit AppTest startup and customer-message/abstention smoke checks: passed
  with no app exceptions (base environment Streamlit 1.55.0).
- Production boot with no signed active release: correctly refused startup.
- Offline extraction/chunk reports and draft summary: `docs/reports/`.
- Model-backed generation and external judge quality scores: not measured locally.

## Server findings and release status

SSH alias `mitao` is reachable using the owner's SSH configuration. The existing
systemd service at `/home/ubuntu/shopassist` was healthy at inspection, serving
through `api.friendzx.app`. Its directory has no Git metadata and contains
server-specific `deploy/` and tests absent from the original local checkout.
Preserve them and checkpoint code/configuration before a source update.

The server's Ollama confirms `shopassist-finetuned:latest` is **1.5B Q4_K_M**;
preserve it. It also has `qwen2.5:1.5b`; `qwen2.5:3b` is not installed at inspection.
Server RAM: about 11 GiB total, 10 GiB available. Local RAM: about 7.7 GiB total,
0.8 GiB available at inspection; BGE models were not loaded locally.

The grounded production release gate remains **closed**. Compatibility deployment
may use the existing legacy answer pipeline while new capabilities are staged.
No claim is made that the acceptance targets are met or all live integrations are
validated. Commits `a137d80` and `2ff9cd7` were pushed to GitHub `main`.

## Deployment completed: 2026-11-11

- Deployed code commit: `2ff9cd72c7fcc751bbaf138c0adfd37d743bcdd3`.
- Server: SSH alias `mitao`; existing `shopassist.service` and HTTPS proxy retained.
- URL: https://api.friendzx.app . Local health returned `ok` after restart.
- Server preflight: **42 tests passed** in 5.973 seconds; compilation and package
  dependency check passed. Server runtime is Python 3.12.3 / Pydantic 2.14.2.
  Its working virtual environment was preserved rather than unnecessarily replaced.
- Actual Streamlit/Ollama policy-answer smoke passed in 40.3 seconds. This checks
  functional chat and source retrieval, not factual acceptance or latency targets.
- Public HTTPS, WebSocket and chat rendering passed in 0.8 seconds.
- Live pipeline: explicit **legacy compatibility mode**. New platform source,
  operations and feedback are installed; the grounded retrieval/verification path
  is **not activated** and its release gate remains closed.
- Existing model weights, policy documents, vector collection, deployment scripts,
  TLS/proxy configuration and server voice tests were preserved.
- Rollback checkpoint:
  `/home/ubuntu/shopassist-releases/20261111T080852Z-2ff9cd7/checkpoint`.
- Deployment manifest: `docs/reports/server_deployment.json`; server also keeps
  `/home/ubuntu/shopassist/ai-release.json`.

## Remaining work and exact next steps

1. Provision the separate Qwen 3B FAQ/verifier model on suitable server hardware.
2. Execute real new-index Chroma/embedding, OCR/Tesseract, FAQ and Ragas/DeepEval integration
   checks. Optional judge dependency ranges need a validated resolved lock.
3. Stage a complete new index, regenerate benchmark IDs against it, human-review
   and expand the 54 drafts toward the supported target, including at least 30
   held-out reviewed cases. Human ground truth cannot be approved by this agent.
4. Run A–E and 400/500/700/MiniLM/BGE experiments, diagnose failures and repeat.
   Keep release closed until correctness/security/regression gates pass.
5. Activate the measured index, switch the server to grounded mode, verify the
   actual fine-tuned model digest, then execute live customer/voice/feedback tests.

Primary changed files are `app.py`, `src/ingest.py`, `src/rag_memory.py`, new
`src/{documents,chunking,ingestion,store,vector_index,faq,datasets,retrieval,
chat_service,verification,evaluation,evaluation_judges,experiments,benchmarks,
calibration,observability,jobs,security,release,admin}.py`, configuration/schema
modules, tests, scripts, dependency extras, deployment files and documentation.
