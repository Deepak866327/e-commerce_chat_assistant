# AI upgrade implementation plan

Preserve Python/Streamlit, local Ollama, Chroma, MiniLM, the fine-tuned model
registration, voice and the `ask_shopassist` interface. There is no justification
for adding a Node server, external database, duplicate admin UI or remote tracing
service. Use a small local SQLite control store for transactions, jobs, document
versions, workflow history and redacted traces; keep Chroma for dense vectors.

1. A/B foundation: reusable extraction, bounded uploads, versioned ingestion,
   metadata, FAQ/heading/table-aware token chunks, staging and safe activation,
   document deletion/rollback, tests and operation commands.
2. C/D: schema-constrained Ollama FAQ generation, evidence/duplicate checks,
   lifecycle transitions, editing/approval/publication, strictly separate
   versioned human-reviewed benchmark files and reproducible grouped splits.
3. E/F/G: lazy model/index loading, dense + BM25/RRF, optional CPU BGE reranker,
   guarded rewriting, source-isolated structured generation, validated citations,
   independent verification, one correction and fail-closed abstention.
4. H/I: deterministic retrieval/citation/abstention metrics, explicit optional
   Ragas/DeepEval judges, A–E experiments and chunk/model benchmarks, immutable
   reports and regression/acceptance gates. No proxy metric may be represented as
   measured semantic correctness or calibrated probability.
5. J/L: bounded PII-redacted traces, feedback, authenticated role-checked local
   admin CLI, retryable jobs, rate limits, audit/retention and backup/runbooks.
6. K: no admin frontend exists. Supply operations in the existing Python project,
   not a second frontend. Preserve and extend the customer UI only where useful.

Every phase includes executable offline tests; model-backed checks are reported
separately. Production activation of new embedding indexes requires a measured
validation gate and keeps the previous index for rollback. Generated FAQs and
benchmark ground truth always require a human reviewer. Release stays closed
until reviewed held-out metrics, live model checks and deployment security pass.

Initial quality targets are the user-specified Recall@5 .95, MRR .85, context
precision .85, context recall .90, faithfulness .95, correctness .90, citation
accuracy .98, abstention accuracy .95. They are targets, never default results.
