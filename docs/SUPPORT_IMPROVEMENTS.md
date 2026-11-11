# Incremental customer-support improvements

Based on upstream `ecb664347232e811954ec9aa92ff0bcb339fdc1d`. The repository had
already replaced the original FAQ indexer and added versioned ingestion, hybrid
retrieval, independent verification, human-reviewed FAQ publication and a staff
ticket workspace. This continuation preserves those implementations.

## Implemented in order

1. One publication gate for FAQ suggestions, retrieval aliases and exports.
   FAQs require a non-empty human reviewer, verified content, published status,
   a current undeleted source and a matching source chunk/version. Editing,
   withdrawal, source update or deletion removes customer availability without
   rebuilding a separate FAQ vector collection. Malformed records are skipped.
2. Whole-message Hindi/Hinglish handoff and negation shortcuts, plus richer
   structured-classification instructions. Account-action replies ask a relevant
   question about refund confirmation, dispatch, payment error or tracking instead
   of giving an unrelated policy summary or redirecting users out of support.
3. Explicit English/Hindi/Hinglish chat language, independent of speech language.
   Fixed replies and generated policy prompts honor it in both pipelines.
4. Bounded, redacted customer issue context and attempted steps survive the
   recent-history window. Reports are labeled as unverified user context, never
   official policy or checked account records. The initial issue is retained;
   topic changes, new issues and post-resolution episodes start fresh context.
   Only explicit customer resolution phrases mark customer-reported resolution;
   thanks and assistant assertions do not. Clear conversation clears this context.
   Account-action follow-ups attach to an existing owned open ticket; ownership
   checks prevent cross-session reuse. The staff workspace remains the response
   channel; no external notification, CRM or transaction backend was added.
5. Generic training data becomes a pending-review queue. Reviewed export requires
   a reviewer and leakage group; flagged company claims need approved context.
   Obvious completed/promised backend-action claims are rejected. Flags are
   heuristic, so human review must check meaning, policy conditions and capability
   claims that the patterns miss. Preparation does not train or replace weights.
6. A 25-case synthetic development routing set separates general policies from
   complaints/actions, Hindi/Hinglish requests, negation and contextual follow-ups.
   Offline execution reports model cases as skipped, not passed. This set stays
   separate from production FAQs and human-reviewed release evaluation data.

## Validation and limits

Commands:

```bash
python -m unittest discover -q
python -m compileall -q app.py src scripts tests
python -m pip check
git diff --check
python scripts/evaluate_support_routes.py
```

Baseline: 76 offline tests passed on the current upstream source. Final executed
results are recorded in the pull request. The route fixture has 17 deterministic
conversation/procedure cases and 8 model cases. Tests also exercise real SQLite operations,
source ingestion and Streamlit UI behavior using simulated inference.

Local Ollama is unavailable and the external fine-tuned GGUF is absent. Prompt
changes, actual multilingual generation, retrieval quality and customer resolution
rates are not measured by these offline checks. No live deployment was performed.
Existing production configuration, weights, index activation and release gates
are unchanged. Grounded generation now identifies its prompt as
`shopassist-grounded-v3`; rerun measured acceptance before activating it.

## Remaining measured work

### Run against actual Ollama models

The comprehensive runner uses the configured utility, fine-tuned answer and
verifier models through the real Ollama API. It records their exact digests,
executes all 25 routing cases, builds a temporary dense/BM25 candidate index from
the repository policy PDFs, and exercises 11 actual answer/abstention/action cases.
It saves answers, cited evidence, verification results, failures and timings.
Only cached embeddings are used; it never downloads replacement model weights,
activates an index or writes production tickets/data. Reports are checkpointed
after routing and each answer. Failed infrastructure is exit code 2; failed
checks are exit code 1; a completed passing development run is exit code 0.

Use an isolated checkout on the server where Ollama and the existing models are
installed. The existing server virtual environment can run the evaluation without
replacing the service or its environment:

```bash
git clone --branch improve-support-resolution https://github.com/Deepak866327/e-commerce_chat_assistant.git shopassist-model-eval
cd shopassist-model-eval
/home/ubuntu/shopassist/.venv/bin/python scripts/evaluate_real_models.py --report /tmp/shopassist-real-model-report.json
```

For another configured Python environment, run
`python scripts/evaluate_real_models.py --report ../shopassist-real-model-report.json`.
Defaults are `http://localhost:11434`, answer model `shopassist-finetuned` and
utility/verifier `qwen2.5:1.5b`, unless corresponding `SHOPASSIST_*` variables override
them. CLI overrides include `--ollama-url`, `--answer-model`, `--utility-model`,
`--verifier-model` and `--policies-dir`. Use the existing server environment or
install `requirements.txt` in a separate evaluation virtual environment.

This is an actual model-backed development smoke test, not a human-reviewed
correctness benchmark. A passing model-judged grounding result still needs review.
An unreachable Ollama service or missing model produces an infrastructure error,
never a passing result. The remote machine/endpoint must be reachable from the
environment that runs this command; a laptop's localhost is not this workspace.

1. On the model host, run `python scripts/evaluate_support_routes.py --model-backed`.
   Diagnose all 8 real-classifier cases and add observed customer failures.
2. Human-review the existing policy benchmark drafts and representative English,
   Hindi and Hinglish examples; keep held-out evaluation separate from training.
3. Use the existing A–E/embedding/chunk experiments and release gates to measure
   retrieval relevance, factual support, appropriate abstention and regression.
   Tune thresholds from those results. No arbitrary threshold adjustment or
   unmeasured embedding/model swap was made in this continuation.
4. Stage the changes, exercise actual model/voice/customer/staff interactions,
   then deploy the reviewed commit through the existing release process. Track
   customer-confirmed resolution separately from a helpful response or a saved
   ticket. A ticket is a queued request, not proof of resolution.
