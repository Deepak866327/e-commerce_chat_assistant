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
6. A 24-case synthetic development routing set separates general policies from
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
results are recorded in the pull request. The route fixture has 10 deterministic
shortcut cases and 14 model cases. Tests also exercise real SQLite operations,
source ingestion and Streamlit UI behavior using simulated inference.

Local Ollama is unavailable and the external fine-tuned GGUF is absent. Prompt
changes, actual multilingual generation, retrieval quality and customer resolution
rates are not measured by these offline checks. No live deployment was performed.
Existing production configuration, weights, index activation and release gates
are unchanged. Grounded generation now identifies its prompt as
`shopassist-grounded-v3`; rerun measured acceptance before activating it.

## Remaining measured work

1. On the model host, run `python scripts/evaluate_support_routes.py --model-backed`.
   Diagnose all 14 real-classifier cases and add observed customer failures.
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
