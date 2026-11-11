# Procedural-query escalation hotfix

Base: `ecb664347232e811954ec9aa92ff0bcb339fdc1d`, the reported live deployment.

## Observed live behavior

On https://api.friendzx.app/, a fresh conversation asking
"How can I cancel my order?" returned the exact `ACCOUNT_ACTION` template and
created a support ticket. It did not reach policy retrieval. An independent
"What is your return policy?" request did retrieve sources and generate a reply;
the issue is not universal failure of every query or necessarily a missing index.

The return reply also asserted a 10-business-day refund timeline. The repository
policy states: "Approved refunds are generally processed within 5–7 business days
after inspection." This is an observed legacy-generation quality issue; server
logs and its actual indexed policy bytes were not available for inspection.

## Changes

- Deterministically route unambiguous procedure/eligibility/policy questions to
  knowledge retrieval before consulting the small utility classifier. This rule
  selects evidence retrieval; it does not hardcode any company policy answers.
- Keep action requests, explicit human requests, mixed action/procedure requests,
  unsafe messages and ambiguous follow-ups on the existing model-based route.
- Do not open a ticket solely because useful sourced procedural guidance also
  contains a live-account-access disclaimer. Preserve escalation for a bare
  refusal, insufficient policy evidence, service errors and actual handoffs.
- For deployed compatibility answers, retry once when numerical claims do not
  occur in the retrieved source text. Reject persistent unsupported numbers.
  List ordinals are excluded and decimal digits are normalized across scripts.
  This is a narrow guard, not a claim of complete semantic grounding: spelled-out
  quantities, wrong conditions and unrelated evidence can still require review.
- Leave the controlled historical evaluation path unchanged (`intent_checked=False`).

## Regression verification

`python -m unittest discover -q` includes an erroneous `ACCOUNT_ACTION` classifier
fixture: explicit cancellation procedures must still reach policy retrieval, and
the legacy wrapper must create zero tickets when returning supported guidance.
Additional cases cover human requests, imperative cancellations, Hindi/Hinglish,
mixed requests, bare refusals and the observed unsupported 10-day claim.

Compilation and whitespace checks accompany the suite. New code has not been
deployed or verified against the server's model here. Live observations above
describe the old deployment only. Existing data, models, indexes and rollout mode
are preserved; this hotfix does not activate the grounded pipeline.
