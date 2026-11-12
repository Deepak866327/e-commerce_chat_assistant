# Relevant support without a CRM

The bot can explain approved policies, troubleshoot with customer-reported facts,
and queue an agent request. It cannot verify an order, cancel it, initiate a refund
or inspect a bank transaction. A human also needs an appropriate external process
to perform those actions; creating a ticket does not complete them.

## Fill knowledge gaps with actual procedures

In Staff workspace, reviewers/admins can inspect **Knowledge gaps to review**.
Do not publish a failed customer conversation as an answer. Have the policy owner
write the missing procedure with these fields:

| Field | Required information |
| --- | --- |
| Customer goal | What the customer wants to do |
| Conditions | Eligible products/order states, exceptions and effective date |
| Procedure | Actual website/app steps and labels, verified by the business |
| Failure branch | What to do if each step is unavailable or fails |
| Required details | Only the information needed to choose the next step |
| Agent boundary | Which step requires order/payment access |
| Evidence | Owner-approved source and revision |

Prioritize cancellation with a missing option, a deducted payment without an order,
an approved but delayed refund, stalled tracking, and damaged/wrong-item returns.
Avoid writing fictitious menus, eligibility windows, fees or promised response times.
The repository's general-help articles are not substitutes for these store procedures.

Use the existing ingestion/index activation workflow in AI_PLATFORM_RUNBOOK.md for
updated policy PDFs. Review generated FAQs against those sources, then publish them.
Old FAQs disappear when their source revision is no longer active. This change does
not silently ingest, activate or publish any new business-policy facts.

## Validate the actual deployed model

With the existing fine-tuned model and utility/verifier models installed in Ollama:

```bash
python scripts/evaluate_real_models.py --support-conversations --report /tmp/shopassist-support-evaluation.json
```

The runner uses a temporary index and ticket database. It does not change production
documents, active indexes, queues or model weights. It records real answers and model
digests. Conversation contracts check context, clarification and handoff behavior;
they do not prove factual correctness. Review the answers and their citations against
the policy source, including eligibility exceptions and invented UI steps.

Compare keyword/vector retrieval and optional reranking on the same reviewed queries.
Enable reranking only if retrieval quality improves enough to justify its latency.
Do not change similarity thresholds based on one successful example.

## Agent handoff

Tickets show the customer goal, reported facts, remaining blocker, latest update,
customer-tried steps, bot guidance and FAQ already shown. Reported facts are labelled
unverified, and the brief records that no backend actions were completed. New customer
updates refresh the owned ticket's brief; other sessions cannot update it.
