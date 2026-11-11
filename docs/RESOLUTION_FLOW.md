# Try to resolve before human handoff

The default flow makes up to three support attempts for the same issue before
queuing human help. If distinct available questions/steps are exhausted sooner,
it hands over rather than repeating them just to fill the budget. The bot asks one relevant question at a time, gives only
retrieved-policy guidance, remembers customer-reported details, and checks the
outcome. It still cannot actually cancel orders, process refunds or inspect live
accounts without a backend.

## Example cancellation conversation

1. Customer: "Cancel my order."
   Bot asks whether it has been dispatched, rather than immediately opening a ticket.
2. Customer: "No."
   This is dispatch information, not failed resolution. The next retrieval query
   targets cancellation before dispatch, using the original issue and reported details.
3. The bot offers supported guidance and asks whether it helped. If the customer
   is still blocked, it tries a different question/step rather than the same answer.
4. Once three attempts have not resolved the issue, human handoff includes the
   original issue, reported facts, asked questions, attempted guidance, cited sources,
   failure reasons and available transcript.

An attempt is a customer-facing support cycle (targeted clarification or distinct
guidance), not a token-generation retry. Responses supplying a requested detail
are used before declaring that cycle unsuccessful; this can include a clarification
and the ensuing guidance in the same cycle. Greetings, thanks, out-of-scope turns
and blocked/rate-limited requests do not consume the budget. A plain "thanks" does
not prove resolution. Explicit confirmation, or "yes" to an outcome check, does.
"Yes/no" to dispatch/approval questions is treated as a reported fact instead.

The attempt budget is enforced in code and persistent state; it is not an
instruction the answer model can override. Unambiguous human requests and the
**Talk to a human** form bypass the budget. If the customer explicitly declines
human support, the bot offers handoff again at the limit instead of silently
creating a ticket against that preference. Ticket-creation failure is displayed
honestly; no successful queueing is claimed without a returned ticket reference.

## State and concurrency

SQLite schema v4 adds `support_sessions`. Each row is keyed by a salted session
hash, includes an issue ID, phase, bounded redacted context and attempt count, and
expires after 24 hours of inactivity. Reopening a connection with the same session
keeps the budget. Browser-provided counters cannot reset or inflate server counts.
The app supplies a unique session ID; external callers must do the same.

New issues and **Clear conversation** reset the budget. Existing tickets remain
available through their session-owned controls. Closed/resolved tickets are not
reopened. Distinct issue IDs prevent unrelated generic replies ("No", "still not
working") from deduplicating into the wrong ticket. A version check prevents an
older concurrent response from overwriting a newer turn or opening a stale ticket;
callers should serialize turns per conversation.

Customer details are reported information, never verified account records or
policy evidence. The model receives bounded context to understand short replies.
Fixed follow-up questions contain no invented policy dates, fees or self-service
UI instructions. Policy generation retains the retrieval/verification controls.
Repeated-answer detection is conservative text matching; paraphrases and semantic
quality still need actual model evaluation. User-intent/slot recognition is bounded
and heuristic, with the utility model handling remaining messages.

## Configuration and rollout

```bash
SHOPASSIST_SUPPORT_MAX_ATTEMPTS=3
```

Supported values are 2 and 3. Configure the environment used by the actual service;
the project does not automatically load `.env`. Restart after configuration changes.

The flow is in the `improve-support-resolution` branch / PR #1. The separate
`fix-procedural-routing` hotfix / PR #2 only fixes routing and numerical guard issues;
it does not contain this multi-turn retry flow.

Back up the control SQLite database and source before deployment. Migration is
additive, but rolling back to code that only understands schema v3 also requires
restoring its matching database backup. Preserve models, policy documents,
vector collections, runtime credentials and the current pipeline configuration.
Restart the app and any FAQ worker together so both use code that understands schema v4.
The grounded release gate is unchanged; changed prompt v4 needs re-evaluation.

Run the offline suite with `python -m unittest discover -q`. It includes state,
legacy/grounded routing, cancellation follow-ups, budget two/three, ownership,
reset/expiry, explicit handoff, confirmed resolution, duplicate guidance, stale
concurrent turns, trace updates, redaction and the actual Streamlit chat UI with
simulated inference. These checks do not establish model factual accuracy.

On the Ollama host, the existing isolated real-model runner now also accepts:

```bash
python scripts/evaluate_real_models.py --support-conversations --report /tmp/shopassist-real-model-report.json
```

It exercises an unresolved cancellation conversation and explicit handoff using
real models where needed. Evaluation tickets and the candidate index exist only
in temporary state; the production queue and index are untouched. Actual model
execution remains pending until the runner is executed on the model host.
