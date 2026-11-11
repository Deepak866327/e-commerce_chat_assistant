# ShopAssist AI

An e-commerce customer support assistant built with Streamlit, LangChain, ChromaDB, and local Ollama models. It retrieves company policy information from PDFs and uses a custom Qwen model to answer in Hinglish (Hindi and English).

## Features

- Policy answers covering shipping, returns, refunds, and warranties.
- PDF retrieval with document names and page references.
- Recent conversation history and follow-up question rewriting.
- A scope classifier for customer support questions.
- A Streamlit chat interface with conversation history.
- Local voice transcription with a lightweight multilingual Whisper model, an editable transcript, and browser read-aloud controls.
- Support tickets for unresolved queries, customer replies, and an authenticated staff queue.
- Reviewed, published FAQ suggestion buttons and background FAQ generation using the existing fine-tuned Qwen model.
- Scripts for policy indexing and dataset preparation.

Local FAQ candidates are included in `data/faqs/faqs.json`. Import an entry into
the existing review workspace with `python -m src.admin import-faq
data/faqs/faqs.json FAQ001 SOURCE_CHUNK_ID`. The command prompts for the admin
password and checks the answer against that source chunk. Imported entries remain
pending review until a reviewer approves and an administrator publishes them;
the catalog's `active` status alone does not make an answer public. The convenience
command `python src/ingest_faqs.py` accepts the same three arguments.

The main app uses the platform's existing session-owned support tickets and staff
queue, including additional human-agent request phrases from the local changes.
`src/support_db.py` remains a standalone SQLite utility for local experiments;
its database is excluded from Git and is separate from production tickets.

The assistant provides guidance. It has no connection to an order, payment, cancellation, or refund processing backend.

## Prerequisites

- Python 3.11, matching the development environment.
- [Ollama](https://ollama.com/) installed and running at its default address, `http://localhost:11434`.
- The custom model artifact `shopassist-q4_k_m.gguf`, obtained separately.
- Internet access for the initial dependency, base model, and embedding model downloads.

The custom GGUF model is excluded from Git. A fresh clone requires that artifact before the main chat application can answer questions. This repository contains dataset preparation scripts, but no model training or GGUF conversion pipeline.

## Setup on Windows PowerShell

Run the following commands from the directory where you want to clone the project:

```powershell
git clone https://github.com/Deepak866327/e-commerce_chat_assistant.git
Set-Location e-commerce_chat_assistant
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All remaining commands should be run from the project root unless otherwise indicated. Calling the virtual environment's Python directly avoids needing to activate it in PowerShell.

### 1. Prepare the Ollama models

Start the Ollama desktop application, or run `ollama serve` in a separate terminal. Download the base model used for query rewriting and scope classification:

```powershell
ollama pull qwen2.5:1.5b
```

Place your custom model at `models/shopassist-q4_k_m.gguf`, next to `models/Modelfile`. Register it with the name expected by the application:

```powershell
Push-Location models
ollama create shopassist-finetuned -f Modelfile
Pop-Location
ollama list
```

Both `qwen2.5:1.5b` and `shopassist-finetuned` should be available locally.

### 2. Index the policy documents

The repository includes policy PDFs in `data/policies/`. Build the local Chroma collection:

```powershell
.\.venv\Scripts\python.exe src/ingest.py
```

The upgraded script stages versioned knowledge in a separate Chroma collection and
preserves existing records. Enable `SHOPASSIST_ALLOW_MODEL_DOWNLOAD=true` only
during initial model provisioning. New indexes require a reviewed evaluation
before activation; see [the platform runbook](docs/AI_PLATFORM_RUNBOOK.md).

### 3. Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The sidebar includes a button to clear the conversation.
Choose **Chat language** independently of speech recognition: Hinglish, English,
or Hindi. The preference applies to conversational replies, account guidance,
ticket confirmations, abstentions and generated policy answers in both pipelines.
Customer-reported issue details and attempted steps are kept in bounded session
context separately from verified policy evidence. Account-action follow-ups can
continue the same session-owned ticket; a new issue starts fresh context.

## Voice support

Install the updated dependencies and start the app from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` on the same computer and allow microphone access.
Click **Ask with your voice** to expand the controls, record a question of up to 30 seconds, and stop
recording. Click **Transcribe recording**, edit **Review your question** if needed,
then click **Send voice question**. Voice questions use the same retrieval,
conversation memory, and scope checks as typed questions.

The first transcription downloads the multilingual `tiny` model into
`models/speech/` (excluded from Git). It uses Faster Whisper on the CPU with
INT8 precision and two threads. The model loads only when transcription is
requested and is reused afterward. Later transcriptions work without a speech
API or an internet connection once the model is downloaded. Audio is processed
in memory; the app does not save recording files. The transcript becomes part
of the current chat only after you send it.

Choose **Hindi / Hinglish** or **English** in the sidebar if automatic language
detection is inaccurate. Short clips, background noise, and mixed languages can
reduce accuracy; review the transcript before sending. Hindi speech may appear
in Devanagari rather than Roman letters.

Each assistant reply has **Read aloud**, **Stop**, and a language selector.
Playback uses the browser's available voices; Hindi voice availability and
Hinglish pronunciation vary by device. Depending on the selected device voice,
playback may use an online service. Text chat works without using the microphone
or loading the speech model. Browser microphone access works on localhost and
HTTPS; opening an ordinary HTTP LAN address may prevent recording.

For a hosted deployment, open the app over HTTPS. Speech transcription runs on
the application server; customers only need a browser with microphone access.

Before restarting a hosted app, install the updated requirements and preload the
speech model using the same account and virtual environment as the app service:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'from src.voice import get_transcriber; get_transcriber()'
```

Keep `models/speech/` between deployments so customer requests do not need to
download the model. The model files are deployment artifacts, excluded from Git.

## Dataset preparation

These optional scripts download the Bitext customer support dataset and prepare
filtered **review candidates**. Generic dataset answers are not approved company
policies and may claim backend capabilities the chatbot does not have:

```powershell
.\.venv\Scripts\python.exe src/download_datasets.py
.\.venv\Scripts\python.exe src/prepare_datasets.py
```

The default preparation command writes `review_candidates.jsonl`, never training
splits. Review/edit each example, set `reviewed` to `true`, identify the `reviewer`,
and assign a shared `group_id` to related examples/paraphrases. Supply approved
`policy_context` for company facts or contact details. Obvious backend-action
claims must be rewritten. The flags are heuristic review aids, not factual validation.
Save reviewed entries separately, then export chat-format splits:

```powershell
.\.venv\Scripts\python.exe src/prepare_datasets.py --reviewed-file data/training/reviewed_examples.jsonl
```

Export requires at least 20 reviewed examples in three leakage groups; groups
never cross train/validation/test splits. Evaluation data stays separate.
The downloaded and processed datasets are written under `data/training/` and
excluded from Git. Review the upstream dataset's license and terms before using
or redistributing it. These scripts do not train or replace the custom model.

## Project layout

```text
app.py                       Streamlit chat interface
requirements.txt             Python dependencies
src/voice.py                 Local CPU speech recognition and recording limits
src/read_aloud.html          Browser speech playback controls
models/Modelfile              Ollama custom model configuration
src/rag_memory.py             Main chat pipeline with conversation history
src/ingest.py                 PDF loading, chunking, and indexing
src/download_datasets.py      Download the source training dataset
src/prepare_datasets.py       Prepare filtered chat datasets
data/policies/                Policy documents
```

## Troubleshooting

- **Ollama connection error:** Start Ollama and confirm it is listening at `http://localhost:11434`.
- **Model not found:** Check `ollama list` and complete the model preparation steps above.
- **ChromaDB is empty:** Run `src/ingest.py` from the project root before launching the application.
- **Model artifact missing:** Obtain the custom GGUF separately and place it next to `models/Modelfile` before running `ollama create`.

## Versioned AI platform upgrade

The existing Python, Streamlit, Chroma and Ollama architecture is retained. The
upgrade adds document versions, PDF/DOCX/TXT/CSV/approved-FAQ ingestion, optional
table extraction/OCR, token chunks, grounded FAQ generation and human review,
BM25/dense fusion, optional CPU BGE reranking, structured answers, citation checks,
independent verification, safe abstention, feedback and local monitoring.

Evaluation data is stored separately from production knowledge. The repository
includes **54 unreviewed benchmark drafts**, not 300 approved cases or measured
quality improvements. Ragas/DeepEval are optional local-model judge adapters.
Signed reports, regression checks and index rollback support controlled releases.

Operations extend the Python project through `python -m src.admin --help` and a
staff workspace in the existing Streamlit app. Support staff can respond to tickets;
reviewers can review FAQ drafts; administrators can queue generation and publish
approved FAQs. SQLite holds workflow state; Chroma still holds vectors.

The bot first asks a relevant follow-up or gives evidence-based guidance and
checks whether it helped. It tracks each issue across turns, avoids repeating
questions/steps, and escalates after three unsuccessful support cycles. Set
`SHOPASSIST_SUPPORT_MAX_ATTEMPTS=2` for a two-attempt budget. These are customer
support attempts, not the model's internal generation retries. A reply to a
requested detail is used before declaring that attempt unsuccessful.

Explicit human requests create a ticket immediately. Greetings and thanks do not
consume the budget; a new issue or **Clear conversation** resets it. Resolution
requires customer confirmation. See [the support flow](docs/RESOLUTION_FLOW.md).
Customers
can also use **Talk to a human** and check **Your support tickets** in the sidebar.
Replies are available in the current browser session; click **Refresh ticket** to
check for a staff response. Optional contact details require a configured encryption
key. Ticket creation queues the request; it does not promise an immediate human reply.

The customer **Help center · Frequently asked questions** panel offers search,
topic filters and expandable approved answers. Opening or searching articles does
not call a model, create tickets or consume support attempts. **I still need help**
saves feedback and sends the FAQ ID and revision into the support flow. The server
loads the corresponding answer itself, asks a relevant detail first, and remembers
the unsuccessful article through later replies and any agent handoff. Answers are
shown in their reviewed language; the chat language selector controls chat replies.

**This helped** records article feedback without submitting a chat turn or marking
an unrelated active issue resolved. Votes are retained per session and article
revision, so repeated clicks update a vote rather than inflate the count. Reviewers
and admins can see **FAQ helpfulness** in Staff workspace, including confirmed chat
resolutions, escalated issues and detected repeated guidance. These are retained
feedback/support outcomes, not proof that a backend action was completed.

FAQ search combines keyword matching with the configured local embedding model.
If its dependencies or cached weights are missing, keyword search still works.
No Ollama call is needed to read saved articles. FAQ vectors are cached by article
revision; edited or withdrawn content is excluded by the publication gate before
search. Restart after provisioning missing embedding weights.

In English chat, an unambiguous exact approved FAQ question can return the saved
answer with its source citation without a generation call. Active troubleshooting,
human requests and account actions retain their normal routes. Other chat languages
use the usual answer pipeline so the requested language is preserved. Semantic
auto-answers are available behind `SHOPASSIST_FAQ_SEMANTIC_ANSWERS=true`, but stay off
by default until evaluated on your paraphrases and difficult negative examples.
The similarity and runner-up margin settings are retrieval heuristics, not calibrated
confidence probabilities. Low-confidence, ambiguous and personally qualified queries
use RAG instead. Only current published policy FAQs qualify for automatic answers;
built-in general help does not substitute for a store-specific policy.

Fresh deployments show six built-in general-help FAQs (orders, delivery, returns,
refunds, payments and support), with search and topic filters immediately available.
These explain practical checks without promising store-specific eligibility or
timelines. The published store catalog replaces this starter catalog once available.
If the store catalog cannot load, general help remains available with a notice.
The legacy `data/faqs/faqs.json` file is not automatically published: its policy
claims still need validation and review against current source documents.

To populate the panel, open **Staff workspace**, generate drafts from current policy
chunks, review and edit the answers and categories, select **Approve reviewed FAQ**,
then have an admin select **Publish approved FAQ**. Only reviewed, verified FAQs
whose source version is still active appear. Edits, withdrawal or source changes
remove the article on the next interaction without rebuilding the index. Generated drafts,
rejected FAQs and evaluation data never appear in the help center. See the
[runbook](docs/AI_PLATFORM_RUNBOOK.md) for staff credentials and worker setup.

- [Audit](docs/AI_UPGRADE_AUDIT.md)
- [Plan](docs/AI_UPGRADE_PLAN.md)
- [Progress and remaining validation](docs/AI_UPGRADE_PROGRESS.md)
- [Operations, evaluation and deployment](docs/AI_PLATFORM_RUNBOOK.md)

Run offline checks with `python -m unittest discover -v` and
`python -m compileall -q app.py src scripts tests`. Optional extraction/OCR and
evaluation dependencies are in `requirements-ocr.txt` and `requirements-eval.txt`.
`python scripts/offline_audit.py` generates extraction and chunk-structure reports.
`python scripts/evaluate_support_routes.py` checks deterministic conversation
routes and explicitly reports skipped model cases. Add `--model-backed` with
local Ollama available to test complaint/policy/follow-up classification. This
development routing check does not measure factual accuracy or authorize release.
See [the incremental support changes](docs/SUPPORT_IMPROVEMENTS.md) for scope,
validation and remaining model-backed work.

Existing deployments can explicitly set `SHOPASSIST_PIPELINE=legacy` while staging
the upgrade. This preserves the previous customer answer pipeline and collection;
its replies do **not** carry the new grounding guarantee. The default `grounded`
pipeline requires an activated index. Production boot through `scripts/serve.py`
checks a signed release report and the actual Ollama model digest.
