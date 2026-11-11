# ShopAssist AI

An e-commerce customer support assistant built with Streamlit, LangChain, ChromaDB, and local Ollama models. It retrieves company policy information from PDFs and uses a custom Qwen model to answer in Hinglish (Hindi and English).

## Features

- Policy answers covering shipping, returns, refunds, and warranties.
- PDF retrieval with document names and page references.
- Recent conversation history and follow-up question rewriting.
- A scope classifier for customer support questions.
- A Streamlit chat interface with conversation history.
- Local voice transcription with a lightweight multilingual Whisper model, an editable transcript, and browser read-aloud controls.
- Scripts for policy indexing and dataset preparation.

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

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` for embeddings. The script stores the index in `vector_store/`, which is generated locally and excluded from Git. Run ingestion again after changing the PDFs; it replaces the existing indexed records.

### 3. Start the application

Index the approved FAQ answers before starting the app:

```powershell
.\.venv\Scripts\python.exe src/ingest_faqs.py
```

FAQ answers live in `data/faqs/faqs.json`. Only active entries with a question
and answer are indexed. Run ingestion again after editing them. The app requires
both the policy and FAQ collections. Explicit requests for human support create
a local ticket in `data/support_tickets.db`; this runtime database is excluded
from Git and must be preserved between deployments. Ticket creation does not
notify a support team or connect to an external help desk.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The sidebar includes a button to clear the conversation.

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

These optional scripts download the Bitext customer support dataset and produce filtered chat-format training, validation, and test JSONL files:

```powershell
.\.venv\Scripts\python.exe src/download_datasets.py
.\.venv\Scripts\python.exe src/prepare_datasets.py
```

The downloaded and processed datasets are written under `data/training/` and excluded from Git. Review the upstream dataset's license and terms before using or redistributing it. Running these scripts does not train the custom model.

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
