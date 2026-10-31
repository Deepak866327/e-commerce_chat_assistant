# ShopAssist AI

An e-commerce customer support assistant built with Streamlit, LangChain, ChromaDB, and local Ollama models. It retrieves company policy information from PDFs and uses a custom Qwen model to answer in Hinglish (Hindi and English).

## Features

- Policy answers covering shipping, returns, refunds, and warranties.
- PDF retrieval with document names and page references.
- Recent conversation history and follow-up question rewriting.
- A scope classifier for customer support questions.
- A Streamlit chat interface and command-line chat scripts.
- Scripts for dataset preparation and baseline retrieval evaluation.

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

Start the Ollama desktop application, or run `ollama serve` in a separate terminal. Download the base model used for query rewriting, scope classification, and baseline evaluation:

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

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The sidebar includes a button to clear the conversation.

For the conversation-aware command-line interface:

```powershell
.\.venv\Scripts\python.exe -m src.rag_memory
```

Type `clear` to reset the command-line conversation or `exit` to quit.

## Dataset preparation

These optional scripts download the Bitext customer support dataset and produce filtered chat-format training, validation, and test JSONL files:

```powershell
.\.venv\Scripts\python.exe src/download_datasets.py
.\.venv\Scripts\python.exe src/prepare_datasets.py
```

The downloaded and processed datasets are written under `data/training/` and excluded from Git. Review the upstream dataset's license and terms before using or redistributing it. Running these scripts does not train the custom model.

## Baseline evaluation

After preparing the base model and policy index, run:

```powershell
.\.venv\Scripts\python.exe src/evaluate.py
```

The script reads `data/evaluation/test_cases.json` and overwrites `data/evaluation/baseline_results.csv`. It evaluates the base `qwen2.5:1.5b` retrieval pipeline using retrieval hits and approximate keyword matches. These metrics do not evaluate the full Streamlit conversation, custom model, or scope classifier; generated answers still need manual review.

## Project layout

```text
app.py                       Streamlit chat interface
requirements.txt             Python dependencies
models/Modelfile              Ollama custom model configuration
src/rag_memory.py             Main chat pipeline with conversation history
src/chatbot.py                Reusable chatbot class
src/ingest.py                 PDF loading, chunking, and indexing
src/search.py                 Policy retrieval inspection
src/download_datasets.py      Download the source training dataset
src/prepare_datasets.py       Prepare filtered chat datasets
src/create_eval_dataset.py    Generate baseline evaluation cases
src/evaluate.py               Run baseline evaluation
data/policies/                Policy documents
data/evaluation/              Evaluation cases and saved baseline report
```

## Troubleshooting

- **Ollama connection error:** Start Ollama and confirm it is listening at `http://localhost:11434`.
- **Model not found:** Check `ollama list` and complete the model preparation steps above.
- **ChromaDB is empty:** Run `src/ingest.py` from the project root before launching the application.
- **Model artifact missing:** Obtain the custom GGUF separately and place it next to `models/Modelfile` before running `ollama create`.
