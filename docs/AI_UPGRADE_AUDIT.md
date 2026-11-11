# AI upgrade audit

Audit date: 2026-11-11. Scope: all tracked files, hidden configuration, data,
models and source directories. Initial working tree was clean (Git required a
per-command `safe.directory` override because the sandbox account differs from
the repository owner; no global Git settings were changed).

## Existing architecture

- Python Streamlit UI in `app.py`; no HTTP API, Node/TypeScript manifest, admin
  frontend, application database, migrations, Docker, CI or tests exist.
- Chat entry point: `src/rag_memory.py:ask_shopassist(question, chat_history)`.
  Import eagerly loads embeddings and Chroma and raises if the index is empty.
- Ollama via LangChain: `qwen2.5:1.5b` for scope and rewriting,
  `shopassist-finetuned` for answers. `models/Modelfile` loads the external
  `shopassist-q4_k_m.gguf`, temperature 0.2, context 4096. Runtime temperature is
  zero. Neither weights nor training metadata are present. The fine-tuned
  parameter count (1.5B versus 3B) cannot be verified. Preserve its registered name.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`; persistent Chroma
  collection `shopassist_policies` in `vector_store/`, default distance settings.
- Ingestion loads only PDFs, splits at 500 **characters** with 100-character
  overlap, then deletes every old vector before inserting new vectors.
- Included knowledge: three PDFs (returns/refunds, shipping, warranty) and one
  DOCX of returns/refunds, currently ignored. No reviewed FAQ or evaluation data.
- Retrieval: dense top 3; no scores, threshold, keyword fusion, reranking or
  active-version checks. Every retrieved page is displayed as a source regardless
  of whether the answer uses it. There is no citation validation.
- Scope classifier runs on every query; follow-ups add another serial model
  call. Rewrite validation checks only length and a trailing question mark.
- Hinglish answer prompt uses policy context inside a system message; no
  structured generation or independent verification. History is eight LangChain
  messages, per Streamlit session; history is explicitly not policy evidence.
- Voice and browser read-aloud are separate reusable components. Training data
  preparation is separate and must remain separate from production/evaluation.

## Executed baseline checks

| Command/check | Actual result |
| --- | --- |
| `python --version` | Python 3.14.2 (README requests 3.11) |
| `python -m unittest discover -v` | 0 tests; no existing tests |
| `python -m compileall -q app.py src` | passed |
| `python -m pip check` | passed for installed environment; does not prove project dependencies installed |
| `python src/ingest.py` | failed: missing `langchain_community` |
| `python -c "import src.rag_memory"` | failed: missing `langchain_ollama` |
| Ollama localhost:11434 | not listening |
| fine-tuned GGUF / existing vector index | absent |
| existing build/typecheck | no commands/configuration exist |

Live baseline answers, quality metrics and inference latency cannot be measured
without the external fine-tuned weights, running Ollama, cached embeddings and
missing runtime dependencies. No quality improvement is claimed. Hardware queries
were denied by the sandbox; BGE-M3 suitability is unmeasured.

## Main risks and bottlenecks

Destructive rebuild loses the working index on insertion failure; document
versions and IDs are missing; malformed/scanned PDFs have no handling; embedding
loading blocks import; serial inference dominates latency; no auth, upload
validation, auditing, rate limit, feedback or retention. Retrieved instructions
share system-message authority. There are no release gates or ground-truth
datasets. These are implementation findings, not measured load-test results.
