"""Shared settings. Importing this module never loads models or creates files."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    state_dir: Path = ROOT / "data" / "platform"
    vector_dir: Path = ROOT / "vector_store"
    ollama_url: str = "http://localhost:11434"
    answer_model: str = "shopassist-finetuned"
    utility_model: str = "qwen2.5:1.5b"
    faq_model: str = "shopassist-finetuned"
    verifier_model: str = "qwen2.5:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "local-v1"
    chunk_tokens: int = 500
    chunk_overlap: int = 75
    dense_k: int = 20
    keyword_k: int = 20
    fusion_k: int = 15
    context_k: int = 4
    context_tokens: int = 2000
    rrf_k: int = 60
    min_dense_similarity: float = 0.35
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = False
    model_timeout: float = 90.0
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pages: int = 500
    max_text_chars: int = 2_000_000
    retention_days: int = 30
    max_traces: int = 10000
    requests_per_minute: int = 20
    faq_per_chunk: int = 5
    allow_model_download: bool = False
    require_login: bool = False
    pipeline: str = "grounded"
    ticket_encryption_key: str = field(default="", repr=False)
    ticket_retention_days: int = 90

    def __post_init__(self):
        if self.pipeline not in {"grounded", "legacy"}:
            raise ValueError("Pipeline must be grounded or legacy")
        if not 0 <= self.chunk_overlap < self.chunk_tokens or self.chunk_tokens < 50:
            raise ValueError("Require 0 <= overlap < chunk_tokens, chunk_tokens >= 50")
        for name in ("dense_k", "keyword_k", "fusion_k", "context_k", "context_tokens",
                     "max_upload_bytes", "max_pages", "max_text_chars", "retention_days", "max_traces",
                     "requests_per_minute", "faq_per_chunk", "ticket_retention_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.model_timeout <= 0 or not 0 <= self.min_dense_similarity <= 1:
            raise ValueError("Invalid timeout or similarity threshold")

    @classmethod
    def from_env(cls):
        values = {}
        defaults = cls()
        for name, default in asdict(defaults).items():
            raw = os.getenv("SHOPASSIST_" + name.upper())
            if raw is None:
                continue
            if isinstance(default, Path):
                values[name] = Path(raw).expanduser().resolve()
            elif isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid boolean: {name}")
                values[name] = raw.lower() in {"true", "1"}
            else:
                values[name] = type(default)(raw)
        return cls(**values)

    def index_identity(self):
        # Window pooling avoids silently truncating long MiniLM chunks. Changing
        # any embedding/chunking semantics creates a separate Chroma collection.
        spec = {"model": self.embedding_model, "revision": self.embedding_revision,
                "pooling": "normalized-window-mean-v1", "chunker": "sections-embedding-tokenizer-v2",
                "tokens": self.chunk_tokens, "overlap": self.chunk_overlap}
        return sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20], spec
