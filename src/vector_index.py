"""Lazy Chroma adapter with model identity and dimension isolation."""

from functools import lru_cache
import math


class IndexUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def embedding_model(name, revision, allow_download=False):
    from sentence_transformers import SentenceTransformer
    options = {"local_files_only": not allow_download, "device": "cpu"}
    if revision.startswith("hf:"):
        options["revision"] = revision[3:]
    return SentenceTransformer(name, **options)


class WindowEmbeddings:
    """Pool every token window rather than MiniLM's silently truncated prefix."""

    def __init__(self, settings):
        self.model = embedding_model(settings.embedding_model, settings.embedding_revision,
                                     settings.allow_model_download)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        tokenizer = self.model.tokenizer
        window = max(8, self.model.max_seq_length - 2)
        result = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            pieces = [tokenizer.decode(ids[i:i + window]) for i in range(0, len(ids), window)] or [""]
            vectors = self.model.encode(pieces, normalize_embeddings=True, show_progress_bar=False)
            pooled = [sum(float(v[j]) for v in vectors) / len(vectors) for j in range(self.dimension)]
            norm = math.sqrt(sum(v * v for v in pooled)) or 1.0
            result.append([v / norm for v in pooled])
        return result


class ChromaIndex:
    def __init__(self, settings, index_id=None, expected_spec=None):
        try:
            import chromadb
            self.embeddings = WindowEmbeddings(settings)
            self.client = chromadb.PersistentClient(path=str(settings.vector_dir))
            identity, spec = settings.index_identity()
            if index_id and index_id != identity:
                raise IndexUnavailable("Index identity differs from configured embedding/chunking model")
            if expected_spec is not None and spec != expected_spec:
                raise IndexUnavailable("Stored index specification differs from runtime settings")
            self.id = identity
            self.dimension = self.embeddings.dimension
            self.collection_name = "shopassist_v2_" + identity
            existing = {c.name for c in self.client.list_collections()}
            if self.collection_name in existing:
                self.collection = self.client.get_collection(self.collection_name, embedding_function=None)
                metadata = self.collection.metadata or {}
                if metadata.get("identity") != identity or metadata.get("dimension") != self.dimension:
                    raise IndexUnavailable("Embedding identity/dimension mismatch")
            else:
                self.collection = self.client.create_collection(
                    self.collection_name, embedding_function=None,
                    metadata={"hnsw:space": "cosine", "identity": identity, "dimension": self.dimension})
        except IndexUnavailable:
            raise
        except Exception as exc:
            raise IndexUnavailable(f"Dense index unavailable ({type(exc).__name__}); install dependencies/cache embeddings") from exc

    def upsert(self, chunks):
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            self.collection.upsert(ids=[c.id for c in batch], documents=[c.text for c in batch],
                                   embeddings=self.embeddings.encode([c.text for c in batch]),
                                   metadatas=[{k: v for k, v in c.metadata.items() if v is not None} for c in batch])

    def search(self, query, k, allowed_ids):
        if not allowed_ids:
            return []
        result = self.collection.query(query_embeddings=self.embeddings.encode([query]),
                                       n_results=min(k, len(allowed_ids)),
                                       where={"chunk_id": {"$in": list(allowed_ids)}}, include=["distances"])
        return [(cid, max(0.0, min(1.0, 1.0 - distance)))
                for cid, distance in zip(result["ids"][0], result["distances"][0])]

    def delete(self, ids):
        if ids:
            self.collection.delete(ids=ids)

    def count_tokens(self, text):
        return len(self.embeddings.model.tokenizer.encode(text, add_special_tokens=False))
