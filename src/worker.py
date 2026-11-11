"""OS-authenticated background worker; can ingest/generate, never approve/publish."""

import argparse
from functools import lru_cache
import signal
import threading

from src.config import Settings
from src.faq import FAQService
from src.ingestion import IngestionService
from src.jobs import JobQueue
from src.ollama_client import OllamaClient
from src.security import Principal
from src.store import Store, encode
from src.vector_index import ChromaIndex, WindowEmbeddings
from src.verification import GroundingVerifier


def handlers(settings, store):
    principal = Principal("platform-worker", "worker")

    @lru_cache(maxsize=1)
    def vector():
        return ChromaIndex(settings)

    @lru_cache(maxsize=1)
    def encoder():
        return WindowEmbeddings(settings)

    def ingest(payload):
        service = IngestionService(settings, store, None if payload.get("extract_only", False) else vector())
        return service.ingest(payload["path"], payload.get("source_key"), principal.name, payload.get("ocr", False))

    def generate(payload):
        client = OllamaClient(settings)
        service = FAQService(store, settings, client, GroundingVerifier(client, settings.verifier_model), encoder())
        return service.generate(payload["chunk_id"], principal, payload.get("count"))

    return {"ingest": ingest, "generate_faq": generate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    if args.check:
        client = OllamaClient(settings)
        print(encode({"ready": True, "models": {m: client.model_digest(m)
                                               for m in {settings.faq_model, settings.verifier_model}}}))
        return 0
    queue = JobQueue(store)
    operations = handlers(settings, store)
    if args.once:
        print(encode(queue.run_once(operations)))
        return 0
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    while not stop.is_set():
        result = queue.run_once(operations)
        if result:
            print(encode(result), flush=True)
        stop.wait(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
