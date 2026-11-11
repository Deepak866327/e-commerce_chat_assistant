"""Compatible script entry point; see python -m src.admin for managed operations."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, Settings
from src.documents import SUPPORTED
from src.ingestion import IngestionService
from src.vector_index import ChromaIndex


def main():
    settings = Settings.from_env()
    service = IngestionService(settings, vector=ChromaIndex(settings))
    files = sorted(p for p in (ROOT / "data" / "policies").iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError("No supported documents inside data/policies/")
    for path in files:
        print(service.ingest(path))
    print("Candidate index ready. Validate against reviewed data before activation. Legacy index preserved.")


if __name__ == "__main__":
    main()
