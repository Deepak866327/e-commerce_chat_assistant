"""Import a local FAQ into the existing authenticated review workspace."""

from pathlib import Path
import sys

# Support both python src/ingest_faqs.py and python -m src.ingest_faqs.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.admin import main as admin_main


def main(argv=None):
    return admin_main(["import-faq", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
