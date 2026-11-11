"""Inspect the FAQ questions currently available to customers."""

from src.config import Settings
from src.faq_catalog import published_suggestions
from src.store import Store


def main():
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    suggestions = published_suggestions(store, limit=10)
    print("Reviewed, published FAQs:", len(suggestions))
    for faq in suggestions:
        print(f"{faq['faq_id']}: {faq['question']}")


if __name__ == "__main__":
    main()
