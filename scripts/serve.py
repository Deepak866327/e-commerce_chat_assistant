"""Fail closed before production startup; local development uses Streamlit directly."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.release import check_release
from src.store import Store


def main():
    settings = Settings.from_env()
    try:
        check_release(Store(settings.state_dir / "control.sqlite3"), settings)
    except Exception as exc:
        print(f"Production startup blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py",
              "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=20"])


if __name__ == "__main__":
    raise SystemExit(main())
