"""Production boot validation. Offline unit success does not authorize serving."""

import json
import re

from src.index_state import verify_report
from src.ollama_client import OllamaClient


def check_release(store, settings, check_models=True):
    active = store.setting("active_index")
    run_id = store.setting("active_run_id", "")
    if not active or not re.fullmatch(r"RUN_[a-f0-9]{32}", run_id):
        raise RuntimeError("No validated active index and signed evaluation run")
    if active != settings.index_identity()[0]:
        raise RuntimeError("Runtime embedding/chunk configuration differs from the active index")
    path = settings.state_dir / "evaluation" / "reports" / (run_id + ".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not verify_report(store, report) or report.get("release_gate") is not True or
            report.get("index_id") != active or report.get("model_version") != settings.answer_model):
        raise RuntimeError("Active evaluation report is invalid or does not authorize this release")
    if check_models:
        client = OllamaClient(settings)
        digest = client.model_digest(settings.answer_model)
        if digest != report.get("model_digest"):
            raise RuntimeError("Fine-tuned model artifact changed after evaluation")
        for model in {settings.utility_model, settings.verifier_model}:
            client.model_digest(model)
    return {"ready": True, "index_id": active, "run_id": run_id, "model_digest": report.get("model_digest")}
