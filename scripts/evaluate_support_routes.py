"""Run development routing cases; model-backed checks require local Ollama."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, Settings
from src.conversation import conversational_reply
from src.query_routing import classify_query


def run_cases(cases, client=None, settings=None):
    results = []
    for case in cases:
        reply = conversational_reply(case["question"])
        if reply:
            route = {"human_support": "HUMAN_SUPPORT", "clarification": "CLARIFICATION"}.get(reply["route"], "CONVERSATION")
            result = {"id": case["id"], "actual": route, "model_backed": False}
        elif client is None:
            result = {"id": case["id"], "skipped": "local_model_not_requested"}
        else:
            try:
                route = classify_query(client, settings.utility_model, case["question"], case.get("history", []))
                result = {"id": case["id"], "actual": route, "model_backed": True}
            except Exception as exc:
                result = {"id": case["id"], "actual": None, "model_backed": True, "error": type(exc).__name__}
        if "actual" in result:
            result["passed"] = result["actual"] == case["expected"]
        results.append(result)
    executed = [row for row in results if "passed" in row]
    return {"scope": "Synthetic development routing only; no factual accuracy, retrieval or resolution score",
            "total": len(cases), "executed": len(executed), "skipped": len(cases) - len(executed),
            "passed": sum(row["passed"] for row in executed),
            "model_backed_executed": sum(row.get("model_backed", False) for row in executed),
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/benchmarks/support_routes.json")
    parser.add_argument("--model-backed", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    client = None
    if args.model_backed:
        from src.ollama_client import OllamaClient
        client = OllamaClient(settings)
    report = run_cases(json.loads(args.cases.read_text(encoding="utf-8"))["cases"], client, settings)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["passed"] != report["executed"])


if __name__ == "__main__":
    raise SystemExit(main())
