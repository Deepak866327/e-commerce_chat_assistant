"""Prepare review candidates; export training splits only from reviewed examples."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.training_review import candidate, grouped_splits, reviewed_examples

SELECTED_INTENTS = {"cancel_order", "change_order", "place_order", "track_order", "delivery_options",
    "delivery_period", "check_refund_policy", "get_refund", "track_refund", "payment_issue",
    "check_payment_methods", "complaint", "contact_customer_service", "contact_human_agent"}


def valid_candidate(row):
    return (row.get("intent") in SELECTED_INTENTS
            and all(isinstance(row.get(key), str) and row[key].strip() for key in ("instruction", "response"))
            and len(row["instruction"]) <= 500 and len(row["response"]) <= 2000
            and not re.search(r"\{\{.*?\}\}", row["instruction"] + row["response"]))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training/raw/customer_support.csv")
    parser.add_argument("--reviewed-file", type=Path, help="Human-reviewed JSONL, based on review_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/processed")
    args = parser.parse_args(argv)
    source = (args.reviewed_file or args.input).resolve()
    if source.is_relative_to((ROOT / "data/benchmarks").resolve()) or source.is_relative_to((ROOT / "data/platform/evaluation").resolve()):
        parser.error("Evaluation data must remain separate from training")
    try:
        if args.reviewed_file:
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            examples = reviewed_examples(rows)
            splits = grouped_splits(examples)
            args.output.mkdir(parents=True, exist_ok=True)
            for name, records in splits.items():
                write_jsonl(args.output / f"{name}.jsonl", [{"messages": row["messages"]} for row in records])
            manifest = {"reviewed_examples": len(examples), "reviewers": sorted({row["reviewer"] for row in examples}),
                        "group_counts": {name: len({row["group_id"] for row in records}) for name, records in splits.items()},
                        "counts": {name: len(records) for name, records in splits.items()}, "model_training_performed": False}
            (args.output / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(manifest))
        else:
            seen, candidates = set(), []
            with source.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if not valid_candidate(row):
                        continue
                    key = " ".join(row["instruction"].casefold().split())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate(row))
                    if len(candidates) == 2000:
                        break
            args.output.mkdir(parents=True, exist_ok=True)
            write_jsonl(args.output / "review_candidates.jsonl", candidates)
            flags = Counter(flag for row in candidates for flag in row["flags"])
            print(json.dumps({"pending_review": len(candidates), "flags": dict(flags), "training_splits_written": False}))
            print("Review/edit responses, set reviewed=true, reviewer and group_id, then run with --reviewed-file.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
