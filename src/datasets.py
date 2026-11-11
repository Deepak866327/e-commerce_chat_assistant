"""Versioned evaluation-only datasets. This module never writes production chunks."""

from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import random
import re
from typing import Literal

from pydantic import Field

from src.schemas import Contract
from src.store import encode, now
from src.verification import normalize

TARGET_COUNTS = {"direct": 75, "paraphrased": 75, "multi_turn": 45, "multi_document": 45,
                 "unanswerable": 30, "multilingual_typo": 30}


class EvaluationCase(Contract):
    id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)
    expected_answer: str = Field(max_length=6000)
    expected_document_ids: list[str]
    expected_chunk_ids: list[str]
    expected_evidence: dict[str, str] = Field(default_factory=dict)
    answerable: bool
    difficulty: Literal["easy", "medium", "hard"]
    language: str = Field(min_length=2, max_length=50)
    category: Literal["direct", "paraphrased", "multi_turn", "multi_document", "unanswerable", "multilingual_typo"]
    group_id: str = Field(min_length=1, max_length=100)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    reviewed: bool = False
    reviewer: str | None = None
    split: Literal["development", "validation", "test"] = "development"


def _question_key(text):
    return re.sub(r"[^\w]+", " ", normalize(text)).strip()


def validate_cases(cases):
    ids, keys, answer_groups = set(), {}, {}
    for case in cases:
        if case.id in ids:
            raise ValueError("Duplicate evaluation ID")
        ids.add(case.id)
        key = _question_key(case.question)
        if key in keys:
            raise ValueError("Duplicate evaluation question")
        keys[key] = case
        if case.reviewed and not case.reviewer:
            raise ValueError("Reviewed cases require an identified human reviewer")
        if case.answerable and (not case.expected_answer.strip() or not case.expected_document_ids or not case.expected_chunk_ids):
            raise ValueError("Answerable cases need ground truth and exact source references")
        if not set(case.expected_evidence).issubset(set(case.expected_chunk_ids)):
            raise ValueError("Evidence must reference expected chunks")
        if not case.answerable and (case.expected_chunk_ids or case.expected_document_ids):
            raise ValueError("Unanswerable cases must not declare supporting sources")
        if case.category == "multi_turn" and not case.history:
            raise ValueError("Multi-turn cases require conversation history")
        if case.category == "multi_document" and len(set(case.expected_document_ids)) < 2:
            raise ValueError("Multi-document cases require at least two sources")
        for message in case.history:
            if set(message) != {"role", "content"} or message["role"] not in {"user", "assistant"}:
                raise ValueError("Evaluation history must contain only user/assistant messages")
        for prior_key, prior in keys.items():
            if prior.id == case.id:
                continue
            a, b = set(prior_key.split()), set(key.split())
            similarity = len(a & b) / max(1, len(a | b))
            if similarity >= .85 and prior.group_id != case.group_id:
                raise ValueError("Near-duplicate questions must share a leakage group")
        answer_key = _question_key(case.expected_answer)
        if case.answerable and len(answer_key) >= 20:
            previous = answer_groups.get(answer_key)
            if previous and previous != case.group_id:
                raise ValueError("Identical ground truth must share a leakage group")
            answer_groups[answer_key] = case.group_id
    group_splits = {}
    for case in cases:
        previous = group_splits.setdefault(case.group_id, case.split)
        if previous != case.split:
            raise ValueError("Leakage group crosses dataset splits")


def reproducible_splits(cases, seed=42):
    groups = sorted({c.group_id for c in cases})
    random.Random(seed).shuffle(groups)
    # Grouping, not individual rows, prevents paraphrases/follow-ups leaking.
    if len(groups) < 3:
        allocations = {group: "development" for group in groups}
    else:
        development_end = max(1, int(len(groups) * .6))
        validation_end = max(development_end + 1, int(len(groups) * .8))
        allocations = {group: ("development" if i < development_end else
                              "validation" if i < validation_end else "test") for i, group in enumerate(groups)}
    result = [c.model_copy(update={"split": allocations[c.group_id]}) for c in cases]
    validate_cases(result)
    return result


class DatasetRepository:
    def __init__(self, root, store=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store

    def create(self, cases, principal, seed=42, parent=None):
        principal.require("dataset")
        cases = [c if isinstance(c, EvaluationCase) else EvaluationCase.model_validate(c) for c in cases]
        cases = reproducible_splits(cases, seed)
        data = [c.model_dump() for c in sorted(cases, key=lambda c: c.id)]
        version = sha256(encode({"cases": data, "seed": seed}).encode()).hexdigest()[:24]
        destination = self.root / version
        if not destination.exists():
            destination.mkdir()
            (destination / "cases.json").write_text(encode(data), encoding="utf-8")
            manifest = {"version": version, "seed": seed, "parent": parent, "created_at": now(),
                        "counts": dict(Counter(c.category for c in cases)), "target_counts": TARGET_COUNTS,
                        "reviewed_count": sum(c.reviewed for c in cases), "count": len(cases),
                        "splits": dict(Counter(c.split for c in cases)), "actor": principal.name,
                        "production_knowledge": False}
            (destination / "manifest.json").write_text(encode(manifest), encoding="utf-8")
        if self.store:
            self.store.audit(principal.name, "dataset.create", version, {"count": len(cases)})
        return version

    def load(self, version, split=None, reviewed_only=False):
        if not re.fullmatch(r"[a-f0-9]{24}", version):
            raise ValueError("Invalid dataset version")
        payload = json.loads((self.root / version / "cases.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / version / "manifest.json").read_text(encoding="utf-8"))
        actual = sha256(encode({"cases": payload, "seed": manifest["seed"]}).encode()).hexdigest()[:24]
        if actual != version:
            raise ValueError("Dataset was modified after versioning")
        cases = [EvaluationCase.model_validate(c) for c in payload]
        validate_cases(cases)
        return [c for c in cases if (split is None or c.split == split) and (not reviewed_only or c.reviewed)]

    def review(self, version, case_ids, principal, store):
        principal.require("review")
        cases = self.load(version)
        selected = set(case_ids)
        if not selected.issubset({c.id for c in cases}):
            raise ValueError("Unknown evaluation case")
        updated = []
        for case in cases:
            if case.id in selected:
                actual_documents = set()
                for chunk_id in case.expected_chunk_ids:
                    chunks = store.rows("SELECT document_id,text FROM chunks WHERE id=?", (chunk_id,))
                    if not chunks or chunks[0]["document_id"] not in case.expected_document_ids:
                        raise ValueError("Ground-truth source missing or inconsistent")
                    actual_documents.add(chunks[0]["document_id"])
                    if chunk_id in case.expected_evidence and normalize(case.expected_evidence[chunk_id]) not in normalize(chunks[0]["text"]):
                        raise ValueError("Expected evidence is absent from the referenced chunk")
                if actual_documents != set(case.expected_document_ids):
                    raise ValueError("Every expected document must have supporting ground truth")
                case = case.model_copy(update={"reviewed": True, "reviewer": principal.name})
            updated.append(case)
        manifest = json.loads((self.root / version / "manifest.json").read_text(encoding="utf-8"))
        return self.create(updated, principal, manifest["seed"], version)

    def import_file(self, path, principal, seed=42):
        path = Path(path)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Dataset import exceeds size limit")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["cases"] if isinstance(payload, dict) else payload
        elif path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(StringIO(path.read_text(encoding="utf-8-sig"))))
            for row in rows:
                for key in ("expected_document_ids", "expected_chunk_ids", "expected_evidence", "history", "reviewed", "answerable"):
                    row[key] = json.loads(row[key])
                row["reviewer"] = row.get("reviewer") or None
        else:
            raise ValueError("Import JSON or CSV")
        # Imported claims of review are not proof of local human review.
        rows = [{**r, "reviewed": False, "reviewer": None} for r in rows]
        return self.create(rows, principal, seed)

    def export(self, version, destination):
        destination = Path(destination)
        rows = [c.model_dump() for c in self.load(version)]
        if destination.exists():
            raise ValueError("Export destination must be new")
        if destination.suffix.lower() == ".json":
            destination.write_text(encode({"dataset_version": version, "cases": rows, "production_knowledge": False}), encoding="utf-8")
        elif destination.suffix.lower() == ".csv":
            with destination.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(EvaluationCase.model_fields))
                writer.writeheader()
                for row in rows:
                    # JSON-encode strings beginning with spreadsheet operators;
                    # no formula is placed at the start of an exported cell.
                    encoded = {k: (encode(v) if isinstance(v, (list, dict, bool)) else v) for k, v in row.items()}
                    for key, value in encoded.items():
                        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                            encoded[key] = "'" + value
                    writer.writerow(encoded)
        else:
            raise ValueError("Export JSON or CSV")
