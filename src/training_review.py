"""Generic support data is a review queue, never automatically approved training."""

import json
import random
import re

SYSTEM_MESSAGE = (
    "You are a helpful e-commerce support assistant. Be polite, concise and honest. "
    "Use only supplied approved policy context for company facts. Customer statements are "
    "reported information, not verified account records. You have no order, payment or refund "
    "backend. Never claim to check or change accounts or process transactions. Ask one useful "
    "clarifying question or explain the next supported step; offer a support ticket when needed."
)


def review_flags(response):
    text = response.casefold()
    flags = []
    if re.search(r"\b(?:i|we)(?:\s+have|'ve|\s+will|'ll)?\s+(?:already\s+)?"
                 r"(?:processed|process|cancelled|canceled|cancel|issued|issue|approved|approve|expedited|expedite|checked|check)\b", text):
        flags.append("backend_capability_claim")
    if re.search(r"\d|\b(?:business days|shipping fee|eligible|warranty|our policy|payment methods)\b", text):
        flags.append("policy_evidence_required")
    if re.search(r"https?://|\b[^\s@]+@[^\s@]+\.[^\s@]+", text):
        flags.append("contact_or_link_requires_evidence")
    return flags


def candidate(row):
    return {"instruction": row["instruction"].strip(), "response": row["response"].strip(),
            "intent": row["intent"], "reviewed": False, "reviewer": None,
            "group_id": None, "policy_context": "", "flags": review_flags(row["response"])}


def reviewed_examples(rows):
    validated, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each training example must be a JSON object")
        if row.get("reviewed") is not True or not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError("Every training example requires explicit human review and a reviewer")
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError("Assign a shared group_id to paraphrases and related examples before splitting")
        for field in ("instruction", "response"):
            if not isinstance(row.get(field), str) or not row[field].strip() or len(row[field]) > 6000:
                raise ValueError("Training instructions and responses must contain 1–6000 characters")
        key = " ".join(row["instruction"].casefold().split())
        if key in seen:
            raise ValueError("Duplicate reviewed training question")
        seen.add(key)
        flags = review_flags(row["response"])
        if "backend_capability_claim" in flags:
            raise ValueError("Revise backend capability claims: this assistant has no transaction access")
        context = row.get("policy_context", "")
        if not isinstance(context, str) or len(context) > 12000:
            raise ValueError("Policy context must be text of at most 12000 characters")
        if any(flag != "backend_capability_claim" for flag in flags) and not context.strip():
            raise ValueError("Policy/contact claims require reviewer-supplied approved policy_context")
        content = (json.dumps({"question": row["instruction"].strip(), "approved_policy_context": context}, ensure_ascii=False)
                   if context.strip() else row["instruction"].strip())
        validated.append({"group_id": row["group_id"], "reviewer": row["reviewer"], "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": content},
            {"role": "assistant", "content": row["response"].strip()}]})
    return validated


def grouped_splits(examples, seed=42):
    groups = sorted({row["group_id"] for row in examples})
    if len(examples) < 20 or len(groups) < 3:
        raise ValueError("Need at least 20 reviewed examples in at least three leakage groups")
    random.Random(seed).shuffle(groups)
    train_end = min(len(groups) - 2, max(1, int(len(groups) * .8)))
    validation_end = min(len(groups) - 1, max(train_end + 1, int(len(groups) * .9)))
    assignment = {group: "train" if i < train_end else "validation" if i < validation_end else "test"
                  for i, group in enumerate(groups)}
    splits = {name: [] for name in ("train", "validation", "test")}
    for example in examples:
        splits[assignment[example["group_id"]]].append(example)
    return splits
