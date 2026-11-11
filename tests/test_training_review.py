from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from src.prepare_datasets import main
from src.training_review import candidate, grouped_splits, reviewed_examples


class TrainingReviewTests(unittest.TestCase):
    def example(self, **changes):
        row = {"instruction": "My refund is missing", "response": "Has the refund been approved?",
               "intent": "track_refund", "reviewed": True, "reviewer": "Human reviewer", "group_id": "refund"}
        return {**row, **changes}

    def test_raw_preparation_writes_review_queue_without_training_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.csv", root / "output"
            with source.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["instruction", "response", "intent"])
                writer.writeheader()
                writer.writerow({"instruction": "Cancel my order", "response": "I have cancelled your order.", "intent": "cancel_order"})
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            row = json.loads((output / "review_candidates.jsonl").read_text())
            self.assertFalse(row["reviewed"])
            self.assertIn("backend_capability_claim", row["flags"])
            self.assertFalse((output / "train.jsonl").exists())

    def test_unreviewed_and_false_backend_claims_never_export(self):
        for row in (candidate(self.example()), self.example(response="I have processed your refund."),
                    self.example(response="Refunds take 7 business days."), self.example(group_id=None)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                reviewed_examples([row])

    def test_policy_examples_include_approved_context_and_groups_never_leak(self):
        rows = [self.example(instruction=f"Refund question {i}", group_id=f"group-{i // 2}",
                             response="After approval, refunds take 7 business days.",
                             policy_context="Approved policy: refunds take 7 business days after approval.") for i in range(20)]
        examples = reviewed_examples(rows)
        self.assertIn("approved_policy_context", examples[0]["messages"][1]["content"])
        splits = grouped_splits(examples)
        group_sets = [set(row["group_id"] for row in records) for records in splits.values()]
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[1] & group_sets[2] or group_sets[0] & group_sets[2])
        self.assertEqual(splits, grouped_splits(examples))

    def test_failed_review_does_not_overwrite_existing_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text("existing reviewed training\n")
            source = root / "pending.jsonl"
            source.write_text(json.dumps(candidate(self.example())))
            with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                main(["--reviewed-file", str(source), "--output", str(root)])
            self.assertEqual((root / "train.jsonl").read_text(), "existing reviewed training\n")
