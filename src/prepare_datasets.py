
import json
import re
from pathlib import Path
from datasets import load_dataset

DATASET_PATH = "data/training/raw/customer_support.csv"
OUTPUT_DIR = Path("data/training/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Relevant intents for e-commerce customer support
SELECTED_INTENTS = {
    "cancel_order",
    "change_order",
    "place_order",
    "track_order",
    "delivery_options",
    "delivery_period",
    "check_refund_policy",
    "get_refund",
    "track_refund",
    "payment_issue",
    "check_payment_methods",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
}

SYSTEM_MESSAGE = (
    "You are a helpful e-commerce customer support "
    "assistant. Be polite, concise, and honest. "
    "Do not invent order status, company policies, "
    "refund amounts, or customer information. "
    "If information requires account access, "
    "explain that verification is needed."
)

# Load CSV
dataset = load_dataset(
    "csv",
    data_files=DATASET_PATH,
    split="train"
)

print("Original rows:", len(dataset))

# Keep relevant intents
dataset = dataset.filter(
    lambda row: row["intent"] in SELECTED_INTENTS
)

print("After intent filtering:", len(dataset))

# Remove missing / empty / placeholder-containing rows
def is_valid(row):
    instruction = row.get("instruction")
    response = row.get("response")

    if not isinstance(instruction, str):
        return False

    if not isinstance(response, str):
        return False

    instruction = instruction.strip()
    response = response.strip()

    if not instruction or not response:
        return False

    # Exclude unresolved template placeholders
    if re.search(r"\{\{.*?\}\}", instruction + response):
        return False

    if len(instruction) > 500:
        return False

    if len(response) > 2000:
        return False

    return True

dataset = dataset.filter(is_valid)

# Deduplicate by normalized question
seen = set()
unique_indices = []

for index, row in enumerate(dataset):
    key = row["instruction"].strip().lower()

    if key not in seen:
        seen.add(key)
        unique_indices.append(index)

dataset = dataset.select(unique_indices)

print("After cleaning:", len(dataset))

# Keep small subset for initial training experiment
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(
    range(min(2000, len(dataset)))
)

if len(dataset) < 20:
    raise RuntimeError(
        "Too few clean samples. Inspect filters and dataset."
    )

# 80% training, 10% validation, 10% test
first_split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

second_split = first_split["test"].train_test_split(
    test_size=0.5,
    seed=42
)

splits = {
    "train": first_split["train"],
    "validation": second_split["train"],
    "test": second_split["test"]
}

def to_chat_format(row):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": row["instruction"].strip()
            },
            {
                "role": "assistant",
                "content": row["response"].strip()
            }
        ]
    }

for split_name, split_data in splits.items():

    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in split_data:
            example = to_chat_format(row)
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )

    print(
        f"{split_name}: {len(split_data)} samples "
        f"-> {output_path}"
    )

print("\nDataset preparation complete!")
print("Review samples before training.")
