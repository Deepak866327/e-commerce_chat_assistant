
from pathlib import Path
from collections import Counter
from datasets import load_dataset

DATASET_NAME = (
    "bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset"
)

output_dir = Path("data/training/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading customer support dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print("\nTotal rows:", len(dataset))
print("Columns:", dataset.column_names)

print("\nFirst 3 examples:")

for index in range(3):
    row = dataset[index]

    print(f"\n--- Example {index + 1} ---")
    print("Category:", row["category"])
    print("Intent:", row["intent"])
    print("Customer:", row["instruction"])
    print("Assistant:", row["response"][:500])

print("\nCategories:")
for category, count in Counter(
    dataset["category"]
).most_common():
    print(f"{category}: {count}")

print("\nTop intents:")
for intent, count in Counter(
    dataset["intent"]
).most_common(15):
    print(f"{intent}: {count}")

# Save raw dataset locally
output_path = output_dir / "customer_support.csv"
dataset.to_csv(str(output_path), index=False)

print("\nSaved:", output_path)
