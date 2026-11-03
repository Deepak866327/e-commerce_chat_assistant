
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = (
    "bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset"
)

output_dir = Path("data/training/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading customer support dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

# Save raw dataset locally
output_path = output_dir / "customer_support.csv"
dataset.to_csv(str(output_path), index=False)

print(f"Saved {len(dataset)} rows to {output_path}")
