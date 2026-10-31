
import json
from pathlib import Path

test_cases = [
    {
        "question": "What is the return window?",
        "source": "return_refund.pdf",
        "expected_terms": ["7", "days"],
        "answerable": True
    },
    {
        "question": "Can I return a product after 5 days?",
        "source": "return_refund.pdf",
        "expected_terms": ["7", "days"],
        "answerable": True
    },
    {
        "question": "What condition should returned products be in?",
        "source": "return_refund.pdf",
        "expected_terms": ["unused", "packaging"],
        "answerable": True
    },
    {
        "question": "How soon should damaged products be reported?",
        "source": "return_refund.pdf",
        "expected_terms": ["48", "hours"],
        "answerable": True
    },
    {
        "question": "How long does a refund take after inspection?",
        "source": "return_refund.pdf",
        "expected_terms": ["5", "7", "business days"],
        "answerable": True
    },
    {
        "question": "Are digital gift cards returnable?",
        "source": "return_refund.pdf",
        "expected_terms": ["not", "return"],
        "answerable": True
    },
    {
        "question": "When can I cancel my order?",
        "source": "return_refund.pdf",
        "expected_terms": ["before", "dispatched"],
        "answerable": True
    },
    {
        "question": "What is standard shipping time?",
        "source": "shipping.pdf",
        "expected_terms": ["4", "6", "business days"],
        "answerable": True
    },
    {
        "question": "How long does express shipping take?",
        "source": "shipping.pdf",
        "expected_terms": ["1", "2", "business days"],
        "answerable": True
    },
    {
        "question": "Is shipping free above 999 rupees?",
        "source": "shipping.pdf",
        "expected_terms": ["999", "free"],
        "answerable": True
    },
    {
        "question": "What is the standard shipping fee for a 500 rupee order?",
        "source": "shipping.pdf",
        "expected_terms": ["49"],
        "answerable": True
    },
    {
        "question": "How much does express shipping cost?",
        "source": "shipping.pdf",
        "expected_terms": ["99"],
        "answerable": True
    },
    {
        "question": "How long does order processing take?",
        "source": "shipping.pdf",
        "expected_terms": ["1", "2", "business days"],
        "answerable": True
    },
    {
        "question": "Does ShopAssist offer international shipping?",
        "source": "shipping.pdf",
        "expected_terms": ["not", "international"],
        "answerable": True
    },
    {
        "question": "What is the electronics warranty period?",
        "source": "warranty.pdf",
        "expected_terms": ["12", "month"],
        "answerable": True
    },
    {
        "question": "Does warranty cover liquid damage?",
        "source": "warranty.pdf",
        "expected_terms": ["not", "liquid"],
        "answerable": True
    },
    {
        "question": "Does warranty cover manufacturing defects?",
        "source": "warranty.pdf",
        "expected_terms": ["manufacturing", "defects"],
        "answerable": True
    },
    {
        "question": "What documents are needed for warranty claims?",
        "source": "warranty.pdf",
        "expected_terms": ["invoice", "order"],
        "answerable": True
    },
    {
        "question": "Where is my order number 12345 right now?",
        "source": None,
        "expected_terms": [],
        "answerable": False
    },
    {
        "question": "What is the exact stock count of iPhone 17?",
        "source": None,
        "expected_terms": [],
        "answerable": False
    }
]

output = Path("data/evaluation/test_cases.json")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as file:
    json.dump(test_cases, file, indent=2)

print(f"Created {len(test_cases)} evaluation questions")
print(f"Saved to {output}")
