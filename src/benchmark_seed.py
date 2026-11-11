"""Grounded benchmark drafts from the repository's actual fictional policies.

These are seed specifications, not human-approved ground truth. Resolve them
against stored chunks after ingestion, then review every case before evaluation.
"""

from src.datasets import EvaluationCase
from src.verification import normalize

# Question, filename, exact beginning of a source sentence, shared fact group.
DIRECT = [
    ("How long after delivery can I request a return?", "return_refund.pdf", "Customers may request a return within 7 calendar days", "return-window"),
    ("What condition must a product be in to qualify for return?", "return_refund.pdf", "To qualify for a return, the product must be unused", "return-condition"),
    ("Can opened packaging prevent a damaged-on-delivery return?", "return_refund.pdf", "Products damaged during delivery or items received incorrectly", "opened-damaged"),
    ("When must I report a damaged or incorrect delivery?", "return_refund.pdf", "Customers who receive a damaged, defective, or incorrect product", "damage-report"),
    ("What evidence should I provide for a damaged delivery?", "return_refund.pdf", "Customers should provide their order ID, photographs", "damage-evidence"),
    ("When is a refund initiated?", "return_refund.pdf", "Once a returned product passes inspection", "refund-inspection"),
    ("Which payment method receives a prepaid order refund?", "return_refund.pdf", "Refunds for prepaid orders are normally issued", "refund-method"),
    ("Can I cancel my order before dispatch?", "return_refund.pdf", "Customers may cancel an order before it has been dispatched", "cancel-dispatch"),
    ("What can I do after my order has been dispatched?", "return_refund.pdf", "Once an order has been dispatched, customers must follow", "cancel-after"),
    ("How do I start a return request?", "return_refund.pdf", "Customers can initiate a return request", "return-request"),
    ("What details are needed for a return request?", "return_refund.pdf", "Customers should provide their order ID, registered email", "return-details"),
    ("Is return approval subject to inspection?", "return_refund.pdf", "Return approval is subject to verification", "return-inspection"),
    ("Can standard shipping time vary?", "shipping.pdf", "Delivery time may vary depending on the customer's location", "shipping-variation"),
    ("Is express delivery available for every product and postal code?", "shipping.pdf", "Express delivery is available only for selected products", "express-eligibility"),
    ("What order value qualifies for free standard shipping?", "shipping.pdf", "ShopAssist provides free standard shipping", "free-shipping"),
    ("What shipping fee may apply to an order costing INR 999 or less?", "shipping.pdf", "Orders of INR 999 or less may be subject", "standard-fee"),
    ("What does express shipping cost where available?", "shipping.pdf", "Express shipping costs INR 99", "express-fee"),
    ("When are shipping charges displayed?", "shipping.pdf", "Shipping charges are displayed during checkout", "shipping-checkout"),
    ("What happens to orders placed on public holidays?", "shipping.pdf", "Orders placed on public holidays or non-working days", "holiday-processing"),
    ("When do I receive tracking information?", "shipping.pdf", "Customers receive tracking information after", "tracking-dispatch"),
    ("What can tracking details include?", "shipping.pdf", "Tracking details may include the courier company", "tracking-details"),
    ("When should I contact support about stale tracking information?", "shipping.pdf", "Customers may contact support if their tracking information", "tracking-stale"),
    ("Are estimated delivery dates guaranteed?", "shipping.pdf", "Estimated delivery dates are not guaranteed", "delivery-guarantee"),
    ("Does ShopAssist deliver internationally?", "shipping.pdf", "International shipping is not available", "international"),
    ("Which country does the domestic delivery policy cover?", "shipping.pdf", "ShopAssist currently supports domestic deliveries within India only", "domestic"),
    ("What should I provide if my package appears lost?", "shipping.pdf", "Customers who believe their package has been lost", "lost-package"),
    ("What does the limited warranty cover?", "warranty.pdf", "The limited warranty covers manufacturing defects", "warranty-coverage"),
    ("Does warranty eligibility depend on product category?", "warranty.pdf", "Warranty eligibility depends on the product category", "warranty-eligibility"),
    ("What does support verify when I submit a warranty issue?", "warranty.pdf", "The support team will verify the warranty period", "warranty-verification"),
    ("Can warranty service require sending the product to a service center?", "warranty.pdf", "If necessary, the customer may be instructed", "service-center"),
    ("What outcomes are possible for an eligible warranty claim?", "warranty.pdf", "Eligible warranty claims may result in product repair", "repair-replace"),
    ("Is an immediate warranty replacement guaranteed?", "warranty.pdf", "ShopAssist does not guarantee an immediate replacement", "immediate-replacement"),
    ("What affects replacement availability?", "warranty.pdf", "Replacement availability may depend on stock", "replacement-stock"),
    ("What affects the time needed for repair or replacement?", "warranty.pdf", "Repair or replacement time depends on the nature", "repair-time"),
    ("Can a manufacturing defect reported after the return window qualify for warranty support?", "warranty.pdf", "Customers reporting manufacturing defects after the return window", "warranty-after-return"),
]

PARAPHRASES = {
    "return-window": "How many calendar days do I have to ask for a return?",
    "refund-method": "Where does the refund for a prepaid purchase go?",
    "cancel-dispatch": "Is cancellation possible while the order is still undispatched?",
    "free-shipping": "When is standard delivery free of charge?",
    "express-fee": "How much is the express delivery charge?",
    "tracking-stale": "My tracking has stopped updating. When may I contact support?",
    "international": "Can you ship my purchase outside India?",
    "immediate-replacement": "Will a replacement be sent before the warranty issue is verified?",
}

UNANSWERABLE = [
    "What is the exact current status of my personal order?",
    "What is ShopAssist's cryptocurrency payment policy?",
    "What is the ShopAssist premium membership annual fee?",
    "What is the exact customer support telephone number?",
    "Does ShopAssist offer a student discount and what is its percentage?",
    "Which installment payment providers does ShopAssist support?",
]


def seed_cases(chunks):
    cases, missing, by_group = [], [], {}
    for question, filename, needle, group in DIRECT:
        matching = [c for c in chunks if c["metadata"]["source"].casefold() == filename and normalize(needle) in normalize(c["text"])]
        if not matching:
            missing.append(group)
            continue
        chunk = matching[0]
        text = " ".join(chunk["text"].split())
        start = text.casefold().find(needle.casefold())
        if start < 0:
            missing.append(group)
            continue
        end = text.find(".", start)
        evidence = text[start:end + 1] if end >= 0 else text[start:]
        case = EvaluationCase(id="SEED_" + group, question=question, expected_answer=evidence,
            expected_document_ids=[chunk["document_id"]], expected_chunk_ids=[chunk["id"]], answerable=True,
            expected_evidence={chunk["id"]: evidence},
            difficulty="easy", language="en", category="direct", group_id=group)
        cases.append(case)
        by_group[group] = case
        if group in PARAPHRASES:
            cases.append(case.model_copy(update={"id": case.id + "_paraphrase", "question": PARAPHRASES[group], "category": "paraphrased"}))
    for index, question in enumerate(UNANSWERABLE):
        cases.append(EvaluationCase(id=f"SEED_unknown_{index}", question=question,
            expected_answer="The provided policy documents do not contain this information.",
            expected_document_ids=[], expected_chunk_ids=[], answerable=False,
            difficulty="medium", language="en", category="unanswerable", group_id=f"unknown-{index}"))
    for group, question in [("express-fee", "expres shiping cost?"), ("return-window", "return kitne calendar days mein request kar sakte hain?")]:
        if group in by_group:
            cases.append(by_group[group].model_copy(update={"id": "SEED_" + group + "_variation", "question": question,
                         "category": "multilingual_typo", "language": "en" if group == "express-fee" else "hi-Latn"}))
    for group, question, history in [
        ("express-fee", "And what does express cost?", [{"role": "user", "content": "Tell me about delivery options."}]),
        ("cancel-after", "What about after dispatch?", [{"role": "user", "content": "Can I cancel before dispatch?"}]),
    ]:
        if group in by_group:
            cases.append(by_group[group].model_copy(update={"id": "SEED_" + group + "_turn", "question": question,
                         "category": "multi_turn", "history": history, "difficulty": "medium"}))
    # Reusing component facts across train/test would leak. Tie the combined
    # cases and all constituent paraphrases to the same group.
    if "international" in by_group and "cancel-dispatch" in by_group:
        a, b = by_group["international"], by_group["cancel-dispatch"]
        joined = "shipping-and-cancellation"
        cases = [c.model_copy(update={"group_id": joined}) if c.group_id in {a.group_id, b.group_id} else c for c in cases]
        cases.append(EvaluationCase(id="SEED_multi_shipping_cancel", question="Can I ship internationally and can I cancel before dispatch?",
            expected_answer=a.expected_answer + " " + b.expected_answer,
            expected_document_ids=list(dict.fromkeys(a.expected_document_ids + b.expected_document_ids)),
            expected_chunk_ids=list(dict.fromkeys(a.expected_chunk_ids + b.expected_chunk_ids)), answerable=True,
            expected_evidence=a.expected_evidence | b.expected_evidence,
            difficulty="hard", language="en", category="multi_document", group_id=joined))
    return cases, missing
