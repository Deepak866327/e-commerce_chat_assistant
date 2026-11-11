"""Built-in self-service guidance, without unverified store-policy promises.

These are maintained product copy, not generated or withdrawn policy FAQs.
Published store articles take precedence over this starter catalog.
"""


def starter_faqs():
    content = [
        ("cancel", "Orders", "How can I cancel my order?",
         "1. Check the order confirmation or your account for the order’s current status.\n"
         "2. If a cancellation option is available, follow its instructions and check for confirmation.\n"
         "3. If that option is missing or fails, tell us in chat whether the order has been dispatched and what you see.\n\n"
         "Cancellation eligibility depends on the store’s policy and order status. This chatbot cannot cancel an order directly."),
        ("tracking", "Delivery", "How can I track my order?",
         "1. Check your dispatch email or order details for a tracking link or tracking number.\n"
         "2. Open the courier’s tracking page to see the latest available update.\n"
         "3. If tracking is missing or appears stuck, tell us the latest status and when it last changed.\n\n"
         "We can help you understand the update, but cannot look up live order status here."),
        ("returns", "Returns", "How do I return a damaged or unwanted item?",
         "1. Check the return instructions in your order details or the store’s return policy.\n"
         "2. Keep the packaging and, for damage or a wrong item, take clear photos.\n"
         "3. If return instructions are missing or the process fails, tell us what went wrong and whether the item has been delivered.\n\n"
         "Eligibility and any return window depend on the applicable store policy. Wait for return instructions before sending the item."),
        ("refund", "Refunds", "My refund has not arrived. What should I check?",
         "1. Check whether you received a refund approval or processing confirmation.\n"
         "2. Check the original payment method and any timeline stated in that confirmation.\n"
         "3. If the stated timeline has passed, tell us when the refund was approved and the payment method type.\n\n"
         "We cannot confirm a refund’s status or issue a refund without access to the order and payment system."),
        ("payment", "Payments", "My payment was deducted but the order is not confirmed. What should I do?",
         "1. Check your order history and confirmation email, including spam.\n"
         "2. Check whether your payment provider marks the transaction as pending or completed.\n"
         "3. Avoid retrying the payment until you know whether an order was created. Tell us the payment status and whether an order confirmation exists.\n\n"
         "Do not share card numbers, passwords, PINs or OTPs in chat. We cannot verify a bank transaction directly."),
        ("human", "Support", "How can I contact a support agent?",
         "Describe your issue in chat and we’ll try to help you resolve it. If you prefer a person, use **Talk to a human** in the sidebar. "
         "After a support ticket is created, check **Your support tickets** and use **Refresh ticket** to see replies.\n\n"
         "A ticket places your request in the support queue; it does not guarantee an immediate response."),
    ]
    return [{"faq_id": "starter-" + key, "category": category, "question": question, "answer": answer}
            for key, category, question, answer in content]
