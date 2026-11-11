from src.support_db import (
    create_ticket,
    list_tickets,
    get_ticket,
    add_ticket_message,
    update_ticket_status,
)


# 1. Create a sample support ticket
ticket_id = create_ticket(
    subject="Refund not received",
    customer_message="My refund has not been credited yet.",
    conversation=[
        {
            "role": "user",
            "content": "My refund has not been credited yet."
        },
        {
            "role": "assistant",
            "content": "Your refund generally takes 5-7 business days after approval."
        }
    ],
    customer_name="Test Customer",
    customer_email="test@example.com",
    priority="High",
)

print("\nCreated ticket:", ticket_id)


# 2. Retrieve the ticket
ticket = get_ticket(ticket_id)
print("\nTicket details:")
print(ticket)


# 3. Add an agent response
add_ticket_message(
    ticket_id,
    "agent",
    "We are checking your refund status."
)


# 4. Assign ticket and update status
update_ticket_status(
    ticket_id,
    "In Progress",
    assigned_to="Support Agent 1",
)


# 5. Display updated ticket
print("\nUpdated ticket:")
print(get_ticket(ticket_id))


# 6. Display all open tickets
print("\nOpen tickets:")
print(list_tickets(status="Open"))