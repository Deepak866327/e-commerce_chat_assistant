import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


# Database location: data/support_tickets.db
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "support_tickets.db"

VALID_STATUSES = {"Open", "In Progress", "Resolved", "Closed"}
VALID_PRIORITIES = {"Low", "Normal", "High", "Urgent"}


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    finally:
        connection.close()


def current_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    """Create support ticket tables if they do not exist."""

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                customer_message TEXT NOT NULL,
                customer_name TEXT,
                customer_email TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                priority TEXT NOT NULL DEFAULT 'Normal',
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS ticket_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ticket_id)
                    REFERENCES tickets(id)
                    ON DELETE CASCADE
            )
        """)


def format_ticket_id(database_id):
    return f"SA-{database_id:06d}"


def parse_ticket_id(ticket_id):
    """Accept either SA-000001 or the numeric database ID."""

    if isinstance(ticket_id, str) and ticket_id.upper().startswith("SA-"):
        ticket_id = ticket_id[3:]

    return int(ticket_id)


def create_ticket(
    subject,
    customer_message,
    conversation=None,
    customer_name=None,
    customer_email=None,
    priority="Normal",
):
    """Create a ticket and save its conversation."""

    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority: {priority}")

    if not subject.strip() or not customer_message.strip():
        raise ValueError("Subject and customer message are required.")

    timestamp = current_timestamp()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                subject,
                customer_message,
                customer_name,
                customer_email,
                status,
                priority,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, 'Open', ?, ?, ?)
            """,
            (
                subject.strip(),
                customer_message.strip(),
                customer_name,
                customer_email,
                priority,
                timestamp,
                timestamp,
            ),
        )

        database_id = cursor.lastrowid

        # Save the available conversation history.
        messages = conversation or [
            {"role": "user", "content": customer_message}
        ]

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if content:
                connection.execute(
                    """
                    INSERT INTO ticket_messages (
                        ticket_id, role, content, created_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (database_id, role, content, timestamp),
                )

    return format_ticket_id(database_id)


def list_tickets(status=None):
    """Return tickets, optionally filtered by status."""

    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    with get_connection() as connection:
        if status:
            rows = connection.execute(
                """
                SELECT * FROM tickets
                WHERE status = ?
                ORDER BY created_at DESC
                """,
                (status,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM tickets
                ORDER BY created_at DESC
                """
            ).fetchall()

    tickets = []

    for row in rows:
        ticket = dict(row)
        ticket["ticket_id"] = format_ticket_id(ticket.pop("id"))
        tickets.append(ticket)

    return tickets


def get_ticket(ticket_id):
    """Return one ticket and its complete message history."""

    database_id = parse_ticket_id(ticket_id)

    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE id = ?",
            (database_id,),
        ).fetchone()

        if row is None:
            return None

        ticket = dict(row)
        ticket["ticket_id"] = format_ticket_id(ticket.pop("id"))

        messages = connection.execute(
            """
            SELECT role, content, created_at
            FROM ticket_messages
            WHERE ticket_id = ?
            ORDER BY id ASC
            """,
            (database_id,),
        ).fetchall()

        ticket["messages"] = [dict(message) for message in messages]

    return ticket


def add_ticket_message(ticket_id, role, content):
    """Add a customer or agent message to an existing ticket."""

    database_id = parse_ticket_id(ticket_id)

    if role not in {"user", "assistant", "agent"}:
        raise ValueError("Role must be user, assistant, or agent.")

    if not content.strip():
        raise ValueError("Message cannot be empty.")

    timestamp = current_timestamp()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO ticket_messages (
                ticket_id, role, content, created_at
            )
            SELECT id, ?, ?, ?
            FROM tickets
            WHERE id = ?
            """,
            (role, content.strip(), timestamp, database_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Ticket {ticket_id} does not exist.")

        connection.execute(
            """
            UPDATE tickets
            SET updated_at = ?
            WHERE id = ?
            """,
            (timestamp, database_id),
        )


def update_ticket_status(ticket_id, status, assigned_to=None):
    """Update ticket status and optionally assign it to an agent."""

    database_id = parse_ticket_id(ticket_id)

    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE tickets
            SET status = ?,
                assigned_to = COALESCE(?, assigned_to),
                updated_at = ?
            WHERE id = ?
            """,
            (status, assigned_to, current_timestamp(), database_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Ticket {ticket_id} does not exist.")


# Create database tables when this module is imported.
init_db()
