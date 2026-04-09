import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "conversations.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id              TEXT PRIMARY KEY,
            started_at      TEXT,
            ended_at        TEXT,
            customer_contact TEXT,
            customer_name   TEXT,
            turn_count      INTEGER DEFAULT 0,
            escalated       INTEGER DEFAULT 0,
            sentiment       TEXT,
            resolved        INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            timestamp       TEXT,
            role            TEXT,
            content         TEXT,
            tool_used       TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id  TEXT,
            message_id       INTEGER,
            timestamp        TEXT,
            article          TEXT,
            section          TEXT,
            similarity_score REAL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id  TEXT,
            timestamp        TEXT,
            issue_type       TEXT,
            issue_summary    TEXT,
            sentiment        TEXT,
            ticket_id        TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def create_conversation() -> str:
    """Insert a new conversation row and return its ID."""
    conversation_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (id, started_at) VALUES (?, ?)",
        (conversation_id, now())
    )
    conn.commit()
    conn.close()
    return conversation_id


def save_message(conversation_id: str, role: str, content: str, tool_used: str = None) -> int:
    """
    Write a single message to the database immediately.
    Returns the inserted row's ID so callers can link related records (e.g. citations) back to it.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO messages (conversation_id, timestamp, role, content, tool_used) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, now(), role, str(content), tool_used)
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id


def save_citation(conversation_id: str, message_id: int, article: str, section: str, similarity_score: float):
    """Write a single KB citation linked to the specific tool_call message that triggered it."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO citations (conversation_id, message_id, timestamp, article, section, similarity_score) VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, message_id, now(), article, section, round(similarity_score, 4))
    )
    conn.commit()
    conn.close()


def save_escalation(conversation_id: str, issue_type: str, issue_summary: str, sentiment: str, ticket_id: str):
    """Write an escalation record and mark the conversation as escalated."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO escalations (conversation_id, timestamp, issue_type, issue_summary, sentiment, ticket_id) VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, now(), issue_type, issue_summary, sentiment, ticket_id)
    )
    conn.execute(
        "UPDATE conversations SET escalated = 1 WHERE id = ?",
        (conversation_id,)
    )
    conn.commit()
    conn.close()


def get_transcript(conversation_id: str) -> str:
    """
    Build a readable transcript for the CCAS payload.
    Only includes user and assistant turns — tool calls and results are internal.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,)
    ).fetchall()
    conn.close()

    lines = []
    for role, content in rows:
        if role == "user":
            lines.append(f"Customer: {content}")
        elif role == "assistant":
            lines.append(f"Agent: {content}")

    return "\n".join(lines)


def close_conversation(
    conversation_id: str,
    customer_contact: str = None,
    customer_name: str = None,
    turn_count: int = 0,
    escalated: bool = False,
    sentiment: str = None,
    resolved: bool = None,
):
    """Update conversation metadata when the session ends."""
    resolved_int = None if resolved is None else int(resolved)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE conversations
           SET ended_at=?, customer_contact=?, customer_name=?, turn_count=?, escalated=?, sentiment=?, resolved=?
           WHERE id=?""",
        (now(), customer_contact, customer_name, turn_count, int(escalated), sentiment, resolved_int, conversation_id)
    )
    conn.commit()
    conn.close()
