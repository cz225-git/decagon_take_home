# tools.py
# Defines the tools available to the agent and the functions that execute them.
# Order data lives in order_data.py and is imported below.

import json
import re
import uuid
from datetime import datetime, timezone

from order_data import MOCK_ORDERS, save_orders

RECENT_ORDER_WINDOW_DAYS = 90


# ---------------------------------------------------------------------------
# Tool definitions — sent to the Claude API so the model knows what tools exist
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "identify_customer",
        "description": (
            "Call this as soon as the customer provides their email or phone number. "
            "It verifies the contact and automatically retrieves their most recent order. "
            "Present the returned order to the customer and ask if that is the order "
            "they are contacting support about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_contact": {
                    "type": "string",
                    "description": "The email address or phone number the customer provided",
                },
            },
            "required": ["customer_contact"],
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Look up the full details of a specific order by order ID. "
            "Use this when the customer has confirmed their order ID — either because "
            "find_recent_order returned the wrong order, or they provided one directly. "
            "Requires the order ID and the customer's contact info for verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The Bookly order ID, e.g. BK-1001",
                },
                "customer_contact": {
                    "type": "string",
                    "description": "The email address or phone number the customer provided",
                },
            },
            "required": ["order_id", "customer_contact"],
        },
    },
    {
        "name": "submit_refund",
        "description": (
            "Submit a return or refund request for a customer's order. "
            "Use this when a customer wants to return an item or request a refund. "
            "First use lookup_order to confirm the order exists and is eligible for a refund. "
            "Requires the order ID, the customer's contact info, and their reason for returning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The Bookly order ID",
                },
                "customer_contact": {
                    "type": "string",
                    "description": "The email address or phone number the customer provided",
                },
                "reason": {
                    "type": "string",
                    "description": "The customer's reason for requesting a return or refund",
                },
            },
            "required": ["order_id", "customer_contact", "reason"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search Bookly's knowledge base for answers to general questions about "
            "shipping, return policy, password reset, payment methods, order cancellation, "
            "or damaged/missing items. Use this before answering any policy or process question. "
            "Do not answer policy questions from memory — always search first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question, rephrased as a clear search query",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "escalate_to_agent",
        "description": (
            "Escalate the conversation to a human support agent. "
            "Use this when the customer explicitly asks to speak to a human, "
            "is frustrated and cannot be helped with the available tools, "
            "or has a complex issue beyond your scope. "
            "Provide an issue_type and a brief issue_summary based on the conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_contact": {
                    "type": "string",
                    "description": "The email address or phone number the customer provided",
                },
                "issue_type": {
                    "type": "string",
                    "enum": [
                        "order_status", "refund_dispute", "shipping_issue",
                        "account_issue", "damaged_item", "policy_question", "other",
                    ],
                    "description": "Category that best describes the customer's issue",
                },
                "issue_summary": {
                    "type": "string",
                    "description": "1-2 sentence summary of the customer's issue and what was attempted",
                },
            },
            "required": ["customer_contact", "issue_type", "issue_summary"],
        },
    },
]

# ---------------------------------------------------------------------------
# Helper — checks whether a contact string matches the order's email or phone
# ---------------------------------------------------------------------------

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def contact_matches(order: dict, customer_contact: str) -> bool:
    """
    Return True if the provided contact matches the order's email or phone.
    Phone comparison strips all non-digit characters so formats like
    5556969631, 555-696-9631, (555) 696-9631 all match the stored value.
    """
    contact = customer_contact.strip().lower()
    if contact == order["customer_email"].lower():
        return True
    return _digits_only(contact) == _digits_only(order["customer_phone"])

# ---------------------------------------------------------------------------
# Tool execution functions — called by agent.py when the model requests a tool
# ---------------------------------------------------------------------------

def identify_customer(customer_contact: str) -> dict:
    """
    Called by the script once contact info is parsed.
    Returns the customer's name, their most recent order, and all orders
    placed within the last 90 days — so the agent can resolve order references
    by description (e.g. "my Great Gatsby order") without asking for an order ID.
    """
    from datetime import timedelta
    matching = [o for o in MOCK_ORDERS.values() if contact_matches(o, customer_contact)]
    if not matching:
        return {"verified": False, "result": "No orders found for this contact information."}

    cutoff = datetime.now() - timedelta(days=RECENT_ORDER_WINDOW_DAYS)
    recent = [
        o for o in matching
        if datetime.strptime(o["order_date"], "%Y-%m-%d") >= cutoff
    ]
    recent_sorted = sorted(recent, key=lambda o: o["order_date"], reverse=True)
    most_recent = recent_sorted[0] if recent_sorted else max(matching, key=lambda o: o["order_date"])

    def order_summary(o):
        return {
            "order_id": o["order_id"],
            "order_date": o["order_date"],
            "items": o["items"],
            "status": o["status"],
            "total": o["total"],
            "eligible_for_refund": o["eligible_for_refund"],
        }

    return {
        "verified": True,
        "customer_name": most_recent["customer_name"],
        "most_recent_order": order_summary(most_recent),
        "recent_orders": [order_summary(o) for o in recent_sorted],
    }


def lookup_order(order_id: str, customer_contact: str) -> dict:
    """
    Return full order details if the order ID exists and contact info matches.
    Returns an auth error if the contact doesn't match — never reveals which part was wrong.
    """
    order = MOCK_ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"No order found with ID '{order_id}'. Please check the order ID and try again."}
    if not contact_matches(order, customer_contact):
        return {"error": "We couldn't verify your identity for this order. Please check the email or phone number you provided."}
    return order


def submit_refund(order_id: str, customer_contact: str, reason: str) -> dict:
    """
    Submit a refund request. Verifies identity and eligibility before approving.
    On approval, mutates the in-memory order so the updated state is visible
    for the rest of the session.
    """
    order = MOCK_ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"No order found with ID '{order_id}'."}
    if not contact_matches(order, customer_contact):
        return {"error": "We couldn't verify your identity for this order. Please check the email or phone number you provided."}
    if not order["eligible_for_refund"]:
        return {
            "status": "denied",
            "reason": "This order is outside the 30-day return window and is no longer eligible for a refund.",
        }

    # Mutate the order and persist to disk so the state survives across sessions
    order["eligible_for_refund"] = False
    order["refund_status"] = "approved"
    order["refund_submitted_at"] = datetime.now(timezone.utc).isoformat()
    save_orders()

    return {
        "status": "approved",
        "order_id": order_id,
        "refund_amount": order["total"],
        "message": f"Refund request submitted successfully. {order['total']} will be returned to your original payment method within 5-7 business days.",
    }


def escalate_to_agent(
    customer_contact: str,
    issue_type: str,
    issue_summary: str,
    conversation_id: str,
    client,
) -> dict:
    """
    Mock a CCAS escalation API call.
    Builds the full payload (contact, transcript, issue_type, issue_summary, sentiment)
    and logs it to the console and to the escalations table.
    In production this would be an HTTP POST to the CCAS escalation endpoint.
    """
    from storage import get_transcript, save_escalation
    from analysis import analyze_sentiment

    transcript = get_transcript(conversation_id)
    sentiment = analyze_sentiment(transcript, client)
    ticket_id = str(uuid.uuid4())[:8].upper()

    payload = {
        "endpoint": "ccas.bookly.com/api/v1/escalate",
        "customer_contact": customer_contact,
        "issue_type": issue_type,
        "issue_summary": issue_summary,
        "sentiment": sentiment,
        "transcript_lines": len(transcript.splitlines()),
    }

    print(f"\n[CCAS ESCALATION] Sending to {payload['endpoint']}:")
    print(json.dumps({k: v for k, v in payload.items() if k != "endpoint"}, indent=2))

    save_escalation(conversation_id, issue_type, issue_summary, sentiment, ticket_id)

    return {
        "status": "escalated",
        "ticket_id": ticket_id,
        "estimated_wait_minutes": 2,
        "message": "Escalation successful. Inform the customer that a human agent will join the chat shortly, with an estimated wait time of approximately 2 minutes. Do not offer further assistance — the human agent will take over from here.",
    }


def execute_tool(
    tool_name: str,
    tool_input: dict,
    conversation_id: str = None,
    message_id: int = None,
    client=None,
) -> str:
    """Route a tool call from the model to the correct function and return the result as a string."""
    if tool_name == "identify_customer":
        result = identify_customer(tool_input["customer_contact"])
    elif tool_name == "lookup_order":
        result = lookup_order(tool_input["order_id"], tool_input["customer_contact"])
    elif tool_name == "submit_refund":
        result = submit_refund(tool_input["order_id"], tool_input["customer_contact"], tool_input["reason"])
    elif tool_name == "search_knowledge_base":
        from knowledge_base import search_knowledge_base
        from storage import save_citation
        text, citations = search_knowledge_base(tool_input["query"])
        if conversation_id and message_id:
            for c in citations:
                save_citation(conversation_id, message_id, c["article"], c["section"], c["similarity_score"])
        return text
    elif tool_name == "escalate_to_agent":
        result = escalate_to_agent(
            tool_input["customer_contact"],
            tool_input["issue_type"],
            tool_input["issue_summary"],
            conversation_id,
            client,
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return str(result)
