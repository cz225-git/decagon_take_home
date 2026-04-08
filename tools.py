# tools.py
# Defines the tools available to the agent and the functions that execute them.
# In a real system these would call a database or external API.
# Here they return hardcoded mock data so we can demo without a real backend.

# ---------------------------------------------------------------------------
# Mock data — fake orders the agent can look up
# order_date is used to determine the most recent order per customer
# ---------------------------------------------------------------------------

MOCK_ORDERS = {
    "BK-1001": {
        "order_id": "BK-1001",
        "customer_name": "Alex Johnson",
        "customer_email": "alex.johnson@email.com",
        "customer_phone": "555-101-2020",
        "order_date": "2026-03-01",
        "status": "shipped",
        "items": ["The Great Gatsby", "1984"],
        "total": "$24.99",
        "estimated_delivery": "April 10, 2026",
        "tracking_number": "USPS-9400111899223456789012",
        "eligible_for_refund": True,
    },
    "BK-1002": {
        "order_id": "BK-1002",
        "customer_name": "Maria Garcia",
        "customer_email": "maria.garcia@email.com",
        "customer_phone": "555-202-3131",
        "order_date": "2026-02-14",
        "status": "delivered",
        "items": ["Atomic Habits"],
        "total": "$18.99",
        "estimated_delivery": "April 3, 2026",
        "tracking_number": "USPS-9400111899223456789099",
        "eligible_for_refund": False,  # outside 30-day return window
    },
    "BK-1003": {
        "order_id": "BK-1003",
        "customer_name": "Sam Lee",
        "customer_email": "sam.lee@email.com",
        "customer_phone": "555-303-4242",
        "order_date": "2026-04-01",
        "status": "processing",
        "items": ["Dune", "Foundation"],
        "total": "$34.50",
        "estimated_delivery": "April 12, 2026",
        "tracking_number": None,
        "eligible_for_refund": True,
    },
    "BK-1004": {
        "order_id": "BK-1004",
        "customer_name": "Alex Johnson",
        "customer_email": "alex.johnson@email.com",
        "customer_phone": "555-101-2020",
        "order_date": "2026-04-05",
        "status": "processing",
        "items": ["The Hobbit"],
        "total": "$14.99",
        "estimated_delivery": "April 14, 2026",
        "tracking_number": None,
        "eligible_for_refund": True,
    },
}

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
]

# ---------------------------------------------------------------------------
# Helper — checks whether a contact string matches the order's email or phone
# ---------------------------------------------------------------------------

def contact_matches(order: dict, customer_contact: str) -> bool:
    """Return True if the provided contact matches the order's email or phone."""
    contact = customer_contact.strip().lower()
    return (
        contact == order["customer_email"].lower()
        or contact == order["customer_phone"].lower()
    )

# ---------------------------------------------------------------------------
# Tool execution functions — called by agent.py when the model requests a tool
# ---------------------------------------------------------------------------

def identify_customer(customer_contact: str) -> dict:
    """
    Called by the script (not chosen by the LLM) once contact info is parsed.
    Finds the most recent order for the contact and returns a summary.
    The LLM's only job was to extract the contact string — the lookup runs automatically.
    """
    matching = [o for o in MOCK_ORDERS.values() if contact_matches(o, customer_contact)]
    if not matching:
        return {"result": "No orders found for this contact information."}
    most_recent = max(matching, key=lambda o: o["order_date"])
    # Return a summary only — full details come via lookup_order once the customer confirms
    return {
        "verified": True,
        "customer_name": most_recent["customer_name"],
        "recent_order": {
            "order_id": most_recent["order_id"],
            "order_date": most_recent["order_date"],
            "items": most_recent["items"],
            "status": most_recent["status"],
            "total": most_recent["total"],
        }
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
    """Submit a refund request. Verifies identity and eligibility before approving."""
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
    return {
        "status": "approved",
        "order_id": order_id,
        "refund_amount": order["total"],
        "message": f"Refund request submitted successfully. {order['total']} will be returned to your original payment method within 5-7 business days.",
    }


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Route a tool call from the model to the correct function and return the result as a string."""
    if tool_name == "identify_customer":
        result = identify_customer(tool_input["customer_contact"])
    elif tool_name == "lookup_order":
        result = lookup_order(tool_input["order_id"], tool_input["customer_contact"])
    elif tool_name == "submit_refund":
        result = submit_refund(tool_input["order_id"], tool_input["customer_contact"], tool_input["reason"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return str(result)
