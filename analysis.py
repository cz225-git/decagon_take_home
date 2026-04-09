"""
LLM-based conversation analysis functions.
Used at escalation time (sentiment only) and conversation close (full analysis).
"""

import json
import re


def _parse_json(text: str) -> dict:
    """Strip markdown code fences if present, then parse JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def analyze_sentiment(transcript: str, client) -> str:
    """
    Classify the customer's overall sentiment as 'positive', 'negative', or 'neutral'.
    Used at escalation time — quick call, returns a single label.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": (
                "Classify the customer's sentiment in this support conversation as "
                "'positive', 'negative', or 'neutral'. "
                "Respond with JSON only: {\"sentiment\": \"...\"}\n\n"
                f"Transcript:\n{transcript}"
            ),
        }],
    )
    try:
        return _parse_json(response.content[0].text)["sentiment"]
    except Exception:
        return "neutral"


def analyze_conversation(transcript: str, client) -> dict:
    """
    Full conversation analysis run at close time.
    Returns {sentiment, issue_type, issue_summary} for the CCAS completion report.

    issue_type is one of: order_status, refund_dispute, shipping_issue,
    account_issue, damaged_item, policy_question, other
    """
    issue_types = [
        "order_status", "refund_dispute", "shipping_issue",
        "account_issue", "damaged_item", "policy_question", "other",
    ]
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Analyze this customer support conversation and return JSON with:\n"
                "- sentiment: 'positive', 'negative', or 'neutral'\n"
                f"- issue_type: one of {issue_types}\n"
                "- issue_summary: 1-2 sentence summary of what the customer needed "
                "and how it was resolved\n"
                "- resolved: true if the customer's issue was fully addressed, "
                "false if it was abandoned, unresolved, or ended without a clear outcome\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Respond with JSON only."
            ),
        }],
    )
    try:
        result = _parse_json(response.content[0].text)
        return {
            "sentiment": result.get("sentiment", "neutral"),
            "issue_type": result.get("issue_type", "other"),
            "issue_summary": result.get("issue_summary", ""),
            "resolved": result.get("resolved", None),
        }
    except Exception:
        return {"sentiment": "neutral", "issue_type": "other", "issue_summary": "", "resolved": None}
