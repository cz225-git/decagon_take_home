import re
import json

# Patterns for specific factual tokens the agent might state
FACT_PATTERNS = [
    r'BK-\d+',                    # Order IDs
    r'\$[\d,]+\.\d{2}',           # Prices
    r'USPS-\w+',                   # Tracking numbers
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Dates
]


def validate_regex(reply: str, tool_results: list) -> list:
    """
    Extract fact tokens from the reply using regex patterns.
    Returns a list of tokens that appear in the reply but not in any tool result.
    These are potential hallucinations or transformations of ground-truth data.
    """
    combined = " ".join(tool_results)
    violations = []
    for pattern in FACT_PATTERNS:
        for match in re.finditer(pattern, reply):
            token = match.group()
            if token not in combined:
                violations.append(token)
    return violations


def validate_llm(reply: str, tool_results: list, client) -> list:
    """
    Use a verifier LLM call to identify claims in the reply not supported by tool results.
    Returns a list of violation strings.

    Note: Adds ~200ms latency per tool-use turn. Not appropriate for voice agent deployments
    where sub-100ms response times are required. In those contexts, rely on regex validation
    and structural tool-gating instead.
    """
    if not tool_results:
        return []

    combined = "\n\n".join(tool_results)
    prompt = (
        "You are a fact-checker for a customer support agent.\n\n"
        f"Tool Results (ground truth):\n{combined}\n\n"
        f"Agent Reply:\n{reply}\n\n"
        "List any specific facts in the Agent Reply (order IDs, prices, dates, statuses, "
        "tracking numbers) that are NOT directly supported by the Tool Results. "
        "Do not flag general policy paraphrasing — only flag specific factual claims.\n\n"
        'Respond with JSON only: {"violations": ["..."]}. '
        'If there are no violations: {"violations": []}'
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        parsed = json.loads(response.content[0].text)
        return parsed.get("violations", [])
    except Exception:
        return []
