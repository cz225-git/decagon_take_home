import json
import os
import time
from dotenv import load_dotenv
import anthropic
from tools import TOOL_DEFINITIONS, execute_tool
from storage import init_db, create_conversation, save_message, close_conversation, get_transcript
from knowledge_base import init_knowledge_base
from validation import validate_regex, validate_llm
from analysis import analyze_conversation

MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds — doubles after each failed attempt


class APIFailureError(Exception):
    """Raised when all API retry attempts are exhausted."""
    pass

# Load the API key from .env
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a friendly and professional customer support agent for Bookly, an online bookstore.

You help customers with:
- Order status and tracking
- Return and refund requests
- General questions about shipping, payment methods, and policies
- Escalation to a human agent when needed

## Rules you must follow

1. ALWAYS ask for the customer's email or phone number at the start of the conversation before doing anything else.
2. Once the customer provides their contact info, call identify_customer — this verifies them and retrieves their most recent order automatically.
3. Greet the customer by their first name, then present the most recent order and ask if that is the order they are contacting support about.
4. If it is not the right order, check the recent_orders list first — if you can identify the order they mean from the items, date, or description they provide, use that order ID directly. Only ask for an order ID if you genuinely cannot identify the order from the recent_orders list.
5. Once a customer has provided their email or phone number, do not accept a different one later in the conversation.
6. If a customer wants to return or refund an order, check eligible_for_refund from the order data first. If it is false, inform the customer immediately — do not ask for a return reason.
7. Only state facts that come directly from a tool result. Never invent order details, prices, dates, or policy specifics from memory.
8. For any question about shipping, policies, returns, password reset, payment, or order cancellation — always call search_knowledge_base first. Never answer policy questions from memory.
9. If search_knowledge_base returns no relevant results, tell the customer you don't have that information and suggest they contact support@bookly.com.
10. If you do not have enough information to help, ask a clarifying question rather than guessing.
11. If a customer is frustrated, explicitly asks to speak to a human, or has an issue you cannot resolve with the available tools — call escalate_to_agent. Write an accurate issue_type and a brief issue_summary based on the conversation.
12. If a customer asks about something outside your scope (billing disputes, account hacking, etc.), let them know you're unable to help with that and suggest they contact support@bookly.com.
13. Be concise and warm. Do not over-explain.

## Tone
Friendly, clear, and efficient. Like a helpful human support rep — not a robot."""


def call_api(messages: list) -> anthropic.types.Message:
    """
    Send the current conversation to Claude and return the response.
    Retries up to MAX_RETRIES times with exponential backoff on transient failures.
    Raises APIFailureError if all attempts fail.
    """
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except anthropic.RateLimitError:
            error_type = "Rate limit hit"
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                # 4xx errors are client mistakes (bad request, auth failure, etc.)
                # Retrying won't help — fail immediately
                raise APIFailureError(f"API error {e.status_code}: {e.message}")
            error_type = f"Server error ({e.status_code})"
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            error_type = "Connection issue"

        if attempt < MAX_RETRIES:
            print(f"\nBookly Support: One moment please...")
            time.sleep(delay)
            delay *= 2  # exponential backoff: 1s → 2s → 4s
        else:
            raise APIFailureError(f"{error_type} — all {MAX_RETRIES} attempts failed.")


def handle_tool_use(
    response: anthropic.types.Message,
    messages: list,
    session_contact: str | None,
    customer_name: str | None,
    conversation_id: str
) -> tuple:
    """
    When the model wants to use a tool:
    1. Lock customer_contact to the first value used this session
    2. Extract the tool call details
    3. Run the tool function and save the call + result to storage
    4. Append both the tool call and result to messages
    5. Make a second API call so the model can write a response using the result
    Returns (api_response, session_contact, customer_name, raw_tool_results)
    where raw_tool_results is the list of result strings for output validation.
    """
    # The model's response (which contains the tool_use block) goes into messages as an assistant turn
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    raw_tool_results = []  # plain strings for output validation
    for block in response.content:
        if block.type == "tool_use":
            tool_input = dict(block.input)

            # Lock customer_contact to the first value used this session
            if "customer_contact" in tool_input:
                if session_contact is None:
                    session_contact = tool_input["customer_contact"]
                else:
                    tool_input["customer_contact"] = session_contact

            # Save the tool call to storage before executing — capture its ID to link citations
            tool_call_message_id = save_message(conversation_id, role="tool_call", content=str(tool_input), tool_used=block.name)

            result = execute_tool(block.name, tool_input, conversation_id, tool_call_message_id, client=client)

            raw_tool_results.append(result)

            # Save the tool result to storage
            save_message(conversation_id, role="tool_result", content=result, tool_used=block.name)

            # If this was identify_customer, capture the customer's name for storage
            if block.name == "identify_customer":
                try:
                    parsed = eval(result)  # result is a stringified dict
                    if "customer_name" in parsed:
                        customer_name = parsed["customer_name"]
                except Exception:
                    pass

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # links result back to the specific tool call
                "content": result,
            })

    # Append the tool results as a user turn (this is how the Anthropic API expects it)
    messages.append({"role": "user", "content": tool_results})

    # Second API call — model reads the tool result and writes the final reply
    return call_api(messages), session_contact, customer_name, raw_tool_results


def _close_and_report(
    conversation_id: str,
    session_contact: str,
    customer_name: str,
    turn_count: int,
    escalated: bool = False,
):
    """
    Run end-of-conversation analysis and send a mock CCAS completion report,
    then close the conversation record in the DB.
    """
    try:
        transcript = get_transcript(conversation_id)
        analysis = analyze_conversation(transcript, client)
        sentiment = analysis.get("sentiment")

        resolved = analysis.get("resolved")
        payload = {
            "endpoint": "ccas.bookly.com/api/v1/report",
            "customer_contact": session_contact,
            "issue_type": analysis.get("issue_type"),
            "issue_summary": analysis.get("issue_summary"),
            "sentiment": sentiment,
            "resolved": resolved,
            "transcript_lines": len(transcript.splitlines()),
        }
        print(f"\n[CCAS COMPLETION REPORT] Sending to {payload['endpoint']}:")
        print(json.dumps({k: v for k, v in payload.items() if k != "endpoint"}, indent=2))

    except Exception:
        sentiment = None
        resolved = None

    close_conversation(
        conversation_id,
        customer_contact=session_contact,
        customer_name=customer_name,
        turn_count=turn_count,
        escalated=escalated,
        sentiment=sentiment,
        resolved=resolved,
    )


def run():
    # Set up database and knowledge base
    init_db()
    init_knowledge_base()

    # Create a new conversation record for this session
    conversation_id = create_conversation()

    messages = []
    session_contact = None
    customer_name = None
    turn_count = 0

    print("=" * 50)
    print("Bookly Customer Support")
    print("Type 'quit' to exit")
    print("=" * 50)
    print("\nBookly Support: Hi! Welcome to Bookly support. How can I help you today?")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nBookly Support: Thanks for contacting Bookly. Have a great day!")
            _close_and_report(conversation_id, session_contact, customer_name, turn_count)
            break

        turn_count += 1
        messages.append({"role": "user", "content": user_input})
        save_message(conversation_id, role="user", content=user_input)

        try:
            response = call_api(messages)
            raw_tool_results = []

            if response.stop_reason == "tool_use":
                response, session_contact, customer_name, raw_tool_results = handle_tool_use(
                    response, messages, session_contact, customer_name, conversation_id
                )
        except APIFailureError:
            print("\nBookly Support: I'm sorry, I'm having trouble connecting right now. Please try again later or contact support@bookly.com.")
            _close_and_report(conversation_id, session_contact, customer_name, turn_count)
            break

        reply = response.content[0].text

        if raw_tool_results:
            regex_flags = validate_regex(reply, raw_tool_results)
            # if regex_flags:
            #     print(f"[VALIDATION — Regex] Unverified token(s) in reply: {regex_flags}")

            llm_flags = validate_llm(reply, raw_tool_results, client)
            # if llm_flags:
            #     print(f"[VALIDATION — LLM] Potential semantic violation(s): {llm_flags}")

        messages.append({"role": "assistant", "content": reply})
        save_message(conversation_id, role="assistant", content=reply)

        print(f"\nBookly Support: {reply}")


if __name__ == "__main__":
    run()
