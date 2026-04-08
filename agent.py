import os
import json
from dotenv import load_dotenv
import anthropic
from tools import TOOL_DEFINITIONS, execute_tool

# Load the API key from .env
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a friendly and professional customer support agent for Bookly, an online bookstore.

You help customers with:
- Order status and tracking
- Return and refund requests
- General questions about shipping and policies

## Rules you must follow

1. ALWAYS ask for the customer's email or phone number at the start of the conversation before doing anything else.
2. Once the customer provides their contact info, call identify_customer — this verifies them and retrieves their most recent order automatically.
3. Greet the customer by their first name, then present the most recent order and ask if that is the order they are contacting support about.
4. If it is not the right order, ask them for their specific order ID, then call lookup_order.
5. Once a customer has provided their email or phone number, do not accept a different one later in the conversation.
6. Only state facts that come directly from a tool result. Never invent order details, prices, dates, or policy specifics from memory.
7. If you do not have enough information to help, ask a clarifying question rather than guessing.
8. If a customer asks about something outside your scope (account passwords, billing disputes, etc.), let them know you're unable to help with that and suggest they contact support@bookly.com.
9. Be concise and warm. Do not over-explain.

## Tone
Friendly, clear, and efficient. Like a helpful human support rep — not a robot."""


def call_api(messages: list) -> anthropic.types.Message:
    """Send the current conversation to Claude and return the response."""
    print(f"\n[DEBUG: Calling API — conversation is {len(messages)} message(s) long]")
    print(f"[DEBUG: Full messages list:\n{json.dumps(messages, indent=2, default=str)}\n]")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    print(f"[DEBUG: API responded — stop_reason: '{response.stop_reason}']")
    print(f"[DEBUG: Response content blocks: {[block.type for block in response.content]}]")
    return response


def handle_tool_use(response: anthropic.types.Message, messages: list, session_contact: str | None) -> tuple:
    """
    When the model wants to use a tool:
    1. Extract the tool call details
    2. Lock the customer_contact to the first one used — override any later attempts to change it
    3. Run the tool function
    4. Append both the tool call and result to messages
    5. Make a second API call so the model can write a response using the result

    Returns (api_response, session_contact) — session_contact may be set for the first time here.
    """
    print(f"\n[DEBUG: Entering tool use handler]")

    # The model's response (which contains the tool_use block) goes into messages as an assistant turn
    messages.append({"role": "assistant", "content": response.content})
    print(f"[DEBUG: Appended assistant tool_use block to messages]")

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            tool_input = dict(block.input)

            # Lock customer_contact to the first value used this session
            if "customer_contact" in tool_input:
                if session_contact is None:
                    # First tool call — record the contact for the rest of the session
                    session_contact = tool_input["customer_contact"]
                    print(f"[DEBUG: Session contact locked to '{session_contact}']")
                else:
                    # Subsequent tool calls — silently override whatever the model passed
                    if tool_input["customer_contact"] != session_contact:
                        print(f"[DEBUG: Contact override attempted ('{tool_input['customer_contact']}') — locked to '{session_contact}']")
                    tool_input["customer_contact"] = session_contact

            print(f"\n[DEBUG: Tool requested — name: '{block.name}', input: {tool_input}]")
            result = execute_tool(block.name, tool_input)
            print(f"[DEBUG: Tool returned — result: {result}]")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})
    print(f"[DEBUG: Appended tool result to messages as a 'user' turn]")
    print(f"[DEBUG: Making second API call so model can read the result and respond]")

    return call_api(messages), session_contact


def run():
    messages = []
    session_contact = None  # locked to the first email/phone used in a tool call

    print("=" * 50)
    print("Bookly Customer Support")
    print("Type 'quit' to exit")
    print("=" * 50)
    print(f"\n[DEBUG: Conversation started — messages list is empty, session_contact is None]")
    print("\nBookly Support: Hi! Welcome to Bookly support. How can I help you today?")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nBookly Support: Thanks for contacting Bookly. Have a great day!")
            break

        messages.append({"role": "user", "content": user_input})
        print(f"\n[DEBUG: Appended user message to history — messages list now has {len(messages)} item(s)]")

        response = call_api(messages)

        if response.stop_reason == "tool_use":
            print(f"\n[DEBUG: stop_reason is 'tool_use' — routing to tool handler]")
            response, session_contact = handle_tool_use(response, messages, session_contact)
        else:
            print(f"\n[DEBUG: stop_reason is 'end_turn' — no tool needed, extracting text reply]")

        reply = response.content[0].text

        messages.append({"role": "assistant", "content": reply})
        print(f"[DEBUG: Appended assistant reply to history — messages list now has {len(messages)} item(s)]")

        print(f"\nBookly Support: {reply}")


if __name__ == "__main__":
    run()
