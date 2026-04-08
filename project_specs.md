# Bookly Customer Support Agent — Project Specs

## Assignment
Decagon Solutions Engineering take-home. Build a customer support AI agent for "Bookly" (fictional online bookstore). Deliverables: working prototype + one-page design doc.

## Key Constraints
- Must call APIs directly — no all-in-one agentic platforms (no LangChain, AutoGen, etc.)
- Use Anthropic Claude API
- Code does not need to be production-ready
- Tools can be mocked

## Use Cases (focus on depth, not breadth)
1. **Order status** — agent asks for order ID (clarifying question), calls lookup_order tool, returns status
2. **Return/refund request** — multi-turn: collect order ID + reason, call submit_refund tool
3. **Policy questions** — search knowledge base via RAG, answer from KB content only

## Build Strategy: Incremental
### Phase 1 (Basic CLI Agent) — BUILD FIRST, PAUSE FOR TESTING
- `agent.py` — while loop, message history, Anthropic API call, tool_use handling
- `tools.py` — mocked lookup_order and submit_refund tools
- System prompt — persona, behavioral rules, tool-gating instructions
- `.env` — API key loaded via python-dotenv

### Phase 2 (Add-ons) — only after Phase 1 is tested and understood
- **2a. SQLite storage** (`storage.py`) — write each message as it happens, crash-safe
- **2b. API error handling** — exponential backoff retry, graceful user messages
- **2c. RAG with FAISS** (`knowledge_base.py`) — chunk KB articles, embed, similarity search
- **2d. Hallucination guardrails** — tool-gating (structural) + output validation (programmatic)

## File Structure
```
Decagon_Take_Home/
├── agent.py
├── tools.py
├── knowledge_base.py   # Phase 2
├── storage.py          # Phase 2
├── data/
│   └── kb_articles.md  # Phase 2
├── .env                # API key — never commit
├── .gitignore
├── requirements.txt
└── design_doc.md       # Final deliverable
```

## Why Anthropic API
Previous experience with OpenAI and Gemini APIs in a prior role — chose Anthropic specifically to learn a new API and become familiar with Claude's tool use patterns.

## Tech Stack
- Python 3.14
- anthropic SDK
- python-dotenv
- faiss-cpu (Phase 2)
- openai (Phase 2, embeddings)

## Conversation Flow
```
User message
  → Append to messages[]
  → API call (system prompt + messages + tools)
  → stop_reason == "tool_use"?
      YES → run tool function → append result → second API call
      NO  → print response
  → Append assistant response to messages[]
  → Loop
```

## Design Doc Sections (deliverable)
1. Architecture overview + flow diagram
2. Conversation & decision design
3. Hallucination & safety controls
4. Production readiness tradeoffs
5. Example system prompt

## Key Design Decisions & Rationale
- **Intent routing via tool descriptions** — no separate classifier; model matches user intent to tool based on description
- **Tool-gating for hallucination** — structural guarantee: agent can only state specifics from tool results
- **SQLite for storage** — simple, local, queryable; production would use PostgreSQL + encryption
- **FAISS for RAG** — local, no external dependencies; production would use Pinecone/Weaviate
- **Paragraph chunking** — preserves semantic coherence vs. fixed token chunking
