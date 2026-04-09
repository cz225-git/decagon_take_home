# Bookly AI Support Agent — Design Document

---

## Architecture Overview

The agent is a Python CLI application built directly on the Anthropic Claude API. It uses no agentic frameworks — all orchestration is handled in code.

**Flow for each conversation turn:**
```
User input
  → Append to messages[]
  → API call (system prompt + full message history + tool definitions)
  → stop_reason == "tool_use"?
      YES → execute tool locally → append result → second API call → text reply
      NO  → text reply directly
  → Append reply to messages[], write to SQLite
  → Loop
```

**Components:**
| File | Role |
|---|---|
| `agent.py` | Conversation loop, API calls, retry logic, completion reporting |
| `tools.py` | Tool definitions (sent to API) + execution functions |
| `order_data.py` | Seeded order generation, JSON persistence, mutable state |
| `knowledge_base.py` | FAISS vector index, embedding, semantic search |
| `storage.py` | SQLite: conversations, messages, citations, escalations |
| `analysis.py` | LLM-based sentiment analysis + conversation classification |
| `validation.py` | Post-response output validation (regex + LLM-as-judge) |
| `data/articles/` | KB articles as individual markdown files |

**Why Anthropic over OpenAI or Gemini:** I had prior experience with both OpenAI and Gemini APIs in a previous role and chose Anthropic specifically to work with a new API and Claude's tool use patterns.

---

## Conversation & Decision Design

**Intent routing via tool descriptions, not a classifier**

Rather than a separate intent classification step, routing is handled by how tool descriptions are written. The model reads all tool descriptions on every turn and matches the user's intent to the most appropriate tool. This keeps the architecture simple — there's no additional latency or API call for routing — and means routing logic lives in one place (the tool definition) rather than being split across a classifier and an executor. The tradeoff is that ambiguous intent can cause the model to pick the wrong tool, which is mitigated by writing precise, non-overlapping descriptions.

**LLM extracts, script executes**

For `identify_customer`, the LLM's only job is parsing the customer's contact info from natural language. The moment it calls the tool, the script runs the order lookup automatically — the LLM never decides whether to look up orders. This is a general principle applied throughout: use the LLM for language understanding, use code for deterministic operations.

**Recent orders window (90 days)**

On identification, the agent loads all orders placed in the last 90 days, not just the most recent. This allows the agent to resolve order references by description ("my Great Gatsby order") without requiring the customer to know their order ID. Orders outside the window require an explicit ID — a reasonable tradeoff between convenience and scope.

**Eligibility check before collecting return reason**

The agent checks `eligible_for_refund` from the order data before asking for a return reason. Asking a customer to explain their return only to immediately tell them it's ineligible is a poor experience. This is handled via a system prompt rule, not code — which means it's behavioral rather than guaranteed, but it's appropriate here since the consequence of failure is UX friction, not data integrity.

**Session contact locking**

Once a customer provides their email or phone and a tool call fires, that contact value is locked for the session at the code level in `handle_tool_use`. Even if the model receives a different contact later in the conversation, the code silently overwrites it with the original before executing any tool. This is intentionally a code-level guarantee, not a prompt instruction — prompt instructions can be overridden by a sufficiently persistent user.

**Clarifying question before acting**

The system prompt instructs the agent to ask for missing information rather than guess. The primary trigger is the `identify_customer` flow — the agent won't call any order tool until it has collected and locked a contact value. This satisfies the multi-turn requirement while keeping the constraint structural.

---

## Hallucination & Safety Controls

**Tool-gating (structural)**

The most reliable guardrail is that the agent cannot state order-specific or policy-specific information without first calling a tool. Order details only come from `lookup_order` results. Policy details only come from `search_knowledge_base` results. The system prompt reinforces this with explicit rules ("only state facts that come directly from a tool result"), but the structural dependency is what actually enforces it — if the tool isn't called, the information isn't available.

**RAG threshold filtering**

`search_knowledge_base` only returns chunks with a cosine similarity score ≥ 0.4. Below this threshold, the function returns "No relevant information found" and the agent is instructed to say so rather than attempt an answer from memory. This prevents the agent from synthesizing a plausible-sounding but incorrect policy answer when the query doesn't match the KB.

**Output validation (post-response)**

Even with tool-gating, the LLM can still transform facts when generating its reply — misstating an order ID, rounding a price, or paraphrasing a date. Two layers catch this:

1. **Regex extraction:** After each reply, regex patterns extract specific factual tokens (order IDs, prices, tracking numbers, dates). Any token found in the reply but absent from the tool result strings is flagged and logged. Zero latency, zero cost. Misses semantic drift (e.g., "your order is on its way" when status is "processing").

2. **LLM-as-judge:** A second API call to a verifier model (haiku) receives the tool results and the reply and is asked to list any specific factual claims not supported by the tool data. Catches semantic transformations regex cannot. Adds ~200ms latency and a small per-turn cost — this makes it inappropriate for voice agent deployments, where sub-100ms response times are expected and the UX cost of added latency would outweigh the benefit. In voice contexts, rely on regex validation and structural tool-gating instead.

Both layers log warnings only and do not block the user-facing response — appropriate for a prototype where visibility matters more than enforcement.

**Why prompt instructions alone are insufficient**

In long conversations or with adversarial inputs, models can violate behavioral instructions. The tool-gating approach is preferred because it's structural — the agent literally does not have specific data to hallucinate from unless a tool has been called and returned it. Prompt instructions serve as a secondary layer, not the primary control.

**Session contact immutability**

Accepting a different contact mid-conversation would allow a user to access another customer's order data. This is prevented at the code level, not the prompt level, for the same reason: prompt-level instructions can be bypassed.

---

## Example System Prompt

```
You are a friendly and professional customer support agent for Bookly, an online bookstore.

You help customers with:
- Order status and tracking
- Return and refund requests
- General questions about shipping and policies
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
11. If a customer is frustrated, explicitly asks to speak to a human, or has an issue you cannot resolve with the available tools — call escalate_to_agent.
12. If a customer asks about something outside your scope, let them know and suggest they contact support@bookly.com.
13. Be concise and warm. Do not over-explain.
```

---

## RAG Design

**Embedding model: `all-MiniLM-L6-v2` (local, via sentence-transformers)**

Chose a local model over OpenAI's `text-embedding-3-small` for three reasons: no additional API key or cost, embeddings are computed once and cached to disk so startup cost is a one-time hit, and quality difference is negligible for a small, domain-specific KB. Production would use a managed embedding API for versioning, multilingual support, and operational simplicity.

**Chunking strategy: paragraph-level**

Articles are split on double newlines (markdown paragraph boundaries). Each chunk is prefixed with its section name to preserve context. Paragraph chunking was chosen over fixed-size token chunking (which splits sentences mid-thought) and sentence-level chunking (which loses context). Section-level chunking would produce larger, richer chunks but risks diluting the embedding signal. Paragraph is the best balance for this KB size.

**Retrieval: FAISS with cosine similarity, top_k=3, threshold=0.4**

FAISS retrieves up to 3 candidates per query. Only those scoring ≥ 0.4 cosine similarity are passed to the LLM — below this score, results are considered too weakly related to be useful. All chunks that pass the threshold are sent to the LLM in a single tool result; the model synthesizes from them. Production would add a reranking step (cross-encoder model) between retrieval and generation to improve precision at the cost of latency.

**Source traceability**

KB articles are stored as separate files in `data/articles/`. Each chunk carries metadata (article name, section). Search results include `[Source: Article — Section]` tags so the agent can cite them. Citations are written to a dedicated `citations` SQLite table with a `message_id` foreign key linking each citation to the exact tool call message that triggered it, enabling turn-level audit queries.

---

## Order Data & Mutable State

**Seeded generation for reproducibility**

`order_data.py` generates 1000 orders across ~600 customers using `random.Random(42)`. The seeded RNG ensures the same dataset every time the file is regenerated, which makes debugging and demo flows repeatable. 4 original hardcoded orders (BK-1001 through BK-1004) are preserved for test compatibility; the remaining 996 are generated.

**JSON persistence with load-or-generate pattern**

On first run, the generated dataset is written to `data/orders.json`. On subsequent runs, it loads from that file. This means the data is stable across sessions without requiring a database for order storage. The tradeoff: JSON isn't suitable for concurrent writes at scale, but it's appropriate here because the prototype runs as a single-user CLI.

**Mutable refund state**

When `submit_refund` approves a refund, it mutates the in-memory order dict (`eligible_for_refund = False`, adds `refund_status` and `refund_submitted_at`) and writes the full dataset back to disk via `save_orders()`. This means:
- Within the same session, the agent correctly reports an order as already refunded if the customer asks again
- Across sessions, the refund persists — the order won't be offered for refund again

The write-back-on-mutation approach is intentionally simple. Production would use a database transaction with proper locking.

**Phone number normalization**

Contact matching strips all non-digit characters before comparing phone numbers, so `555-696-9631`, `(555) 696-9631`, and `5556969631` all match. Email comparison is case-insensitive. This is handled in `contact_matches()` at the code level — the LLM doesn't need to normalize input formats.

---

## CCAS Escalation & Conversation Lifecycle

**Escalation tool**

`escalate_to_agent` mimics a handoff to a Contact Center as a Service (CCAS) platform. The tool accepts `customer_contact`, `issue_type` (enum), and `issue_summary` (LLM-written). When called, it:

1. Retrieves the full transcript from SQLite (only user/assistant turns — tool calls are internal)
2. Runs a sentiment analysis LLM call on the transcript
3. Logs the payload (contact, transcript, issue type, summary, sentiment) as a mock API POST
4. Saves an escalation record to the `escalations` table and marks the conversation as escalated

The LLM writes `issue_type` and `issue_summary` based on conversation context — these aren't hardcoded or classifier-derived. The return message instructs the agent to tell the customer a human will join shortly with an estimated 2-minute wait, and to stop offering further assistance.

**Completion reporting**

Every conversation — whether it ends cleanly (user types quit) or due to an API failure — runs through `analyze_conversation()`, a single LLM call that returns sentiment, issue type, issue summary, and whether the issue was resolved. This is logged as a mock CCAS completion report. The analysis runs in a try/except so failures don't block the conversation from closing cleanly.

**Sentiment analysis**

Two separate LLM calls serve different purposes:
- `analyze_sentiment()` — lightweight, returns a single label (`positive`/`negative`/`neutral`). Used at escalation time because a full analysis call isn't worth the latency mid-conversation.
- `analyze_conversation()` — full analysis returning sentiment, issue type, summary, and resolved status. Used at conversation close.

Both are persisted: sentiment goes to the `conversations` table; escalation sentiment goes to the `escalations` table.

**Resolved tracking**

The `resolved` field (stored as `INTEGER` in SQLite: 1 = resolved, 0 = not, NULL = analysis failed) allows admins to identify abandoned or unresolved conversations. This distinguishes between a customer who got their answer and left satisfied vs. one who gave up. Stored as a nullable integer rather than a boolean so the "analysis didn't run" case is distinguishable from "analysis ran and said unresolved."

---

## Production Readiness

| Prototype decision | Production change | Reason |
|---|---|---|
| SQLite, no encryption | PostgreSQL + column encryption, role-based access | PII at scale requires access controls, encryption at rest, and right-to-deletion support |
| JSON key-value auth (email/phone) | Verified `customer_id` from authenticated session | Current approach is trust-based — real auth requires session tokens or OAuth |
| FAISS local index | Pinecone or Weaviate | Managed vector store adds persistence, horizontal scale, and metadata filtering |
| Local embedding model | Managed embedding API (e.g. OpenAI) | Versioning, multilingual support, no local compute dependency |
| Manual exponential backoff | Anthropic SDK `max_retries` + circuit breaker | SDK handles retry logic correctly; circuit breaker stops hammering a failing service |
| LLM-based sentiment analysis | LLM scoring calibrated against explicit CSAT ratings | Explicit ratings have <30% response rate; LLM scoring covers 100% and can be calibrated against the ones you do collect |
| Paragraph chunking, no reranking | Hybrid search (keyword + semantic) + cross-encoder reranking | Improves retrieval precision significantly for larger, more diverse KBs |
| Conversation data unencrypted | Automatic retention policy + PII scrubbing | GDPR compliance requires time-bounded retention and deletion on request |
| JSON file for order data | Database with transactional writes | JSON write-back doesn't support concurrent users or partial updates |
| Mock CCAS API (logged payloads) | HTTP POST to real CCAS endpoint | Prototype logs the payload; production sends it to the actual escalation/reporting service |
| Validation logs warnings only | Configurable enforcement (block, rewrite, or flag) | Prototype prioritizes visibility; production should act on detected hallucinations |

---

## Production Roadmap

Beyond the prototype-to-production swaps above, several features would make this a complete production system.

### Cost Tracking (Token Logging)

The Anthropic SDK already returns `input_tokens` and `output_tokens` on every response via `response.usage`. Logging these per API call and aggregating per conversation gives direct visibility into cost. Implementation: add `input_tokens` and `output_tokens` columns to the `conversations` table, increment them after each `call_api()` return, and write the totals on conversation close. Downstream, aggregate by customer, issue type, and time period to build cost dashboards. This data also enables per-conversation token budgets — if a conversation exceeds a threshold (e.g., a runaway tool loop), the system can force an escalation to a human rather than continuing to burn tokens. Cost-per-resolution becomes a trackable metric alongside CSAT and resolution rate.

### CSAT Scoring (Hybrid: Explicit + LLM-Inferred)

The prototype runs LLM-based sentiment analysis on every conversation, but sentiment is a weak proxy for satisfaction — a customer can be neutral in tone and still leave unsatisfied. Production CSAT requires two inputs:

1. **Explicit ratings.** After conversation close, prompt the customer for a 1-5 rating via the UI. Response rates for post-chat surveys are typically under 30%, and skew toward extremes (very satisfied or very dissatisfied), so this alone is insufficient.

2. **LLM-inferred CSAT.** For the ~70% of conversations without an explicit rating, run an LLM call on the transcript that predicts a CSAT score. The key is calibration: use the explicit ratings as ground truth to tune the LLM scorer's prompts and thresholds. Store `explicit_csat` and `inferred_csat` as separate columns so calibration drift is detectable — if the inferred distribution diverges from the explicit one over time, the prompt or model needs adjustment.

This hybrid approach gives 100% coverage while maintaining a ground-truth anchor.

### Customer Memory and Conversation History

When `identify_customer` verifies a returning customer, the system should query past conversations for that contact and inject context into the current session. Raw transcripts are too token-heavy — instead, use an LLM to generate a summary of recent interactions (last 3-5 conversations, issues raised, resolutions, any open follow-ups) and inject that summary into the system prompt as a "customer context" block.

This enables continuity ("I called about this last week") without the customer re-explaining their history. It also lets the agent avoid repeating information — if a customer was already told the return policy on a previous call, the agent can reference that rather than reciting it again.

Privacy matters here: conversation history needs a retention policy (auto-delete after N days), and customers must be able to request deletion of their history. The summary should be regenerated on each visit rather than cached, so deletions take effect immediately.

### Autonomous Follow-Up Actions

The agent should be able to promise and deliver on follow-ups — "I'll email you when your order is delivered" or "I'll send you a summary of what we discussed." Implementation:

- **New tool: `schedule_followup`** — the agent calls it during the conversation with parameters: trigger event (e.g., `order_delivered`), channel (`email` or `sms`), and content context (what to include). The agent decides whether a follow-up is appropriate based on conversation context.
- **Backend: async task queue** (Celery or Bull) with event-driven triggers. For order-status follow-ups, the queue subscribes to order status change events and fires the message when the condition is met. For immediate follow-ups (conversation summary), the task fires on conversation close.
- **Delivery:** Email via SendGrid or SES, SMS via Twilio. Messages use templates with order-specific variables (order ID, tracking number, delivery date) injected from the order data at send time — not at schedule time, so the data is current.

The agent's role is deciding *what* to follow up on and *when*. The actual delivery is handled asynchronously by the task queue — no LLM involved at send time.

### Multi-Provider LLM Support

To avoid vendor lock-in and enable cost optimization, abstract the LLM call behind a provider interface. Each provider adapter implements a common contract: `create_message(messages, tools, system) -> Response`, handling the translation between the internal schema and the provider's API format.

The main complexity is tool definitions — Anthropic, OpenAI, and Google each use different schemas for tool/function calling. The adapter layer translates the canonical tool definitions (stored once) into each provider's format at call time. Response parsing also differs: extracting tool calls, text content, and usage stats from each provider's response shape.

Configuration would be per-tenant or per-conversation-type: use a cheaper, faster model (e.g., Haiku, GPT-4o-mini) for simple FAQ lookups, and a stronger model for complex disputes or multi-step resolutions. This also allows tenants with data residency requirements to select providers that operate in specific regions.

### Embeddable UI Widget

To make this accessible to non-technical customers, package the agent as an embeddable chat widget:

- **Backend:** Wrap the agent in a FastAPI (or Express) API with WebSocket support for streaming responses. Each WebSocket connection maps to a conversation session. The backend manages message history, tool execution, and conversation lifecycle — the frontend is stateless.
- **Frontend widget:** A lightweight JavaScript bundle served from a CDN. Customers embed it with a single `<script>` tag and a tenant API key. The widget renders a chat bubble in the corner of the page, opens a conversation panel on click, and communicates with the backend over WebSocket.
- **Tenant configuration:** Each tenant configures branding (colors, logo, greeting message), which tools are available, which knowledge base to use, and behavioral rules (e.g., always escalate after 5 minutes). Configuration is loaded from the backend on widget initialization.
- **Authentication:** The widget generates a session token scoped to the tenant's API key, passed with every message. Customer identity is established during the conversation (email/phone), not at widget load time.
- **Deployment:** The backend runs as a managed service with horizontal scaling (stateless workers + shared DB). The widget JS is versioned and served from a CDN so updates propagate without customers changing their embed code.
