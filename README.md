# Bookly AI Support Agent

Customer support agent for Bookly (fictional online bookstore), built directly on the Anthropic Claude API. No agentic frameworks — all orchestration is handled in code.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):
```
ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
python agent.py
```

On first run, the agent generates 1000 mock orders (`data/orders.json`) and builds the knowledge base index (`data/kb_index.faiss`). Subsequent runs load from disk.

## Files

| File | Role |
|---|---|
| `agent.py` | Conversation loop, API calls, retry logic, completion reporting |
| `tools.py` | Tool definitions + execution functions (order lookup, refund, escalation) |
| `order_data.py` | Seeded order generation, JSON persistence, mutable state |
| `knowledge_base.py` | FAISS vector index, embedding, semantic KB search |
| `storage.py` | SQLite: conversations, messages, citations, escalations |
| `analysis.py` | LLM-based sentiment analysis + conversation classification |
| `validation.py` | Post-response output validation (regex + LLM-as-judge) |
| `design_doc.md` | Design document (deliverable) |
| `data/articles/` | Knowledge base articles (markdown) |
