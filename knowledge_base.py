import os
import re
import json
import logging
import warnings
import numpy as np
import faiss

# Suppress HuggingFace/transformers noise before importing sentence_transformers
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _logger in ("sentence_transformers", "huggingface_hub", "transformers", "torch"):
    logging.getLogger(_logger).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

KB_DIR = "data/articles"
INDEX_PATH = "data/kb_index.faiss"
CHUNKS_PATH = "data/kb_chunks.json"

# Cosine similarity threshold — results below this score are considered not relevant enough.
# Range is 0 to 1. 0.4 is a reasonable starting point for this model.
SIMILARITY_THRESHOLD = 0.4

_model = None
_index = None
_chunks = None  # list of dicts: {"text": ..., "source": ..., "section": ...}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _article_title(filename: str) -> str:
    """Convert a filename like 'shipping_policy.md' to 'Shipping Policy'."""
    return filename.replace(".md", "").replace("_", " ").title()


def _chunk_articles(directory: str) -> list[dict]:
    """
    Read all .md files in the articles directory.
    Split each file into paragraph-level chunks.
    Each chunk is stored as a dict with the text, source filename, and section heading
    so results can be traced back to a specific article and section.
    """
    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(directory, filename)
        article_title = _article_title(filename)
        current_section = article_title  # default to article title if no section heading found

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for block in re.split(r'\n\n+', content):
            block = block.strip()
            if not block:
                continue
            if block.startswith('#'):
                # Extract section heading — skip the top-level article title (single #)
                heading_text = block.split('\n')[0].lstrip('#').strip()
                if block.startswith('##'):
                    current_section = heading_text
                # If there's body text after the heading, chunk it
                body = '\n'.join(block.split('\n')[1:]).strip()
                if body:
                    chunks.append({
                        "text": f"{current_section}: {body}",
                        "source": filename,
                        "section": current_section,
                        "article": article_title,
                    })
            else:
                chunks.append({
                    "text": f"{current_section}: {block}",
                    "source": filename,
                    "section": current_section,
                    "article": article_title,
                })

    return chunks


def _build_index() -> tuple:
    """
    Embed all KB chunks and build a FAISS index.
    Saves both to disk so we don't re-embed on every startup.
    """
    chunks = _chunk_articles(KB_DIR)
    model = _get_model()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Knowledge base built: {len(chunks)} chunks indexed from {KB_DIR}.")
    return index, chunks


def init_knowledge_base():
    """Load the FAISS index from disk if it exists, otherwise build it."""
    global _index, _chunks
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            _chunks = json.load(f)
    else:
        _index, _chunks = _build_index()


def search_knowledge_base(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """
    Search the KB and return (formatted_text, citations).

    formatted_text — sent to the LLM, includes [Source: ...] tags
    citations      — list of dicts with article, section, score for storage

    Returns a no-match string and empty citations if nothing exceeds the threshold.
    """
    if _index is None or _chunks is None:
        return "Knowledge base not initialized.", []

    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = _index.search(query_embedding, top_k)

    results = []
    citations = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= SIMILARITY_THRESHOLD:
            chunk = _chunks[idx]
            results.append(
                f"[Source: {chunk['article']} — {chunk['section']}]\n{chunk['text']}"
            )
            citations.append({
                "article": chunk["article"],
                "section": chunk["section"],
                "similarity_score": float(score),
            })

    if not results:
        return "No relevant information found in the knowledge base for this query.", []

    return "\n\n".join(results), citations
