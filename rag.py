"""
rag.py – Local knowledge base retrieval for AutoStream agent.
Loads the JSON knowledge base and provides a simple keyword-based
retrieval function that returns the most relevant context chunks.
"""

import json
import os
from typing import List

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")


def _load_kb() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _kb_to_chunks(kb: dict) -> List[str]:
    """Flatten the knowledge base into searchable text chunks."""
    chunks = []

    # Product overview
    chunks.append(f"AutoStream Overview: {kb['description']}")

    # Plans
    for plan in kb["plans"]:
        features_str = ", ".join(plan["features"])
        chunks.append(
            f"{plan['name']}: Costs {plan['price']}. Features include: {features_str}."
        )

    # Policies
    for policy in kb["policies"]:
        chunks.append(f"{policy['topic']}: {policy['detail']}")

    return chunks


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Simple keyword-overlap retrieval.
    Returns the top_k most relevant chunks joined as a single context string.
    """
    kb = _load_kb()
    chunks = _kb_to_chunks(kb)

    query_tokens = set(query.lower().split())

    scored: List[tuple[int, str]] = []
    for chunk in chunks:
        chunk_tokens = set(chunk.lower().split())
        overlap = len(query_tokens & chunk_tokens)
        scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k] if _ > 0]

    if not top_chunks:
        # Fall back: return all chunks so the agent always has context
        top_chunks = [chunk for _, chunk in scored[:top_k]]

    return "\n\n".join(top_chunks)
