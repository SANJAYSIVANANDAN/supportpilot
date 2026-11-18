"""Provider-agnostic embedding generation.

Kept separate from `llm.py` (which handles chat-completion calls) so
swapping embedding providers later is a one-function change, not a change
to every caller. `graph_store.generate_issue_embedding` is the only caller
today.
"""
from __future__ import annotations

from openai import OpenAI

from kg_agent.config import settings

_client = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def generate_embedding(text: str) -> list[float]:
    """Returns an embedding vector for `text` using the configured provider."""
    if settings.embedding_provider == "openai":
        response = _get_openai_client().embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )
        return response.data[0].embedding
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
