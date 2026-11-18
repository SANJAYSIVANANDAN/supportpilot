"""Loads kg_agent/.env (falls back to the repo-root .env) and exposes settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "ontodesk123")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Legacy Jaccard gate, repurposed as a cheap pre-filter floor on fused
    # candidate scores before anything is sent to LLM verification.
    novelty_threshold: float = float(os.getenv("NOVELTY_THRESHOLD", "0.6"))

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "issue_embedding_index")

    # Reciprocal Rank Fusion smoothing constant and retrieval widths.
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    semantic_top_k: int = int(os.getenv("SEMANTIC_TOP_K", "5"))
    entity_top_k: int = int(os.getenv("ENTITY_TOP_K", "5"))

    # Fused candidates scoring below this (on either raw score) aren't worth
    # spending an LLM verification call on. Calibrate with kg_agent/eval/run_eval.py.
    min_candidate_score: float = float(os.getenv("MIN_CANDIDATE_SCORE", "0.15"))


settings = Settings()
