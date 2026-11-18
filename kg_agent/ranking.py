"""Reciprocal Rank Fusion for combining semantic and entity retrieval.

Pure functions, no Neo4j/OpenAI/Streamlit imports, so they're trivially
unit-testable (see kg_agent/tests/test_ranking.py) and reusable from both
`brain.py` and `eval/run_eval.py`.
"""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int) -> dict[str, float]:
    """RRF(d) = sum over lists containing d of 1 / (k + rank(d)), rank 1-based."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_candidates(
    semantic_candidates: list[dict[str, Any]],
    entity_candidates: list[dict[str, Any]],
    k: int,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Dedupes semantic + entity candidate lists by issue id, fuses their
    ranks with RRF, and returns the top `top_n` merged candidates with both
    original scores preserved for debugging/evaluation."""
    semantic_ranked = [c["id"] for c in semantic_candidates]
    entity_ranked = [c["id"] for c in entity_candidates]
    rrf_scores = reciprocal_rank_fusion([semantic_ranked, entity_ranked], k)

    by_id: dict[str, dict[str, Any]] = {}
    for c in semantic_candidates:
        by_id[c["id"]] = {**c, "semantic_score": c.get("score"), "entity_score": None}
    for c in entity_candidates:
        existing = by_id.get(c["id"])
        if existing is None:
            by_id[c["id"]] = {**c, "semantic_score": None, "entity_score": c.get("score")}
        else:
            existing["entity_score"] = c.get("score")

    fused = []
    for doc_id, merged in by_id.items():
        merged = dict(merged)
        merged.pop("score", None)
        merged["rrf_score"] = rrf_scores.get(doc_id, 0.0)
        fused.append(merged)

    fused.sort(key=lambda c: c["rrf_score"], reverse=True)
    for rank, c in enumerate(fused, start=1):
        c["hybrid_rank"] = rank
    return fused[:top_n]
