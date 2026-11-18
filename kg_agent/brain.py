"""Single orchestration entrypoint shared by the MCP server and the dashboard.

Hybrid retrieval + LLM verification pipeline:
  1. extract structured info from the query (llm.extract)
  2. embed its problem_summary (graph_store.generate_issue_embedding)
  3. retrieve candidates two ways: semantic (vector index) and entity
     (Jaccard overlap), both pure reads
  4. fuse them with Reciprocal Rank Fusion (ranking.fuse_candidates)
  5. if any candidates clear the pre-filter floor, ask the LLM to verify
     whether one of them is really the same underlying problem
  6. branch on that verification: reuse, create new, or ask for clarification

The novelty gate is still enforced by graph_store (only create_issue writes
new nodes, only on the create-new branch below) - but *which* branch is
taken is now decided by explicit LLM verification over ranked evidence,
not a bare similarity threshold.
"""
from __future__ import annotations

from kg_agent import graph_store, llm, ranking
from kg_agent.config import settings


def _create_new_issue(query: str, extraction: llm.ExtractionResult) -> dict:
    embedding = graph_store.generate_issue_embedding(extraction.problem_summary)
    structured = llm.resolve(query, extraction.entities, extraction.problem_summary)
    resolution_text = llm.render_resolution_text(structured)
    issue_id = graph_store.create_issue(
        query=query,
        entities=extraction.entities,
        category=extraction.category,
        problem_summary=extraction.problem_summary,
        embedding=embedding,
        resolution_text=resolution_text,
        resolution_structured=structured.model_dump(),
    )
    return {
        "answer": resolution_text,
        "is_novel": True,
        "graph_updated": True,
        "matched_issue_id": issue_id,
        "semantic_score": None,
        "entity_score": None,
        "hybrid_rank": None,
        "verification_decision": None,
        "verification_reasoning": None,
        "times_seen": 1,
        "entities": extraction.entities,
        "category": extraction.category,
        "problem_summary": extraction.problem_summary,
    }


def handle_query(query: str) -> dict:
    extraction = llm.extract(query)

    embedding = graph_store.generate_issue_embedding(extraction.problem_summary)
    semantic_candidates = graph_store.find_semantically_similar_issues(
        embedding, extraction.category, settings.semantic_top_k
    )
    entity_candidates = graph_store.find_entity_similar_issues(
        extraction.entities, extraction.category, settings.entity_top_k
    )

    fused = ranking.fuse_candidates(semantic_candidates, entity_candidates, settings.rrf_k, top_n=5)
    fused = [
        c
        for c in fused
        if (c.get("semantic_score") or 0) >= settings.min_candidate_score
        or (c.get("entity_score") or 0) >= settings.min_candidate_score
    ]
    debug = {
        "semantic_candidates": semantic_candidates,
        "entity_candidates": entity_candidates,
        "fused_candidates": fused,
    }

    if not fused:
        # Nothing worth verifying against - straight to novel, no wasted LLM call.
        return {**_create_new_issue(query, extraction), **debug}

    verification = llm.verify_issue_match(extraction, fused)
    top = fused[0]

    if verification.decision == llm.VerificationDecision.SAME_ISSUE:
        matched = next(c for c in fused if c["id"] == verification.matched_issue_id)
        graph_store.bump_seen(matched["id"])
        return {
            "answer": matched["resolution"],
            "is_novel": False,
            "graph_updated": False,
            "matched_issue_id": matched["id"],
            "semantic_score": matched.get("semantic_score"),
            "entity_score": matched.get("entity_score"),
            "hybrid_rank": matched.get("hybrid_rank"),
            "verification_decision": verification.decision.value,
            "verification_reasoning": verification.reasoning,
            "times_seen": (matched.get("times_seen") or 0) + 1,
            "entities": extraction.entities,
            "category": extraction.category,
            "problem_summary": extraction.problem_summary,
            **debug,
        }

    if verification.decision == llm.VerificationDecision.NEEDS_CLARIFICATION:
        return {
            "answer": (
                "I need a bit more detail before I can confidently reuse an existing "
                "answer or confirm this is a new issue: " + verification.reasoning
            ),
            "is_novel": False,
            "graph_updated": False,
            "matched_issue_id": None,
            "semantic_score": top.get("semantic_score"),
            "entity_score": top.get("entity_score"),
            "hybrid_rank": top.get("hybrid_rank"),
            "verification_decision": verification.decision.value,
            "verification_reasoning": verification.reasoning,
            "times_seen": None,
            "entities": extraction.entities,
            "category": extraction.category,
            "problem_summary": extraction.problem_summary,
            **debug,
        }

    # RELATED_BUT_DIFFERENT or NEW_ISSUE
    result = _create_new_issue(query, extraction)
    result["verification_decision"] = verification.decision.value
    result["verification_reasoning"] = verification.reasoning
    result["semantic_score"] = top.get("semantic_score")
    result["entity_score"] = top.get("entity_score")
    result["hybrid_rank"] = top.get("hybrid_rank")
    result.update(debug)
    return result
