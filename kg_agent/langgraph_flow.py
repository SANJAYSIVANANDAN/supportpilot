"""OntoDesk's orchestration: a LangGraph `StateGraph` wired exactly to the
hybrid retrieval + LLM verification pipeline described in ARCHITECTURE.md.

This is the ONLY stateful orchestration in the project - the MCP server,
the dashboard, the manual smoke test, and the eval harness all call
`handle_query` here, never hand-roll branching of their own.

Graph shape:

    extract -> retrieve -> [no candidates]-------------------> create_issue
                  |
                  `-> [candidates] -> verify -> SAME_ISSUE ---> reuse
                                           |--> NEEDS_CLARIFICATION -> clarify
                                           `--> RELATED_BUT_DIFFERENT
                                                or NEW_ISSUE ----------> create_issue

    reuse / clarify / create_issue -> END

Each node reads/writes a shared `OntoDeskState` TypedDict - that shared,
explicitly-typed state (not function arguments threaded by hand) is what
"stateful orchestration" means here. Every node is a thin wrapper around
the pure/IO functions already in `llm.py`, `graph_store.py`, and
`ranking.py`; this module adds no business logic of its own, only the
graph topology and the state shape that connects them.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from kg_agent import graph_store, llm, ranking
from kg_agent.config import settings


class OntoDeskState(TypedDict, total=False):
    query: str
    extraction: llm.ExtractionResult
    embedding: list[float]
    semantic_candidates: list[dict[str, Any]]
    entity_candidates: list[dict[str, Any]]
    fused_candidates: list[dict[str, Any]]
    verification: llm.VerificationResult | None
    result: dict[str, Any]


def _base_fields(state: OntoDeskState) -> dict[str, Any]:
    extraction = state["extraction"]
    return {
        "entities": extraction.entities,
        "category": extraction.category,
        "problem_summary": extraction.problem_summary,
        "semantic_candidates": state.get("semantic_candidates", []),
        "entity_candidates": state.get("entity_candidates", []),
        "fused_candidates": state.get("fused_candidates", []),
    }


def extract_node(state: OntoDeskState) -> dict[str, Any]:
    return {"extraction": llm.extract(state["query"])}


def retrieve_node(state: OntoDeskState) -> dict[str, Any]:
    extraction = state["extraction"]
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

    return {
        "embedding": embedding,
        "semantic_candidates": semantic_candidates,
        "entity_candidates": entity_candidates,
        "fused_candidates": fused,
    }


def route_after_retrieve(state: OntoDeskState) -> str:
    return "verify" if state["fused_candidates"] else "create_issue"


def verify_node(state: OntoDeskState) -> dict[str, Any]:
    extraction = state["extraction"]
    fused = state["fused_candidates"]
    return {"verification": llm.verify_issue_match(extraction, fused)}


def route_after_verify(state: OntoDeskState) -> str:
    decision = state["verification"].decision
    if decision == llm.VerificationDecision.SAME_ISSUE:
        return "reuse"
    if decision == llm.VerificationDecision.NEEDS_CLARIFICATION:
        return "clarify"
    return "create_issue"  # RELATED_BUT_DIFFERENT or NEW_ISSUE


def reuse_node(state: OntoDeskState) -> dict[str, Any]:
    verification = state["verification"]
    fused = state["fused_candidates"]
    matched = next(c for c in fused if c["id"] == verification.matched_issue_id)
    graph_store.bump_seen(matched["id"])

    result = {
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
        **_base_fields(state),
    }
    return {"result": result}


def clarify_node(state: OntoDeskState) -> dict[str, Any]:
    verification = state["verification"]
    fused = state["fused_candidates"]
    top = fused[0] if fused else {}

    result = {
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
        **_base_fields(state),
    }
    return {"result": result}


def create_issue_node(state: OntoDeskState) -> dict[str, Any]:
    extraction = state["extraction"]
    verification = state.get("verification")
    fused = state.get("fused_candidates") or []
    top = fused[0] if fused else {}

    # The embedding was already computed in retrieve_node for the same,
    # unchanged problem_summary - reuse it instead of paying for it twice.
    embedding = state["embedding"]
    structured = llm.resolve(state["query"], extraction.entities, extraction.problem_summary)
    resolution_text = llm.render_resolution_text(structured)
    issue_id = graph_store.create_issue(
        query=state["query"],
        entities=extraction.entities,
        category=extraction.category,
        problem_summary=extraction.problem_summary,
        embedding=embedding,
        resolution_text=resolution_text,
        resolution_structured=structured.model_dump(),
    )

    result = {
        "answer": resolution_text,
        "is_novel": True,
        "graph_updated": True,
        "matched_issue_id": issue_id,
        "semantic_score": top.get("semantic_score"),
        "entity_score": top.get("entity_score"),
        "hybrid_rank": top.get("hybrid_rank"),
        "verification_decision": verification.decision.value if verification else None,
        "verification_reasoning": verification.reasoning if verification else None,
        "times_seen": 1,
        **_base_fields(state),
    }
    return {"result": result}


def build_graph():
    graph = StateGraph(OntoDeskState)

    graph.add_node("extract", extract_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify", verify_node)
    graph.add_node("reuse", reuse_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("create_issue", create_issue_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"verify": "verify", "create_issue": "create_issue"}
    )
    graph.add_conditional_edges(
        "verify", route_after_verify, {"reuse": "reuse", "clarify": "clarify", "create_issue": "create_issue"}
    )
    graph.add_edge("reuse", END)
    graph.add_edge("clarify", END)
    graph.add_edge("create_issue", END)

    return graph.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def handle_query(query: str) -> dict[str, Any]:
    """Runs one query through the compiled LangGraph and returns the same
    result shape OntoDesk's MCP tools and dashboard expect: answer,
    is_novel, graph_updated, matched_issue_id, semantic_score, entity_score,
    hybrid_rank, verification_decision, verification_reasoning, times_seen,
    entities, category, problem_summary, semantic_candidates,
    entity_candidates, fused_candidates."""
    final_state = get_compiled_graph().invoke({"query": query})
    return final_state["result"]
