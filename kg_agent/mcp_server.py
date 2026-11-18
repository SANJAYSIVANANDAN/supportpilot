"""Exposes OntoDesk as a private MCP server: a client sends a support query
in and gets back an answer plus whether that query taught the knowledge
graph anything new. Run over stdio, so it's only reachable by whatever MCP
client is configured to spawn this process (e.g. Claude Desktop) — it is not
network-exposed.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from kg_agent import graph_store, langgraph_flow

mcp = FastMCP("ontodesk")


@mcp.tool()
def ask_support(query: str) -> dict:
    """Answers a customer support query using OntoDesk's Neo4j knowledge
    graph. Retrieves candidates both semantically (embedding similarity) and
    by entity overlap, fuses them with Reciprocal Rank Fusion, and asks an
    LLM to verify whether a candidate is really the same underlying problem.
    A confirmed match reuses the stored resolution without modifying the
    graph; an unclear match returns a clarification request instead of
    guessing; a genuinely new problem gets a generated resolution and a new
    Issue subgraph - so the agent only gets smarter on unique cases. Returns
    answer, is_novel, graph_updated, matched_issue_id, semantic_score,
    entity_score, hybrid_rank, verification_decision, verification_reasoning,
    times_seen, entities, category, problem_summary."""
    return langgraph_flow.handle_query(query)


@mcp.tool()
def graph_stats() -> dict:
    """Returns current knowledge graph size: number of distinct issue kinds,
    entities, categories, resolutions, and total queries served overall."""
    return graph_store.get_stats()


def main() -> None:
    graph_store.init_constraints()
    graph_store.create_vector_index()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
