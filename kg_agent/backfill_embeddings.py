"""Migration: backfills `embedding`/`problem_summary` on Issue nodes created
before the hybrid-retrieval upgrade (or any time the vector index needs
rebuilding). Idempotent - re-running finds nothing left to do once every
Issue has an embedding.

Usage:
    python -m kg_agent.backfill_embeddings
"""
from __future__ import annotations

from kg_agent import graph_store


def run() -> None:
    graph_store.init_constraints()

    pending = graph_store.issues_missing_embedding()
    print(f"Found {len(pending)} issue(s) missing an embedding.")

    for issue in pending:
        problem_summary = issue.get("problem_summary") or issue["text"]
        embedding = graph_store.generate_issue_embedding(problem_summary)
        graph_store.set_issue_embedding(issue["id"], problem_summary, embedding)
        print(f"  backfilled {issue['id']}")

    graph_store.create_vector_index()
    print("Vector index ensured. Stats:", graph_store.get_stats())


if __name__ == "__main__":
    run()
