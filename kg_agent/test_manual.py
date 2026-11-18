"""Quick smoke test of the hybrid retrieval + verification pipeline, no MCP
client required.

Run after `seed.py`:
    python -m kg_agent.test_manual
Expect: query 1 reuses a seeded issue (SAME_ISSUE, graph_updated=False),
query 2 is novel (NEW_ISSUE, graph_updated=True), and re-asking query 2
verbatim then reuses it (SAME_ISSUE).
"""
from __future__ import annotations

import json

from kg_agent import graph_store, langgraph_flow

SAMPLE_QUERIES = [
    "I got billed two times this month for the same plan",  # should match seeded billing issue
    "The mobile app crashes whenever I open the analytics tab",  # novel
    "The mobile app crashes whenever I open the analytics tab",  # repeat of the above -> should reuse
]


def run() -> None:
    before = graph_store.get_stats()
    print("Before:", before)

    for query in SAMPLE_QUERIES:
        result = langgraph_flow.handle_query(query)
        print(f"\nQuery: {query}")
        print(json.dumps({k: v for k, v in result.items() if k != "answer"}, indent=2))
        print("Answer:", str(result["answer"])[:200], "...")

    after = graph_store.get_stats()
    print("\nAfter:", after)


if __name__ == "__main__":
    run()
