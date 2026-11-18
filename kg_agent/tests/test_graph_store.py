"""Unit tests for graph_store's pure logic, plus optional integration tests
against a real local Neo4j (skipped automatically if it's not reachable)."""
from __future__ import annotations

import pytest

from kg_agent import graph_store


def test_jaccard_basic():
    assert graph_store._jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert graph_store._jaccard({"a"}, {"b"}) == 0.0
    assert graph_store._jaccard(set(), set()) == 0.0


def test_jaccard_partial_overlap():
    score = graph_store._jaccard({"a", "b", "c"}, {"b", "c", "d"})
    assert score == pytest.approx(2 / 4)


def _neo4j_reachable() -> bool:
    try:
        graph_store.get_driver().verify_connectivity()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _neo4j_reachable(), reason="local Neo4j not reachable")
class TestGraphStoreIntegration:
    def test_stats_shape(self):
        stats = graph_store.get_stats()
        assert set(stats) == {"issues", "entities", "categories", "resolutions", "total_queries_served"}

    def test_create_vector_index_is_idempotent(self):
        graph_store.create_vector_index()
        graph_store.create_vector_index()  # must not raise on second call
