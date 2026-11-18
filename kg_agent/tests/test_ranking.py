"""Pure unit tests for Reciprocal Rank Fusion - no Neo4j/OpenAI needed."""
from __future__ import annotations

from kg_agent.ranking import fuse_candidates, reciprocal_rank_fusion


def test_rrf_single_list_preserves_order():
    scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_rewards_agreement_across_lists():
    # "a" ranked top in both lists should beat "b" which is only in one.
    scores = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def test_fuse_candidates_dedupes_by_id_and_keeps_both_scores():
    semantic = [{"id": "x1", "score": 0.92, "text": "a"}, {"id": "x2", "score": 0.80, "text": "b"}]
    entity = [{"id": "x2", "score": 0.5, "text": "b"}, {"id": "x3", "score": 0.4, "text": "c"}]

    fused = fuse_candidates(semantic, entity, k=60, top_n=5)
    ids = {c["id"] for c in fused}
    assert ids == {"x1", "x2", "x3"}

    by_id = {c["id"]: c for c in fused}
    assert by_id["x2"]["semantic_score"] == 0.80
    assert by_id["x2"]["entity_score"] == 0.5
    assert by_id["x1"]["entity_score"] is None
    assert by_id["x3"]["semantic_score"] is None


def test_fuse_candidates_ranks_agreement_highest():
    # x2 appears in both lists (agreement), x1/x3 only in one each.
    semantic = [{"id": "x1", "score": 0.95}, {"id": "x2", "score": 0.70}]
    entity = [{"id": "x2", "score": 0.60}, {"id": "x3", "score": 0.50}]

    fused = fuse_candidates(semantic, entity, k=60, top_n=5)
    assert fused[0]["id"] == "x2"
    assert fused[0]["hybrid_rank"] == 1


def test_fuse_candidates_respects_top_n():
    semantic = [{"id": f"s{i}", "score": 1.0 - i * 0.1} for i in range(5)]
    entity: list[dict] = []
    fused = fuse_candidates(semantic, entity, k=60, top_n=2)
    assert len(fused) == 2


def test_fuse_candidates_handles_empty_inputs():
    assert fuse_candidates([], [], k=60, top_n=5) == []
