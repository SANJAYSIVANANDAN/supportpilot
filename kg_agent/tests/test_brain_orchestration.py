"""Unit tests for brain.handle_query's branching logic. Every call that
would hit Neo4j/OpenAI is monkeypatched to a fake, so these run with no
external services and should always pass."""
from __future__ import annotations

import pytest

from kg_agent import brain, graph_store, llm


def _extraction(**overrides):
    defaults = dict(
        entities=["billing", "duplicate_charge"],
        category="billing",
        problem_summary="Customer was charged twice for the same subscription.",
        intent="report_problem",
        context=[],
    )
    defaults.update(overrides)
    return llm.ExtractionResult(**defaults)


def _candidate(id_="issue-1", score=0.9, resolution="Refund the duplicate charge."):
    return {
        "id": id_,
        "text": "I was double charged",
        "problem_summary": "Customer charged twice.",
        "category": "billing",
        "entities": ["billing", "duplicate_charge"],
        "resolution": resolution,
        "times_seen": 1,
        "score": score,
    }


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch):
    monkeypatch.setattr(graph_store, "generate_issue_embedding", lambda text: [0.1, 0.2, 0.3])


def test_same_issue_reuses_without_writing_new_node(monkeypatch):
    extraction = _extraction()
    candidate = _candidate()

    monkeypatch.setattr(llm, "extract", lambda query: extraction)
    monkeypatch.setattr(graph_store, "find_semantically_similar_issues", lambda *a, **k: [candidate])
    monkeypatch.setattr(graph_store, "find_entity_similar_issues", lambda *a, **k: [candidate])

    bump_calls = []
    create_calls = []
    monkeypatch.setattr(graph_store, "bump_seen", lambda issue_id: bump_calls.append(issue_id))
    monkeypatch.setattr(graph_store, "create_issue", lambda **k: create_calls.append(k) or "should-not-be-called")
    monkeypatch.setattr(
        llm,
        "verify_issue_match",
        lambda incoming, candidates: llm.VerificationResult(
            decision=llm.VerificationDecision.SAME_ISSUE,
            matched_issue_id=candidate["id"],
            reasoning="Same underlying problem.",
            confidence=0.9,
        ),
    )

    result = brain.handle_query("I got billed twice")

    assert result["graph_updated"] is False
    assert result["is_novel"] is False
    assert result["matched_issue_id"] == candidate["id"]
    assert result["answer"] == candidate["resolution"]
    assert result["times_seen"] == candidate["times_seen"] + 1
    assert bump_calls == [candidate["id"]]
    assert create_calls == []


def test_new_issue_creates_node_when_no_candidates(monkeypatch):
    extraction = _extraction()
    monkeypatch.setattr(llm, "extract", lambda query: extraction)
    monkeypatch.setattr(graph_store, "find_semantically_similar_issues", lambda *a, **k: [])
    monkeypatch.setattr(graph_store, "find_entity_similar_issues", lambda *a, **k: [])

    monkeypatch.setattr(
        llm,
        "resolve",
        lambda query, entities, problem_summary: llm.StructuredResolution(
            problem="p", diagnosis="d", resolution_steps=["step"], escalation_required=False, limitations="none"
        ),
    )
    create_calls = []
    monkeypatch.setattr(graph_store, "create_issue", lambda **k: create_calls.append(k) or "new-issue-id")

    verify_calls = []
    monkeypatch.setattr(
        llm,
        "verify_issue_match",
        lambda incoming, candidates: verify_calls.append(candidates) or llm.VerificationResult(
            decision=llm.VerificationDecision.NEW_ISSUE, reasoning="no candidates", confidence=1.0
        ),
    )

    result = brain.handle_query("The analytics tab crashes")

    assert result["graph_updated"] is True
    assert result["is_novel"] is True
    assert result["matched_issue_id"] == "new-issue-id"
    assert len(create_calls) == 1
    # No candidates existed, so verification should never even be called.
    assert verify_calls == []


def test_related_but_different_creates_new_node(monkeypatch):
    extraction = _extraction()
    candidate = _candidate(id_="issue-unrelated", resolution="Unrelated resolution")
    monkeypatch.setattr(llm, "extract", lambda query: extraction)
    monkeypatch.setattr(graph_store, "find_semantically_similar_issues", lambda *a, **k: [candidate])
    monkeypatch.setattr(graph_store, "find_entity_similar_issues", lambda *a, **k: [])
    monkeypatch.setattr(
        llm,
        "resolve",
        lambda query, entities, problem_summary: llm.StructuredResolution(
            problem="p", diagnosis="d", resolution_steps=["step"], escalation_required=False, limitations="none"
        ),
    )
    create_calls = []
    monkeypatch.setattr(graph_store, "create_issue", lambda **k: create_calls.append(k) or "new-issue-id-2")
    monkeypatch.setattr(graph_store, "bump_seen", lambda issue_id: pytest.fail("must not bump_seen"))
    monkeypatch.setattr(
        llm,
        "verify_issue_match",
        lambda incoming, candidates: llm.VerificationResult(
            decision=llm.VerificationDecision.RELATED_BUT_DIFFERENT,
            reasoning="Shares entities but different problem.",
            confidence=0.7,
        ),
    )

    result = brain.handle_query("Something related but different")

    assert result["graph_updated"] is True
    assert result["is_novel"] is True
    assert len(create_calls) == 1
    assert result["verification_decision"] == "RELATED_BUT_DIFFERENT"


def test_needs_clarification_does_not_touch_graph(monkeypatch):
    extraction = _extraction()
    candidate = _candidate()
    monkeypatch.setattr(llm, "extract", lambda query: extraction)
    monkeypatch.setattr(graph_store, "find_semantically_similar_issues", lambda *a, **k: [candidate])
    monkeypatch.setattr(graph_store, "find_entity_similar_issues", lambda *a, **k: [])
    monkeypatch.setattr(graph_store, "bump_seen", lambda issue_id: pytest.fail("must not bump_seen"))
    monkeypatch.setattr(graph_store, "create_issue", lambda **k: pytest.fail("must not create_issue"))
    monkeypatch.setattr(
        llm,
        "verify_issue_match",
        lambda incoming, candidates: llm.VerificationResult(
            decision=llm.VerificationDecision.NEEDS_CLARIFICATION,
            reasoning="Missing key detail about which plan.",
            confidence=0.3,
        ),
    )

    result = brain.handle_query("I was charged weirdly")

    assert result["graph_updated"] is False
    assert result["is_novel"] is False
    assert result["matched_issue_id"] is None
    assert "Missing key detail" in result["answer"]


def test_verification_cannot_select_id_outside_candidates(monkeypatch):
    """llm.verify_issue_match itself guards against a hallucinated id -
    this test locks that behavior in."""
    extraction = _extraction()
    candidate = _candidate(id_="real-candidate")

    class _FakeParsed:
        def __init__(self, parsed):
            self.message = type("M", (), {"parsed": parsed})()

    class _FakeResponse:
        def __init__(self, parsed):
            self.choices = [_FakeParsed(parsed)]

    hallucinated = llm.VerificationResult(
        decision=llm.VerificationDecision.SAME_ISSUE,
        matched_issue_id="made-up-id-not-in-candidates",
        reasoning="hallucinated",
        confidence=0.99,
    )

    class _FakeCompletions:
        def parse(self, **kwargs):
            return _FakeResponse(hallucinated)

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(llm, "get_client", lambda: _FakeClient())

    result = llm.verify_issue_match(extraction, [candidate])

    assert result.decision == llm.VerificationDecision.NEEDS_CLARIFICATION
    assert result.matched_issue_id is None
