"""OntoDesk demo dashboard: ask a question, watch the knowledge graph only
grow when the question is genuinely new, and inspect the hybrid retrieval
(semantic + entity) and LLM verification behind each answer."""
from __future__ import annotations

import streamlit as st

from kg_agent import brain, graph_store

st.set_page_config(page_title="OntoDesk", page_icon="🧠", layout="centered")

graph_store.init_constraints()
graph_store.create_vector_index()

st.title("🧠 OntoDesk")
st.caption("A support agent whose brain is a Neo4j ontology — it only learns from unique, LLM-verified cases.")

with st.sidebar:
    st.subheader("Knowledge graph")
    stats = graph_store.get_stats()
    st.metric("Issue kinds", stats["issues"])
    st.metric("Entities", stats["entities"])
    st.metric("Categories", stats["categories"])
    st.metric("Queries served", stats["total_queries_served"])
    st.caption("Issue kinds only increase on a verified-novel case.")

tab_ask, = st.tabs(["Ask"])

DECISION_BADGES = {
    "SAME_ISSUE": ("info", "♻️ Known case — reused existing answer"),
    "RELATED_BUT_DIFFERENT": ("success", "🆕 Related but different — new issue created"),
    "NEW_ISSUE": ("success", "🆕 Novel case — knowledge graph updated"),
    "NEEDS_CLARIFICATION": ("warning", "❓ Needs clarification — graph not updated"),
}


def _candidate_table(candidates: list[dict], score_key: str) -> list[dict]:
    return [
        {
            "issue_id": c["id"][:8],
            "problem_summary": c.get("problem_summary") or c.get("text"),
            score_key: c.get("score"),
        }
        for c in candidates
    ]


with tab_ask:
    query = st.text_area("Ask a support question", placeholder="e.g. My invoice shows the wrong tax amount")
    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Thinking..."):
            result = brain.handle_query(query.strip())

        st.session_state["last_result"] = result
        st.rerun()

    result = st.session_state.get("last_result")
    if result is not None:
        decision = result.get("verification_decision")
        if decision is None:
            # No candidates existed at all - straight-to-novel path.
            level, label = "success", "🆕 Novel case — knowledge graph updated"
        else:
            level, label = DECISION_BADGES.get(decision, ("info", decision))
        getattr(st, level)(label)

        st.markdown("**Answer**")
        st.write(result["answer"])

        cols = st.columns(2)
        cols[0].caption(f"Category: `{result['category']}`")
        cols[1].caption(f"Entities: {', '.join(result['entities'])}")
        st.caption(f"Problem summary: {result.get('problem_summary')}")
        st.caption(f"Issue id: `{result['matched_issue_id']}`")

        with st.expander("Hybrid retrieval & verification detail"):
            st.write(
                f"Semantic score: `{result.get('semantic_score')}` · "
                f"Entity score: `{result.get('entity_score')}` · "
                f"Hybrid rank: `{result.get('hybrid_rank')}`"
            )
            if decision:
                st.write(f"Verification decision: **{decision}**")
                st.caption(result.get("verification_reasoning"))

            sem_candidates = result.get("semantic_candidates") or []
            ent_candidates = result.get("entity_candidates") or []
            st.markdown("Semantic candidates (embedding cosine similarity):")
            st.table(_candidate_table(sem_candidates, "semantic_score") or [{"info": "none"}])
            st.markdown("Entity candidates (Jaccard overlap):")
            st.table(_candidate_table(ent_candidates, "entity_score") or [{"info": "none"}])
