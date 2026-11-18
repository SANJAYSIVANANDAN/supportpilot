"""OntoDesk's knowledge graph: the agent's ontology lives here, not in prose.

Schema:
  (:Issue {id, text, category, problem_summary, embedding,
            created_at, last_seen, times_seen})
    -[:MENTIONS]->(:Entity {name})
    -[:BELONGS_TO]->(:Category {name})
    -[:RESOLVED_BY]->(:Resolution {text, problem, diagnosis,
                                    resolution_steps, escalation_required,
                                    limitations, created_at})

An Issue is a *kind* of problem, not a raw log of every query: repeats of an
existing kind bump times_seen on a confirmed match instead of creating a new
node. Retrieval (find_semantically_similar_issues / find_entity_similar_issues)
only ever ranks candidates - it never writes. The actual reuse-vs-create
decision is made by LLM verification in brain.py; this module just supplies
and stores the candidates it's asked for.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase

from kg_agent import embeddings
from kg_agent.config import settings

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def init_constraints() -> None:
    with get_driver().session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")


def create_vector_index() -> None:
    """Creates the native vector index backing semantic retrieval. Safe to
    call repeatedly (IF NOT EXISTS) - called at startup and by the backfill
    script."""
    cypher = f"""
    CREATE VECTOR INDEX {settings.vector_index_name} IF NOT EXISTS
    FOR (i:Issue) ON (i.embedding)
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: $dims,
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    with get_driver().session() as session:
        session.run(cypher, dims=settings.embedding_dimensions)


def generate_issue_embedding(text: str) -> list[float]:
    """Thin wrapper so callers import graph_store, not the embedding
    provider directly; the actual provider call lives in embeddings.py."""
    return embeddings.generate_embedding(text)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def find_entity_similar_issues(entities: list[str], category: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Ranks existing Issues in the same category by Jaccard overlap between
    their entity set and the query's. Pure retrieval - never writes."""
    query_entities = {e.lower().strip() for e in entities if e.strip()}

    cypher = """
    MATCH (i:Issue)-[:BELONGS_TO]->(:Category {name: $category})
    MATCH (i)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (i)-[:RESOLVED_BY]->(r:Resolution)
    RETURN i.id AS id, i.text AS text, i.problem_summary AS problem_summary,
           i.category AS category, i.times_seen AS times_seen,
           collect(DISTINCT e.name) AS entities, r.text AS resolution
    """
    with get_driver().session() as session:
        records = session.run(cypher, category=category).data()

    scored = []
    for rec in records:
        issue_entities = {e.lower().strip() for e in rec["entities"]}
        score = _jaccard(query_entities, issue_entities)
        scored.append({**rec, "score": round(score, 3)})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def find_semantically_similar_issues(embedding: list[float], category: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Queries the native vector index for the nearest Issues by cosine
    similarity, then filters to the matching category in Python (portable
    across Neo4j versions rather than relying on native metadata
    pre-filtering). Issues without an embedding (not yet backfilled) are
    simply absent from the vector index and so never appear here."""
    cypher = f"""
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node AS i, score
    OPTIONAL MATCH (i)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH (i)-[:RESOLVED_BY]->(r:Resolution)
    RETURN i.id AS id, i.text AS text, i.problem_summary AS problem_summary,
           i.category AS category, i.times_seen AS times_seen,
           collect(DISTINCT e.name) AS entities, r.text AS resolution, score
    """
    overfetch = max(top_k * 4, top_k)
    try:
        with get_driver().session() as session:
            records = session.run(
                cypher, index_name=settings.vector_index_name, k=overfetch, embedding=embedding
            ).data()
    except Exception:
        # Index not created yet (e.g. before the first create_vector_index()
        # call, or a Neo4j version/edition without vector index support).
        return []

    matching = [{**rec, "score": round(rec.pop("score"), 3)} for rec in records if rec["category"] == category]
    matching.sort(key=lambda r: r["score"], reverse=True)
    return matching[:top_k]


def bump_seen(issue_id: str) -> None:
    cypher = """
    MATCH (i:Issue {id: $id})
    SET i.times_seen = i.times_seen + 1, i.last_seen = $now
    """
    with get_driver().session() as session:
        session.execute_write(lambda tx: tx.run(cypher, id=issue_id, now=datetime.now(timezone.utc).isoformat()))


def create_issue(
    query: str,
    entities: list[str],
    category: str,
    problem_summary: str,
    embedding: list[float],
    resolution_text: str,
    resolution_structured: dict[str, Any] | None = None,
) -> str:
    """Writes a brand-new Issue subgraph. Only called for genuinely novel
    (or related-but-different) cases, after LLM verification has ruled out
    reuse. `resolution_structured`, if given, should contain problem,
    diagnosis, resolution_steps, escalation_required, limitations - stored
    alongside the always-present `resolution_text` back-compat string."""
    issue_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    resolution_structured = resolution_structured or {}

    cypher = """
    MERGE (c:Category {name: $category})
    CREATE (i:Issue {id: $id, text: $text, category: $category,
                      problem_summary: $problem_summary, embedding: $embedding,
                      created_at: $now, last_seen: $now, times_seen: 1})
    MERGE (i)-[:BELONGS_TO]->(c)
    CREATE (r:Resolution {
      text: $resolution_text, created_at: $now,
      problem: $problem, diagnosis: $diagnosis, resolution_steps: $resolution_steps,
      escalation_required: $escalation_required, limitations: $limitations
    })
    MERGE (i)-[:RESOLVED_BY]->(r)
    WITH i
    UNWIND $entities AS ent
    MERGE (e:Entity {name: ent})
    MERGE (i)-[:MENTIONS]->(e)
    """
    params = dict(
        id=issue_id,
        text=query,
        category=category,
        problem_summary=problem_summary,
        embedding=embedding,
        now=now,
        resolution_text=resolution_text,
        problem=resolution_structured.get("problem"),
        diagnosis=resolution_structured.get("diagnosis"),
        resolution_steps=resolution_structured.get("resolution_steps") or [],
        escalation_required=resolution_structured.get("escalation_required"),
        limitations=resolution_structured.get("limitations"),
        entities=[e.lower().strip() for e in entities if e.strip()],
    )
    with get_driver().session() as session:
        session.execute_write(lambda tx: tx.run(cypher, **params))
    return issue_id


def issues_missing_embedding() -> list[dict[str, Any]]:
    """Used by backfill_embeddings.py: existing Issues (e.g. from before this
    upgrade) that have no embedding yet, so semantic retrieval can't see
    them until backfilled."""
    cypher = """
    MATCH (i:Issue)
    WHERE i.embedding IS NULL
    RETURN i.id AS id, i.text AS text, i.problem_summary AS problem_summary
    """
    with get_driver().session() as session:
        return session.run(cypher).data()


def set_issue_embedding(issue_id: str, problem_summary: str, embedding: list[float]) -> None:
    cypher = """
    MATCH (i:Issue {id: $id})
    SET i.embedding = $embedding, i.problem_summary = $problem_summary
    """
    with get_driver().session() as session:
        session.execute_write(
            lambda tx: tx.run(cypher, id=issue_id, embedding=embedding, problem_summary=problem_summary)
        )


def get_stats() -> dict[str, Any]:
    cypher = """
    CALL () { MATCH (i:Issue) RETURN count(i) AS issues }
    CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
    CALL () { MATCH (c:Category) RETURN count(c) AS categories }
    CALL () { MATCH (r:Resolution) RETURN count(r) AS resolutions }
    CALL () { MATCH (i2:Issue) RETURN coalesce(sum(i2.times_seen), 0) AS total_queries_served }
    RETURN issues, entities, categories, resolutions, total_queries_served
    """
    with get_driver().session() as session:
        result = session.run(cypher).data()
    if not result:
        return {"issues": 0, "entities": 0, "categories": 0, "resolutions": 0, "total_queries_served": 0}
    return result[0]
