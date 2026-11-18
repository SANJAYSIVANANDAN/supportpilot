# OntoDesk — Architecture

One-page reference. For the full narrative see [`GUIDE.md`](GUIDE.md); for
operational commands see [`README.md`](README.md).

## System overview

```mermaid
flowchart LR
    subgraph Clients
        MCP["MCP client\n(Claude Desktop, etc.)"]
        UI["Streamlit dashboard"]
    end

    MCP -- "stdio, ask_support(query)" --> Server[mcp_server.py]
    UI -- "handle_query(query)" --> Brain

    Server --> Brain[brain.handle_query]

    Brain --> LLM[llm.py\nextract / verify / resolve]
    Brain --> Graph[graph_store.py]
    LLM --> OpenAI[(OpenAI API)]
    Graph --> Embed[embeddings.py]
    Embed --> OpenAI
    Graph --> Neo4j[(Neo4j\nIssue / Entity / Category / Resolution\n+ vector index)]
```

Everything funnels through **one function**, `brain.handle_query`, so the
MCP server and the dashboard can never disagree on behavior.

## Graph schema (the ontology)

```mermaid
graph LR
    I["Issue\nid, text, category,\nproblem_summary, embedding,\ntimes_seen, last_seen"]
    E["Entity\nname"]
    C["Category\nname"]
    R["Resolution\ntext, problem, diagnosis,\nresolution_steps,\nescalation_required, limitations"]

    I -- MENTIONS --> E
    I -- BELONGS_TO --> C
    I -- RESOLVED_BY --> R
```

`Entity` and `Category` are `MERGE`d, so they're shared across every
`Issue` that involves them — the graph grows denser, not just bigger.

## Decision logic — the only thing that gates a write

```mermaid
flowchart TD
    Q[Incoming query] --> EX["llm.extract\nentities, category,\nproblem_summary, intent, context"]
    EX --> EMB["embed(problem_summary)"]

    EMB --> SEM["find_semantically_similar_issues\n(vector index, cosine, same category)"]
    EX --> ENT["find_entity_similar_issues\n(Jaccard overlap, same category)"]

    SEM --> FUSE["fuse_candidates\nReciprocal Rank Fusion"]
    ENT --> FUSE

    FUSE --> FILTER{"Any candidate clears\nMIN_CANDIDATE_SCORE?"}
    FILTER -- no --> CREATE
    FILTER -- yes --> VERIFY["llm.verify_issue_match\n(reads problem + intent,\nnot just scores)"]

    VERIFY --> SAME{decision}
    SAME -- SAME_ISSUE --> REUSE["bump_seen(issue_id)\nreturn stored resolution\ngraph_updated = False"]
    SAME -- NEEDS_CLARIFICATION --> CLARIFY["return clarifying message\nNO write at all"]
    SAME -- "RELATED_BUT_DIFFERENT\nor NEW_ISSUE" --> CREATE

    CREATE["llm.resolve + create_issue\nnew Issue/Entity/Resolution\ngraph_updated = True"]

    style REUSE fill:#1f6feb,color:#fff
    style CREATE fill:#da3633,color:#fff
    style CLARIFY fill:#9a6700,color:#fff
```

**The rule that matters:** the LLM never writes to Neo4j directly. It only
returns a decision enum; `graph_store.bump_seen` and
`graph_store.create_issue` are the only two write paths, called explicitly
from `brain.py`'s branching. A verification result that names a candidate
id outside the ones it was shown is rejected and forced to
`NEEDS_CLARIFICATION` before it can reach that branching at all.

## One query, sequenced

```mermaid
sequenceDiagram
    participant C as Client (MCP/UI)
    participant B as brain.py
    participant L as llm.py
    participant G as graph_store.py
    participant N as Neo4j

    C->>B: handle_query(query)
    B->>L: extract(query)
    L-->>B: entities, category, problem_summary, intent
    B->>G: generate_issue_embedding(problem_summary)
    G-->>B: embedding
    B->>G: find_semantically_similar_issues()
    B->>G: find_entity_similar_issues()
    G->>N: vector index + Jaccard reads
    N-->>G: candidates
    G-->>B: semantic_candidates, entity_candidates
    B->>B: fuse_candidates (RRF) + MIN_CANDIDATE_SCORE filter
    alt candidates remain
        B->>L: verify_issue_match(extraction, candidates)
        L-->>B: SAME_ISSUE | RELATED_BUT_DIFFERENT | NEW_ISSUE | NEEDS_CLARIFICATION
    end
    alt SAME_ISSUE
        B->>G: bump_seen(issue_id)
        G->>N: SET times_seen += 1
    else NEEDS_CLARIFICATION
        Note over B: no write
    else novel (RELATED_BUT_DIFFERENT / NEW_ISSUE / no candidates)
        B->>L: resolve(query, entities, problem_summary)
        L-->>B: structured resolution
        B->>G: create_issue(...)
        G->>N: CREATE Issue + Entity + Resolution
    end
    B-->>C: answer, is_novel, graph_updated, scores, decision, reasoning
```

## Why hybrid retrieval, in one sentence

Semantic similarity catches paraphrases that share no keywords; entity
(Jaccard) overlap catches exact-vocabulary repeats that phrase themselves
oddly enough to embed differently. **Reciprocal Rank Fusion** rewards a
candidate both signals agree on; **LLM verification** — not either score —
makes the final same/different/unsure call, because only it can read
*intent*, not just measure word or vector overlap.

## Module map

| File | Responsibility |
|---|---|
| `brain.py` | Orchestrates the flowchart above. The only thing that decides which branch runs. |
| `llm.py` | `extract`, `verify_issue_match`, `resolve` — three structured-output calls. Never writes to Neo4j. |
| `embeddings.py` | The one place that calls an embedding provider. |
| `graph_store.py` | All Cypher. Retrieval functions only read; `bump_seen`/`create_issue` are the only writes. |
| `ranking.py` | Pure Reciprocal Rank Fusion — no external calls, fully unit-testable. |
| `mcp_server.py` / `dashboard.py` | Thin callers of `brain.handle_query`. |
| `seed.py` / `backfill_embeddings.py` | Populate / migrate the graph. |
| `eval/` | Labeled dataset + precision/recall/F1 harness for calibrating thresholds. |
