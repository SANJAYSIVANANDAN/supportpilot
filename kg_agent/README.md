# OntoDesk

A support agent whose knowledge is literally a Neo4j graph: `(:Issue)-[:MENTIONS]->(:Entity)`,
`(:Issue)-[:BELONGS_TO]->(:Category)`, `(:Issue)-[:RESOLVED_BY]->(:Resolution)`.

Every incoming query is checked against the graph for a near-duplicate `Issue`
using **hybrid retrieval**: semantic embedding similarity (Neo4j native vector
index) and entity/Jaccard overlap are both retrieved, fused with Reciprocal
Rank Fusion, and an **LLM verification** step decides whether a candidate is
really the same underlying problem — not a bare similarity score. A confirmed
match is **reused** (`times_seen` bumped, no new nodes); an unclear match
returns a **clarification** request instead of guessing; a genuinely new
problem is **novel** and gets a new `Issue`/`Entity`/`Resolution` subgraph.
The graph only grows on verified-unique cases. See
[`explanation.md`](explanation.md) for the full architecture writeup.

Standalone from the rest of `supportpilot` (no shared Postgres/Redis/LangGraph
code), but reuses the same local Neo4j instance via new node labels.

## Setup

```powershell
copy kg_agent\.env.example kg_agent\.env
# edit kg_agent\.env with your OPENAI_API_KEY (Neo4j creds default to the
# same local instance already used by the rest of supportpilot)

# make sure Neo4j is running locally (see repo root README.md step 4)

.venv\Scripts\python -m kg_agent.seed                  # preload 3 canonical issues (with embeddings)
.venv\Scripts\python -m kg_agent.backfill_embeddings    # no-op right after seed; needed for pre-upgrade Issues
.venv\Scripts\python -m kg_agent.test_manual            # smoke test the hybrid pipeline
.venv\Scripts\python -m kg_agent.eval.run_eval          # precision/recall/F1 against the labeled eval set
.venv\Scripts\streamlit run kg_agent\dashboard.py
```

## Calibrating thresholds

`NOVELTY_THRESHOLD` and `MIN_CANDIDATE_SCORE` (in `kg_agent/.env`) are not the
reuse decision itself — that's made by LLM verification — but they control
which candidates are even worth sending to verification. To calibrate them
for your own data: add labeled rows to `kg_agent/eval/eval_dataset.jsonl`
(query + expected decision), run `kg_agent.eval.run_eval`, and adjust
`MIN_CANDIDATE_SCORE` to the value that maximizes F1 on the reuse decision
without raising the false-positive rate (an unsafe auto-reuse is worse than
an unnecessary new Issue).

## MCP server

```powershell
.venv\Scripts\python -m kg_agent.mcp_server
```

Tools exposed: `ask_support(query)`, `graph_stats()`. Register with an MCP
client (e.g. Claude Desktop) the same way as `app/mcp/server.py` in the repo
root README:

```json
{
  "mcpServers": {
    "ontodesk": {
      "command": "C:\\Users\\sanja\\projects\\supportpilot\\.venv\\Scripts\\python.exe",
      "args": ["-m", "kg_agent.mcp_server"],
      "cwd": "C:\\Users\\sanja\\projects\\supportpilot"
    }
  }
}
```

## Tests

```powershell
.venv\Scripts\python -m pytest kg_agent\tests -q
```

`test_ranking.py` and `test_brain_orchestration.py` are pure/monkeypatched
and always run. `test_graph_store.py`'s integration class auto-skips if a
local Neo4j isn't reachable.
