# OntoDesk — how it works

OntoDesk is a customer-support agent whose "memory" is a Neo4j knowledge
graph instead of a transcript log or a plain vector store. The core idea:
**the graph only grows when an LLM verification step confirms a query is a
genuinely new kind of problem.** Anything confirmed as a repeat gets
answered instantly from the graph, with no write — so the knowledge base
gets denser and more reusable over time instead of accumulating duplicate
or near-duplicate noise.

It lives in `kg_agent/`, is independent of the rest of the `supportpilot`
repo (no shared Postgres/Redis/LangGraph code), and is exposed as a private
MCP server so any MCP client (Claude Desktop, another agent, a support desk
tool) can call it directly.

> **v2 note:** the first version of OntoDesk gated reuse purely on Jaccard
> keyword overlap against a fixed threshold. That was precision-fragile:
> paraphrases with different vocabulary scored too low to reuse, and
> coincidental shared keywords could over-match. This version retrieves
> candidates two ways (semantic + entity), fuses them, and only reuses a
> candidate after an explicit LLM verification call confirms it's really
> the same underlying problem — see §2 and §5.

---

## 1. The ontology (graph schema)

Four node labels, three relationship types:

```
(:Issue {id, text, category, problem_summary, embedding,
          created_at, last_seen, times_seen})
   -[:MENTIONS]->        (:Entity {name})
   -[:BELONGS_TO]->      (:Category {name})
   -[:RESOLVED_BY]->     (:Resolution {text, problem, diagnosis,
                                        resolution_steps,
                                        escalation_required, limitations,
                                        created_at})
```

- **`Issue`** — a *kind* of problem, not a raw log row. "I was charged
  twice this month" and "double charge on my March invoice" should become
  the *same* `Issue` node, not two.
  - `id`: a UUID, generated when the issue is first created.
  - `text`: the original query that first created this issue (kept for
    traceability/debugging, not shown to users as "the" canonical phrasing).
  - `problem_summary`: an LLM-normalized, canonical 1–2 sentence
    description of the underlying problem (see §2 Step 1). This is what
    gets embedded, not the raw query text, so two differently-worded
    queries about the same problem produce similar embeddings.
  - `embedding`: a vector (1536 floats by default, `text-embedding-3-small`)
    of `problem_summary`, indexed by the native vector index (§2 Step 3) for
    semantic retrieval. `NULL` on Issues created before this upgrade until
    `backfill_embeddings.py` runs (see §4).
  - `category`: a coarse bucket like `billing`, `login`, `data_export`.
  - `created_at` / `last_seen`: ISO timestamps.
  - `times_seen`: starts at 1, incremented every time a later query is
    LLM-*verified* as the same issue recurring (not just similarity-scored
    as such). This is the number that proves the graph is being *reused*,
    not just grown.

- **`Entity`** — a normalized keyword/concept extracted from the query
  (e.g. `"duplicate_charge"`, `"password_reset"`, `"csv_export"`). Entities
  are shared across issues via `MERGE`, so the same entity node is reused
  whenever it recurs — this is what makes entity-overlap comparison
  possible (§2 Step 3) and makes the graph genuinely graph-shaped rather
  than a flat table.

- **`Category`** — one label per general class of issue, also `MERGE`d so
  it's shared across issues in the same bucket. Used to scope both
  retrieval paths (§2 Step 3) so a billing question is never matched
  against a login issue just because of an incidental shared keyword.

- **`Resolution`** — the stored answer for an `Issue`, as both a single
  back-compat `text` string (what gets displayed/returned) and structured
  fields (`problem`, `diagnosis`, `resolution_steps`, `escalation_required`,
  `limitations`) written by `llm.resolve` (§2 Step 5). Kept as its own node
  (not just properties on `Issue`) so the schema can grow later (resolution
  versioning, feedback/ratings) without changing the `Issue` node shape.
  Deliberately holds no customer-identifying information — it's phrased to
  be reusable across different customers with the same problem.

Uniqueness constraints (`graph_store.init_constraints()`) are created on
`Issue.id`, `Entity.name`, and `Category.name`. A native vector index
(`graph_store.create_vector_index()`) is created on `Issue.embedding` for
cosine-similarity search. Both are idempotent (`IF NOT EXISTS`) and re-run
safely at every startup (MCP server `main()`, the dashboard, `seed.py`,
`backfill_embeddings.py`).

---

## 2. What happens on a query, end to end

Entry point: `brain.handle_query(query: str) -> dict`. Both the MCP server
and the Streamlit dashboard call this one function.

```
query
  │
  ▼
llm.extract(query)   → entities, category, problem_summary, intent, context
  │
  ▼
embed(problem_summary)
  │
  ├──────────────────────┬──────────────────────┐
  ▼                       ▼                      │
graph_store.find_        graph_store.find_       │
semantically_similar_    entity_similar_         │
issues() (vector index,  issues() (Jaccard,       │
cosine, same category)   same category)          │
  │                       │                       │
  └──────────┬────────────┘                       │
             ▼                                    │
     ranking.fuse_candidates()  (Reciprocal Rank Fusion)
             │
             ▼
    filter by MIN_CANDIDATE_SCORE (cheap pre-filter)
             │
     ┌───────┴────────┐
     │ no candidates   │ candidates remain
     ▼                 ▼
  (skip verification)  llm.verify_issue_match(extraction, candidates)
     │                 │
     │         ┌───────┼────────────┬──────────────────┐
     │         ▼        ▼            ▼                  ▼
     │     SAME_ISSUE  RELATED_BUT_  NEW_ISSUE      NEEDS_CLARIFICATION
     │         │       DIFFERENT       │                  │
     │         ▼          │            │                  ▼
     │   bump_seen()      └─────┬──────┘          return clarification,
     │   reuse resolution       ▼                  no write, is_novel=False
     │   graph_updated=False  llm.resolve() +
     │                        create_issue()
     └──────────────────────────┼─────────────────
                                 ▼
                      graph_updated=True, is_novel=True
```

`handle_query` returns a dict with `answer, is_novel, graph_updated,
matched_issue_id, semantic_score, entity_score, hybrid_rank,
verification_decision, verification_reasoning, times_seen, entities,
category, problem_summary`, plus the raw `semantic_candidates` /
`entity_candidates` / `fused_candidates` lists (used by the dashboard's
debug panel, §3).

### Step 1 — Extraction (`llm.extract`)

One `client.chat.completions.parse(response_format=ExtractionResult)` call
(structured Pydantic output, `temperature=0`) turns the raw query into:
- `entities`: 2–6 short, lowercase, normalized keyword/concept strings
  (synonyms normalized to the same canonical entity).
- `category`: one coarse, consistent label.
- `problem_summary`: a canonical 1–2 sentence restatement, written so two
  phrasings of the same problem converge to similar summaries. **This**,
  not the raw query, is what gets embedded in Step 2.
- `intent`: what the customer is trying to accomplish.
- `context`: a list of `"key: value"` strings for distinguishing details
  actually present in the query (never invented).

This step never decides novelty itself — it only supplies raw material to
retrieval and verification.

### Step 2 — Embedding (`graph_store.generate_issue_embedding`)

Delegates to `embeddings.generate_embedding(problem_summary)`
(`kg_agent/embeddings.py`), which calls the configured provider (OpenAI
`text-embedding-3-small`, 1536 dims, by default — swappable via
`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS`). Embedding
the normalized `problem_summary` rather than the raw query is what makes
paraphrases land close together in vector space.

### Step 3 — Dual retrieval (pure reads, no writes)

- **`graph_store.find_semantically_similar_issues(embedding, category,
  top_k)`**: `CALL db.index.vector.queryNodes('issue_embedding_index', ...)`
  against the native vector index, over-fetching (`top_k * 4`) and then
  filtering to the matching category in Python — portable across Neo4j
  versions rather than depending on native metadata pre-filtering. Issues
  without an embedding (not yet backfilled) simply aren't in the index, so
  they're never returned until backfilled.
- **`graph_store.find_entity_similar_issues(entities, category, top_k)`**:
  the original Jaccard overlap, now returning a *ranked list* of
  candidates instead of picking a single best match.

Both are read-only and category-scoped.

### Step 4 — Fusion (`ranking.fuse_candidates`)

Combines the two ranked id-lists with **Reciprocal Rank Fusion**:

```
RRF(d) = Σ 1 / (k + rank_in_list(d))     (summed over whichever lists contain d)
```

`k` (`RRF_K`, default 60) is the standard smoothing constant from the RRF
literature. A candidate that ranks well in *both* semantic and entity
retrieval outranks one that only appears in one list — agreement between
the two signals is rewarded. Each returned candidate keeps its original
`semantic_score` (cosine) and `entity_score` (Jaccard) alongside the fused
`rrf_score`, for debugging/evaluation (visible in the dashboard, §3).

Candidates are then filtered by `MIN_CANDIDATE_SCORE`: a cheap floor (on
either raw score) so the LLM verification call in Step 5 isn't wasted on
obviously-irrelevant candidates. If nothing clears it, OntoDesk skips
straight to creating a new Issue (Step 6) — this is also what happens for
a query whose category has no existing issues at all.

### Step 5 — LLM verification (`llm.verify_issue_match`)

This is the actual reuse decision, and it's the part that fixes v1's
precision problem: it's not a bare threshold, it's an LLM reading the
*problem*, not just the keywords. Given the incoming problem summary,
category, entities, intent, and context, plus the fused candidates (each
with its id, text, problem summary, category, entities, existing
resolution, and both scores), the model returns one of:

- **`SAME_ISSUE`** — reuse a specific candidate's resolution.
  `matched_issue_id` must be one of the ids it was given.
- **`RELATED_BUT_DIFFERENT`** — shares concepts but is a different problem
  or needs a different resolution → treated as novel (new node).
- **`NEW_ISSUE`** — none of the candidates are the same problem → novel.
- **`NEEDS_CLARIFICATION`** — not enough information to decide safely →
  no graph write at all, a clarifying message is returned instead.

**Guardrail:** if the model returns `SAME_ISSUE` with an id that wasn't in
the candidate list it was given (a hallucination), `verify_issue_match`
overrides it to `NEEDS_CLARIFICATION` before it ever reaches `brain.py` —
a fabricated match can never be trusted (see
`llm.py::verify_issue_match`, and
`tests/test_brain_orchestration.py::test_verification_cannot_select_id_outside_candidates`).

`confidence` is returned for display only — the code never branches on it
(the prompt explicitly says not to treat it as a calibrated probability).

### Step 6a — `SAME_ISSUE` (reuse path)

- `graph_store.bump_seen(issue_id)`: `times_seen += 1`, `last_seen = now()`.
  This is the *only* write — no new `Issue`, `Entity`, `Category`, or
  `Resolution` node.
- The matched candidate's stored resolution text is returned as the answer.
- `graph_updated: False`, `is_novel: False`.

### Step 6b — `NEEDS_CLARIFICATION`

- No graph write at all (not even `bump_seen` — the match wasn't
  confirmed). Returns a clarification message built from the model's
  `reasoning`. `graph_updated: False`, `is_novel: False`,
  `matched_issue_id: None`.

### Step 6c — `RELATED_BUT_DIFFERENT` / `NEW_ISSUE` (learn path)

- `llm.resolve(query, entities, problem_summary)`: a structured
  (`StructuredResolution`) call — `problem`, `diagnosis`,
  `resolution_steps`, `escalation_required`, `limitations` — explicitly
  prompted to stay reusable and to omit customer-specific facts.
  `llm.render_resolution_text` flattens it into the back-compat `text`
  string stored on `Resolution.text`.
- `graph_store.create_issue(...)`: `MERGE`s `Category`/`Entity` nodes,
  `CREATE`s a new `Issue` (with its embedding) and `Resolution`, in one
  `session.execute_write` transaction.
- `graph_updated: True`, `is_novel: True`, `times_seen: 1`.

---

## 3. Interfaces

Both interfaces are thin wrappers around `brain.handle_query`.

### MCP server (`mcp_server.py`)

Built with `mcp.server.fastmcp.FastMCP("ontodesk")`, run over **stdio**
(`mcp.run(transport="stdio")`) — reachable only by a process that spawns it
directly (e.g. Claude Desktop reading its MCP config), no network port
opened, which is what makes it "private."

Two tools:
- `ask_support(query: str) -> dict` — calls `brain.handle_query`, returning
  all the fields listed in §2.
- `graph_stats() -> dict` — current counts of issues/entities/categories/
  resolutions and the running total of queries served
  (`sum(Issue.times_seen)`).

`main()` runs `init_constraints()` and `create_vector_index()` (both
idempotent) before starting the stdio loop.

### Streamlit dashboard (`dashboard.py`)

The "Ask" tab shows a decision badge (🆕 novel / ♻️ known / ❓ needs
clarification), the answer, extracted category/entities/problem_summary,
and an expandable **"Hybrid retrieval & verification detail"** panel with
the semantic score, entity score, hybrid rank, the verification decision +
reasoning, and two tables listing every semantic and entity candidate that
was retrieved (not just the winning one) — so you can see *why* a
particular Issue was or wasn't matched.

The sidebar shows live graph size. Because Streamlit reruns the whole
script on every interaction, the "Ask" result is stashed in
`st.session_state["last_result"]` and the script calls `st.rerun()`
immediately after handling a query, so the sidebar counters reflect the
just-processed query on the very next render.

---

## 4. Supporting scripts

- **`seed.py`** — writes three canonical issues (billing double-charge,
  login/password reset, CSV export timeout), each with a real embedding
  and a structured resolution, via `graph_store.create_issue` — so the
  graph is immediately usable by both retrieval paths without needing the
  backfill script first.
- **`backfill_embeddings.py`** — migration for Issues that predate this
  upgrade (or a fresh re-index): finds every `Issue` with
  `embedding IS NULL` (`graph_store.issues_missing_embedding`), computes
  `problem_summary` (falling back to the stored `text` if that's all
  there is) and its embedding, writes them back
  (`graph_store.set_issue_embedding`), then ensures the vector index
  exists. Idempotent — safe to re-run; reports 0 pending once everything's
  backfilled.
- **`test_manual.py`** — calls `brain.handle_query` directly (no MCP
  client needed) against three queries: one that should match a seeded
  issue, one genuinely new query, and a verbatim repeat of that new query
  — which should flip from novel to known. Prints `graph_store.get_stats()`
  before/after.
- **`eval/eval_dataset.jsonl` + `eval/run_eval.py`** — a small labeled set
  covering paraphrases (expect `SAME_ISSUE`), same-entities-different-
  problem (expect `RELATED_BUT_DIFFERENT`), unseen problems (expect
  `NEW_ISSUE`), and ambiguous queries (expect `NEEDS_CLARIFICATION`).
  `run_eval.py` runs each through the real pipeline and reports
  precision/recall/F1 on the reuse decision (treating `SAME_ISSUE` as
  "positive"), plus the exact mismatches — this is the tool to use when
  calibrating `MIN_CANDIDATE_SCORE`/`NOVELTY_THRESHOLD`: sweep the value
  and pick whichever maximizes F1 without raising the false-positive rate,
  since an unsafe auto-reuse is worse than one extra Issue node.
- **`ranking.py`** — pure Reciprocal Rank Fusion logic (`
  reciprocal_rank_fusion`, `fuse_candidates`), with no Neo4j/OpenAI
  imports, so `tests/test_ranking.py` can unit-test it directly.
- **`embeddings.py`** — the one place that calls an embedding provider's
  API; everything else goes through `graph_store.generate_issue_embedding`.
- **`config.py`** — loads `kg_agent/.env` first, then falls back to the
  repo-root `.env` for any variable not set there. Exposes a single frozen
  `Settings` dataclass (`settings`), including every retrieval knob
  (`RRF_K`, `SEMANTIC_TOP_K`, `ENTITY_TOP_K`, `MIN_CANDIDATE_SCORE`,
  `EMBEDDING_*`, `VECTOR_INDEX_NAME`).

---

## 5. Design decisions worth knowing about

- **Verification, not a threshold, is the reuse gate.** v1 used a bare
  Jaccard threshold. This version only reuses after an explicit LLM
  decision over ranked evidence (§2 Step 5) — `MIN_CANDIDATE_SCORE` is
  merely a pre-filter to avoid wasting a verification call, not the
  decision itself. This is what lets paraphrases with low keyword overlap
  still get reused correctly when they're semantically close, and what
  lets an entity-overlapping-but-different problem *not* get reused
  incorrectly.
- **Hybrid retrieval over either signal alone.** Semantic similarity alone
  can be fooled by surface-level phrasing similarity that isn't the same
  problem; entity overlap alone can be fooled by shared vocabulary.
  Fusing both via RRF and feeding the *union* to verification means a
  correct match found by only one signal still gets a chance, while a
  candidate both signals agree on ranks highest.
- **The LLM can't force or block a write by itself.** `verify_issue_match`
  only *decides between branches*; the actual Cypher write only happens in
  `graph_store.create_issue` on the create-new branches, and only
  `bump_seen` runs on reuse. A hallucinated match id is explicitly
  rejected (§2 Step 5 guardrail) rather than trusted.
- **One database, one package.** Unlike the rest of `supportpilot`
  (Postgres + Redis + Neo4j + LangGraph), OntoDesk uses Neo4j as its only
  store (now also as its vector index — no separate vector DB) and has no
  dependency on the rest of the `app/` package.
- **Resolutions are written to be reusable, not personalized**, and
  explicitly exclude customer-identifying facts (§1, `Resolution`).
- **Thresholds are configurable, not hard-coded guesses** — `RRF_K`,
  `SEMANTIC_TOP_K`, `ENTITY_TOP_K`, `MIN_CANDIDATE_SCORE`,
  `NOVELTY_THRESHOLD` all live in `.env` and are meant to be calibrated
  against `eval/run_eval.py`'s output, not tuned by intuition.
- **Known limitation — concurrency.** `create_issue` is a single atomic
  transaction, but two near-simultaneous requests for the *same* new
  problem could each independently decide "novel" and create two Issue
  nodes before either write lands (no distributed lock across the
  retrieve-then-write gap). Acceptable for a single local process
  (Streamlit/MCP, low concurrency); a production deployment with real
  concurrent traffic would need either a short-lived lock keyed on
  `(category, top entity)` or a periodic dedup pass over near-identical
  `Issue` embeddings.
- **Known limitation — embedding drift across providers/models.** If
  `EMBEDDING_MODEL` or its dimensions change, previously-stored embeddings
  become incomparable to new ones. There's no automatic re-embedding on a
  config change; re-run `backfill_embeddings.py` after clearing
  `Issue.embedding` (or rebuilding the graph) if you switch models.

---

## 6. Worked example (from an actual run against the seeded graph)

1. `seed.py` creates `Issue A` — problem_summary "Customer was charged more
   than once for the same subscription period," category `billing`,
   entities `[duplicate_charge, subscription, billing]`.
2. A user asks *"I got billed two times this month for the same plan."*
   - Extraction: entities `[duplicate_charge, billing, plan]`, category
     `billing`, problem_summary "Customer was billed twice for the same
     plan in one month."
   - Semantic retrieval: `Issue A` at cosine **0.886**.
   - Entity retrieval: `Issue A` at Jaccard **0.5**.
   - Fused → verification: `SAME_ISSUE`, reasoning: *"The incoming query
     and the candidate issue both describe a situation where the customer
     was billed twice for the same plan or subscription period... the
     entities and category align well, and there are no distinguishing
     factors that suggest a different underlying problem."*
   - `bump_seen`: `Issue A.times_seen → 2`. No new nodes.
3. A user asks *"The mobile app crashes whenever I open the analytics
   tab."* No candidates in the `performance` category at all →
   verification is skipped entirely, straight to novel. `llm.resolve`
   generates a structured resolution, `create_issue` writes `Issue B`.
4. The same crash query, asked again: semantic score **1.0**, entity score
   **1.0** against `Issue B` → `SAME_ISSUE`, `Issue B.times_seen → 2`, no
   new write.
5. A trickier case from the eval set: *"I want a refund for my duplicate
   charge, not just an explanation"* — shares `billing`/`duplicate_charge`
   entities with `Issue A`, but differs in **intent** (refund request vs.
   an explanatory resolution). Verification correctly separates intent
   from topic overlap in most runs, though the eval run recorded in
   `eval/eval_dataset.jsonl` shows this is exactly the kind of case worth
   watching with `run_eval.py` — it's a genuinely hard boundary between
   `SAME_ISSUE` and `RELATED_BUT_DIFFERENT` that benefits from more labeled
   examples over time, which is the whole point of having the eval
   harness rather than trusting intuition.
