# OntoDesk, from the beginning — a complete guide

This document tells the whole story of OntoDesk: why it exists, what
problem it solves, how it was originally built, why that first version
wasn't enough, how it was upgraded, and exactly how every piece of it
works today. It's meant to be read start to finish by someone who has
never seen this codebase before.

If you just need a quick operational reference, see
[`README.md`](README.md) (setup/run commands) or
[`explanation.md`](explanation.md) (the current architecture, without the
history). This document is the long-form version of both, plus the "why."

---

## Part 1 — The problem and the idea

### 1.1 The problem

A typical customer-support knowledge base is a flat pile of articles,
FAQs, or past tickets. Nothing in that pile knows that two tickets are
"the same kind of problem" unless a human tags them that way. Every
incoming question either:

- gets manually matched to an existing article by a support agent (slow,
  inconsistent), or
- gets answered from scratch every time, even if the exact same question
  was answered yesterday (wasteful, and answers drift in quality/wording
  over repeated instances of the same problem).

What's missing is a structure that represents *problems* as first-class,
reusable things — not documents, not tickets, but a map of "kinds of
issues" and how they relate to the concepts (products, error codes,
symptoms) that appear in them.

### 1.2 The idea: an agent whose memory is a graph

**OntoDesk**'s premise: model the support agent's knowledge as a graph
whose shape *is* the ontology — `Issue` nodes connected to the `Entity`
and `Category` concepts they involve, each linked to its `Resolution`.
When a new query comes in:

1. Check whether the graph already has this *kind* of issue.
2. If yes — reuse the existing resolution. Don't touch the graph.
3. If no — generate a resolution, and *now* grow the graph by one new
   `Issue` (plus whatever new `Entity`/`Category` nodes it introduces).

The entire value proposition hinges on step 1 being done well. If it's
too permissive, the agent gives wrong answers by reusing an unrelated
resolution. If it's too strict, the graph never reuses anything and just
becomes an ever-growing list of near-duplicates — no smarter than a flat
ticket log. Almost everything interesting about this project is the
engineering of that one decision: **"is this query the same underlying
problem as something I've already resolved?"**

### 1.3 Why a graph database specifically

A relational table of "resolved issues" could technically store the same
data, but a graph makes two things natural that a table makes awkward:

- **Shared concepts as real nodes.** `Entity` and `Category` nodes are
  shared (`MERGE`d) across every `Issue` that involves them. This means
  the graph can answer questions like "what other kinds of problems
  mention `password_reset`?" just by walking relationships — no join
  logic to maintain.
- **The schema doubles as the retrieval structure.** Scoping a search to
  `Issue` nodes under the same `Category`, or walking from an `Entity` to
  every `Issue` that mentions it, is a one-hop graph traversal. The same
  schema that represents the ontology is what both retrieval paths (§3.3)
  query directly.

Neo4j specifically was chosen because it's local, free, has a mature
Python driver, and — critically for the v2 upgrade (§2) — has a native
vector index built in, so semantic search didn't require bolting on a
separate vector database.

### 1.4 Why expose it as an MCP server

The agent's logic (`brain.handle_query`) is a plain Python function. It's
wrapped as an **MCP (Model Context Protocol) server** so that any MCP
client — Claude Desktop, another agent, a support-desk tool — can call
`ask_support(query)` directly, without needing a REST API, authentication
layer, or network exposure. It runs over **stdio**: the client spawns the
Python process itself and talks to it over its stdin/stdout, so there's no
listening network port at all. That's what makes it "private" — only
whatever explicitly launches the process can talk to it.

---

## Part 2 — Version 1: the MVP, and why it wasn't enough

### 2.1 What v1 built

The first version, built as a same-day MVP, used the simplest possible
version of the "is this the same issue?" decision: **pure keyword overlap**.

1. An LLM call extracted a handful of normalized keywords (`entities`) and
   a coarse `category` from the query.
2. A Cypher query pulled every existing `Issue` in that same `category`.
3. For each one, compute **Jaccard similarity** between the query's entity
   set and that issue's entity set:
   ```
   overlap(A, B) = |A ∩ B| / |A ∪ B|
   ```
4. If the best-scoring issue cleared a fixed threshold (`NOVELTY_THRESHOLD`,
   default `0.6`), reuse it. Otherwise, treat the query as novel: generate
   a resolution and write a brand-new `Issue` node.

This is the schema from §1.2, implemented directly: `(:Issue)-[:MENTIONS]->
(:Entity)`, `(:Issue)-[:BELONGS_TO]->(:Category)`, `(:Issue)-[:RESOLVED_BY]->
(:Resolution)`. Supporting pieces: `seed.py` to preload a few canonical
issues, `test_manual.py` for a no-MCP-client smoke test, a Streamlit
dashboard with an "Ask" tab and a live graph-size counter, and the MCP
server exposing `ask_support`/`graph_stats`.

### 2.2 Why pure keyword overlap wasn't good enough

Jaccard-over-keywords is cheap, deterministic, and fully auditable — you
can read exactly which keywords overlapped. But it has two failure modes
that matter a lot for a support agent:

- **False negatives on paraphrase.** "I was charged twice for my
  subscription" and "there are two identical charges on my invoice"
  describe the exact same problem but might extract to barely-overlapping
  keyword sets (`duplicate_charge, subscription` vs. `invoice, charge`).
  Below threshold → treated as novel → the graph grows a near-duplicate
  `Issue` instead of reusing the existing one. Over time this means the
  "self-learning" property degrades: the graph gets *bigger*, not
  *smarter*, because it keeps re-learning things it already knew in
  slightly different words.
- **False positives on shared vocabulary.** Two problems in the same
  category can share keywords without being the same problem at all — "my
  password was reset but login still fails" and "I can't set up two-factor
  authentication" both involve `login`/`authentication`, but need
  completely different resolutions. Pure keyword overlap has no concept of
  *intent* or *what's actually different* between two problems that happen
  to share vocabulary.

Both failure modes trace back to the same root cause: a bare numeric
threshold over one signal (keyword overlap) is being asked to make a
judgment call that really requires understanding the *problem*, not just
comparing word sets. That's the gap v2 closes.

---

## Part 3 — Version 2: hybrid retrieval + LLM verification

### 3.1 The core change

v2 keeps the same ontology (extended, see §3.2) and the same non-negotiable
rule — **the graph only gets a new node when the query is confirmed novel**
— but changes *how* that confirmation happens, in two ways:

1. **Retrieve candidates two independent ways**, not one: semantic
   embedding similarity *and* keyword/entity Jaccard overlap, each scoped
   to the same category, each returning a *ranked list* of candidates
   rather than a single best guess.
2. **Replace the bare threshold with an LLM verification call.** The
   fused candidates (from both retrieval paths) are handed to an LLM
   whose only job is to decide, in light of the actual problem description
   — not just overlapping words — whether one of them is really the same
   underlying issue, a related-but-different one, a brand-new one, or too
   ambiguous to tell.

This directly fixes both v1 failure modes: a paraphrase that scores low on
Jaccard can still be retrieved by semantic similarity (and vice versa for
cases keyword overlap catches that embeddings might miss), and the actual
same-vs-different judgment is made by something that can read the problem
statement, not just measure it.

### 3.2 The extended ontology

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

Everything from v1 is still there (`text`, `category`, `times_seen`, etc.);
what's new:

- **`Issue.problem_summary`** — an LLM-normalized, canonical 1–2 sentence
  restatement of the underlying problem (not the raw query). This exists
  specifically so that two different phrasings of the same problem produce
  *similar text to embed* — embedding the raw query directly would still
  carry along each customer's specific wording, punctuation, and extra
  detail, diluting the semantic signal.
- **`Issue.embedding`** — a vector (1536 floats, `text-embedding-3-small`
  by default) of `problem_summary`, backing the native vector index used
  for semantic retrieval. `NULL` on any `Issue` created before this
  upgrade, until the backfill script (§3.7) runs — retrieval simply never
  returns un-embedded issues rather than erroring on them.
- **`Resolution`'s structured fields** (`problem`, `diagnosis`,
  `resolution_steps`, `escalation_required`, `limitations`) — the old
  single `text` string is still there for backward compatibility and
  display, but the resolution is now generated and stored as structured
  data too, which is both more useful for a dashboard and forces the LLM
  resolution prompt to actually produce each of those parts rather than
  an unstructured paragraph.

Constraints (`Issue.id`, `Entity.name`, `Category.name` all unique) and the
native vector index (`CREATE VECTOR INDEX ... FOR (i:Issue) ON
(i.embedding)`, cosine similarity) are both created with `IF NOT EXISTS`
and re-run at every startup — safe to call repeatedly.

### 3.3 The full pipeline, step by step

Entry point: `brain.handle_query(query: str) -> dict`, called identically
by the MCP server and the dashboard.

**Step 1 — Extraction (`llm.extract`).** One structured-output call
(`client.chat.completions.parse(response_format=ExtractionResult)`,
`temperature=0`) turns the raw query into:
- `entities` — 2–6 normalized keyword/concept strings, synonyms collapsed
  to the same canonical form (e.g. "charged twice" and "double billed"
  both become `duplicate_charge`).
- `category` — one coarse, consistent label.
- `problem_summary` — the canonical restatement described in §3.2.
- `intent` — what the customer is trying to accomplish (e.g.
  `report_problem`, `request_refund`). This is what lets verification later
  distinguish "report a duplicate charge" from "ask for a refund on a
  duplicate charge" even though they share every keyword.
- `context` — a list of `"key: value"` strings for details actually
  present in the query; the model is explicitly instructed never to invent
  anything here.

This step never decides novelty by itself — it only supplies the raw
material that both retrieval paths and verification work from.

**Step 2 — Embedding.** `graph_store.generate_issue_embedding` delegates
to `embeddings.generate_embedding(problem_summary)`, which calls the
configured provider (OpenAI by default; swappable via
`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` so a
different provider is a one-function change, not a rewrite).

**Step 3 — Dual retrieval**, both pure reads, both scoped to the same
`category` so cross-category coincidental overlap can't produce a match:

- `graph_store.find_semantically_similar_issues(embedding, category,
  top_k)` — queries the native vector index
  (`db.index.vector.queryNodes`), over-fetching (`top_k * 4`) and then
  filtering to the right category in Python. (Filtering in Python rather
  than relying on native pre-filtering was a deliberate portability choice
  — it works the same regardless of exact Neo4j version/edition features.)
- `graph_store.find_entity_similar_issues(entities, category, top_k)` —
  the original Jaccard overlap from v1, now returning a ranked list of
  candidates instead of picking one winner.

**Step 4 — Fusion (`ranking.fuse_candidates`).** The two ranked id-lists
are combined with **Reciprocal Rank Fusion**:

```
RRF(d) = Σ 1 / (k + rank_in_list(d))
```

summed over whichever of the two lists contain candidate `d`. `k`
(`RRF_K`, default 60) is the standard smoothing constant from the RRF
literature — it controls how much rank position matters versus how much
simple presence-in-a-list matters. A candidate that both signals agree on
(appears, and ranks well, in *both* lists) outranks one that only shows up
in one — agreement between semantic and keyword evidence is explicitly
rewarded, which is the whole point of doing both retrievals instead of
picking one. Each fused candidate keeps its original `semantic_score`
(cosine) and `entity_score` (Jaccard) for transparency/debugging, visible
in the dashboard.

The fused list is then filtered by `MIN_CANDIDATE_SCORE` — a cheap floor
(on either raw score) that exists purely to avoid spending an LLM call
verifying a candidate that's obviously irrelevant. If nothing clears it
(including the case where the category has no existing issues at all),
OntoDesk skips verification entirely and goes straight to "novel."

**Step 5 — LLM verification (`llm.verify_issue_match`) — the actual
decision.** This is the part that didn't exist in v1 at all, and it's the
single most important function in the codebase. Given the incoming
problem's summary/category/entities/intent/context, and the fused
candidates (each with its id, text, problem summary, category, entities,
*existing resolution*, and both scores), the model returns exactly one of:

- **`SAME_ISSUE`** — this candidate really is the same underlying problem;
  its resolution is potentially applicable. Must name a `matched_issue_id`
  that was actually in the candidate list it was given.
- **`RELATED_BUT_DIFFERENT`** — shares concepts/entities with a candidate
  but is a different problem, or needs a different resolution (e.g. same
  topic, different intent — "report a duplicate charge" vs. "request a
  refund for one"). Treated as novel.
- **`NEW_ISSUE`** — none of the candidates describe the same problem at
  all. Treated as novel.
- **`NEEDS_CLARIFICATION`** — not enough information to tell safely. No
  graph write happens at all — not even a `bump_seen` — because nothing
  was actually confirmed.

The prompt is explicit that `confidence` is **not** a calibrated
probability and must never alone justify a risky match, and that the model
must never conclude `SAME_ISSUE` merely because entities overlap — it has
to weigh intent and distinguishing context too.

**Guardrail against hallucination.** If the model ever returns
`SAME_ISSUE` with a `matched_issue_id` that wasn't actually in the
candidates it was shown, `verify_issue_match` catches this in code and
overrides the result to `NEEDS_CLARIFICATION` before it ever reaches
`brain.py`. A fabricated match is never trusted, no matter how confident
the model claims to be. This is locked in by a dedicated test
(`tests/test_brain_orchestration.py::
test_verification_cannot_select_id_outside_candidates`).

**Step 6 — Branching on the decision, in `brain.py`:**

- *`SAME_ISSUE`* → `graph_store.bump_seen(issue_id)` (the only write: `
  times_seen += 1`, `last_seen = now()`), return the existing resolution.
  `graph_updated: False`, `is_novel: False`.
- *`NEEDS_CLARIFICATION`* → no write of any kind. Return a clarifying
  message built from the model's reasoning. `graph_updated: False`,
  `is_novel: False`, `matched_issue_id: None`.
- *`RELATED_BUT_DIFFERENT` / `NEW_ISSUE`* → `llm.resolve(query, entities,
  problem_summary)` generates a structured resolution (`problem`,
  `diagnosis`, `resolution_steps`, `escalation_required`, `limitations`),
  explicitly prompted to be generally reusable and to omit any
  customer-specific facts. `graph_store.create_issue(...)` then `MERGE`s
  the `Category`/`Entity` nodes and `CREATE`s the new `Issue` + embedding
  + `Resolution`, all inside one `session.execute_write` transaction.
  `graph_updated: True`, `is_novel: True`, `times_seen: 1`.

The final dict returned to callers includes every field a consumer might
want: `answer, is_novel, graph_updated, matched_issue_id, semantic_score,
entity_score, hybrid_rank, verification_decision, verification_reasoning,
times_seen, entities, category, problem_summary`, plus the raw
`semantic_candidates`/`entity_candidates`/`fused_candidates` lists (used by
the dashboard's debug panel, §3.4) — these are additive relative to v1's
response shape, so nothing that read the old fields breaks.

### 3.4 The interfaces

**MCP server (`mcp_server.py`)** — `FastMCP("ontodesk")`, run over stdio
(no network port). Two tools: `ask_support(query)` (calls
`brain.handle_query`) and `graph_stats()` (current node/relationship
counts and total queries served). `main()` ensures constraints and the
vector index exist before starting the stdio loop.

**Streamlit dashboard (`dashboard.py`)** — an "Ask" tab showing a decision
badge (🆕 novel / ♻️ known / ❓ needs clarification / 🆕 related-but-
different), the answer, extracted category/entities/problem_summary, and
an expandable **"Hybrid retrieval & verification detail"** panel: the
semantic score, entity score, hybrid rank, the verification decision and
its reasoning, and two full tables — every semantic candidate and every
entity candidate that was retrieved, not just the one that won — so you
can actually see *why* a match was or wasn't made, not just the outcome.
A sidebar shows live graph size, refreshed via `st.rerun()` right after
each query so the counters never lag by a click.

### 3.5 Structured outputs and why some Pydantic defaults had to change

Every LLM call in `llm.py` uses `client.chat.completions.parse(response_
format=<PydanticModel>)` — OpenAI's structured-output mode, which returns
a typed object instead of hand-parsed JSON. Two real bugs surfaced while
wiring this up, worth knowing about if you extend these schemas:

- A field typed as a **free-form dict** (`context: dict[str, str]`) made
  OpenAI's strict-schema generation fail outright (`BadRequestError:
  'required' is required to be supplied and to be an array including
  every key in properties`) — strict mode can't represent an open-ended
  object with arbitrary keys. The fix was to change `context` to
  `list[str]` of `"key: value"` strings instead of a dict — a type the
  strict schema converter handles cleanly.
- Fields with **`default_factory=list`** (plain lists) are fine; it was
  specifically the dict type that broke things, not "having a default" in
  general. `distinguishing_factors: list[str] = Field(default_factory=
  list, ...)` on `VerificationResult` works correctly.

### 3.6 Guardrails against the LLM forcing or blocking a write

This matters enough to restate explicitly, since it's the actual mechanism
behind "only update the graph on a unique case": the LLM never writes to
Neo4j, directly or indirectly. `verify_issue_match` only returns a
*decision enum*; the only functions that ever call Cypher writes are
`graph_store.bump_seen` (on `SAME_ISSUE`) and `graph_store.create_issue`
(on `RELATED_BUT_DIFFERENT`/`NEW_ISSUE`), both called from `brain.py`'s
explicit branching, never from inside `llm.py`. A model that "sounds
confident" about novelty can't force a write by itself, and a
hallucinated match id is rejected before it can trigger a reuse (§3.3,
Step 5 guardrail).

### 3.7 Supporting scripts, in depth

- **`seed.py`** — writes three canonical issues (billing double-charge,
  login/password-reset, CSV-export timeout), each with a real computed
  embedding and a fully structured resolution, directly via
  `graph_store.create_issue`. This exists so a fresh graph isn't empty —
  the first live demo query can show the *reuse* path immediately, rather
  than needing to ask the same thing twice first.
- **`backfill_embeddings.py`** — the migration path for any `Issue` node
  that predates embeddings (or whenever the vector index needs
  rebuilding). It calls `graph_store.issues_missing_embedding()` to find
  every `Issue` with `embedding IS NULL`, computes a `problem_summary`
  (falling back to the stored `text` if nothing better exists) and its
  embedding, writes them back with `graph_store.set_issue_embedding`, then
  ensures the vector index exists. Fully idempotent — re-running it after
  everything's backfilled just reports zero pending.
- **`test_manual.py`** — a no-MCP-client smoke test: ask a query that
  should match a seeded issue, ask a genuinely new one, then repeat that
  new one verbatim and confirm it flips from novel to known. Prints graph
  stats before and after.
- **`eval/eval_dataset.jsonl` + `eval/run_eval.py`** — a small labeled set
  deliberately covering the hard cases: straightforward paraphrases
  (expected `SAME_ISSUE`), same-topic-different-intent queries (expected
  `RELATED_BUT_DIFFERENT`), genuinely unseen problems (expected
  `NEW_ISSUE`), and ambiguous one-liners (expected `NEEDS_CLARIFICATION`).
  `run_eval.py` runs every row through the real pipeline and reports
  precision/recall/F1 **on the reuse decision specifically** (treating
  `SAME_ISSUE` as "positive"), because an incorrect automatic reuse is a
  worse failure than an unnecessary extra `Issue` node — a false positive
  gives the customer a wrong answer with false confidence, a false
  negative just costs a bit of graph density. This is also the intended
  tool for calibrating `MIN_CANDIDATE_SCORE`/`NOVELTY_THRESHOLD`: sweep the
  value, rerun, and pick whichever maximizes F1 without raising the
  false-positive count.
- **`ranking.py`** — pure Reciprocal Rank Fusion, zero Neo4j/OpenAI
  imports, which is exactly why `tests/test_ranking.py` can test it without
  any external service.
- **`embeddings.py`** — the single place that calls an embedding
  provider's API; everything else routes through
  `graph_store.generate_issue_embedding`, so swapping providers later
  never touches calling code.
- **`config.py`** — loads `kg_agent/.env` first, then falls back to the
  repo-root `.env` (so a shared Neo4j/OpenAI config at the repo root
  doesn't need to be duplicated into `kg_agent/.env`). Exposes one frozen
  `Settings` dataclass covering every tunable: `RRF_K`, `SEMANTIC_TOP_K`,
  `ENTITY_TOP_K`, `MIN_CANDIDATE_SCORE`, `NOVELTY_THRESHOLD`,
  `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS`,
  `VECTOR_INDEX_NAME`.

### 3.8 Testing strategy

`kg_agent/tests/` is split deliberately by what it needs to run:

- **`test_ranking.py`** — pure function tests, no mocking needed at all
  (RRF math, dedup, agreement-ranks-highest, top-n truncation, empty
  inputs).
- **`test_brain_orchestration.py`** — every branch of `brain.handle_query`
  (`SAME_ISSUE` reuse with no new node, straight-to-novel when there are
  no candidates at all, `RELATED_BUT_DIFFERENT` creating a new node,
  `NEEDS_CLARIFICATION` touching nothing, and the hallucination guardrail)
  with `llm.*` and `graph_store.*` monkeypatched to fakes. These always
  run — no live Neo4j or OpenAI key needed — which is what makes them safe
  to run in CI or before every commit.
- **`test_graph_store.py`** — unit tests for the pure `_jaccard` helper,
  plus an integration class that automatically `skipif`s when a local
  Neo4j isn't reachable (checked once at collection time via
  `verify_connectivity()`), so the suite degrades gracefully rather than
  failing when no database is running.

Running `pytest kg_agent/tests -q` with no services running yields 14
pure/mocked passes and 2 skips; with a local Neo4j up, all 16 pass,
including the real `create_vector_index()` idempotency check.

### 3.9 Known limitations, stated plainly

- **Concurrency.** `create_issue` writes atomically in one transaction,
  but the *retrieve → verify → write* sequence as a whole isn't locked.
  Two near-simultaneous requests describing the same brand-new problem
  could both independently reach "novel" and each create their own
  `Issue` node before either write lands — there's no distributed lock
  across that gap. This is an accepted trade-off for a single local
  process (Streamlit/MCP, low concurrency in practice); a production
  deployment facing real concurrent traffic would need either a
  short-lived lock keyed on `(category, top entity)` or a periodic dedup
  pass over near-identical `Issue` embeddings.
- **Embedding drift.** If `EMBEDDING_MODEL` (or its dimensionality)
  changes, previously stored embeddings become incomparable to newly
  generated ones — there's no automatic re-embedding trigger on a config
  change. Switching models means re-running `backfill_embeddings.py`
  after clearing `Issue.embedding` on existing nodes.
- **Verification isn't infallible.** The eval run recorded in
  `eval/eval_dataset.jsonl` surfaced genuinely hard boundary cases (e.g.
  "I want a refund for my duplicate charge" vs. an existing
  duplicate-charge explanation — same topic, different intent) where the
  model's `SAME_ISSUE`/`RELATED_BUT_DIFFERENT` call isn't always
  consistent. This is exactly why `eval/run_eval.py` exists: to make that
  error rate visible and trackable rather than assumed away.

---

## Part 4 — Running it yourself, end to end

```powershell
copy kg_agent\.env.example kg_agent\.env
# edit kg_agent\.env with your OPENAI_API_KEY
# (NEO4J_* defaults already point at a local Neo4j instance)

# make sure a local Neo4j is running (bolt://localhost:7687)

.venv\Scripts\python -m kg_agent.seed                  # preload 3 canonical issues
.venv\Scripts\python -m kg_agent.backfill_embeddings    # no-op right after seed
.venv\Scripts\python -m kg_agent.test_manual            # smoke test the pipeline
.venv\Scripts\python -m kg_agent.eval.run_eval          # precision/recall/F1
.venv\Scripts\streamlit run kg_agent\dashboard.py        # interactive demo
.venv\Scripts\python -m kg_agent.mcp_server              # as an MCP server
.venv\Scripts\python -m pytest kg_agent\tests -q         # unit + integration tests
```

Each of these is explained in more operational detail in
[`README.md`](README.md); this guide is the narrative and reasoning behind
*why* each of those commands does what it does.

---

## Part 5 — Glossary / quick lookup

| Term | Meaning |
|---|---|
| `Issue` | A *kind* of problem (not a raw query log entry). |
| `times_seen` | How many times a query has been LLM-*verified* (not just scored) as a recurrence of this `Issue`. |
| `problem_summary` | The canonical, LLM-normalized restatement of a query's underlying problem — what actually gets embedded. |
| Jaccard / entity score | `\|A ∩ B\| / \|A ∪ B\|` between two entity sets — the keyword-overlap retrieval signal. |
| Cosine / semantic score | Embedding similarity between two `problem_summary` vectors — the meaning-based retrieval signal. |
| RRF (Reciprocal Rank Fusion) | `Σ 1/(k + rank)` across both ranked candidate lists — how the two signals are combined into one ranking. |
| Verification decision | The LLM's `SAME_ISSUE` / `RELATED_BUT_DIFFERENT` / `NEW_ISSUE` / `NEEDS_CLARIFICATION` call — the actual reuse-vs-create decision. |
| `graph_updated` | `True` only when a new `Issue`/`Entity`/`Resolution` subgraph was written — the single field that proves the "only on unique cases" rule held. |
| MCP (stdio) | How the agent is exposed privately: a client spawns the process and talks over stdin/stdout, no network port. |
