"""The three LLM calls OntoDesk needs: understand the query, verify whether
a retrieved candidate is really the same underlying problem, and (only for
genuinely new problems) generate a resolution.

All three use `client.chat.completions.parse(response_format=<PydanticModel>)`
for structured output, so callers get typed objects instead of hand-parsed
JSON.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from kg_agent.config import settings

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# ---------------------------------------------------------------------------
# Phase 1: extraction
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    entities: list[str] = Field(description="2-6 short, lowercase, normalized canonical concepts")
    category: str = Field(description="One short, consistent general problem category")
    problem_summary: str = Field(description="Concise, canonical description of the underlying problem")
    intent: str = Field(description="What the customer is trying to accomplish, e.g. report_problem, request_refund")
    context: list[str] = Field(
        description="Distinguishing details explicitly present in the query, as 'key: value' strings "
        "(never invented); [] if none"
    )


EXTRACT_PROMPT = """You triage incoming customer support queries for a knowledge graph.

Extract:
- entities: 2-6 short normalized keyword/entity strings (lowercase, no punctuation,
  underscores instead of spaces) capturing the specific products, features, error
  codes or symptoms mentioned. Normalize synonyms to the same canonical entity
  (e.g. "charged twice" and "double billed" both become "duplicate_charge").
- category: one short, consistent label for the general kind of issue (e.g.
  "billing", "login", "performance", "data_export", "integration"). Reuse the
  same category label for the same kind of issue across different queries.
- problem_summary: a concise (1-2 sentence), canonical description of the
  underlying problem, written so that two different phrasings of the same
  problem produce very similar summaries.
- intent: what the customer is trying to accomplish (e.g. "report_problem",
  "request_refund", "request_explanation", "request_cancellation").
- context: a short list of "key: value" strings for distinguishing details
  ACTUALLY present in the query (e.g. "product: mobile_app", "plan_tier: pro",
  "error_code: E502"). Never invent information the customer didn't state.
  Omit anything not explicitly present; use [] if nothing distinguishing is stated.

Distinguish queries that are related but describe a different underlying
problem or require a different resolution - do not force them into the same
category/summary just because they share surface keywords."""


def extract(query: str) -> ExtractionResult:
    response = get_client().chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format=ExtractionResult,
        temperature=0,
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Phase 5: candidate verification
# ---------------------------------------------------------------------------


class VerificationDecision(str, Enum):
    SAME_ISSUE = "SAME_ISSUE"
    RELATED_BUT_DIFFERENT = "RELATED_BUT_DIFFERENT"
    NEW_ISSUE = "NEW_ISSUE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class VerificationResult(BaseModel):
    decision: VerificationDecision
    matched_issue_id: str | None = Field(
        default=None, description="Must be one of the candidate issue ids, or null unless decision is SAME_ISSUE"
    )
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, description="Not a calibrated probability - relative confidence only")
    distinguishing_factors: list[str] = Field(
        default_factory=list, description="Key factors that drove this decision; [] if none"
    )


VERIFY_PROMPT = """You verify whether an incoming customer support query describes
the SAME underlying problem as a previously-seen, candidate issue in a knowledge
graph - not merely a similar-sounding one.

You will be given the incoming query's problem summary, category, entities and
context, plus a short list of candidate issues (each with its id, original text,
problem summary, category, entities, and existing resolution).

Decide exactly one of:
- SAME_ISSUE: the incoming query describes the same underlying problem as one
  specific candidate, and that candidate's existing resolution is potentially
  applicable. Set matched_issue_id to that candidate's id (it MUST be one of
  the ids given to you - never invent or guess an id).
- RELATED_BUT_DIFFERENT: the incoming query shares concepts/entities with one
  or more candidates but describes a different problem, or would need a
  different resolution. Do not set matched_issue_id.
- NEW_ISSUE: none of the candidates describe the same underlying problem at all.
- NEEDS_CLARIFICATION: important information is missing to tell whether an
  existing resolution genuinely applies - prefer this over an unsafe match.

Rules:
- Never pick a candidate id that wasn't given to you.
- Never conclude SAME_ISSUE merely because entities or keywords overlap -
  weigh the customer's intent and any distinguishing contextual details.
- Treat related-but-different problems as different issues, even if a
  resolution might partially apply.
- Do not fabricate information not present in the query or candidates.
- confidence is a relative signal for a human/dashboard to read, not a
  calibrated probability - never let it alone justify a risky SAME_ISSUE call."""


def _format_candidate(candidate: dict[str, Any]) -> str:
    return (
        f"- id: {candidate['id']}\n"
        f"  text: {candidate.get('text')}\n"
        f"  problem_summary: {candidate.get('problem_summary')}\n"
        f"  category: {candidate.get('category')}\n"
        f"  entities: {candidate.get('entities')}\n"
        f"  existing_resolution: {candidate.get('resolution')}\n"
        f"  semantic_score: {candidate.get('semantic_score')}, entity_score: {candidate.get('entity_score')}"
    )


def verify_issue_match(incoming: ExtractionResult, candidates: list[dict[str, Any]]) -> VerificationResult:
    if not candidates:
        return VerificationResult(decision=VerificationDecision.NEW_ISSUE, reasoning="No candidates retrieved.", confidence=1.0)

    candidate_ids = {c["id"] for c in candidates}
    user_content = (
        f"Incoming query problem_summary: {incoming.problem_summary}\n"
        f"Incoming category: {incoming.category}\n"
        f"Incoming entities: {incoming.entities}\n"
        f"Incoming intent: {incoming.intent}\n"
        f"Incoming context: {incoming.context}\n\n"
        f"Candidates:\n" + "\n".join(_format_candidate(c) for c in candidates)
    )
    response = get_client().chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=VerificationResult,
        temperature=0,
    )
    result = response.choices[0].message.parsed

    if result.decision == VerificationDecision.SAME_ISSUE and result.matched_issue_id not in candidate_ids:
        # The model hallucinated or omitted an id that wasn't offered to it -
        # never trust a match outside the candidate set.
        return VerificationResult(
            decision=VerificationDecision.NEEDS_CLARIFICATION,
            reasoning="Model selected a SAME_ISSUE match outside the candidate set; treated as unsafe.",
            confidence=0.0,
        )
    return result


# ---------------------------------------------------------------------------
# Phase 7: resolution generation
# ---------------------------------------------------------------------------


class StructuredResolution(BaseModel):
    problem: str = Field(description="Restatement of the underlying problem")
    diagnosis: str = Field(description="Likely cause, grounded only in information available")
    resolution_steps: list[str] = Field(description="Concrete, actionable steps to resolve it")
    escalation_required: bool = Field(description="Whether this needs human/engineering escalation")
    limitations: str = Field(description="Assumptions made and what this resolution does NOT cover")


RESOLVE_PROMPT = """You are a senior customer support agent writing a resolution
that will be stored in a knowledge graph and reused for future customers with
the same underlying problem - not just answering this one person.

Requirements:
- Generally reusable: do not reference this specific customer's personal
  details; describe the problem and fix in general terms.
- Structured and actionable: concrete steps, not vague advice.
- Accurate and grounded: do not invent system behavior you weren't told about.
- Explicit about escalation_required when the fix needs a human/engineer.
- Explicit in `limitations` about assumptions made and what this does NOT cover.
- Never include customer-specific facts (names, account ids, amounts) - those
  belong in the ticket, not in shared knowledge-graph resolutions."""


def resolve(query: str, entities: list[str], problem_summary: str) -> StructuredResolution:
    response = get_client().chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": RESOLVE_PROMPT},
            {
                "role": "user",
                "content": f"Problem summary: {problem_summary}\nKey entities: {', '.join(entities)}\nOriginal query (for context only, do not quote customer specifics): {query}",
            },
        ],
        response_format=StructuredResolution,
        temperature=0.3,
    )
    return response.choices[0].message.parsed


def render_resolution_text(resolution: StructuredResolution) -> str:
    """Single display/back-compat string, kept on Resolution.text."""
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(resolution.resolution_steps, start=1))
    parts = [
        f"Problem: {resolution.problem}",
        f"Diagnosis: {resolution.diagnosis}",
        f"Resolution steps:\n{steps}",
    ]
    if resolution.escalation_required:
        parts.append("Escalation required: yes")
    if resolution.limitations:
        parts.append(f"Limitations: {resolution.limitations}")
    return "\n\n".join(parts)
