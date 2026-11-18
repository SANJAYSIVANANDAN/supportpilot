"""Preloads a few canonical issues (with real embeddings) so the first demo
queries can show the 'known case, reused' path immediately, not just
'novel' every time, and so semantic retrieval has something to find without
needing the backfill script first."""
from __future__ import annotations

from kg_agent import graph_store

SEED_ISSUES = [
    {
        "query": "I was charged twice for my monthly subscription",
        "entities": ["billing", "duplicate_charge", "subscription"],
        "category": "billing",
        "problem_summary": "Customer was charged more than once for the same subscription period.",
        "resolution_structured": {
            "problem": "Customer was charged more than once for the same subscription period.",
            "diagnosis": "Usually caused by a retried payment succeeding twice after a transient gateway timeout.",
            "resolution_steps": [
                "Check Billing > Transaction History for two charges on the same date.",
                "The duplicate charge is automatically refunded within 5-7 business days.",
                "If not reversed after a week, open a billing ticket with both transaction IDs.",
            ],
            "escalation_required": False,
            "limitations": "Assumes both charges are for the identical plan/amount; differing amounts need manual billing review.",
        },
    },
    {
        "query": "I can't log in, it says my password is incorrect even though I just reset it",
        "entities": ["login", "password_reset", "authentication"],
        "category": "login",
        "problem_summary": "Customer's new password is rejected immediately after a password reset.",
        "resolution_structured": {
            "problem": "A freshly reset password is rejected as incorrect at login.",
            "diagnosis": "Almost always a cached old session or a password manager autofilling the previous password, or an expired/already-used reset link.",
            "resolution_steps": [
                "Clear the login form and type the new password manually (avoid autofill).",
                "Confirm caps lock is off.",
                "If it still fails, request a fresh reset link - links expire after 30 minutes and silently fail on re-use.",
            ],
            "escalation_required": False,
            "limitations": "Does not cover account lockouts after repeated failed attempts, which need a separate unlock flow.",
        },
    },
    {
        "query": "Exporting my report to CSV times out on large datasets",
        "entities": ["csv_export", "timeout", "large_dataset"],
        "category": "data_export",
        "problem_summary": "CSV export requests time out when the dataset being exported is large.",
        "resolution_structured": {
            "problem": "CSV export times out for large datasets.",
            "diagnosis": "Exports over roughly 50k rows can exceed the synchronous request timeout.",
            "resolution_steps": [
                "Use the 'Export as background job' option instead of the inline export.",
                "An email with a download link is sent once the background job finishes.",
            ],
            "escalation_required": False,
            "limitations": "Assumes the background export feature is enabled for the customer's plan tier.",
        },
    },
]


def run() -> None:
    graph_store.init_constraints()
    for item in SEED_ISSUES:
        embedding = graph_store.generate_issue_embedding(item["problem_summary"])
        resolution_text = "\n\n".join(
            [
                f"Problem: {item['resolution_structured']['problem']}",
                f"Diagnosis: {item['resolution_structured']['diagnosis']}",
                "Resolution steps:\n" + "\n".join(
                    f"{i}. {s}" for i, s in enumerate(item["resolution_structured"]["resolution_steps"], start=1)
                ),
                f"Limitations: {item['resolution_structured']['limitations']}",
            ]
        )
        graph_store.create_issue(
            query=item["query"],
            entities=item["entities"],
            category=item["category"],
            problem_summary=item["problem_summary"],
            embedding=embedding,
            resolution_text=resolution_text,
            resolution_structured=item["resolution_structured"],
        )
    graph_store.create_vector_index()
    print(f"Seeded {len(SEED_ISSUES)} issues. Stats: {graph_store.get_stats()}")


if __name__ == "__main__":
    run()
