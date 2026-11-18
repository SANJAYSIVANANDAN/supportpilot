"""Evaluates the hybrid retrieval + verification pipeline against a labeled
dataset, and reports precision/recall/F1 so NOVELTY_THRESHOLD/
MIN_CANDIDATE_SCORE can be calibrated from evidence rather than guessed.

The primary metric this optimizes for is the *reuse* decision, since an
incorrect automatic reuse (false positive) is worse than an unnecessary new
Issue (false negative) - a duplicate Issue just costs some graph density,
a bad reuse gives the customer the wrong answer. "Positive" here means
"the pipeline decided SAME_ISSUE".

Usage:
    python -m kg_agent.eval.run_eval [path/to/eval_dataset.jsonl]

Requires a seeded graph (run `python -m kg_agent.seed` first) plus
NEO4J_*/OPENAI_API_KEY configured, since this exercises the real pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kg_agent import brain

DEFAULT_DATASET = Path(__file__).parent / "eval_dataset.jsonl"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(path: Path = DEFAULT_DATASET) -> None:
    rows = load_dataset(path)

    # "Positive" = pipeline said SAME_ISSUE (i.e. it would auto-reuse).
    tp = fp = tn = fn = 0
    mismatches = []

    for row in rows:
        result = brain.handle_query(row["query"])
        predicted = result.get("verification_decision") or "NEW_ISSUE"  # None means no-candidates -> straight novel
        expected = row["expected_decision"]

        predicted_reuse = predicted == "SAME_ISSUE"
        expected_reuse = expected == "SAME_ISSUE"

        if predicted_reuse and expected_reuse:
            tp += 1
        elif predicted_reuse and not expected_reuse:
            fp += 1
        elif not predicted_reuse and not expected_reuse:
            tn += 1
        else:
            fn += 1

        if predicted != expected:
            mismatches.append(
                {"query": row["query"], "expected": expected, "predicted": predicted, "notes": row.get("notes")}
            )

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    print(f"Rows evaluated: {len(rows)}")
    print(f"Reuse decision: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"False-positive rate (unsafe auto-reuse): {fp}/{len(rows)}")
    print(f"False-negative rate (missed valid reuse): {fn}/{len(rows)}")

    if mismatches:
        print(f"\n{len(mismatches)} exact-decision mismatch(es) (includes non-reuse confusions):")
        for m in mismatches:
            print(f"  - {m['query']!r}: expected={m['expected']} got={m['predicted']} ({m['notes']})")


if __name__ == "__main__":
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    run(dataset_path)
