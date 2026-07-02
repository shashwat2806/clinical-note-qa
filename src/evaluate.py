"""
evaluate.py
Runs the 32-question eval set through the RAG pipeline and reports accuracy.
"""

import csv
import json
from vectorstore import ClinicalVectorStore
from rag_qa import answer_question

EVAL_PATH = "../data/eval/eval_qa.csv"
OUTPUT_PATH = "../outputs/eval_results.json"


def run_evaluation():
    store = ClinicalVectorStore()

    with open(EVAL_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running evaluation on {len(rows)} questions...\n")

    results = []
    correct = 0

    for i, row in enumerate(rows):
        question = row["question"]
        expected = row["expected_answer"].lower()
        q_type = row["question_type"]

        result = answer_question(store, question)
        answer = result["answer"].lower()

        # Simple grading: does the answer contain key words from expected answer?
        # This is a keyword-overlap heuristic — not perfect, but honest and fast.
        expected_keywords = [w for w in expected.split() if len(w) > 4]
        matches = sum(1 for kw in expected_keywords if kw in answer)
        score = matches / len(expected_keywords) if expected_keywords else 0
        passed = score >= 0.3  # at least 30% keyword overlap

        if passed:
            correct += 1

        results.append({
            "question": question,
            "expected": row["expected_answer"],
            "got": result["answer"],
            "type": q_type,
            "passed": passed,
            "score": round(score, 2),
        })

        status = "✓" if passed else "✗"
        print(f"[{i+1:02d}] {status} ({q_type}) {question[:60]}")

    # Summary
    total = len(rows)
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1f}%")

    # Breakdown by question type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r["type"]]["total"] += 1
        if r["passed"]:
            by_type[r["type"]]["correct"] += 1

    print("\nBy question type:")
    for qtype, counts in sorted(by_type.items()):
        pct = counts["correct"] / counts["total"] * 100
        print(f"  {qtype:<20} {counts['correct']}/{counts['total']} = {pct:.0f}%")

    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(accuracy, 1),
            "correct": correct,
            "total": total,
            "by_type": {k: v for k, v in by_type.items()},
            "results": results,
        }, f, indent=2)

    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
