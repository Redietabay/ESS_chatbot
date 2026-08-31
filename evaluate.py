"""
Automated answer-quality evaluation for the ESS Chatbot RAG pipeline.

WHY THIS EXISTS: up to now, answer quality was checked manually — ask a
question in the browser, eyeball the answer, maybe take a screenshot. That
doesn't scale and doesn't catch regressions (e.g. a change to chunking or
the year-filter logic silently breaking a question that worked last week).
This script runs a small fixed "gold set" of real questions with known-
correct expected facts through the exact same get_answer() the Flask app
uses, and reports which ones still pass.

USAGE:
    python evaluate.py                  # run the whole gold set
    python evaluate.py --category inflation   # only questions tagged with this category
    python evaluate.py --save results.json    # also write full results to a file

HOW MATCHING WORKS: each gold-set item lists one or more `expect` keywords
that MUST all appear (case-insensitive substring match) somewhere in the
answer for it to count as a PASS. This is intentionally simple — no LLM
grading, no embeddings — because a plain substring check is transparent
enough to trust in a live demo, and "does the actual number/keyword show
up" is the failure mode that has actually happened in this project so far
(wrong year, wrong month, "not indexed" for something that IS indexed).

EXTENDING THE GOLD SET: add new dicts to GOLD_SET below as new report
categories get indexed. Keep `expect` short and specific — a literal
figure ("33.8%"), a place name, a report title fragment — not a full
sentence, since exact answer phrasing can vary between runs.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

# Importing rag triggers the same startup as the Flask app (DB pool,
# ChromaDB client, embedding model load) — run this from the project root,
# with the same .env the app uses, not from inside a fresh/empty venv.
from rag import get_answer


GOLD_SET = [
    {
        "id": "inflation_dec_2022",
        "category": "inflation",
        "question": "What was the inflation rate in December 2022?",
        "expect": ["33.8"],
    },
    {
        "id": "inflation_july_2023",
        "category": "inflation",
        "question": "What was the inflation rate in July 2023?",
        "expect": ["33.5"],
    },
    # ── Add more as they're validated against a real indexed report ──
    # {
    #     "id": "population_total",
    #     "category": "population",
    #     "question": "What is the population of Ethiopia?",
    #     "expect": ["million"],   # replace with an actual figure once verified
    # },
    # {
    #     "id": "land_use_amhara",
    #     "category": "land",
    #     "question": "What is the land utilization in Amhara region?",
    #     "expect": ["hectare"],
    # },
]


def run_case(case: dict) -> dict:
    t0 = time.time()
    try:
        result = get_answer(case["question"]) or {}
        answer = (result.get("answer") or "").strip()
        error = None
    except Exception as e:
        answer = ""
        error = str(e)
        result = {}

    elapsed = round(time.time() - t0, 2)
    answer_lower = answer.lower()
    missing = [kw for kw in case["expect"] if kw.lower() not in answer_lower]
    passed = error is None and not missing and bool(answer)

    return {
        "id": case["id"],
        "category": case.get("category", "general"),
        "question": case["question"],
        "expected": case["expect"],
        "missing": missing,
        "passed": passed,
        "error": error,
        "elapsed_s": elapsed,
        "source": result.get("source"),
        "page": result.get("page"),
        "route": result.get("route"),
        "answer_preview": answer[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the ESS Chatbot answer-quality gold set.")
    parser.add_argument("--category", help="Only run cases tagged with this category.")
    parser.add_argument("--save", help="Write full JSON results to this path.")
    args = parser.parse_args()

    cases = GOLD_SET
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
        if not cases:
            print(f"No gold-set cases tagged category='{args.category}'.")
            sys.exit(1)

    print(f"Running {len(cases)} gold-set question(s)...\n")
    results = [run_case(c) for c in cases]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"{'ID':<24} {'Status':<6} {'Time':>6}  Question")
    print("-" * 80)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<24} {status:<6} {r['elapsed_s']:>5}s  {r['question'][:60]}")
        if not r["passed"]:
            if r["error"]:
                print(f"    error: {r['error']}")
            if r["missing"]:
                print(f"    missing expected keyword(s): {r['missing']}")
            if r["answer_preview"]:
                print(f"    got: {r['answer_preview']}")

    print("-" * 80)
    rate = round(100 * passed / total, 1) if total else 0.0
    print(f"\n{passed}/{total} passed ({rate}%)")

    if args.save:
        payload = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "total": total,
            "pass_rate": rate,
            "results": results,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Full results written to {args.save}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
