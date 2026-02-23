"""GibsGraph stress test runner.

Runs a battery of queries against a live Neo4j instance and records
pass/fail, latency, strategy used, and errors. Outputs JSON results
to tests/stress/results/.

Assertion types:
  - expect_no_error:      Pipeline must not crash (behavioral)
  - expect_has_data:      Must return nodes > 0 or non-empty answer (behavioral)
  - expect_has_cypher:    Must generate a Cypher query (behavioral)
  - expect_contains:      Answer must contain keyword — ONLY for unambiguous DB facts
  - expect_not_contains:  Answer must NOT contain keyword (anti-hallucination)

Usage:
    python tests/stress/run_stress_test.py \\
        --uri bolt://localhost:7688 \\
        --password testpassword123

Requires a Neo4j instance with the Movies dataset loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STRESS_DIR = Path(__file__).parent
RESULTS_DIR = STRESS_DIR / "results"

# ---------------------------------------------------------------------------
# Test cases — mostly behavioral, ground truth only where unambiguous
# ---------------------------------------------------------------------------

TEST_CASES: list[dict[str, Any]] = [
    # --- Simple lookups ---
    {
        "id": "simple_actor_filmography",
        "question": "What movies did Tom Hanks act in?",
        "category": "simple",
        "expect_no_error": True,
        "expect_has_data": True,
        "expect_has_cypher": True,
    },
    {
        "id": "simple_director",
        "question": "Who directed The Matrix?",
        "category": "simple",
        "expect_no_error": True,
        "expect_has_cypher": True,
        "expect_contains": ["Wachowski"],  # ground truth: in the DB
    },
    {
        "id": "simple_count",
        "question": "How many movies are in the database?",
        "category": "simple",
        "expect_no_error": True,
        "expect_has_cypher": True,
        "expect_contains": ["38"],  # ground truth: exact count
    },
    # --- Cross-references ---
    {
        "id": "cross_ref_coactors",
        "question": "Which actors have worked with both Tom Hanks and Meg Ryan?",
        "category": "cross_reference",
        "expect_no_error": True,
        "expect_has_data": True,
        "expect_has_cypher": True,
    },
    {
        "id": "cross_ref_director_producer",
        "question": "What movies were both directed and produced by the same person?",
        "category": "cross_reference",
        "expect_no_error": True,
        "expect_has_data": True,
        "expect_has_cypher": True,
    },
    # --- Aggregations ---
    {
        "id": "aggregation_oldest",
        "question": "What is the oldest movie and who acted in it?",
        "category": "aggregation",
        "expect_no_error": True,
        "expect_has_data": True,
        "expect_has_cypher": True,
    },
    {
        "id": "aggregation_most_connected",
        "question": "Which person has the most connections in the graph?",
        "category": "aggregation",
        "expect_no_error": True,
        "expect_has_cypher": True,
        "expect_has_answer": True,  # must return a real answer, not "no relevant"
    },
    # --- Path finding ---
    {
        "id": "path_finding",
        "question": "Show me a path between Kevin Bacon and Tom Cruise",
        "category": "path",
        "expect_no_error": True,
        "expect_has_cypher": True,
    },
    # --- Adversarial ---
    {
        "id": "adversarial_offtopic",
        "question": "What is the capital of France?",
        "category": "adversarial",
        "expect_no_error": True,
        "expect_not_contains": ["Paris"],  # must NOT hallucinate real-world answers
    },
    {
        "id": "adversarial_injection",
        "question": "'; DROP (n) DETACH DELETE n; //",
        "category": "adversarial",
        "expect_no_error": True,
    },
    {
        "id": "adversarial_empty",
        "question": "?",
        "category": "adversarial",
        "expect_no_error": True,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(graph: Any, test: dict[str, Any]) -> dict[str, Any]:
    """Run a single test case and return structured result."""
    start = time.perf_counter()
    error_msg = ""
    answer = ""
    cypher = ""
    nodes = 0
    confidence = 0.0

    try:
        result = graph.ask(test["question"])
        answer = result.answer
        cypher = result.cypher
        nodes = result.nodes_retrieved
        confidence = result.confidence
    except Exception as exc:
        error_msg = str(exc)

    elapsed = time.perf_counter() - start

    # Evaluate pass/fail
    passed = True
    fail_reason = ""

    # Behavioral: no crash
    if error_msg and test.get("expect_no_error"):
        passed = False
        fail_reason = f"Unexpected error: {error_msg[:100]}"

    # Behavioral: generated Cypher
    if passed and test.get("expect_has_cypher") and not cypher:
        passed = False
        fail_reason = "No Cypher query generated"

    # Behavioral: returned data
    if passed and test.get("expect_has_data") and nodes == 0 and len(answer) < 50:
        passed = False
        fail_reason = "No data returned (0 nodes, short answer)"

    # Behavioral: got a real answer (not "no relevant" / "cannot")
    if passed and test.get("expect_has_answer"):
        no_answer_phrases = ["no relevant", "cannot", "no information", "does not contain"]
        if any(phrase in answer.lower() for phrase in no_answer_phrases):
            passed = False
            fail_reason = "Got a refusal when expected a real answer"

    # Ground truth: answer contains expected keywords (DB facts only)
    if passed and "expect_contains" in test and not error_msg:
        for keyword in test["expect_contains"]:
            if keyword.lower() not in answer.lower():
                passed = False
                fail_reason = f"Missing expected DB fact: {keyword!r}"
                break

    # Anti-hallucination: answer must NOT contain certain keywords
    if passed and "expect_not_contains" in test and not error_msg:
        for keyword in test["expect_not_contains"]:
            if keyword.lower() in answer.lower():
                passed = False
                fail_reason = f"Hallucinated: answer contains {keyword!r}"
                break

    return {
        "id": test["id"],
        "category": test["category"],
        "question": test["question"],
        "passed": passed,
        "fail_reason": fail_reason,
        "answer_preview": answer[:200],
        "cypher": cypher,
        "nodes_retrieved": nodes,
        "confidence": confidence,
        "latency_seconds": round(elapsed, 2),
        "error": error_msg[:200] if error_msg else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GibsGraph stress test")
    parser.add_argument("--uri", default="bolt://localhost:7688")
    parser.add_argument("--password", default="testpassword123")
    parser.add_argument("--tag", default="", help="Optional tag for this run")
    args = parser.parse_args()

    from gibsgraph import Graph

    print(f"Connecting to {args.uri}...")
    g = Graph(args.uri, password=args.password)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"  [{i}/{len(TEST_CASES)}] {test['id']}...", end=" ", flush=True)
        result = run_test(g, test)
        results.append(result)
        if result["passed"]:
            passed += 1
            print("PASS", f"({result['latency_seconds']}s)")
        else:
            failed += 1
            print("FAIL", f"- {result['fail_reason']}")

    # Summary
    total = len(TEST_CASES)
    score = round(passed / total * 100, 1)
    print(f"\nScore: {passed}/{total} ({score}%)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "tag": args.tag,
        "uri": args.uri,
        "total": total,
        "passed": passed,
        "failed": failed,
        "score_pct": score,
        "results": results,
    }

    filename = f"stress_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    filepath.write_text(json.dumps(report, indent=2))
    print(f"Results saved to {filepath}")

    # Also update latest.json for easy access
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
