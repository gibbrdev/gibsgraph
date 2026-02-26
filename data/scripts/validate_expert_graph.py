"""Expert graph data quality checker.

Queries the live Neo4j expert graph directly for actual data quality
issues. This is NOT closed-loop validation (we don't run our own grading
system on our own homework). Instead, we query the database for real
problems: null fields, orphan nodes, missing relationships, coverage gaps.

Usage:
  python data/scripts/validate_expert_graph.py [--uri bolt://localhost:7687] [--password ...]
"""

from __future__ import annotations

import argparse
import sys

from neo4j import GraphDatabase


def check_data_quality(session: object) -> dict[str, dict]:
    """Run data quality checks against the expert graph.

    Returns a dict of {check_name: {expected, actual, passed, detail}}.
    """
    checks = {}

    # --- Coverage: do we have enough data? ---

    result = session.run("MATCH (c:CypherClause) RETURN count(c) AS total")
    total = result.single()["total"]
    checks["clause_count"] = {
        "expected": ">=30",
        "actual": total,
        "passed": total >= 30,
        "detail": "Missing common clauses" if total < 30 else "",
    }

    result = session.run("MATCH (f:CypherFunction) RETURN count(f) AS total")
    total = result.single()["total"]
    checks["function_count"] = {
        "expected": ">=100",
        "actual": total,
        "passed": total >= 100,
        "detail": "",
    }

    result = session.run("MATCH (ex:CypherExample) RETURN count(ex) AS total")
    total = result.single()["total"]
    checks["example_count"] = {
        "expected": ">=200",
        "actual": total,
        "passed": total >= 200,
        "detail": "",
    }

    result = session.run("MATCH (bp:BestPractice) RETURN count(bp) AS total")
    total = result.single()["total"]
    checks["best_practice_count"] = {
        "expected": ">=100",
        "actual": total,
        "passed": total >= 100,
        "detail": "",
    }

    result = session.run("MATCH (mp:ModelingPattern) RETURN count(mp) AS total")
    total = result.single()["total"]
    checks["modeling_pattern_count"] = {
        "expected": ">=10",
        "actual": total,
        "passed": total >= 10,
        "detail": "",
    }

    # --- Null/empty required fields ---

    result = session.run(
        "MATCH (c:CypherClause) WHERE c.description IS NULL OR c.description = '' "
        "RETURN count(c) AS nulls"
    )
    nulls = result.single()["nulls"]
    checks["clauses_with_description"] = {
        "expected": "0 nulls",
        "actual": nulls,
        "passed": nulls == 0,
        "detail": f"{nulls} clauses missing description" if nulls > 0 else "",
    }

    result = session.run(
        "MATCH (f:CypherFunction) WHERE f.signature IS NULL OR f.signature = '' "
        "RETURN count(f) AS nulls"
    )
    nulls = result.single()["nulls"]
    checks["functions_with_signature"] = {
        "expected": "0 nulls",
        "actual": nulls,
        "passed": nulls == 0,
        "detail": f"{nulls} functions missing signature" if nulls > 0 else "",
    }

    result = session.run(
        "MATCH (ex:CypherExample) WHERE ex.cypher IS NULL OR ex.cypher = '' "
        "RETURN count(ex) AS nulls"
    )
    nulls = result.single()["nulls"]
    checks["examples_with_cypher"] = {
        "expected": "0 nulls",
        "actual": nulls,
        "passed": nulls == 0,
        "detail": f"{nulls} examples missing cypher" if nulls > 0 else "",
    }

    result = session.run(
        "MATCH (bp:BestPractice) WHERE bp.title IS NULL OR bp.title = '' "
        "RETURN count(bp) AS nulls"
    )
    nulls = result.single()["nulls"]
    checks["practices_with_title"] = {
        "expected": "0 nulls",
        "actual": nulls,
        "passed": nulls == 0,
        "detail": f"{nulls} best practices missing title" if nulls > 0 else "",
    }

    # --- Orphan nodes (no relationships at all) ---

    result = session.run(
        "MATCH (ex:CypherExample) WHERE NOT (ex)-[:DEMONSTRATES]->() "
        "RETURN count(ex) AS orphans"
    )
    orphans = result.single()["orphans"]
    result2 = session.run("MATCH (ex:CypherExample) RETURN count(ex) AS total")
    total = result2.single()["total"]
    checks["orphan_examples"] = {
        "expected": "<50% orphans",
        "actual": f"{orphans}/{total}",
        "passed": total == 0 or (orphans / total) < 0.5,
        "detail": f"{orphans} examples not linked via DEMONSTRATES" if orphans > 0 else "",
    }

    result = session.run(
        "MATCH (c:CypherClause) WHERE NOT (c)-[:SOURCED_FROM]->() "
        "RETURN count(c) AS orphans"
    )
    orphans = result.single()["orphans"]
    result2 = session.run("MATCH (c:CypherClause) RETURN count(c) AS total")
    total = result2.single()["total"]
    checks["clauses_with_source"] = {
        "expected": "0 orphans",
        "actual": f"{orphans}/{total}",
        "passed": orphans == 0,
        "detail": f"{orphans} clauses not linked to a Source node" if orphans > 0 else "",
    }

    result = session.run(
        "MATCH (bp:BestPractice) WHERE NOT (bp)-[:SOURCED_FROM]->() "
        "RETURN count(bp) AS orphans"
    )
    orphans = result.single()["orphans"]
    result2 = session.run("MATCH (bp:BestPractice) RETURN count(bp) AS total")
    total = result2.single()["total"]
    checks["practices_with_source"] = {
        "expected": "0 orphans",
        "actual": f"{orphans}/{total}",
        "passed": orphans == 0,
        "detail": f"{orphans} best practices not linked to a Source node" if orphans > 0 else "",
    }

    # --- Relationship integrity ---

    result = session.run(
        "MATCH (ex:CypherExample)-[:DEMONSTRATES]->(target) "
        "RETURN count(ex) AS linked, labels(target)[0] AS target_type "
        "ORDER BY linked DESC LIMIT 5"
    )
    demonstrates_rows = list(result)
    total_demonstrates = sum(r["linked"] for r in demonstrates_rows)
    checks["demonstrates_relationships"] = {
        "expected": ">0",
        "actual": total_demonstrates,
        "passed": total_demonstrates > 0,
        "detail": ", ".join(f"{r['target_type']}={r['linked']}" for r in demonstrates_rows),
    }

    result = session.run(
        "MATCH ()-[r:SOURCED_FROM]->() RETURN count(r) AS total"
    )
    sourced = result.single()["total"]
    checks["sourced_from_relationships"] = {
        "expected": ">0",
        "actual": sourced,
        "passed": sourced > 0,
        "detail": "",
    }

    # --- Distribution: category spread ---

    result = session.run(
        "MATCH (bp:BestPractice)-[:BELONGS_TO]->(cat:PracticeCategory) "
        "RETURN cat.name AS category, count(bp) AS cnt ORDER BY cnt DESC"
    )
    categories = {r["category"]: r["cnt"] for r in result}
    total_categorized = sum(categories.values())
    result2 = session.run("MATCH (bp:BestPractice) RETURN count(bp) AS total")
    total_bp = result2.single()["total"]
    uncategorized = total_bp - total_categorized
    checks["practice_categorization"] = {
        "expected": "100% categorized",
        "actual": f"{total_categorized}/{total_bp} categorized",
        "passed": uncategorized == 0,
        "detail": f"Categories: {categories}" if categories else "No categories found",
    }

    # --- Index and constraint existence ---

    result = session.run("SHOW INDEXES YIELD name RETURN count(name) AS cnt")
    idx_count = result.single()["cnt"]
    checks["indexes_exist"] = {
        "expected": ">=3",
        "actual": idx_count,
        "passed": idx_count >= 3,
        "detail": "",
    }

    result = session.run("SHOW CONSTRAINTS YIELD name RETURN count(name) AS cnt")
    constraint_count = result.single()["cnt"]
    checks["constraints_exist"] = {
        "expected": ">=5",
        "actual": constraint_count,
        "passed": constraint_count >= 5,
        "detail": "",
    }

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check expert graph data quality in Neo4j"
    )
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="neo4j")
    args = parser.parse_args()

    print("=" * 60)
    print("Expert Graph — Data Quality Report")
    print("=" * 60)

    auth = (args.username, args.password) if args.password else None
    driver = GraphDatabase.driver(args.uri, auth=auth)

    try:
        with driver.session(database=args.database) as session:
            # Quick connectivity check
            result = session.run("MATCH (n) RETURN count(n) AS total")
            total = result.single()["total"]
            print(f"\nConnected: {total} nodes in database")

            if total == 0:
                print("\nERROR: No nodes found. Load the expert graph first.")
                print("  python data/scripts/load_expert_graph.py --password <pw>")
                sys.exit(1)

            checks = check_data_quality(session)

        # Print results
        passed = sum(1 for c in checks.values() if c["passed"])
        failed = len(checks) - passed

        print(f"\nResults: {passed}/{len(checks)} checks passed\n")

        for name, check in checks.items():
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {name}")
            print(f"         expected: {check['expected']}, actual: {check['actual']}")
            if check["detail"]:
                print(f"         {check['detail']}")

        print("\n" + "=" * 60)
        if failed == 0:
            print(f"ALL {len(checks)} CHECKS PASSED")
        else:
            print(f"{failed} CHECK(S) FAILED — review data quality issues above")

        sys.exit(0 if failed == 0 else 1)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
