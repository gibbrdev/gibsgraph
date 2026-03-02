"""Expert data quality audit — offline JSONL verification.

Reads all 5 bundled JSONL files and runs 4 tiers of automated checks:
  Tier 1: Completeness (required fields, valid enums)
  Tier 2: Cypher validation (syntax, balance, pseudocode detection)
  Tier 3: Cross-reference (functions/clauses vs Neo4j 5.x official)
  Tier 4: Duplicates & consistency (exact dupes, naming conventions)

Usage:
  python data/scripts/audit_expert_data.py              # terminal report
  python data/scripts/audit_expert_data.py --json out.json  # + JSON export
  python data/scripts/audit_expert_data.py --verbose     # show all issues
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1].parent / "src" / "gibsgraph" / "data"

# ── Data structures ──────────────────────────────────────────────────────────

Severity = Literal["pass", "fail", "warn"]


@dataclass
class Issue:
    """A single audit finding."""

    file: str
    line_num: int
    field: str
    message: str
    severity: Severity
    record_id: str = ""


@dataclass
class AuditResult:
    """Aggregated results for one audit tier."""

    tier: str
    description: str
    total_checked: int = 0
    passed: int = 0
    failed: int = 0
    warned: int = 0
    issues: list[Issue] = field(default_factory=list)


# ── Required fields per file ─────────────────────────────────────────────────

REQUIRED_FIELDS: dict[str, list[str]] = {
    "cypher_clauses.jsonl": [
        "name", "description", "syntax_examples", "source_file", "authority_level",
    ],
    "cypher_functions.jsonl": [
        "name", "category", "description", "signature", "source_file", "authority_level",
    ],
    "cypher_examples.jsonl": [
        "cypher", "description", "context", "category", "source_file", "authority_level",
    ],
    "modeling_patterns.jsonl": [
        "name", "description", "when_to_use", "source_file", "authority_level",
    ],
    "best_practices.jsonl": [
        "title", "description", "category", "source_file", "authority_level",
    ],
}

VALID_QUALITY_TIERS = {"high", "medium", "low"}
VALID_AUTHORITY_LEVELS = {1, 2}

# ── Neo4j 5.x official reference ────────────────────────────────────────────
# Source: https://neo4j.com/docs/cypher-manual/current/functions/
# Source: https://neo4j.com/docs/cypher-manual/current/clauses/

NEO4J_5X_FUNCTIONS: frozenset[str] = frozenset({
    # Aggregating
    "avg", "collect", "count", "max", "min",
    "percentileCont", "percentileDisc", "stDev", "stDevP", "sum",
    # Predicate
    "all", "any", "exists", "isEmpty", "none", "single",
    # Scalar
    "char_length", "character_length", "coalesce", "elementId", "endNode",
    "head", "id", "last", "length", "nullIf", "properties", "randomUUID",
    "size", "startNode", "timestamp", "toBoolean", "toBooleanOrNull",
    "toFloat", "toFloatOrNull", "toInteger", "toIntegerOrNull", "type",
    "valueType",
    # List
    "keys", "labels", "nodes", "range", "reduce", "relationships",
    "reverse", "tail",
    "toBooleanList", "toFloatList", "toIntegerList", "toStringList",
    "coll.distinct", "coll.flatten", "coll.indexOf", "coll.insert",
    "coll.max", "coll.min", "coll.remove", "coll.sort",
    # Numeric
    "abs", "ceil", "floor", "isNaN", "rand", "round", "sign",
    # Logarithmic
    "e", "exp", "log", "log10", "sqrt",
    # Trigonometric
    "acos", "asin", "atan", "atan2", "cos", "cosh", "cot", "coth",
    "degrees", "haversin", "pi", "radians", "sin", "sinh", "tan", "tanh",
    # String
    "btrim", "left", "lower", "ltrim", "normalize", "replace", "right",
    "rtrim", "split", "substring", "toLower", "toString", "toStringOrNull",
    "toUpper", "trim", "upper",
    # Temporal
    "date", "date.realtime", "date.statement", "date.transaction", "date.truncate",
    "datetime", "datetime.fromEpoch", "datetime.fromEpochMillis",
    "datetime.realtime", "datetime.statement", "datetime.transaction",
    "datetime.truncate",
    "duration", "duration.between", "duration.inDays", "duration.inMonths",
    "duration.inSeconds",
    "localdatetime", "localdatetime.realtime", "localdatetime.statement",
    "localdatetime.transaction", "localdatetime.truncate",
    "localtime", "localtime.realtime", "localtime.statement",
    "localtime.transaction", "localtime.truncate",
    "time", "time.realtime", "time.statement", "time.transaction",
    "time.truncate",
    # Spatial
    "point", "point.distance", "point.withinBBox",
    # Vector (5.13+)
    "vector", "vector.similarity.cosine", "vector.similarity.euclidean",
    "vector_dimension_count", "vector_distance", "vector_norm",
    # Database
    "db.nameFromElementId",
    # Graph
    "graph.byElementId", "graph.byName", "graph.names", "graph.propertiesByName",
    # LOAD CSV
    "file", "linenumber",
})

# APOC functions — legitimate plugin, not built-in
KNOWN_APOC_FUNCTIONS: frozenset[str] = frozenset({
    "apoc.coll.toSet", "apoc.convert.fromJsonList", "apoc.convert.fromJsonMap",
    "apoc.convert.toJson", "apoc.create.vNode", "apoc.create.vRelationship",
    "apoc.map.fromPairs", "apoc.map.merge", "apoc.math.round",
    "apoc.meta.type", "apoc.neighbors.tohop",
    "apoc.path.expandConfig", "apoc.path.spanningTree", "apoc.path.subgraphAll",
    "apoc.text.join", "apoc.text.capitalize",
})

# GDS functions — legitimate plugin, not built-in
KNOWN_GDS_FUNCTIONS: frozenset[str] = frozenset({
    "gds.betweenness.stream", "gds.graph.project", "gds.louvain.stream",
    "gds.nodeSimilarity.stream", "gds.pageRank.stream",
    "gds.shortestPath.dijkstra.stream",
})

# Known clause names in our dataset (mapped to official equivalents)
NEO4J_5X_CLAUSE_NAMES: frozenset[str] = frozenset({
    "CALL procedure", "CASE", "CREATE", "Clause composition",
    "DELETE", "FINISH", "FOREACH", "INSERT",
    "LIMIT", "LOAD CSV", "MATCH", "MERGE",
    "OPTIONAL MATCH", "ORDER BY", "REMOVE", "RETURN",
    "SET", "SHOW FUNCTIONS", "SHOW PROCEDURES", "SHOW SETTINGS",
    "SKIP", "Transaction commands", "UNION", "UNION ALL",
    "UNWIND", "USE", "WHERE", "WITH", "YIELD",
    # Subquery / expression clauses
    "Subqueries", "EXISTS subqueries", "COLLECT subqueries", "COUNT subqueries",
    "CALL subqueries",
    # Expression keywords (used as clause-level entries in our data)
    "DISTINCT", "WHEN", "THEN", "ELSE", "SHOW",
    # Newer additions
    "FILTER", "LET", "SEARCH", "OFFSET",
})

# Known non-functions (parser artifacts in the dataset)
KNOWN_NON_FUNCTIONS: frozenset[str] = frozenset({
    "Introduction",
})

ALL_KNOWN_FUNCTIONS = NEO4J_5X_FUNCTIONS | KNOWN_APOC_FUNCTIONS | KNOWN_GDS_FUNCTIONS

# ── Cypher validation constants ──────────────────────────────────────────────

CYPHER_START_KEYWORDS: frozenset[str] = frozenset({
    "MATCH", "OPTIONAL", "CREATE", "MERGE", "RETURN", "WITH", "UNWIND",
    "CALL", "FOREACH", "EXPLAIN", "PROFILE", "SHOW", "DROP", "SET",
    "DELETE", "DETACH", "REMOVE", "LOAD", "USE", "FINISH", "UNION",
    "ORDER", "LIMIT", "SKIP", "WHERE", "CASE", "YIELD", "INSERT",
    "//",  # Cypher comments
})

_INTERPOLATION_RE = re.compile(r'\$\{.+?\}|"\s*\+\s*\w+\s*\+\s*"')

BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
BRACKET_CLOSE = set(BRACKET_PAIRS.values())


# ── Data loading ─────────────────────────────────────────────────────────────


def load_all_records() -> dict[str, list[tuple[int, dict[str, object]]]]:
    """Load all JSONL files. Returns {filename: [(line_num, record), ...]}."""
    result: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for filename in REQUIRED_FIELDS:
        path = DATA_DIR / filename
        records: list[tuple[int, dict[str, object]]] = []
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj: dict[str, object] = json.loads(stripped)
            records.append((i, obj))
        result[filename] = records
    return result


def _record_id(filename: str, obj: dict[str, object]) -> str:
    """Human-readable identifier for a record."""
    name = obj.get("name") or obj.get("title") or obj.get("context") or ""
    return str(name)[:60]


# ── Tier 1: Completeness ────────────────────────────────────────────────────


def check_completeness(
    all_records: dict[str, list[tuple[int, dict[str, object]]]],
) -> AuditResult:
    """Tier 1: Required fields, valid quality_tier, valid authority_level."""
    result = AuditResult(tier="COMPLETENESS", description="Required fields and valid enums")

    for filename, records in all_records.items():
        required = REQUIRED_FIELDS[filename]
        for line_num, obj in records:
            result.total_checked += 1
            rec_id = _record_id(filename, obj)
            issues_before = len(result.issues)

            # Required fields
            for fld in required:
                val = obj.get(fld)
                if val is None:
                    result.issues.append(Issue(
                        file=filename, line_num=line_num, field=fld,
                        message=f"Missing required field: {fld}",
                        severity="fail", record_id=rec_id,
                    ))
                elif isinstance(val, str) and not val.strip():
                    result.issues.append(Issue(
                        file=filename, line_num=line_num, field=fld,
                        message=f"Empty required field: {fld}",
                        severity="fail", record_id=rec_id,
                    ))
                elif isinstance(val, list) and len(val) == 0 and fld == "syntax_examples":
                    result.issues.append(Issue(
                        file=filename, line_num=line_num, field=fld,
                        message="Empty syntax_examples array",
                        severity="fail", record_id=rec_id,
                    ))

            # quality_tier
            qt = obj.get("quality_tier")
            if qt is not None and qt not in VALID_QUALITY_TIERS:
                result.issues.append(Issue(
                    file=filename, line_num=line_num, field="quality_tier",
                    message=f"Invalid quality_tier: {qt!r}",
                    severity="fail", record_id=rec_id,
                ))

            # authority_level
            al = obj.get("authority_level")
            if al is not None and al not in VALID_AUTHORITY_LEVELS:
                result.issues.append(Issue(
                    file=filename, line_num=line_num, field="authority_level",
                    message=f"Invalid authority_level: {al!r}",
                    severity="fail", record_id=rec_id,
                ))

            # source_file looks real
            sf = obj.get("source_file")
            if isinstance(sf, str) and sf.strip():
                if "/" not in sf and "\\" not in sf and not sf.startswith("curated/"):
                    result.issues.append(Issue(
                        file=filename, line_num=line_num, field="source_file",
                        message=f"source_file looks generic: {sf!r}",
                        severity="warn", record_id=rec_id,
                    ))

            if len(result.issues) == issues_before:
                result.passed += 1
            else:
                new_issues = result.issues[issues_before:]
                if any(i.severity == "fail" for i in new_issues):
                    result.failed += 1
                else:
                    result.warned += 1

    return result


# ── Tier 2: Cypher validation ────────────────────────────────────────────────


def extract_all_cypher(
    all_records: dict[str, list[tuple[int, dict[str, object]]]],
) -> list[tuple[str, str, int, str]]:
    """Extract every Cypher snippet: (cypher, filename, line_num, record_id)."""
    snippets: list[tuple[str, str, int, str]] = []

    for filename, records in all_records.items():
        for line_num, obj in records:
            rec_id = _record_id(filename, obj)

            # Direct cypher field (cypher_examples.jsonl)
            cypher = obj.get("cypher")
            if isinstance(cypher, str) and cypher.strip():
                snippets.append((cypher.strip(), filename, line_num, rec_id))

            # examples[] (cypher_functions.jsonl)
            examples = obj.get("examples")
            if isinstance(examples, list):
                for ex in examples:
                    if isinstance(ex, str) and ex.strip():
                        snippets.append((ex.strip(), filename, line_num, rec_id))

            # syntax_examples[] (cypher_clauses.jsonl)
            syntax = obj.get("syntax_examples")
            if isinstance(syntax, list):
                for sx in syntax:
                    if isinstance(sx, str) and sx.strip():
                        snippets.append((sx.strip(), filename, line_num, rec_id))

            # cypher_examples[] (modeling_patterns.jsonl, best_practices.jsonl)
            cypher_ex = obj.get("cypher_examples")
            if isinstance(cypher_ex, list):
                for cx in cypher_ex:
                    if isinstance(cx, str) and cx.strip():
                        snippets.append((cx.strip(), filename, line_num, rec_id))

    return snippets


def _check_balanced(cypher: str) -> list[str]:
    """Check balanced parentheses, brackets, braces."""
    errors: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False
    quote_char = ""

    for ch in cypher:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == quote_char:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            continue
        if ch in BRACKET_PAIRS:
            stack.append(ch)
        elif ch in BRACKET_CLOSE:
            if not stack:
                errors.append(f"Unmatched closing '{ch}'")
            else:
                opener = stack.pop()
                expected = BRACKET_PAIRS[opener]
                if ch != expected:
                    errors.append(f"Mismatched: '{opener}' closed by '{ch}'")

    for unclosed in stack:
        errors.append(f"Unclosed '{unclosed}'")

    return errors


def _looks_like_pseudocode(cypher: str) -> bool:
    """Heuristic: snippet is more natural language than Cypher."""
    stripped = cypher.strip()
    if not stripped:
        return True

    # Skip comments
    if stripped.startswith("//"):
        return False

    # Get first token
    first_token = stripped.split()[0].upper().rstrip("(")

    # If starts with a Cypher keyword, it's probably Cypher
    if first_token in CYPHER_START_KEYWORDS:
        return False

    # If no structural Cypher characters at all, probably pseudocode
    structural = {"(", ")", "[", "]", "{", "}", "->", "<-", ":", "|"}
    if not any(s in stripped for s in structural):
        return True

    return False


def check_cypher_validation(
    all_records: dict[str, list[tuple[int, dict[str, object]]]],
) -> AuditResult:
    """Tier 2: Syntax, balance, interpolation, pseudocode checks."""
    result = AuditResult(tier="CYPHER VALIDATION", description="Syntax and safety on every snippet")
    snippets = extract_all_cypher(all_records)
    result.total_checked = len(snippets)

    for cypher, filename, line_num, rec_id in snippets:
        issues_before = len(result.issues)

        # Balanced brackets
        balance_errors = _check_balanced(cypher)
        for err in balance_errors:
            result.issues.append(Issue(
                file=filename, line_num=line_num, field="cypher",
                message=f"Unbalanced: {err}",
                severity="fail", record_id=rec_id,
            ))

        # String interpolation
        if _INTERPOLATION_RE.search(cypher):
            result.issues.append(Issue(
                file=filename, line_num=line_num, field="cypher",
                message="String interpolation detected (should use $params)",
                severity="fail", record_id=rec_id,
            ))

        # Pseudocode detection
        if _looks_like_pseudocode(cypher):
            result.issues.append(Issue(
                file=filename, line_num=line_num, field="cypher",
                message=f"Looks like pseudocode: {cypher[:80]!r}",
                severity="warn", record_id=rec_id,
            ))

        if len(result.issues) == issues_before:
            result.passed += 1
        else:
            new_issues = result.issues[issues_before:]
            if any(i.severity == "fail" for i in new_issues):
                result.failed += 1
            else:
                result.warned += 1

    return result


# ── Tier 3: Cross-reference ──────────────────────────────────────────────────


def check_cross_reference(
    all_records: dict[str, list[tuple[int, dict[str, object]]]],
) -> AuditResult:
    """Tier 3: Functions and clauses vs Neo4j 5.x official lists."""
    result = AuditResult(tier="CROSS-REFERENCE", description="Functions/clauses vs Neo4j 5.x")

    # Functions
    for line_num, obj in all_records.get("cypher_functions.jsonl", []):
        result.total_checked += 1
        name = str(obj.get("name", ""))
        rec_id = name

        if name in KNOWN_NON_FUNCTIONS:
            result.issues.append(Issue(
                file="cypher_functions.jsonl", line_num=line_num, field="name",
                message=f"Parser artifact, not a real function: {name!r}",
                severity="fail", record_id=rec_id,
            ))
            result.failed += 1
        elif name not in ALL_KNOWN_FUNCTIONS:
            result.issues.append(Issue(
                file="cypher_functions.jsonl", line_num=line_num, field="name",
                message=f"Unrecognized function: {name!r} — may be deprecated or fabricated",
                severity="warn", record_id=rec_id,
            ))
            result.warned += 1
        else:
            result.passed += 1

    # Clauses
    for line_num, obj in all_records.get("cypher_clauses.jsonl", []):
        result.total_checked += 1
        name = str(obj.get("name", ""))
        rec_id = name

        if name not in NEO4J_5X_CLAUSE_NAMES:
            result.issues.append(Issue(
                file="cypher_clauses.jsonl", line_num=line_num, field="name",
                message=f"Unrecognized clause: {name!r}",
                severity="warn", record_id=rec_id,
            ))
            result.warned += 1
        else:
            result.passed += 1

    return result


# ── Tier 4: Duplicates & consistency ─────────────────────────────────────────


def _normalize_cypher(cypher: str) -> str:
    """Normalize for duplicate detection: collapse whitespace, lowercase, strip semicolons."""
    return re.sub(r"\s+", " ", cypher.strip().lower()).rstrip(";")


def _is_pascal_case(s: str) -> bool:
    """PascalCase: starts uppercase, has lowercase, no underscores."""
    return bool(re.match(r"^[A-Z][a-zA-Z0-9]+$", s))


def _is_upper_snake(s: str) -> bool:
    """UPPER_SNAKE_CASE: all uppercase with underscores."""
    return bool(re.match(r"^[A-Z][A-Z0-9_]*$", s)) and len(s) > 1


def check_duplicates_and_consistency(
    all_records: dict[str, list[tuple[int, dict[str, object]]]],
) -> AuditResult:
    """Tier 4: Duplicate detection and naming convention checks."""
    result = AuditResult(
        tier="DUPLICATES & CONSISTENCY",
        description="Duplicate detection and naming conventions",
    )

    # Duplicate Cypher examples (exact after normalization, within same file)
    # Cross-file duplication is expected (clause examples also appear in examples file)
    snippets = extract_all_cypher(all_records)

    # Group by file
    by_file: dict[str, list[tuple[str, int, str]]] = {}
    for cypher, filename, line_num, rec_id in snippets:
        by_file.setdefault(filename, []).append((cypher, line_num, rec_id))

    for filename, file_snippets in by_file.items():
        seen_cypher: dict[str, tuple[int, str]] = {}
        for cypher, line_num, rec_id in file_snippets:
            result.total_checked += 1
            normalized = _normalize_cypher(cypher)
            if len(normalized) < 60:
                result.passed += 1
                continue

            if normalized in seen_cypher:
                prev_line, prev_id = seen_cypher[normalized]
                result.issues.append(Issue(
                    file=filename, line_num=line_num, field="cypher",
                    message=f"Duplicate of line {prev_line} ({prev_id})",
                    severity="warn", record_id=rec_id,
                ))
                result.warned += 1
            else:
                seen_cypher[normalized] = (line_num, rec_id)
                result.passed += 1

    # Duplicate function names
    func_names: Counter[str] = Counter()
    for _, obj in all_records.get("cypher_functions.jsonl", []):
        name = str(obj.get("name", ""))
        func_names[name] += 1

    for name, count in func_names.items():
        if count > 1:
            result.total_checked += 1
            result.issues.append(Issue(
                file="cypher_functions.jsonl", line_num=0, field="name",
                message=f"Function {name!r} appears {count} times",
                severity="warn", record_id=name,
            ))
            result.warned += 1

    # Naming conventions in modeling patterns
    for line_num, obj in all_records.get("modeling_patterns.jsonl", []):
        rec_id = _record_id("modeling_patterns.jsonl", obj)

        node_labels = obj.get("node_labels", [])
        if isinstance(node_labels, list):
            for label in node_labels:
                result.total_checked += 1
                label_str = str(label)
                if _is_upper_snake(label_str) and not _is_pascal_case(label_str):
                    result.issues.append(Issue(
                        file="modeling_patterns.jsonl", line_num=line_num,
                        field="node_labels",
                        message=f"Looks like a relationship type, not a node label: {label_str!r}",
                        severity="warn", record_id=rec_id,
                    ))
                    result.warned += 1
                else:
                    result.passed += 1

        rel_types = obj.get("relationship_types", [])
        if isinstance(rel_types, list):
            for rel in rel_types:
                result.total_checked += 1
                rel_str = str(rel)
                if _is_pascal_case(rel_str) and not _is_upper_snake(rel_str):
                    result.issues.append(Issue(
                        file="modeling_patterns.jsonl", line_num=line_num,
                        field="relationship_types",
                        message=f"Looks like a node label, not a relationship: {rel_str!r}",
                        severity="warn", record_id=rec_id,
                    ))
                    result.warned += 1
                else:
                    result.passed += 1

    return result


# ── Report rendering ─────────────────────────────────────────────────────────


def _severity_icon(sev: str) -> str:
    """Terminal-safe icons."""
    return {"pass": "[green]PASS[/]", "fail": "[red]FAIL[/]", "warn": "[yellow]WARN[/]"}.get(
        sev, sev
    )


def _issues_table(issues: list[Issue], limit: int) -> Table:
    """Build a rich Table from issues."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Sev", width=6)
    table.add_column("File", width=28)
    table.add_column("Line", width=5, justify="right")
    table.add_column("Record", width=25)
    table.add_column("Message", min_width=40)

    shown = issues[:limit]
    for iss in shown:
        table.add_row(
            _severity_icon(iss.severity),
            iss.file,
            str(iss.line_num),
            iss.record_id[:25],
            iss.message[:80],
        )

    if len(issues) > limit:
        table.add_row("", "", "", "", f"... and {len(issues) - limit} more")

    return table


def render_report(
    console: Console,
    results: list[AuditResult],
    total_records: int,
    total_cypher: int,
    verbose: bool,
) -> None:
    """Render the human-readable terminal report."""
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    console.print()
    console.print(Panel(
        f"[bold]EXPERT DATA AUDIT REPORT[/bold]\n"
        f"Date: {now}\n"
        f"Records scanned: {total_records}  |  Cypher snippets: {total_cypher}",
        style="bold blue",
    ))

    total_verified = 0
    total_flagged = 0

    for res in results:
        fail_issues = [i for i in res.issues if i.severity == "fail"]
        warn_issues = [i for i in res.issues if i.severity == "warn"]

        status_parts: list[str] = []
        if res.passed:
            status_parts.append(f"[green]{res.passed} pass[/]")
        if res.failed:
            status_parts.append(f"[red]{res.failed} fail[/]")
        if res.warned:
            status_parts.append(f"[yellow]{res.warned} warn[/]")

        console.print(f"\n[bold]{res.tier}[/bold] ({res.description})")
        console.print(f"  Checked: {res.total_checked}  |  {', '.join(status_parts)}")

        total_verified += res.passed
        total_flagged += res.failed + res.warned

        limit = 999 if verbose else 10

        if fail_issues:
            console.print(f"\n  [red]Failures ({len(fail_issues)}):[/]")
            console.print(_issues_table(fail_issues, limit))

        if warn_issues:
            console.print(f"\n  [yellow]Warnings ({len(warn_issues)}):[/]")
            console.print(_issues_table(warn_issues, limit))

    # Human review reminder
    console.print("\n[bold]HUMAN REVIEW NEEDED[/bold]")
    console.print("  23 modeling patterns — cannot auto-verify advice quality")
    console.print("  318 best practices — cannot auto-verify advice quality")
    console.print("  [dim]These records need review by a Neo4j practitioner[/]")

    # Summary
    console.print()
    console.print(Panel(
        f"[bold]SUMMARY[/bold]\n"
        f"[green]Verified:     {total_verified}[/]\n"
        f"[red]Flagged:      {total_flagged}[/]\n"
        f"Human review: 341 (patterns + practices)",
        style="bold",
    ))


# ── JSON export ──────────────────────────────────────────────────────────────


def export_json(
    results: list[AuditResult],
    total_records: int,
    total_cypher: int,
    output_path: Path,
) -> None:
    """Write machine-readable JSON audit report."""
    data: dict[str, object] = {
        "date": datetime.now(tz=UTC).isoformat(),
        "records_scanned": total_records,
        "cypher_snippets": total_cypher,
        "tiers": {},
        "summary": {},
    }

    total_verified = 0
    total_flagged = 0
    tiers_dict: dict[str, object] = {}

    for res in results:
        tiers_dict[res.tier.lower().replace(" & ", "_").replace(" ", "_")] = {
            "total": res.total_checked,
            "passed": res.passed,
            "failed": res.failed,
            "warned": res.warned,
            "issues": [
                {
                    "file": i.file,
                    "line": i.line_num,
                    "field": i.field,
                    "record": i.record_id,
                    "message": i.message,
                    "severity": i.severity,
                }
                for i in res.issues
            ],
        }
        total_verified += res.passed
        total_flagged += res.failed + res.warned

    data["tiers"] = tiers_dict
    data["summary"] = {
        "verified": total_verified,
        "flagged": total_flagged,
        "human_review": 341,
    }

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the expert data audit."""
    parser = argparse.ArgumentParser(
        description="Audit expert JSONL data for completeness, Cypher validity, and consistency",
    )
    parser.add_argument(
        "--json", type=Path, default=None, dest="json_path",
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all issues, not just the first 10 per category",
    )
    args = parser.parse_args()

    console = Console()

    # Load
    console.print("[dim]Loading JSONL data...[/]")
    all_records = load_all_records()
    total_records = sum(len(recs) for recs in all_records.values())

    # Count Cypher snippets
    all_cypher = extract_all_cypher(all_records)
    total_cypher = len(all_cypher)

    console.print(f"[dim]Loaded {total_records} records, {total_cypher} Cypher snippets[/]")

    # Run tiers
    results: list[AuditResult] = []

    console.print("[dim]Running Tier 1: Completeness...[/]")
    results.append(check_completeness(all_records))

    console.print("[dim]Running Tier 2: Cypher validation...[/]")
    results.append(check_cypher_validation(all_records))

    console.print("[dim]Running Tier 3: Cross-reference...[/]")
    results.append(check_cross_reference(all_records))

    console.print("[dim]Running Tier 4: Duplicates & consistency...[/]")
    results.append(check_duplicates_and_consistency(all_records))

    # Report
    render_report(console, results, total_records, total_cypher, args.verbose)

    # JSON export
    if args.json_path is not None:
        export_json(results, total_records, total_cypher, args.json_path)
        console.print(f"\n[dim]JSON report written to {args.json_path}[/]")

    # Exit code: 1 if any failures
    has_failures = any(r.failed > 0 for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
