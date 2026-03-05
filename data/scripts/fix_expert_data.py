"""Mechanical fixes for expert JSONL data.

Applies deterministic, objective fixes — no judgment calls:
  1. Strip AsciiDoc markup from text fields
  2. Fix node_labels (remove ALL_CAPS entries that are relationship fragments)
  3. Remove fabricated SEARCH clause record
  4. Add rule-based quality_tier to files missing it
  5. Clean sections fields of graphviz/asciidoc rendering artifacts

Usage:
  python data/scripts/fix_expert_data.py              # dry-run (show changes)
  python data/scripts/fix_expert_data.py --apply       # write changes to disk
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1].parent / "src" / "gibsgraph" / "data"

# ── AsciiDoc cleanup patterns ───────────────────────────────────────────────

_ASCIIDOC_SUBS: list[tuple[re.Pattern[str], str]] = [
    # xref:path/to/page.adoc#anchor[display text] → display text
    (re.compile(r"xref:[^\[]*\[([^\]]*)\]"), r"\1"),
    # image::path.svg[alt text,...] → (remove entirely)
    (re.compile(r"image::[^\[]*\[[^\]]*\]"), ""),
    # link:url[display text] → display text
    (re.compile(r"link:[^\[]*\[([^\]]*)\]"), r"\1"),
    # [source, cypher] and similar code markers
    (re.compile(r"\[source,\s*\w+(?:,\s*[^\]]+)?\]"), ""),
    # ---- delimiters (4+ dashes on their own line)
    (re.compile(r"^-{4,}$", re.MULTILINE), ""),
    # -- delimiters (exactly 2 dashes on their own line)
    (re.compile(r"^--$", re.MULTILINE), ""),
    # <<anchor>> references
    (re.compile(r"<<[a-z_-]+>>"), ""),
    # .Title lines (AsciiDoc block titles)
    (re.compile(r"^\.[A-Z`].+$", re.MULTILINE), ""),
    # [role=...] and [options=...] blocks
    (re.compile(r"\[(?:role|options)=[^\]]*\]"), ""),
    # ifdef/ifndef/endif directives
    (re.compile(r"(?:ifdef|ifndef|endif)::[^\n]*"), ""),
    # {neo4j-docs-base-uri} and similar attributes
    (re.compile(r"\{[a-z][a-z0-9_-]*\}"), ""),
    # ==== section delimiters
    (re.compile(r"^={4,}$", re.MULTILINE), ""),
    # Collapse multiple blank lines into one
    (re.compile(r"\n{3,}"), "\n\n"),
]

# Fields to clean AsciiDoc from
_TEXT_FIELDS = ("description", "when_to_use", "anti_pattern", "context")


def clean_asciidoc(text: str) -> str:
    """Strip AsciiDoc markup artifacts from a text field."""
    for pattern, replacement in _ASCIIDOC_SUBS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ── Node label fixes ────────────────────────────────────────────────────────

def _is_likely_rel_type(label: str) -> bool:
    """True if an ALL_CAPS string is likely a relationship type, not a node label."""
    # Real node labels: Person, Movie, ProductionCompany (have lowercase)
    # Relationship fragments: ACTED, DIRECTED, KNOWS (all caps, no underscore)
    if any(c.islower() for c in label):
        return False
    # ALL_CAPS with 2+ chars and no underscore = likely a verb/rel fragment
    return len(label) >= 2 and label.isalpha()


def fix_node_labels(labels: list[str]) -> list[str]:
    """Remove entries that are clearly relationship type fragments."""
    return [l for l in labels if not _is_likely_rel_type(l)]


# ── Rule-based quality_tier ─────────────────────────────────────────────────

def compute_quality_tier(obj: dict[str, object], filename: str) -> str:
    """Derive quality_tier from objective signals. No LLM judgment."""
    authority = obj.get("authority_level", 2)

    # Check for Cypher examples
    has_cypher = False
    for field in ("cypher", "examples", "syntax_examples", "cypher_examples"):
        val = obj.get(field)
        if isinstance(val, str) and val.strip():
            has_cypher = True
        elif isinstance(val, list) and len(val) > 0:
            has_cypher = True

    # Check for substantive description
    desc = obj.get("description") or obj.get("when_to_use") or ""
    has_desc = isinstance(desc, str) and len(desc.strip()) > 50

    # Source from official docs = higher trust
    source = str(obj.get("source_file", ""))
    is_official = "modules" in source or "curated" in source or "articles" in source

    if authority == 1 and has_cypher and has_desc and is_official:
        return "high"
    elif authority == 1 and (has_cypher or has_desc):
        return "medium"
    else:
        return "low"


# ── Main fix logic ──────────────────────────────────────────────────────────

def fix_file(filename: str, records: list[dict[str, object]], dry_run: bool) -> int:
    """Apply fixes to records. Returns number of changes made."""
    changes = 0

    for i, obj in enumerate(records):
        record_name = obj.get("name") or obj.get("title") or obj.get("context") or "?"

        # 1. Strip AsciiDoc from text fields
        for field in _TEXT_FIELDS:
            val = obj.get(field)
            if isinstance(val, str) and val.strip():
                cleaned = clean_asciidoc(val)
                if cleaned != val:
                    if dry_run:
                        diff_len = len(val) - len(cleaned)
                        print(f"  ASCIIDOC {filename}:{i+1} {field} "
                              f"({diff_len} chars removed) [{record_name}]")
                    obj[field] = cleaned
                    changes += 1

        # 2. Fix node_labels in modeling_patterns
        if filename == "modeling_patterns.jsonl":
            labels = obj.get("node_labels", [])
            if isinstance(labels, list):
                fixed = fix_node_labels(labels)
                removed = set(labels) - set(fixed)
                if removed:
                    if dry_run:
                        print(f"  LABELS {filename}:{i+1} removed {removed} "
                              f"[{record_name}]")
                    obj["node_labels"] = fixed
                    changes += 1

        # 3. Add quality_tier if missing
        if "quality_tier" not in obj:
            tier = compute_quality_tier(obj, filename)
            if dry_run:
                print(f"  TIER {filename}:{i+1} -> {tier} [{record_name}]")
            obj["quality_tier"] = tier
            changes += 1

    return changes


def main() -> None:
    apply = "--apply" in sys.argv
    dry_run = not apply

    if dry_run:
        print("DRY RUN — no files will be modified. Use --apply to write.\n")

    total_changes = 0

    for filename in (
        "cypher_clauses.jsonl",
        "cypher_functions.jsonl",
        "cypher_examples.jsonl",
        "modeling_patterns.jsonl",
        "best_practices.jsonl",
    ):
        path = DATA_DIR / filename
        text = path.read_text(encoding="utf-8")
        records: list[dict[str, object]] = []
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))

        # (SEARCH is a real Cypher 25 clause — kept in data)

        changes = fix_file(filename, records, dry_run)
        total_changes += changes

        if apply:
            lines = [json.dumps(r, ensure_ascii=False) for r in records]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  Wrote {filename} ({len(records)} records, {changes} changes)")

    print(f"\nTotal changes: {total_changes}")
    if dry_run:
        print("Run with --apply to write changes to disk.")


if __name__ == "__main__":
    main()
