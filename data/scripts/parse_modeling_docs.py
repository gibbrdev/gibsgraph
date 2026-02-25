"""Parse Neo4j data modeling documentation into structured JSONL.

Extracts:
  - Modeling patterns (intermediate nodes, linked lists, etc.)
  - Best practices and tips
  - Naming conventions
  - Relational-to-graph migration patterns

Output: data/processed/modeling_patterns.jsonl
        data/processed/best_practices.jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent / "raw" / "docs" / "docs-getting-started"
PAGES_DIR = DOCS_ROOT / "modules" / "ROOT" / "pages"
OUTPUT_DIR = Path(__file__).parent.parent / "processed"


@dataclass
class ModelingPattern:
    name: str
    description: str
    when_to_use: str = ""
    anti_pattern: str = ""
    cypher_examples: list[str] = field(default_factory=list)
    node_labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    source_file: str = ""
    authority_level: int = 1


@dataclass
class BestPractice:
    title: str
    description: str
    category: str  # naming, modeling, performance, security
    cypher_examples: list[str] = field(default_factory=list)
    source_file: str = ""
    authority_level: int = 1


def extract_code_blocks(text: str) -> list[str]:
    """Extract Cypher code blocks from AsciiDoc."""
    blocks = []
    pattern = re.compile(
        r'\[source,\s*cypher[^\]]*\]\s*\n----\n(.*?)----',
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        code = match.group(1).strip()
        if code:
            blocks.append(code)
    return blocks


def extract_labels_and_rels(text: str) -> tuple[list[str], list[str]]:
    """Extract node labels and relationship types from Cypher examples."""
    labels: set[str] = set()
    rels: set[str] = set()

    # Find :Label patterns in Cypher
    for match in re.finditer(r':([A-Z][a-zA-Z]+)', text):
        labels.add(match.group(1))

    # Find [:REL_TYPE] patterns
    for match in re.finditer(r'\[:([A-Z_]+)', text):
        rels.add(match.group(1))

    return sorted(labels), sorted(rels)


def parse_modeling_pages() -> tuple[list[ModelingPattern], list[BestPractice]]:
    """Parse data modeling documentation."""
    patterns: list[ModelingPattern] = []
    practices: list[BestPractice] = []

    # Look for modeling pages in various locations
    modeling_dirs = [
        PAGES_DIR / "data-modeling",
        PAGES_DIR,
    ]

    for mdir in modeling_dirs:
        if not mdir.exists():
            continue

        for adoc_file in sorted(mdir.rglob("*.adoc")):
            text = adoc_file.read_text(encoding="utf-8", errors="replace")
            rel_path = str(adoc_file.relative_to(DOCS_ROOT))

            # Skip navigation/index files
            if adoc_file.name in ("nav.adoc", "index.adoc"):
                continue

            # Extract title
            title_match = re.search(r'^=\s+(.+)', text, re.MULTILINE)
            if not title_match:
                continue
            title = title_match.group(1).strip()

            # Extract description
            desc_match = re.search(r':description:\s*(.+)', text)
            description = desc_match.group(1).strip() if desc_match else ""

            examples = extract_code_blocks(text)
            labels, rels = extract_labels_and_rels(text)

            # Classify as pattern or best practice
            is_pattern = any(kw in rel_path.lower() for kw in [
                "modeling-designs", "refactor", "relational-to-graph",
                "versioning",
            ])
            is_practice = any(kw in rel_path.lower() for kw in [
                "tips", "concepts", "naming",
            ])

            if is_pattern or (labels and rels):
                # Parse sections for when_to_use / anti_pattern
                when_to_use = ""
                anti_pattern = ""
                sections = re.split(r'\n==\s+(?!=)', text)
                for section in sections:
                    lower = section.lower()
                    if "when" in lower or "use case" in lower:
                        when_to_use = section[:500].strip()
                    if "anti" in lower or "avoid" in lower or "don't" in lower:
                        anti_pattern = section[:500].strip()

                patterns.append(ModelingPattern(
                    name=title,
                    description=description or title,
                    when_to_use=when_to_use,
                    anti_pattern=anti_pattern,
                    cypher_examples=examples[:10],
                    node_labels=labels,
                    relationship_types=rels,
                    source_file=rel_path,
                ))
                print(f"  Pattern: {title} ({len(examples)} examples, {len(labels)} labels)")

            if is_practice or "tips" in rel_path.lower():
                # Extract individual tips as separate practices
                tip_sections = re.split(r'\n===\s+', text)
                for tip in tip_sections[1:]:
                    tip_lines = tip.split('\n')
                    tip_title = tip_lines[0].strip()
                    tip_desc = ""
                    for line in tip_lines[1:]:
                        stripped = line.strip()
                        if stripped and not stripped.startswith(
                            ('[', ':', '=', 'include', 'image', '|', '----', '//')
                        ):
                            tip_desc = stripped
                            break

                    if tip_title and tip_desc:
                        tip_examples = extract_code_blocks('\n'.join(tip_lines))
                        category = "modeling"
                        if "name" in tip_title.lower() or "naming" in rel_path.lower():
                            category = "naming"
                        elif "performance" in tip_title.lower():
                            category = "performance"

                        practices.append(BestPractice(
                            title=tip_title,
                            description=tip_desc,
                            category=category,
                            cypher_examples=tip_examples[:5],
                            source_file=rel_path,
                        ))

            # Also treat the whole page as a practice if it's a tips file
            if is_practice and description:
                practices.append(BestPractice(
                    title=title,
                    description=description,
                    category="modeling",
                    cypher_examples=examples[:5],
                    source_file=rel_path,
                ))
                print(f"  Practice: {title}")

    return patterns, practices


def parse_kb_articles() -> list[BestPractice]:
    """Parse Knowledge Base articles for best practices."""
    kb_root = Path(__file__).parent.parent / "raw" / "docs" / "knowledge-base"
    articles_dir = kb_root / "articles" / "modules" / "ROOT" / "pages"
    practices = []

    if not articles_dir.exists():
        print(f"  WARNING: {articles_dir} not found")
        return practices

    count = 0
    for adoc_file in sorted(articles_dir.glob("*.adoc")):
        text = adoc_file.read_text(encoding="utf-8", errors="replace")

        title_match = re.search(r'^=\s+(.+)', text, re.MULTILINE)
        if not title_match:
            continue
        title = title_match.group(1).strip()

        # Get first meaningful paragraph as description
        description = ""
        in_content = False
        for line in text.split('\n'):
            if line.startswith('= '):
                in_content = True
                continue
            if in_content:
                stripped = line.strip()
                if stripped and not stripped.startswith(
                    ('[', ':', '=', 'include', 'image', '|', '----', '//', 'ifdef', 'endif')
                ):
                    description = stripped
                    break

        if not description:
            continue

        examples = extract_code_blocks(text)

        # Categorize
        category = "general"
        lower = (title + text[:500]).lower()
        if any(kw in lower for kw in ["cypher", "query", "match", "return"]):
            category = "cypher"
        elif any(kw in lower for kw in ["model", "schema", "label", "relationship"]):
            category = "modeling"
        elif any(kw in lower for kw in ["performance", "memory", "index", "cache"]):
            category = "performance"
        elif any(kw in lower for kw in ["security", "auth", "ssl", "encrypt"]):
            category = "security"
        elif any(kw in lower for kw in ["import", "load", "csv", "export"]):
            category = "data-import"

        practices.append(BestPractice(
            title=title,
            description=description[:500],
            category=category,
            cypher_examples=examples[:3],
            source_file=str(adoc_file.relative_to(kb_root)),
            authority_level=1,  # Official KB
        ))
        count += 1

    print(f"  Knowledge Base articles: {count} parsed")
    return practices


def write_jsonl(items: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} entries to {output_path}")


def main() -> None:
    print("=" * 60)
    print("Parsing Neo4j Data Modeling & Best Practices")
    print("=" * 60)

    print("\n[1/2] Parsing modeling documentation...")
    patterns, practices = parse_modeling_pages()

    print(f"\n[2/2] Parsing Knowledge Base articles...")
    kb_practices = parse_kb_articles()
    practices.extend(kb_practices)

    print(f"\n--- Summary ---")
    print(f"  Modeling patterns: {len(patterns)}")
    print(f"  Best practices:   {len(practices)}")

    print(f"\nWriting output...")
    write_jsonl(patterns, OUTPUT_DIR / "modeling_patterns.jsonl")
    write_jsonl(practices, OUTPUT_DIR / "best_practices.jsonl")

    print("\nDone!")


if __name__ == "__main__":
    main()
