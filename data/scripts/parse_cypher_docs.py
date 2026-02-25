"""Parse Neo4j Cypher AsciiDoc files into structured JSONL for the expert graph.

Extracts:
  - Cypher clauses (MATCH, CREATE, WHERE, etc.) with descriptions and examples
  - Cypher functions (aggregating, string, math, etc.) with signatures
  - Cypher patterns and best practices
  - Code examples with their context

Output: data/processed/cypher_clauses.jsonl
        data/processed/cypher_functions.jsonl
        data/processed/cypher_examples.jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent / "raw" / "docs" / "docs-cypher"
PAGES_DIR = DOCS_ROOT / "modules" / "ROOT" / "pages"
OUTPUT_DIR = Path(__file__).parent.parent / "processed"


@dataclass
class CypherClause:
    name: str
    description: str
    syntax_examples: list[str] = field(default_factory=list)
    sections: list[dict[str, str]] = field(default_factory=list)
    source_file: str = ""
    authority_level: int = 1  # Official docs = highest


@dataclass
class CypherFunction:
    name: str
    category: str
    description: str
    signature: str = ""
    returns: str = ""
    examples: list[str] = field(default_factory=list)
    source_file: str = ""
    authority_level: int = 1


@dataclass
class CypherExample:
    cypher: str
    description: str
    context: str  # clause or function name
    category: str  # clause, function, pattern, etc.
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


def extract_description(text: str) -> str:
    """Extract the :description: field from AsciiDoc header."""
    match = re.search(r':description:\s*(.+)', text)
    return match.group(1).strip() if match else ""


def parse_sections(text: str) -> list[dict[str, str]]:
    """Parse AsciiDoc into sections by heading level."""
    sections = []
    current_heading = ""
    current_content: list[str] = []

    for line in text.split('\n'):
        heading_match = re.match(r'^(={2,4})\s+(.+)', line)
        if heading_match:
            if current_heading:
                sections.append({
                    "heading": current_heading,
                    "content": '\n'.join(current_content).strip(),
                })
            current_heading = heading_match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_heading:
        sections.append({
            "heading": current_heading,
            "content": '\n'.join(current_content).strip(),
        })

    return sections


def parse_clause_file(filepath: Path) -> CypherClause | None:
    """Parse a single clause .adoc file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    name_match = re.search(r'^=\s+(.+)', text, re.MULTILINE)
    if not name_match:
        return None

    name = name_match.group(1).strip()
    # Clean up backticks from name
    name = name.replace('`', '')

    description = extract_description(text)
    if not description:
        # Fall back to first paragraph after the heading
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('= ') and i + 2 < len(lines):
                # Skip blank lines after heading
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith(('=', '[', ':', 'include', 'image')):
                        description = lines[j].strip()
                        break
                break

    examples = extract_code_blocks(text)
    sections = parse_sections(text)

    return CypherClause(
        name=name,
        description=description,
        syntax_examples=examples[:10],  # Cap at 10 most relevant
        sections=sections[:20],  # Cap sections
        source_file=str(filepath.relative_to(DOCS_ROOT)),
    )


def parse_function_file(filepath: Path, category: str) -> list[CypherFunction]:
    """Parse a functions .adoc file — may contain multiple functions."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    functions = []

    # Functions use == headings (h2). Split on them.
    # Match lines like: == char_length()  or  == `size()`
    func_sections = re.split(r'\n==\s+(?!=)', text)

    for section in func_sections[1:]:  # Skip preamble
        lines = section.split('\n')
        raw_name = lines[0].strip().replace('`', '')

        # Extract function name (before parentheses)
        func_name = raw_name.split('(')[0].strip()

        # Skip non-function headings (like "Example graph")
        if ' ' in func_name and not func_name.endswith(')'):
            if not any(c in func_name for c in '.'):
                continue

        # Extract from Details table: *Description*, *Syntax*, *Returns*
        section_text = '\n'.join(lines)
        desc = ""
        sig = ""
        ret = ""

        # Look for table-based description: | *Description* 3+| ...
        desc_match = re.search(
            r'\*Description\*[^|]*\|\s*(.+?)(?:\n|$)', section_text,
        )
        if desc_match:
            desc = desc_match.group(1).strip().rstrip('.')

        # Look for table-based syntax: | *Syntax* 3+| `func(...)`
        sig_match = re.search(
            r'\*Syntax\*[^|]*\|\s*`([^`]+)`', section_text,
        )
        if sig_match:
            sig = sig_match.group(1).strip()

        # Look for table-based returns: | *Returns* 3+| `TYPE`
        ret_match = re.search(
            r'\*Returns\*[^|]*\|\s*`?(\w+)`?', section_text,
        )
        if ret_match:
            ret = ret_match.group(1).strip()

        # Fallback description from first paragraph
        if not desc:
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and not stripped.startswith(
                    ('[', ':', '=', 'include', 'image', '|', '----', '.', '//')
                ):
                    desc = stripped
                    break

        examples = extract_code_blocks(section_text)

        if func_name and len(func_name) < 80 and not func_name.startswith('Example'):
            functions.append(CypherFunction(
                name=func_name,
                category=category,
                description=desc,
                signature=sig,
                returns=ret,
                examples=examples[:5],
                source_file=str(filepath.relative_to(DOCS_ROOT)),
            ))

    return functions


def parse_all_clauses() -> list[CypherClause]:
    """Parse all clause files."""
    clauses_dir = PAGES_DIR / "clauses"
    clauses = []
    if not clauses_dir.exists():
        print(f"  WARNING: {clauses_dir} not found")
        return clauses

    for adoc_file in sorted(clauses_dir.glob("*.adoc")):
        if adoc_file.name == "index.adoc":
            continue
        clause = parse_clause_file(adoc_file)
        if clause:
            clauses.append(clause)
            print(f"  Clause: {clause.name} ({len(clause.syntax_examples)} examples)")

    return clauses


def parse_all_functions() -> list[CypherFunction]:
    """Parse all function files."""
    funcs_dir = PAGES_DIR / "functions"
    all_functions = []
    if not funcs_dir.exists():
        print(f"  WARNING: {funcs_dir} not found")
        return all_functions

    for adoc_file in sorted(funcs_dir.glob("*.adoc")):
        if adoc_file.name == "index.adoc":
            continue
        category = adoc_file.stem.replace("-", " ")
        functions = parse_function_file(adoc_file, category)
        all_functions.extend(functions)
        if functions:
            print(f"  Functions [{category}]: {len(functions)} found")

    return all_functions


def collect_all_examples(clauses: list[CypherClause], functions: list[CypherFunction]) -> list[CypherExample]:
    """Collect all Cypher examples from clauses and functions."""
    examples = []

    for clause in clauses:
        for cypher in clause.syntax_examples:
            examples.append(CypherExample(
                cypher=cypher,
                description=clause.description,
                context=clause.name,
                category="clause",
                source_file=clause.source_file,
            ))

    for func in functions:
        for cypher in func.examples:
            examples.append(CypherExample(
                cypher=cypher,
                description=func.description,
                context=func.name,
                category="function",
                source_file=func.source_file,
            ))

    return examples


def write_jsonl(items: list, output_path: Path) -> None:
    """Write list of dataclasses to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} entries to {output_path}")


def main() -> None:
    print("=" * 60)
    print("Parsing Neo4j Cypher Documentation")
    print("=" * 60)

    if not PAGES_DIR.exists():
        print(f"ERROR: {PAGES_DIR} not found. Run 'git clone' first.")
        return

    print("\n[1/3] Parsing Cypher clauses...")
    clauses = parse_all_clauses()

    print(f"\n[2/3] Parsing Cypher functions...")
    functions = parse_all_functions()

    print(f"\n[3/3] Collecting examples...")
    examples = collect_all_examples(clauses, functions)

    print(f"\n--- Summary ---")
    print(f"  Clauses:   {len(clauses)}")
    print(f"  Functions: {len(functions)}")
    print(f"  Examples:  {len(examples)}")

    print(f"\nWriting output...")
    write_jsonl(clauses, OUTPUT_DIR / "cypher_clauses.jsonl")
    write_jsonl(functions, OUTPUT_DIR / "cypher_functions.jsonl")
    write_jsonl(examples, OUTPUT_DIR / "cypher_examples.jsonl")

    print("\nDone!")


if __name__ == "__main__":
    main()
