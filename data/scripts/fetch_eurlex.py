"""Shared EUR-Lex HTML fetcher and parser for regulation training pairs.

Fetches consolidated regulation text from EUR-Lex, parses articles, recitals,
and annexes, extracts cross-references from hyperlinks and text patterns.

Used by build_dora_pairs.py, build_nis2_pairs.py, build_mica_pairs.py.

The parsed output follows the same training pair format as build_training_pairs.py
(Vestio-based) but works directly from EUR-Lex HTML — no Vestio dependency.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

# EUR-Lex Cellar XHTML endpoint (bypasses WAF JS challenge on main site)
# Pattern: cellar/{cellar_id}.{expression}.{manifestation}/DOC_1
# Expression .0006 = English, Manifestation .03 = XHTML
EURLEX_HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

# Cellar IDs for regulations (CELEX → cellar UUID)
CELLAR_IDS: dict[str, str] = {
    "32022R2554": "0caf473a-85bd-11ed-9887-01aa75ed71a1",  # DORA
    "32022L2555": "9b84d482-85bd-11ed-9887-01aa75ed71a1",  # NIS2
    "32023R1114": "01d55833-0660-11ee-b12e-01aa75ed71a1",  # MiCA
    "32024R1689": "cebc64dc-4562-11ef-8c46-01aa75ed71a1",  # AI Act
    "32016R0679": "02513a3a-d802-11e5-8fea-01aa75ed71a1",  # GDPR
}

CELLAR_XHTML_URL = "https://publications.europa.eu/resource/cellar/{cellar_id}.0006.03/DOC_1"

# Known CELEX → regulation name mapping
CELEX_TO_REG: dict[str, str] = {
    "32022R2554": "dora",
    "32022L2555": "nis2",
    "32023R1114": "mica",
    "32024R1689": "ai_act",
    "32016R0679": "gdpr",
}

# Edge weight scheme (matches build_training_pairs.py)
EDGE_WEIGHTS: dict[str, float] = {
    "AMENDS": 1.0,
    "SUPPLEMENTS": 0.95,
    "CROSS_REGULATES": 0.9,
    "IMPLIES": 0.85,
    "REFERENCES": 0.8,
    "DEFINES": 0.7,
    "INTERPRETS": 0.5,
    "CONTAINS": 0.3,
    "HAS_ARTICLE": 0.1,
}

# Cross-reference patterns for other EU regulations in text
CROSS_REG_PATTERNS: dict[str, str] = {
    r"Regulation\s*\(EU\)\s*2016/679": "gdpr",
    r"Regulation\s*\(EU\)\s*2024/1689": "ai_act",
    r"Regulation\s*\(EU\)\s*2022/2554": "dora",
    r"Directive\s*\(EU\)\s*2022/2555": "nis2",
    r"Regulation\s*\(EU\)\s*2023/1114": "mica",
    r"Directive\s*\(EU\)\s*2015/2366": "psd2",
    r"Directive\s*2014/65/EU": "mifid2",
    r"Directive\s*2009/138/EC": "solvency2",
    r"Regulation\s*\(EU\)\s*No\s*575/2013": "crr",
    r"Directive\s*2013/36/EU": "crd4",
    r"Directive\s*2009/65/EC": "ucits",
    r"Directive\s*2011/61/EU": "aifmd",
    r"Regulation\s*\(EU\)\s*No\s*648/2012": "emir",
    r"Regulation\s*\(EU\)\s*No\s*909/2014": "csdr",
    r"Regulation\s*\(EU\)\s*2019/2088": "sfdr",
    r"Regulation\s*\(EU\)\s*2020/852": "eu_taxonomy",
    r"Regulation\s*\(EU\)\s*No\s*1907/2006": "reach",
    r"Regulation\s*\(EU\)\s*No\s*305/2011": "cpr",
    r"Directive\s*2002/58/EC": "eprivacy",
    r"Regulation\s*\(EU\)\s*910/2014": "eidas",
}


@dataclass
class ParsedChunk:
    """A parsed article, recital, or annex from a regulation."""

    id: str
    regulation: str
    article_id: str
    title: str
    chapter: str
    chunk_type: str  # article, paragraph, recital, annex
    text: str
    source_celex: str
    binding: bool = True
    cross_references: list[str] = field(default_factory=list)
    cross_regulation_refs: list[str] = field(default_factory=list)


def fetch_regulation_html(celex: str, cache_dir: Path | None = None) -> str:
    """Fetch regulation XHTML from EU Cellar, with optional caching.

    EUR-Lex main site uses AWS WAF JS challenges that block programmatic access.
    Instead we use the Publications Office Cellar API which serves XHTML directly.
    """
    if cache_dir:
        cache_file = cache_dir / f"{celex}.html"
        if cache_file.exists():
            content = cache_file.read_text(encoding="utf-8")
            if len(content) > 1000:  # Skip empty/corrupt cache
                print(f"  Using cached HTML: {cache_file}")
                return content

    # Use Cellar XHTML endpoint
    cellar_id = CELLAR_IDS.get(celex)
    if cellar_id:
        url = CELLAR_XHTML_URL.format(cellar_id=cellar_id)
    else:
        # Fallback: try EUR-Lex directly (may fail with WAF)
        url = EURLEX_HTML_URL.format(celex=celex)

    print(f"  Fetching {url}...")

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url, headers={
            "Accept": "application/xhtml+xml,text/html,*/*",
        })
        resp.raise_for_status()

    # Replace non-breaking spaces (common in EU publications)
    html = resp.text.replace("\xa0", " ").replace("\u00a0", " ")

    if len(html) < 1000:
        raise ValueError(f"Fetched content too short ({len(html)} bytes) — may be an error page")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{celex}.html"
        cache_file.write_text(html, encoding="utf-8")
        print(f"  Cached to {cache_file} ({len(html)} bytes)")

    return html


def _clean_text(text: str) -> str:
    """Clean extracted text — normalize whitespace, strip artifacts."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation artifacts
    text = re.sub(r"^[.\s,;]+", "", text)
    return text


def _extract_article_id(text: str) -> str | None:
    """Extract article number from heading text like 'Article 5' or 'Article 12a'."""
    m = re.search(r"Article\s+(\d+[a-z]?)", text)
    return f"Article {m.group(1)}" if m else None


def _extract_cross_refs(text: str, own_regulation: str) -> tuple[list[str], list[str]]:
    """Extract internal article refs and cross-regulation refs from text.

    Returns (internal_refs, cross_reg_names).
    """
    # Internal: "Article 5", "Article 12(3)", "Annex I", "Annex III"
    internal = []
    for m in re.finditer(r"Article\s+(\d+[a-z]?)(?:\(\d+\))?", text):
        ref = f"Article {m.group(1)}"
        if ref not in internal:
            internal.append(ref)

    for m in re.finditer(r"Annex\s+([IVX]+|[A-Z])", text):
        ref = f"Annex {m.group(1)}"
        if ref not in internal:
            internal.append(ref)

    # Cross-regulation references
    cross_regs = []
    for pattern, reg_name in CROSS_REG_PATTERNS.items():
        if reg_name == own_regulation:
            continue
        if re.search(pattern, text):
            if reg_name not in cross_regs:
                cross_regs.append(reg_name)

    return internal, cross_regs


def parse_regulation(
    html: str,
    regulation: str,
    celex: str,
) -> list[ParsedChunk]:
    """Parse EUR-Lex HTML into structured chunks.

    Strategy: EUR-Lex HTML uses consistent patterns:
    - Articles in <div> or <p> with "Article N" headings
    - Recitals numbered (1), (2), etc. in the preamble
    - Annexes after the main body
    """
    soup = BeautifulSoup(html, "html.parser")
    chunks: list[ParsedChunk] = []

    # Get the main content body
    body = soup.find("body") or soup
    all_text = body.get_text(" ", strip=True)

    # === Parse articles ===
    # EUR-Lex uses various structures. Find all "Article N" headings.
    article_pattern = re.compile(r"^Article\s+(\d+[a-z]?)$", re.MULTILINE)

    # Strategy: split the text by article headings
    # First, try structured HTML parsing
    articles_parsed = _parse_articles_structured(soup, regulation, celex)
    if articles_parsed:
        chunks.extend(articles_parsed)
    else:
        # Fallback: text-based splitting
        chunks.extend(_parse_articles_text(all_text, regulation, celex))

    # === Parse recitals ===
    recitals = _parse_recitals(soup, all_text, regulation, celex)
    chunks.extend(recitals)

    # === Parse annexes ===
    annexes = _parse_annexes(soup, all_text, regulation, celex)
    chunks.extend(annexes)

    return chunks


def _parse_articles_structured(
    soup: BeautifulSoup,
    regulation: str,
    celex: str,
) -> list[ParsedChunk]:
    """Try to parse articles from HTML structure."""
    chunks: list[ParsedChunk] = []

    # Find all elements containing "Article N" as heading
    # EUR-Lex typically uses <p class="oj-ti-art"> or <div class="eli-subdivision">
    article_headings = []

    # Method 1: Look for elements with article class patterns
    for el in soup.find_all(["p", "div", "span"]):
        text = el.get_text(strip=True)
        m = re.match(r"^Article\s+(\d+[a-z]?)$", text)
        if m:
            article_headings.append((m.group(1), el))

    if not article_headings:
        return []

    current_chapter = ""

    for i, (art_num, heading_el) in enumerate(article_headings):
        # Collect text until next article heading
        parts = []
        title = ""

        # The title is usually the next sibling element
        sibling = heading_el.find_next_sibling()
        if sibling and len(sibling.get_text(strip=True)) < 200:
            candidate = sibling.get_text(strip=True)
            # Title shouldn't start with a number (that's paragraph content)
            if candidate and not re.match(r"^\d+\.", candidate) and not candidate.startswith("("):
                title = candidate
                sibling = sibling.find_next_sibling()

        # Look for chapter headings above
        prev = heading_el.find_previous(["p", "div"])
        if prev:
            prev_text = prev.get_text(strip=True)
            chapter_m = re.match(r"(?:CHAPTER|Chapter|TITLE|Title|SECTION|Section)\s+([IVX\d]+)", prev_text)
            if chapter_m:
                current_chapter = prev_text

        # Collect paragraphs until next article
        if i < len(article_headings) - 1:
            next_heading_el = article_headings[i + 1][1]
        else:
            next_heading_el = None

        current = sibling
        while current:
            if next_heading_el and current == next_heading_el:
                break
            # Check if we hit another article heading
            current_text = current.get_text(strip=True)
            if re.match(r"^Article\s+\d+[a-z]?$", current_text):
                break
            if current_text:
                parts.append(current_text)
            current = current.find_next_sibling()

        full_text = f"Article {art_num} {title}\n\n" + "\n".join(parts)
        full_text = _clean_text(full_text)

        if len(full_text) < 30:
            continue

        article_id = f"Article {art_num}"
        chunk_id = f"{regulation}_art{art_num}"
        internal_refs, cross_regs = _extract_cross_refs(full_text, regulation)

        chunks.append(ParsedChunk(
            id=chunk_id,
            regulation=regulation,
            article_id=article_id,
            title=title,
            chapter=current_chapter,
            chunk_type="article",
            text=full_text,
            source_celex=celex,
            binding=True,
            cross_references=internal_refs,
            cross_regulation_refs=cross_regs,
        ))

        # Also create paragraph-level chunks for longer articles
        para_pattern = re.compile(r"(\d+)\.\s+(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
        for pm in para_pattern.finditer("\n".join(parts)):
            para_num = pm.group(1)
            para_text = _clean_text(pm.group(2))
            if len(para_text) < 50:
                continue

            para_id = f"{regulation}_art{art_num}_para{para_num}"
            p_internal, p_cross = _extract_cross_refs(para_text, regulation)

            chunks.append(ParsedChunk(
                id=para_id,
                regulation=regulation,
                article_id=article_id,
                title=title,
                chapter=current_chapter,
                chunk_type="paragraph",
                text=para_text,
                source_celex=celex,
                binding=True,
                cross_references=p_internal,
                cross_regulation_refs=p_cross,
            ))

    return chunks


def _parse_articles_text(
    all_text: str,
    regulation: str,
    celex: str,
) -> list[ParsedChunk]:
    """Fallback: parse articles from plain text using regex splitting."""
    chunks: list[ParsedChunk] = []

    # Split on "Article N" patterns
    parts = re.split(r"(Article\s+\d+[a-z]?)\s+", all_text)

    current_chapter = ""
    i = 1  # Skip preamble (index 0)
    while i < len(parts) - 1:
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        m = re.match(r"Article\s+(\d+[a-z]?)", heading)
        if not m:
            continue

        art_num = m.group(1)

        # Extract title (first line before paragraph content)
        lines = content.split("\n")
        title = ""
        body_start = 0
        if lines and not re.match(r"^\d+\.", lines[0]) and len(lines[0]) < 200:
            title = lines[0].strip()
            body_start = 1

        body = "\n".join(lines[body_start:])
        full_text = f"{heading} {title}\n\n{body}".strip()
        full_text = _clean_text(full_text)

        if len(full_text) < 30:
            continue

        chunk_id = f"{regulation}_art{art_num}"
        article_id = f"Article {art_num}"
        internal_refs, cross_regs = _extract_cross_refs(full_text, regulation)

        chunks.append(ParsedChunk(
            id=chunk_id,
            regulation=regulation,
            article_id=article_id,
            title=title,
            chapter=current_chapter,
            chunk_type="article",
            text=full_text,
            source_celex=celex,
            binding=True,
            cross_references=internal_refs,
            cross_regulation_refs=cross_regs,
        ))

    return chunks


def _parse_recitals(
    soup: BeautifulSoup,
    all_text: str,
    regulation: str,
    celex: str,
) -> list[ParsedChunk]:
    """Parse recitals — numbered (1), (2), etc. in the preamble."""
    chunks: list[ParsedChunk] = []

    # Recitals appear before Article 1, numbered like (1), (2), etc.
    # Find the preamble section
    art1_pos = all_text.find("Article 1")
    if art1_pos < 0:
        return chunks

    preamble = all_text[:art1_pos]

    # Split on recital numbers
    recital_parts = re.split(r"\((\d+)\)\s+", preamble)

    i = 1
    while i < len(recital_parts) - 1:
        rec_num = recital_parts[i]
        rec_text = recital_parts[i + 1].strip()
        i += 2

        rec_text = _clean_text(rec_text)
        if len(rec_text) < 30:
            continue

        chunk_id = f"{regulation}_rec{rec_num}"
        internal_refs, cross_regs = _extract_cross_refs(rec_text, regulation)

        chunks.append(ParsedChunk(
            id=chunk_id,
            regulation=regulation,
            article_id=f"Recital {rec_num}",
            title="",
            chapter="Preamble",
            chunk_type="recital",
            text=f"({rec_num}) {rec_text}",
            source_celex=celex,
            binding=False,
            cross_references=internal_refs,
            cross_regulation_refs=cross_regs,
        ))

    return chunks


def _parse_annexes(
    soup: BeautifulSoup,
    all_text: str,
    regulation: str,
    celex: str,
) -> list[ParsedChunk]:
    """Parse annexes from the regulation."""
    chunks: list[ParsedChunk] = []

    # Find annex markers
    annex_pattern = re.compile(r"ANNEX\s+([IVX]+|[A-Z])\b")
    matches = list(annex_pattern.finditer(all_text))

    for i, m in enumerate(matches):
        annex_id = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(all_text)

        annex_text = all_text[start:end].strip()
        # Limit to first 5000 chars per annex (they can be very long)
        if len(annex_text) > 5000:
            annex_text = annex_text[:5000] + "..."

        annex_text = _clean_text(annex_text)
        if len(annex_text) < 50:
            continue

        chunk_id = f"{regulation}_annex{annex_id.lower()}"
        internal_refs, cross_regs = _extract_cross_refs(annex_text, regulation)

        chunks.append(ParsedChunk(
            id=chunk_id,
            regulation=regulation,
            article_id=f"Annex {annex_id}",
            title="",
            chapter="Annexes",
            chunk_type="annex",
            text=annex_text,
            source_celex=celex,
            binding=True,
            cross_references=internal_refs,
            cross_regulation_refs=cross_regs,
        ))

    return chunks


def build_cross_reference_graph(
    chunks: list[ParsedChunk],
) -> dict[str, dict[str, list[str]]]:
    """Build a cross-reference graph from parsed chunks.

    Returns {chunk_id: {"references": [...], "referenced_by": [...]}}.
    """
    # Build article_id → chunk_id lookup
    article_to_chunk: dict[str, str] = {}
    for chunk in chunks:
        if chunk.chunk_type == "article":
            article_to_chunk[chunk.article_id] = chunk.id

    # Also map annexes
    for chunk in chunks:
        if chunk.chunk_type == "annex":
            article_to_chunk[chunk.article_id] = chunk.id

    graph: dict[str, dict[str, list[str]]] = {}
    for chunk in chunks:
        graph.setdefault(chunk.id, {"references": [], "referenced_by": []})

    # Wire up internal cross-references
    for chunk in chunks:
        for ref in chunk.cross_references:
            target_id = article_to_chunk.get(ref)
            if target_id and target_id != chunk.id:
                if target_id not in graph[chunk.id]["references"]:
                    graph[chunk.id]["references"].append(target_id)
                graph.setdefault(target_id, {"references": [], "referenced_by": []})
                if chunk.id not in graph[target_id]["referenced_by"]:
                    graph[target_id]["referenced_by"].append(chunk.id)

    return graph


def chunks_to_training_pairs(
    chunks: list[ParsedChunk],
    graph: dict[str, dict[str, list[str]]],
    *,
    min_edges: int = 0,
    min_text_length: int = 50,
) -> list[dict]:
    """Convert parsed chunks + graph into GibsGraph training pairs.

    Same format as build_training_pairs.py output.
    """
    chunk_map = {c.id: c for c in chunks}
    pairs = []

    for chunk in chunks:
        if len(chunk.text) < min_text_length:
            continue

        node_data = graph.get(chunk.id, {"references": [], "referenced_by": []})
        references = node_data["references"]
        referenced_by = node_data["referenced_by"]

        # Build forward edges
        edges = []
        target_nodes = []
        has_cross_reg = len(chunk.cross_regulation_refs) > 0

        for target_id in references:
            target_chunk = chunk_map.get(target_id)
            if not target_chunk:
                continue

            # Determine edge type
            if target_chunk.regulation != chunk.regulation:
                edge_type = "CROSS_REGULATES"
            elif target_chunk.chunk_type == "recital" or chunk.chunk_type == "recital":
                edge_type = "INTERPRETS"
            elif target_chunk.chunk_type == "annex":
                edge_type = "REFERENCES"
            else:
                edge_type = "REFERENCES"

            edges.append({
                "source": chunk.id,
                "target": target_id,
                "type": edge_type,
                "weight": EDGE_WEIGHTS.get(edge_type, 0.8),
            })
            target_nodes.append({
                "id": target_id,
                "label": _chunk_type_to_label(target_chunk.chunk_type),
                "properties": _chunk_to_props(target_chunk),
            })

        # Build reverse edges
        reverse_edges = []
        for source_id in referenced_by:
            source_chunk = chunk_map.get(source_id)
            if not source_chunk:
                continue

            if source_chunk.regulation != chunk.regulation:
                edge_type = "CROSS_REGULATES"
            elif source_chunk.chunk_type == "recital" or chunk.chunk_type == "recital":
                edge_type = "INTERPRETS"
            else:
                edge_type = "REFERENCES"

            reverse_edges.append({
                "source": source_id,
                "target": chunk.id,
                "type": edge_type,
                "weight": EDGE_WEIGHTS.get(edge_type, 0.8),
            })

        # Add cross-regulation placeholder edges
        for cross_reg in chunk.cross_regulation_refs:
            edge_id = f"{cross_reg}_external"
            edges.append({
                "source": chunk.id,
                "target": edge_id,
                "type": "CROSS_REGULATES",
                "weight": EDGE_WEIGHTS["CROSS_REGULATES"],
            })
            target_nodes.append({
                "id": edge_id,
                "label": "Regulation",
                "properties": {
                    "regulation": cross_reg,
                    "chunk_id": edge_id,
                    "article_id": "",
                    "title": cross_reg.upper().replace("_", " "),
                    "chapter": "",
                    "chunk_type": "regulation",
                    "binding": True,
                },
            })
            has_cross_reg = True

        total_edges = len(edges) + len(reverse_edges)
        if total_edges < min_edges:
            continue

        pair = {
            "id": chunk.id,
            "input_text": chunk.text,
            "metadata": {
                "regulation": chunk.regulation,
                "article_id": chunk.article_id,
                "chunk_type": chunk.chunk_type,
                "chapter": chunk.chapter,
                "title": chunk.title,
                "source_celex": chunk.source_celex,
                "cross_references_in_text": chunk.cross_references,
                "cross_regulation_refs": chunk.cross_regulation_refs,
            },
            "expected_graph": {
                "source_node": {
                    "id": chunk.id,
                    "label": _chunk_type_to_label(chunk.chunk_type),
                    "properties": _chunk_to_props(chunk),
                },
                "target_nodes": target_nodes,
                "edges": edges,
                "reverse_edges": reverse_edges,
            },
            "quality": {
                "has_cross_references": len(chunk.cross_references) > 0,
                "cross_ref_count": len(chunk.cross_references),
                "edge_count": len(edges),
                "reverse_edge_count": len(reverse_edges),
                "has_cross_regulation": has_cross_reg,
                "text_length": len(chunk.text),
                "verified_source": "eurlex_html",
            },
        }
        pairs.append(pair)

    return pairs


def _chunk_type_to_label(chunk_type: str) -> str:
    """Convert chunk type to PascalCase Neo4j label."""
    mapping = {
        "article": "Article",
        "paragraph": "Paragraph",
        "recital": "Recital",
        "annex": "Annex",
        "regulation": "Regulation",
    }
    return mapping.get(chunk_type, "Article")


def _chunk_to_props(chunk: ParsedChunk) -> dict:
    """Convert chunk to node properties dict."""
    return {
        "chunk_id": chunk.id,
        "regulation": chunk.regulation,
        "article_id": chunk.article_id,
        "title": chunk.title,
        "chapter": chunk.chapter,
        "chunk_type": chunk.chunk_type,
        "binding": chunk.binding,
    }


def print_parse_stats(chunks: list[ParsedChunk], regulation: str) -> None:
    """Print parsing statistics."""
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1

    total_refs = sum(len(c.cross_references) for c in chunks)
    total_cross_regs = sum(len(c.cross_regulation_refs) for c in chunks)
    cross_reg_names = set()
    for c in chunks:
        cross_reg_names.update(c.cross_regulation_refs)

    print(f"\n  Parsed {regulation.upper()}:")
    print(f"    Total chunks: {len(chunks)}")
    for ct, count in sorted(by_type.items()):
        print(f"      {ct}: {count}")
    print(f"    Internal cross-references: {total_refs}")
    print(f"    Cross-regulation references: {total_cross_regs}")
    if cross_reg_names:
        print(f"    Referenced regulations: {', '.join(sorted(cross_reg_names))}")


def build_regulation_pairs(
    celex: str,
    regulation: str,
    output_path: Path,
    cache_dir: Path | None = None,
    *,
    min_edges: int = 0,
    min_text_length: int = 50,
    stats_only: bool = False,
) -> list[dict]:
    """Full pipeline: fetch → parse → build graph → generate training pairs."""
    print(f"Building training pairs for {regulation.upper()} (CELEX: {celex})...")

    # Fetch
    html = fetch_regulation_html(celex, cache_dir=cache_dir)

    # Parse
    chunks = parse_regulation(html, regulation, celex)
    print_parse_stats(chunks, regulation)

    if not chunks:
        print(f"  WARNING: No chunks parsed for {regulation}!")
        return []

    # Build cross-reference graph
    graph = build_cross_reference_graph(chunks)

    # Generate training pairs
    pairs = chunks_to_training_pairs(
        chunks, graph,
        min_edges=min_edges,
        min_text_length=min_text_length,
    )

    # Statistics
    total = len(pairs)
    with_edges = sum(1 for p in pairs if p["quality"]["edge_count"] > 0)
    with_cross = sum(1 for p in pairs if p["quality"]["has_cross_regulation"])
    total_fwd = sum(p["quality"]["edge_count"] for p in pairs)
    total_rev = sum(p["quality"]["reverse_edge_count"] for p in pairs)

    print(f"\n  Training pairs: {total}")
    print(f"    With forward edges:     {with_edges}")
    print(f"    With cross-regulation:  {with_cross}")
    print(f"    Total forward edges:    {total_fwd}")
    print(f"    Total reverse edges:    {total_rev}")

    if stats_only:
        return pairs

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Written {total} pairs to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return pairs
