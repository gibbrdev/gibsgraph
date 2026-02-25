"""Generate vector embeddings for the expert knowledge graph.

Uses sentence-transformers (all-MiniLM-L6-v2) to create embeddings
for all parsed Neo4j knowledge entries. Produces:

  - data/processed/embeddings.npz       (numpy arrays)
  - data/processed/embeddings_meta.jsonl (text + metadata per vector)

This is the first open Neo4j expert embedding dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality


def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_documents() -> list[dict]:
    """Build text documents from all JSONL sources with metadata."""
    docs: list[dict] = []

    # Cypher clauses
    for c in load_jsonl(PROCESSED_DIR / "cypher_clauses.jsonl"):
        text = f"{c['name']}: {c['description']}"
        docs.append({
            "text": text,
            "type": "cypher_clause",
            "name": c["name"],
            "source": c.get("source_file", ""),
        })

    # Cypher functions
    for f in load_jsonl(PROCESSED_DIR / "cypher_functions.jsonl"):
        sig = f.get("signature", "")
        text = f"{f['name']}: {f['description']}"
        if sig:
            text += f" Signature: {sig}"
        docs.append({
            "text": text,
            "type": "cypher_function",
            "name": f["name"],
            "category": f.get("category", ""),
            "source": f.get("source_file", ""),
        })

    # Cypher examples
    seen = set()
    for ex in load_jsonl(PROCESSED_DIR / "cypher_examples.jsonl"):
        cypher = ex["cypher"]
        if cypher in seen:
            continue
        seen.add(cypher)
        text = f"{ex['description']} Cypher: {cypher}"
        docs.append({
            "text": text[:500],
            "type": "cypher_example",
            "category": ex.get("category", ""),
            "context": ex.get("context", ""),
        })

    # Modeling patterns
    seen_p = set()
    for p in load_jsonl(PROCESSED_DIR / "modeling_patterns.jsonl"):
        if p["name"] in seen_p:
            continue
        seen_p.add(p["name"])
        text = f"{p['name']}: {p['description']}"
        docs.append({
            "text": text[:500],
            "type": "modeling_pattern",
            "name": p["name"],
        })

    # Best practices
    seen_bp = set()
    for bp in load_jsonl(PROCESSED_DIR / "best_practices.jsonl"):
        title = bp["title"]
        if not title or title in seen_bp:
            continue
        seen_bp.add(title)
        text = f"{title}: {bp['description']}"
        docs.append({
            "text": text[:500],
            "type": "best_practice",
            "name": title,
            "category": bp.get("category", "general"),
        })

    return docs


def main() -> None:
    docs = build_documents()
    print(f"Built {len(docs)} documents for embedding")

    # Count by type
    from collections import Counter
    type_counts = Counter(d["type"] for d in docs)
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Load model and encode
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings as numpy
    out_npz = PROCESSED_DIR / "embeddings.npz"
    np.savez_compressed(out_npz, embeddings=embeddings)
    print(f"Saved: {out_npz} ({out_npz.stat().st_size / 1024:.0f} KB)")

    # Save metadata
    out_meta = PROCESSED_DIR / "embeddings_meta.jsonl"
    with open(out_meta, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            doc["embedding_index"] = i
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Saved: {out_meta} ({out_meta.stat().st_size / 1024:.0f} KB)")

    print(f"\nDone. {len(docs)} vectors x {embeddings.shape[1]} dimensions")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
