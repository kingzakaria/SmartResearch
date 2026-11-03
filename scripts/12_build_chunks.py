"""RAG step 1: cut the decisions into passages and analyse them for the keyword retriever.
Run from the project root: python scripts/12_build_chunks.py

Reads   data/processed/documents.json (and duplicates.json to skip the copies)
Writes  data/processed/chunks.json
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.extraction.cleaner import strip_panel_names  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.rag.chunker import build_chunks  # noqa: E402


def main() -> None:
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    skip = dup.get("drop", [])
    analyzer = analyzer_from_config(config)

    chunks = build_chunks(docs, skip, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    for chunk in chunks:
        chunk["terms"] = analyzer.analyze(strip_panel_names(chunk["text"]))

    lengths = [len(c["text"].split()) for c in chunks]
    per_doc = {}
    for c in chunks:
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
    print(f"{len(docs)} decisions, {len(skip)} duplicates skipped -> {len(chunks)} passages")
    print(f"passages per decision: min {min(per_doc.values())}, median {statistics.median(per_doc.values())}, "
          f"max {max(per_doc.values())}")
    print(f"words per passage: min {min(lengths)}, median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print(f"passages without any indexed term: {sum(1 for c in chunks if not c['terms'])}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"mode": analyzer.mode, "size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP}
    config.CHUNKS_JSON.write_text(json.dumps({"meta": meta, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    print(f"saved {config.CHUNKS_JSON}")
    print("next: python scripts/13_embed_chunks.py   (needs the Gemini key in .env)")


if __name__ == "__main__":
    main()
