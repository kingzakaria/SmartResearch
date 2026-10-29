"""Step 1: raw .txt files -> data/processed/documents.json
Run from the project root: python scripts/01_extract.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.extraction.loader import list_files, read_file  # noqa: E402
from src.extraction.metadata import parse_document  # noqa: E402

FIELDS = ["decision_no", "date", "case_no", "subject"]


def main() -> None:
    files = list_files(config.RAW_DIR)
    if not files:
        print(f"No .txt files found in {config.RAW_DIR}")
        return

    documents, empty = [], []
    for f in files:
        raw = read_file(f, config.ENCODING)
        if not raw.strip():
            empty.append(f.name)
            continue
        doc = parse_document(raw)
        words = len(doc["body"].split())
        documents.append({
            "doc_id": int(f.stem) if f.stem.isdigit() else f.stem,
            "filename": f.name,
            "decision_no": doc["decision_no"],
            "date": doc["date"],
            "year": doc["year"],
            "case_no": doc["case_no"],
            "subject": doc["subject"],
            "n_words": words,
            "category": None,          # filled later by clustering (Step 5)
            "text": doc["text"],       # full cleaned text (used for RAG chunks and display)
            "body": doc["body"],       # text without the metadata lines (used for indexing)
        })

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DOCUMENTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=1)

    n = len(documents)
    print(f"Files read: {len(files)} | empty (skipped): {len(empty)} {empty}")
    print(f"Documents saved: {n} -> {config.DOCUMENTS_JSON}\n")

    print("Metadata coverage:")
    for field in FIELDS:
        ok = sum(d[field] is not None for d in documents)
        print(f"  {field:12s}: {ok}/{n} ({100 * ok / n:.1f}%)")
    complete = sum(all(d[k] is not None for k in FIELDS) for d in documents)
    print(f"  all 4 fields : {complete}/{n} ({100 * complete / n:.1f}%)")

    years = Counter(d["year"] for d in documents if d["year"])
    print(f"\nDecisions per year: {dict(sorted(years.items()))}")

    for field in FIELDS:
        missing = [d["filename"] for d in documents if d[field] is None]
        if missing:
            print(f"\nMissing {field} ({len(missing)}): {missing[:20]}")

    # these show what a wrong 'subject' looks like: very long or very short ones
    odd = [d for d in documents if d["subject"] and (len(d["subject"]) > 300 or len(d["subject"]) < 8)]
    if odd:
        print(f"\nSuspicious subjects ({len(odd)}), check by eye:")
        for d in odd[:8]:
            print(f"  {d['filename']}: {d['subject'][:120]!r}")


if __name__ == "__main__":
    main()