"""Step 3: terms.json -> inverted index + TF-IDF model (models/index.pkl).
Run from the project root: python scripts/03_build_index.py
Needs data/processed/terms.json (scripts/02_analyze.py) and documents.json (scripts/01_extract.py).
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel  # noqa: E402


def main() -> None:
    terms = json.loads(config.TERMS_JSON.read_text(encoding="utf-8"))
    docs = {str(d["doc_id"]): d for d in json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))}

    start = time.perf_counter()
    index = InvertedIndex.build(terms)
    index.meta = {"mode": config.ANALYZER_MODE, "log_base": config.LOG_BASE}   # checked by the search engine
    model = TfidfModel(index, config.LOG_BASE)
    elapsed = time.perf_counter() - start
    index.save(config.INDEX_PKL)

    n_postings = sum(len(p) for p in index.postings.values())
    zero_idf = sum(1 for v in model.idf.values() if v == 0)
    size_mb = Path(config.INDEX_PKL).stat().st_size / 1e6
    print(f"Index built in {elapsed:.2f}s  ->  {config.INDEX_PKL} ({size_mb:.1f} MB)\n")
    print(f"Documents (|D|)           : {index.n_docs}")
    print(f"Vocabulary                : {len(index.postings)} terms")
    print(f"Postings (term, doc) pairs: {n_postings}")
    print(f"Average document length   : {index.avg_doc_len:.0f} terms")
    print(f"Terms with IDF = 0        : {zero_idf} (in every document, so they carry no weight)")

    print("\nHighest document frequency (these get the lowest IDF):")
    top_df = sorted(index.postings, key=index.df, reverse=True)[:12]
    print("  " + " | ".join(f"{t} (df={index.df(t)}, idf={model.idf[t]:.3f})" for t in top_df))

    print("\nRarest useful terms get the highest IDF, e.g. df=2:")
    rare = [t for t in index.postings if index.df(t) == 2][:8]
    print("  " + " | ".join(f"{t} (idf={model.idf[t]:.2f})" for t in rare))

    random.seed(config.RANDOM_SEED)
    print("\nAutomatic keywords (top TF-IDF terms) of 5 random documents. Compare them with the subject:")
    for doc_id in random.sample(sorted(terms), min(5, len(terms))):
        d = docs.get(doc_id, {})
        keywords = " | ".join(f"{t} ({w:.4f})" for t, w in model.top_terms(doc_id, 8))
        print(f"\n  doc {doc_id}  subject: {d.get('subject')}")
        print(f"    {keywords}")


if __name__ == "__main__":
    main()
