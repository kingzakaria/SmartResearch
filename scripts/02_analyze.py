"""Step 2: documents.json -> terms.json, and a comparison of the morphology modes.
Run from the project root: python scripts/02_analyze.py

Produces:
  data/processed/terms.json        {doc_id: [terms]} for the mode set in config.ANALYZER_MODE
  reports/mode_comparison.csv      vocabulary size per mode (table for the report)
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

MODES = ("word", "light", "root")
BOILERPLATE_DF = 0.80          # terms present in more than 80% of documents are stopword candidates


def main() -> None:
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    n = len(docs)
    print(f"{n} documents loaded. Analyzing the 'body' of each one (header lines excluded).\n")

    rows, main_terms = [], None
    for mode in MODES:
        analyzer = analyzer_from_config(config, mode)
        terms = {str(d["doc_id"]): analyzer.analyze(d["body"]) for d in docs}
        df = Counter(t for ts in terms.values() for t in set(ts))
        total = sum(len(ts) for ts in terms.values())
        rows.append({"mode": mode, "tokens": total, "vocabulary": len(df),
                     "hapax_df1": sum(1 for c in df.values() if c == 1),
                     "avg_terms_per_doc": round(total / n)})
        if mode == config.ANALYZER_MODE:
            main_terms, main_df = terms, df

    print(f"{'mode':8s}{'tokens':>10s}{'vocabulary':>12s}{'df=1 terms':>12s}{'terms/doc':>11s}")
    for r in rows:
        print(f"{r['mode']:8s}{r['tokens']:>10d}{r['vocabulary']:>12d}{r['hapax_df1']:>12d}{r['avg_terms_per_doc']:>11d}")
    base = rows[0]["vocabulary"]
    for r in rows[1:]:
        print(f"  {r['mode']}: vocabulary reduced by {100 * (1 - r['vocabulary'] / base):.1f}% compared with 'word'")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "mode_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.TERMS_JSON, "w", encoding="utf-8") as fh:
        json.dump(main_terms, fh, ensure_ascii=False)
    print(f"\nTerms of mode '{config.ANALYZER_MODE}' saved -> {config.TERMS_JSON}")

    print(f"\n30 most frequent terms (mode '{config.ANALYZER_MODE}'):")
    freq = Counter(t for ts in main_terms.values() for t in ts)
    print("  " + " | ".join(f"{t} ({c})" for t, c in freq.most_common(30)))

    cands = [(t, c) for t, c in main_df.most_common() if c / n > BOILERPLATE_DF]
    print(f"\nTerms present in more than {int(BOILERPLATE_DF * 100)}% of documents "
          f"({len(cands)}): stopword candidates, review them by hand")
    print("  " + " | ".join(f"{t} ({100 * c / n:.0f}%)" for t, c in cands[:40]))


if __name__ == "__main__":
    main()
