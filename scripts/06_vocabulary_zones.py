"""Step 5b: the three zones of the vocabulary (frequent / kept / rare).
Run from the project root: python scripts/06_vocabulary_zones.py
Writes reports/vocabulary_zones.csv. Thresholds: MIN_DOC_FREQ and MAX_DOC_FREQ_RATIO in config.py.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.vocabulary import FREQUENT, KEPT, RARE, classify_terms, summary  # noqa: E402


def main() -> None:
    index = InvertedIndex.load(config.INDEX_PKL)
    zones = classify_terms(index, config.MIN_DOC_FREQ, config.MAX_DOC_FREQ_RATIO)
    stats = summary(index, zones)
    n_terms, n_tokens = len(zones), sum(v["tokens"] for v in stats.values())

    print(f"{index.n_docs} documents. Frequent zone: df > {config.MAX_DOC_FREQ_RATIO:.0%} "
          f"(> {config.MAX_DOC_FREQ_RATIO * index.n_docs:.0f} docs). Rare zone: df < {config.MIN_DOC_FREQ}.\n")
    print(f"{'zone':<10}{'distinct terms':>16}{'share':>9}{'occurrences':>14}{'share':>9}")
    for zone in (FREQUENT, KEPT, RARE):
        s = stats[zone]
        print(f"{zone:<10}{s['terms']:>16}{s['terms'] / n_terms:>9.1%}{s['tokens']:>14}{s['tokens'] / n_tokens:>9.1%}")

    by_df = sorted(index.postings, key=lambda t: (-index.df(t), t))
    for zone in (FREQUENT, KEPT, RARE):
        members = [t for t in by_df if zones[t] == zone]
        sample = members[:12] if zone != RARE else members[-12:]
        print(f"\n{zone} (examples): " + " | ".join(f"{t} ({index.df(t)})" for t in sample))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "vocabulary_zones.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "df", "df_ratio", "occurrences", "zone"])
        for t in by_df:
            writer.writerow([t, index.df(t), f"{index.df(t) / index.n_docs:.3f}", sum(index.postings[t].values()), zones[t]])
    print(f"\nSaved: {config.REPORTS_DIR / 'vocabulary_zones.csv'} (open it in Excel and read the middle zone)")


if __name__ == "__main__":
    main()
