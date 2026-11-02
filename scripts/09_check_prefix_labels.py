"""Step 7a: is the first number of the decision number (1/2/3/4) a real category of the decisions?
Run from the project root: python scripts/09_check_prefix_labels.py
For each prefix: its size, the words most linked to it (chi-square) and three example subjects. If every prefix has
its own coherent vocabulary, the prefix can be used as the classification label.
Needs terms.json, documents.json and (optional) models/partitions.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.analysis.labels import adjusted_rand_index, prefix_of, purity, top_terms_by_class  # noqa: E402


def main() -> None:
    terms = json.loads(config.TERMS_JSON.read_text(encoding="utf-8"))
    docs = {str(d["doc_id"]): d for d in json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))}
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    drop = set(dup.get("drop", []))
    labels = {d: prefix_of(docs[d].get("decision_no")) for d in terms if d not in drop}
    kept = {d: terms[d] for d in labels}
    sizes = Counter(v for v in labels.values() if v)
    print(f"{len(kept)} decisions (duplicates set aside). With a prefix: {sum(sizes.values())}; "
          f"without: {len(kept) - sum(sizes.values())}.  Prefix sizes: {dict(sorted(sizes.items()))}\n")

    top = top_terms_by_class(kept, labels, n=12)
    for prefix in sorted(top):
        print(f"=== Prefix {prefix}  ({sizes[prefix]} decisions)")
        print("  words most linked to it (chi-square): " + " | ".join(f"{t} ({s:.0f})" for t, s in top[prefix]))
        examples = [d for d in kept if labels[d] == prefix][:60]
        for d in sorted(examples, key=lambda x: -len(docs[x].get("subject") or ""))[:3]:
            print(f"    {docs[d].get('decision_no')}: {docs[d].get('subject')}")
        print()

    if config.PARTITIONS_JSON.exists():
        groups = {d: g for d, g in json.loads(config.PARTITIONS_JSON.read_text(encoding="utf-8"))["assignments"].items()
                  if d in labels}
        print(f"Agreement between the K-Means groups and the prefixes: purity = {purity(groups, labels):.1%}, "
              f"adjusted Rand index = {adjusted_rand_index(groups, labels):.3f}  (1 = identical, 0 = random)")
    print("\nDecision: if the word lists above are clearly different from one prefix to another, use the prefix as the label.")


if __name__ == "__main__":
    main()
