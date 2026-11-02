"""Step 5a: find the near-duplicate decisions.
Run from the project root: python scripts/05_find_duplicates.py
Writes data/processed/duplicates.json (groups and documents to drop) and reports/duplicates.csv.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.analysis.duplicates import documents_to_drop, group_duplicates, jaccard_pairs, select_duplicate_pairs  # noqa: E402


def main() -> None:
    terms = json.loads(config.TERMS_JSON.read_text(encoding="utf-8"))
    docs = {str(d["doc_id"]): d for d in json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))}
    pairs = jaccard_pairs(terms, min_score=0.5)
    duplicate_pairs = select_duplicate_pairs(pairs, docs, config.DUPLICATE_THRESHOLD, config.DUPLICATE_STRICT)
    groups = group_duplicates(duplicate_pairs, 0.0)                    # the same decision published twice
    drop = documents_to_drop(groups, docs)
    families = [g for g in group_duplicates(pairs, config.FAMILY_THRESHOLD)    # decisions sharing a template
                if len(set(g) - set(drop)) > 1]

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "duplicates.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_a", "doc_b", "jaccard", "same_case_no", "decision_a", "decision_b", "case_a", "case_b"])
        for a, b, score in pairs:
            da, db = docs.get(a, {}), docs.get(b, {})
            same = bool(da.get("case_no")) and da.get("case_no") == db.get("case_no")
            writer.writerow([a, b, f"{score:.3f}", same, da.get("decision_no"), db.get("decision_no"),
                             da.get("case_no"), db.get("case_no")])
    config.DUPLICATES_JSON.write_text(json.dumps({"threshold": config.DUPLICATE_THRESHOLD, "family_threshold": config.FAMILY_THRESHOLD, "groups": groups,
                                                  "drop": drop, "families": families}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Documents: {len(terms)}   pairs with Jaccard >= 0.5: {len(pairs)}   "
          f"duplicates (same case, >= {config.DUPLICATE_THRESHOLD}): {len(groups)} groups, {len(drop)} documents to drop\n"
          f"Template families (>= {config.FAMILY_THRESHOLD}, kept but never split by cross-validation): {len(families)}, "
          f"largest has {max((len(f) for f in families), default=0)} decisions\n")
    print("Closest pairs (check a few by hand: are they really the same decision?):")
    for a, b, score in pairs[:15]:
        da, db = docs.get(a, {}), docs.get(b, {})
        print(f"  {score:.3f}  doc {a} ({da.get('decision_no')}, {da.get('case_no')})  <->  "
              f"doc {b} ({db.get('decision_no')}, {db.get('case_no')})")
    print("\nLargest template families:")
    for family in sorted(families, key=len, reverse=True)[:5]:
        print("  " + ", ".join(f"{d} ({docs.get(d, {}).get('decision_no')})" for d in family))
    print(f"\nSaved: {config.DUPLICATES_JSON}  and  {config.REPORTS_DIR / 'duplicates.csv'}")


if __name__ == "__main__":
    main()
