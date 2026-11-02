"""Step 7b: category labels from the subject line (data/processed/labels.csv).
Run from the project root: python scripts/10_label_by_subject.py
Rules live in data/resources/category_rules.json: edit them, run again, until the 'other' list is short and
the examples of each category look right.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.analysis.categories import UNLABELLED, assign_category, load_rules, subject_head  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

RULES = config.RESOURCES_DIR / "category_rules.json"


def main() -> None:
    analyzer = analyzer_from_config(config)
    rules = load_rules(RULES, analyzer)
    names_ar = {name: ar for name, ar, _ in rules}
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    drop = set(dup.get("drop", []))
    docs = [d for d in docs if str(d["doc_id"]) not in drop]

    labelled = [(d, *assign_category(d.get("subject"), rules, analyzer)) for d in docs]
    counts = Counter(c for _, c, _ in labelled)
    by_head = sum(1 for _, c, how in labelled if how == "head")
    print(f"{len(docs)} decisions (duplicates set aside)\n")
    print(f"{'category':<14}{'decisions':>10}{'share':>8}")
    for name in [r[0] for r in rules] + [UNLABELLED]:
        print(f"{name:<14}{counts[name]:>10}{counts[name] / len(docs):>8.1%}   {names_ar.get(name, '')}")
    print(f"\nMatched on the head of the subject: {by_head}; on the whole subject: "
          f"{len(docs) - by_head - counts[UNLABELLED]}; not matched: {counts[UNLABELLED]}")

    for name in [r[0] for r in rules]:
        print(f"\n--- {name}: 4 examples (check that they belong)")
        for d, c, _ in [x for x in labelled if x[1] == name][:4]:
            print(f"   {d.get('decision_no')}: {d.get('subject')}")
    others = [d for d, c, _ in labelled if c == UNLABELLED]
    print(f"\n--- other: {len(others)} subjects without a category (add keywords for the frequent ones)")
    for head, n in Counter(subject_head(d.get("subject")) for d in others).most_common(30):
        print(f"   {n:>3}  {head}")

    with open(config.LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "category", "matched_by", "decision_no", "subject"])
        for d, c, how in labelled:
            writer.writerow([d["doc_id"], c, how, d.get("decision_no"), d.get("subject")])
    print(f"\nSaved: {config.LABELS_CSV}")


if __name__ == "__main__":
    main()
