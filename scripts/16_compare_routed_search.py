"""Search in every chamber against search routed by Naive Bayes ("auto").
Run from the project root: python scripts/16_compare_routed_search.py

For each decision with a subject, two queries (the subject, and 3 of its words) are searched twice: in every chamber
and with chamber="auto". Reported: how often the search was narrowed to one chamber, how much of the full top 10 the
routed search keeps, how many decisions were scored, and the time.

Caution: the saved model was trained on all decisions, including the one a query comes from, so the routing here is a
little optimistic. The held-out accuracy of the router is the one printed by scripts/15_train_naive_bayes.py.
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402


def main() -> None:
    engine = SearchEngine.from_config(config)
    if engine.router is None:
        sys.exit("No models/naive_bayes.json: run scripts/15_train_naive_bayes.py first.")
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    drop = {str(d) for d in dup.get("drop", [])}
    rng = random.Random(config.RANDOM_SEED)

    queries = []
    for doc_id, doc in engine.documents.items():
        if doc_id in drop or not doc.get("subject"):
            continue
        terms = engine.analyzer.analyze(doc["subject"])
        if not terms:
            continue
        queries.append(("subject", doc["subject"], engine.chambers[doc_id]))
        queries.append(("3 words", " ".join(rng.sample(terms, min(3, len(terms)))), engine.chambers[doc_id]))
    print(f"{len(engine.documents)} decisions, {len(queries)} queries, confidence threshold {config.CHAMBER_MIN_CONFIDENCE:.0%}\n")

    rows = []
    for kind, q, true_chamber in queries:
        t0 = time.perf_counter()
        full = engine.search(q, k=10, chamber="all")
        t1 = time.perf_counter()
        routed = engine.search(q, k=10, chamber="auto")
        t2 = time.perf_counter()
        if not full.results:
            continue
        narrowed = len(routed.chamber_routing["searched"]) == 1
        top_full = [r.doc_id for r in full.results]
        top_routed = {r.doc_id for r in routed.results}
        rows.append({"kind": kind, "narrowed": narrowed, "correct": narrowed and routed.chamber_routing["searched"][0]["chamber"] == true_chamber,
                     "kept10": sum(d in top_routed for d in top_full) / len(top_full),
                     "top1": bool(routed.results) and routed.results[0].doc_id == top_full[0],
                     "scored": routed.scored / max(1, full.scored), "ms_full": (t1 - t0) * 1000, "ms_routed": (t2 - t1) * 1000})

    def mean(items, key):
        return sum(r[key] for r in items) / len(items) if items else 0.0

    print(f"{'queries':<26}{'n':>6}{'narrowed':>10}{'right chamber':>15}{'top 10 kept':>13}{'top 1 kept':>12}{'scored':>9}{'ms full -> routed':>20}")
    for name, items in [("all", rows)] + [(k, [r for r in rows if r["kind"] == k]) for k in ("subject", "3 words")] + \
                       [("narrowed to one chamber", [r for r in rows if r["narrowed"]]),
                        ("searched in both (not sure)", [r for r in rows if not r["narrowed"]])]:
        if not items:
            continue
        narrowed = [r for r in items if r["narrowed"]]
        right = mean(narrowed, "correct") if narrowed else float("nan")
        print(f"{name:<26}{len(items):>6}{mean(items, 'narrowed'):>10.1%}{right:>15.1%}{mean(items, 'kept10'):>13.1%}"
              f"{mean(items, 'top1'):>12.1%}{mean(items, 'scored'):>9.1%}{mean(items, 'ms_full'):>10.2f} -> {mean(items, 'ms_routed'):.2f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "routed_search_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {config.REPORTS_DIR / 'routed_search_comparison.csv'}")


if __name__ == "__main__":
    main()
