"""Search evaluation, step 2: precision@k, recall, MAP and MRR of each system, from your judgments.
Run from the project root: python scripts/18_evaluate_search.py

Reads    reports/eval_pool.csv   (the column "relevant" filled with 1 / 0 by you)
Writes   data/eval/judgments.csv (query_id, doc_id, relevant: small, no text, can be committed)
         reports/eval_results.csv (every metric of every query and system)

Reading the numbers: P@k = share of the k first results that are relevant; R@10 = share of the relevant decisions found in the 10
first; AP = precision averaged at the rank of each relevant result (its mean over the queries is the MAP); MRR = mean of
1 / rank of the first relevant result. Only the decisions in the pool were judged, the others count as not relevant.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.evaluation.judgments import read_compact, read_pool, write_compact  # noqa: E402
from src.evaluation.metrics import average_precision, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from src.evaluation.systems import build_systems  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402

POOL_CSV = config.REPORTS_DIR / "eval_pool.csv"
JUDGMENTS = config.EVAL_DIR / "judgments.csv"
DEPTH = 50


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    if not POOL_CSV.exists():
        sys.exit("No reports/eval_pool.csv: run python scripts/17_build_eval_pool.py first.")
    rows = read_pool(POOL_CSV)
    judged = read_compact(JUDGMENTS)
    judged.update({(r["query_id"], r["doc_id"]): r["relevant"] for r in rows if r["relevant"] is not None})
    write_compact(JUDGMENTS, judged)
    blank = sum(1 for r in rows if r["relevant"] is None)
    if blank:
        print(f"WARNING: {blank} of {len(rows)} decisions have no judgment yet: they count as NOT relevant.\n")

    queries = json.loads((config.EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    relevant = {q["id"]: {d for (qid, d), v in judged.items() if qid == q["id"] and v == 1} for q in queries}
    usable = [q for q in queries if relevant[q["id"]]]
    for q in queries:
        if not relevant[q["id"]]:
            print(f"  {q['id']} {q['query']!r}: no decision judged relevant, left out of the averages")
    if not usable:
        sys.exit("No query has a relevant decision: fill the column 'relevant' of the pool first.")

    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    exclude = {str(d) for d in dup.get("drop", [])}
    systems = build_systems(config, docs, exclude, SearchEngine.from_config(config))

    results = {s.name: {} for s in systems}
    for system in systems:
        for q in usable:
            ranked = system.rank(q["query"], DEPTH)
            rel = relevant[q["id"]]
            results[system.name][q["id"]] = {
                "P@5": precision_at_k(ranked, rel, 5), "P@10": precision_at_k(ranked, rel, 10),
                "R@10": recall_at_k(ranked, rel, 10), "AP": average_precision(ranked, rel),
                "RR": reciprocal_rank(ranked, rel)}

    print(f"{len(usable)} queries with at least one relevant decision, {sum(len(relevant[q['id']]) for q in usable)} relevant decisions in total\n")
    header = f"{'system':<16}{'P@5':>8}{'P@10':>8}{'R@10':>8}{'MAP':>8}{'MRR':>8}"
    for title, group in [("ALL QUERIES", usable)] + [(f"chamber: {c}", [q for q in usable if q.get("chamber") == c])
                                                     for c in sorted({q.get("chamber") for q in usable if q.get("chamber")})]:
        if not group:
            continue
        print(f"{title} ({len(group)} queries)")
        print(header)
        for s in systems:
            r = results[s.name]
            print(f"{s.name:<16}{mean(r[q['id']]['P@5'] for q in group):>8.3f}{mean(r[q['id']]['P@10'] for q in group):>8.3f}"
                  f"{mean(r[q['id']]['R@10'] for q in group):>8.3f}{mean(r[q['id']]['AP'] for q in group):>8.3f}"
                  f"{mean(r[q['id']]['RR'] for q in group):>8.3f}")
        print()

    print("Average precision (AP) of each query:")
    print(f"{'query':<34}{'rel.':>5}" + "".join(f"{s.name:>15}" for s in systems))
    for q in usable:
        print(f"{q['id'] + ' ' + q['query']:<34}{len(relevant[q['id']]):>5}" + "".join(f"{results[s.name][q['id']]['AP']:>15.3f}" for s in systems))

    base = "light"
    print(f"\nQueries won / tied / lost against '{base}' (by AP, difference above 0.01):")
    for s in systems:
        if s.name == base:
            continue
        diffs = [results[s.name][q["id"]]["AP"] - results[base][q["id"]]["AP"] for q in usable]
        print(f"  {s.name:<16}{sum(d > 0.01 for d in diffs):>3} / {sum(abs(d) <= 0.01 for d in diffs):>2} / {sum(d < -0.01 for d in diffs):>2}")
    print("With so few queries these differences are indications, not proof.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "eval_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["system", "query_id", "query", "relevant", "P@5", "P@10", "R@10", "AP", "RR"])
        for s in systems:
            for q in usable:
                m = results[s.name][q["id"]]
                writer.writerow([s.name, q["id"], q["query"], len(relevant[q["id"]])] + [round(m[k], 4) for k in ("P@5", "P@10", "R@10", "AP", "RR")])
    print(f"\nsaved {config.REPORTS_DIR / 'eval_results.csv'} and {JUDGMENTS}")


if __name__ == "__main__":
    main()
