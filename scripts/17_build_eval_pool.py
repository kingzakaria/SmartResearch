"""Search evaluation, step 1: build the sheet of decisions you have to judge.
Run from the project root: python scripts/17_build_eval_pool.py

Method (pooling, the standard way to evaluate a search engine without judging 2,000 decisions per query):
  * data/eval/queries.json holds the test queries (edit it freely: keep a few for each chamber).
  * for each query, the 10 first results of EVERY system (word, light, root, light + chamber routing) are put in a common pool;
  * you judge the pool in reports/eval_pool.csv: 1 in the column "relevant" if the decision is mainly about what the
    query asks, 0 if not. The rows are shuffled, and the rank or the system that found a decision is not shown, so the
    judgment does not depend on them.
Judgments already given (data/eval/judgments.csv) are filled in again, so adding a query never loses your work.
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.evaluation.judgments import read_compact  # noqa: E402
from src.evaluation.systems import build_systems  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from src.search.highlight import reading_text  # noqa: E402

POOL_DEPTH = 10
POOL_CSV = config.REPORTS_DIR / "eval_pool.csv"
JUDGMENTS = config.EVAL_DIR / "judgments.csv"


def main() -> None:
    queries = json.loads((config.EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    by_id = {str(d["doc_id"]): d for d in docs}
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    exclude = {str(d) for d in dup.get("drop", [])}
    engine = SearchEngine.from_config(config)
    systems = build_systems(config, docs, exclude, engine)
    print(f"{len(queries)} queries, {len(docs) - len(exclude)} decisions, systems: {', '.join(s.name for s in systems)}")
    if engine.router is None:
        print("(no models/naive_bayes.json: 'light+routing' is left out; run scripts/15_train_naive_bayes.py)")

    previous = read_compact(JUDGMENTS)
    rng = random.Random(config.RANDOM_SEED)
    rows = []
    for q in queries:
        pool = []
        for system in systems:
            for doc_id in system.rank(q["query"], POOL_DEPTH):
                if doc_id not in pool:
                    pool.append(doc_id)
        if len(pool) < 3:
            print(f"  warning: {q['id']} {q['query']!r} finds almost nothing ({len(pool)} decisions): change the query")
        rng.shuffle(pool)
        for doc_id in pool:
            d = by_id[doc_id]
            text = reading_text(d["body"], d.get("subject"))
            rows.append({"query_id": q["id"], "query": q["query"], "doc_id": doc_id,
                         "chamber": d.get("chamber", "real_estate"), "decision_no": d.get("decision_no") or "",
                         "subject": d.get("subject") or "", "text_start": " ".join(text.split())[:350],
                         "relevant": previous.get((q["id"], doc_id), "")})

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    done = sum(1 for r in rows if r["relevant"] != "")
    print(f"\n{len(rows)} decisions to judge ({done} already judged) saved to {POOL_CSV}")
    print("Open it in Excel, fill the last column with 1 (relevant) or 0 (not relevant), then save as 'CSV UTF-8'.")
    print("Relevant = the decision is MAINLY about what the query asks (the subject and the first lines usually tell).")
    print("Next: python scripts/18_evaluate_search.py")


if __name__ == "__main__":
    main()
